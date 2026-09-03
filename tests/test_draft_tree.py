import math
import unittest

import torch

from nano_vllm_uno.engine.draft_tree import (
    ancestor_mask_from_parents,
    build_draft_tree_batch,
    build_tree_from_candidates,
    walk_tree,
)


class DraftTreeBuilderTests(unittest.TestCase):

    def test_best_first_can_choose_shallow_branch_over_top1_continuation(self):
        tree = build_tree_from_candidates(
            root_token=5,
            top_tokens=((10, 11), (20, 21)),
            top_log_probs=(
                (math.log(0.6), math.log(0.4)),
                (math.log(0.51), math.log(0.49)),
            ),
            max_nodes=3,
        )

        # The root alternative has mass 0.4, which is greater than the top-1
        # continuation's mass 0.6 * 0.51 = 0.306.
        self.assertEqual(tree.token_ids, (5, 10, 11))
        self.assertEqual(tree.parent_indices, (-1, 0, 0))
        self.assertEqual(tree.depths, (0, 1, 1))

    def test_budget_may_be_shorter_than_draft_horizon(self):
        tree = build_tree_from_candidates(
            root_token=5,
            top_tokens=((10, 11), (20, 21), (30, 31)),
            top_log_probs=(
                (math.log(0.6), math.log(0.4)),
                (math.log(0.7), math.log(0.3)),
                (math.log(0.8), math.log(0.2)),
            ),
            max_nodes=2,
        )
        self.assertEqual(tree.token_ids, (5, 10))
        self.assertEqual(tree.parent_indices, (-1, 0))

    def test_duplicate_candidates_do_not_duplicate_children(self):
        tree = build_tree_from_candidates(
            root_token=1,
            top_tokens=((10, 10, 11),),
            top_log_probs=((-0.1, -0.2, -0.3),),
            max_nodes=3,
        )
        self.assertEqual(tree.token_ids, (1, 10, 11))
        self.assertEqual(tree.parent_indices, (-1, 0, 0))

    def test_log_space_scores_do_not_underflow(self):
        tree = build_tree_from_candidates(
            root_token=1,
            top_tokens=((10, 11), (20, 21)),
            top_log_probs=((-1000.0, -1001.0), (-1000.0, -1002.0)),
            max_nodes=4,
        )
        self.assertEqual(tree.num_nodes, 4)
        self.assertTrue(all(not math.isnan(score) for score in tree.log_masses))

    def test_batch_builder_uses_full_softmax_log_probabilities(self):
        root = torch.tensor([5])
        logits = torch.tensor(
            [
                [
                    [0.0, 5.0, 4.0, 1.0],
                    [0.0, 1.0, 5.0, 4.0],
                ]
            ]
        )
        batch = build_draft_tree_batch(
            root,
            logits,
            max_nodes=5,
            candidate_top_k=3,
            temperature=1.0,
        )
        self.assertEqual(batch.token_ids.shape, (1, 5))
        self.assertEqual(batch.token_ids[0, :3].tolist(), [5, 1, 2])
        first_log_prob = 5.0 - torch.logsumexp(logits[0, 0], dim=0).item()
        self.assertAlmostEqual(batch.log_masses[0, 1].item(), first_log_prob, places=6)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_builder_matches_cpu_best_first_reference(self):
        shapes = (
            (1, 2, 3, 5, 97),
            (3, 4, 4, 12, 97),
            (2, 5, 6, 19, 97),
            (1, 16, 16, 32, 151936),
        )
        for seed, shape in enumerate(shapes):
            batch_size, num_depths, candidate_top_k, max_nodes, vocab_size = shape
            generator = torch.Generator().manual_seed(seed)
            root = torch.randint(0, vocab_size, (batch_size,), generator=generator)
            logits = torch.randn(
                batch_size,
                num_depths,
                vocab_size,
                generator=generator,
            )
            expected = build_draft_tree_batch(
                root,
                logits,
                max_nodes=max_nodes,
                candidate_top_k=candidate_top_k,
                temperature=0.7,
            )
            actual = build_draft_tree_batch(
                root.cuda(),
                logits.cuda(),
                max_nodes=max_nodes,
                candidate_top_k=candidate_top_k,
                temperature=0.7,
            )
            torch.testing.assert_close(actual.token_ids.cpu(), expected.token_ids)
            torch.testing.assert_close(
                actual.parent_indices.cpu(),
                expected.parent_indices,
            )
            torch.testing.assert_close(actual.depths.cpu(), expected.depths)
            torch.testing.assert_close(
                actual.log_masses.cpu(),
                expected.log_masses,
                rtol=1e-5,
                atol=1e-5,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_builder_reuses_workspace(self):
        root = torch.tensor([5], device="cuda")
        logits = torch.randn(1, 4, 97, device="cuda")
        workspace = {}
        first = build_draft_tree_batch(
            root,
            logits,
            max_nodes=12,
            candidate_top_k=4,
            temperature=0.7,
            workspace=workspace,
        )
        expected = (
            first.token_ids.clone(),
            first.parent_indices.clone(),
            first.depths.clone(),
            first.log_masses.clone(),
        )
        pointers = {name: tensor.data_ptr() for name, tensor in workspace.items()}
        second = build_draft_tree_batch(
            root,
            logits,
            max_nodes=12,
            candidate_top_k=4,
            temperature=0.7,
            workspace=workspace,
        )
        self.assertEqual(
            pointers,
            {name: tensor.data_ptr() for name, tensor in workspace.items()},
        )
        for actual, reference in zip(
            (
                second.token_ids,
                second.parent_indices,
                second.depths,
                second.log_masses,
            ),
            expected,
        ):
            torch.testing.assert_close(actual, reference)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_builder_handles_sliced_draft_logits(self):
        root = torch.tensor([5, 6])
        full_logits = torch.randn(2, 5, 97)
        logits = full_logits[:, 1:, :]
        cuda_logits = full_logits.cuda()[:, 1:, :]
        self.assertFalse(cuda_logits.is_contiguous())
        expected = build_draft_tree_batch(
            root,
            logits,
            max_nodes=12,
            candidate_top_k=4,
            temperature=0.7,
        )
        actual = build_draft_tree_batch(
            root.cuda(),
            cuda_logits,
            max_nodes=12,
            candidate_top_k=4,
            temperature=0.7,
            workspace={},
        )
        torch.testing.assert_close(actual.token_ids.cpu(), expected.token_ids)
        torch.testing.assert_close(
            actual.parent_indices.cpu(),
            expected.parent_indices,
        )
        torch.testing.assert_close(actual.depths.cpu(), expected.depths)
        torch.testing.assert_close(
            actual.log_masses.cpu(),
            expected.log_masses,
            rtol=1e-5,
            atol=1e-5,
        )


class DraftTreeMetadataTests(unittest.TestCase):

    def test_attention_mask_contains_only_self_and_ancestors(self):
        parents = torch.tensor([[-1, 0, 1, 0, 3]])
        actual = ancestor_mask_from_parents(parents)[0]
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(actual, expected)

    def test_target_walk_follows_branch_and_emits_uncached_lookahead(self):
        tree_tokens = torch.tensor(
            [
                [5, 10, 20, 11, 21],
                [6, 10, 20, 11, 21],
            ]
        )
        parents = torch.tensor(
            [
                [-1, 0, 1, 0, 3],
                [-1, 0, 1, 0, 3],
            ]
        )
        # Row zero walks 0 -> 3 -> 4, then emits 99. Row one misses at root.
        target_tokens = torch.tensor(
            [
                [11, 0, 0, 21, 99],
                [42, 0, 0, 0, 0],
            ]
        )
        committed, accepted, lengths = walk_tree(
            tree_tokens,
            parents,
            target_tokens,
            max_depth=2,
            pad_token_id=0,
        )
        self.assertEqual(committed[0].tolist(), [5, 11, 21, 99])
        self.assertEqual(accepted[0].tolist(), [0, 3, 4])
        self.assertEqual(lengths[0].item(), 4)
        self.assertEqual(committed[1].tolist(), [6, 42, 0, 0])
        self.assertEqual(accepted[1].tolist(), [0, -1, -1])
        self.assertEqual(lengths[1].item(), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_target_walk_matches_cpu_reference(self):
        tree_tokens = torch.tensor(
            [
                [5, 10, 20, 11, 21],
                [6, 10, 20, 11, 21],
            ]
        )
        parents = torch.tensor(
            [
                [-1, 0, 1, 0, 3],
                [-1, 0, 1, 0, 3],
            ]
        )
        target_tokens = torch.tensor(
            [
                [11, 0, 0, 21, 99],
                [42, 0, 0, 0, 0],
            ]
        )
        expected = walk_tree(
            tree_tokens,
            parents,
            target_tokens,
            max_depth=2,
            pad_token_id=0,
        )
        # Exercise non-contiguous [B, Q] inputs as well as traversal semantics.
        actual = walk_tree(
            torch.stack((tree_tokens, tree_tokens), dim=-1)[..., 0].cuda(),
            torch.stack((parents, parents), dim=-1)[..., 0].cuda(),
            torch.stack((target_tokens, target_tokens), dim=-1)[..., 0].cuda(),
            max_depth=2,
            pad_token_id=0,
        )
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor.cpu(), expected_tensor)


if __name__ == "__main__":
    unittest.main()
