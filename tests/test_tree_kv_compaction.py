import unittest

import torch

from nano_vllm_uno.engine.model_runner import ModelRunner
from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.engine.tree_kv_copy import build_tree_kv_slots


class TreeKvCompactionTests(unittest.TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_slot_builder_matches_reference(self):
        accepted = torch.tensor(
            [[0, 3, 5, -1], [0, 2, -1, -1]],
            device="cuda",
        )
        cache_lengths = torch.tensor([3, 2], device="cuda")
        prefix_lengths = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
        block_tables = torch.tensor(
            [[1, 4, 0], [3, 2, 5]],
            dtype=torch.int32,
            device="cuda",
        )
        source = torch.empty(8, dtype=torch.long, device="cuda")
        destination = torch.empty_like(source)
        build_tree_kv_slots(
            accepted,
            cache_lengths,
            prefix_lengths,
            block_tables,
            source,
            destination,
            kv_block_size=4,
        )
        self.assertEqual(source.cpu().tolist(), [7, 18, 0, 18, 15, 9, 9, 10])
        self.assertEqual(
            destination.cpu().tolist(),
            [7, 16, 17, 18, 15, 8, 9, 10],
        )

    def test_branch_slots_are_compacted_before_final_frontier_rollback(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 4
        runner.kv_cache = torch.arange(
            2 * 1 * 2 * 4,
            dtype=torch.float32,
        ).view(2, 1, 2, 4, 1, 1)

        seq = Sequence([7, 8])
        seq.block_table = [0, 1]
        # Tree Q=5 was written at logical slots [2, 7).
        seq.num_cached_tokens = 7
        original = runner.kv_cache.clone().view(2, 1, 8, 1, 1)

        runner._compact_tree_kv(
            [seq],
            accepted_node_indices=torch.tensor([[0, 3, 4]]),
            cache_lengths=torch.tensor([3]),
            tree_size=5,
        )

        compacted = runner.kv_cache.view(2, 1, 8, 1, 1)
        torch.testing.assert_close(compacted[:, :, 2], original[:, :, 2])
        torch.testing.assert_close(compacted[:, :, 3], original[:, :, 5])
        torch.testing.assert_close(compacted[:, :, 4], original[:, :, 6])
        # Compaction stays device-resident. The cycle's existing final payload
        # synchronization supplies the committed length used for this rollback.
        self.assertEqual(seq.num_cached_tokens, 7)
        seq.rollback_kv_to(5)
        self.assertEqual(seq.num_cached_tokens, 5)

    def test_fixed_shape_compaction_matches_path_gather(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 4
        runner.kv_cache = torch.arange(
            2 * 2 * 6 * 4 * 2,
            dtype=torch.float32,
        ).view(2, 2, 6, 4, 1, 2)

        seqs = [Sequence([1, 2, 3]), Sequence([4, 5, 6])]
        seqs[0].block_table = [1, 4, 0]
        seqs[1].block_table = [3, 2, 5]
        # Q=6 occupies logical positions [3, 9).
        for seq in seqs:
            seq.num_cached_tokens = 9

        accepted = torch.tensor(
            [
                [0, 3, 5, -1],
                [0, 2, -1, -1],
            ]
        )
        cache_lengths = torch.tensor([3, 2])
        original = runner.kv_cache.clone().view(2, 2, 24, 1, 2)

        def physical(seq, logical_position):
            block, offset = divmod(logical_position, runner.block_size)
            return seq.block_table[block] * runner.block_size + offset

        expected = original.clone()
        for batch_idx, seq in enumerate(seqs):
            prefix = seq.num_cached_tokens - 6
            for path_idx in range(int(cache_lengths[batch_idx])):
                source = physical(seq, prefix + int(accepted[batch_idx, path_idx]))
                destination = physical(seq, prefix + path_idx)
                expected[:, :, destination] = original[:, :, source]

        runner._compact_tree_kv(
            seqs,
            accepted_node_indices=accepted,
            cache_lengths=cache_lengths,
            tree_size=6,
        )
        actual = runner.kv_cache.view(2, 2, 24, 1, 2)
        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_tiled_copy_matches_path_gather(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 4
        runner.kv_cache = torch.arange(
            2 * 2 * 6 * 4 * 2,
            dtype=torch.bfloat16,
            device="cuda",
        ).view(2, 2, 6, 4, 1, 2)

        seqs = [Sequence([1, 2, 3]), Sequence([4, 5, 6])]
        seqs[0].block_table = [1, 4, 0]
        seqs[1].block_table = [3, 2, 5]
        for seq in seqs:
            seq.num_cached_tokens = 9

        accepted = torch.tensor(
            [[0, 3, 5, -1], [0, 2, -1, -1]],
            device="cuda",
        )
        cache_lengths = torch.tensor([3, 2], device="cuda")
        original = runner.kv_cache.clone().view(2, 2, 24, 1, 2)
        expected = original.clone()
        for batch_idx, seq in enumerate(seqs):
            prefix = seq.num_cached_tokens - 6
            for path_idx in range(int(cache_lengths[batch_idx].item())):
                src_pos = prefix + int(accepted[batch_idx, path_idx].item())
                dst_pos = prefix + path_idx
                src_block, src_offset = divmod(src_pos, runner.block_size)
                dst_block, dst_offset = divmod(dst_pos, runner.block_size)
                source = seq.block_table[src_block] * runner.block_size + src_offset
                destination = (
                    seq.block_table[dst_block] * runner.block_size + dst_offset
                )
                expected[:, :, destination] = original[:, :, source]

        runner._compact_tree_kv(
            seqs,
            accepted_node_indices=accepted,
            cache_lengths=cache_lengths,
            tree_size=6,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            runner.kv_cache.view(2, 2, 24, 1, 2),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
