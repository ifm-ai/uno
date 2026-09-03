import unittest

import torch

from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.engine.two_pass_decoding import TwoPassDecoder
from nano_vllm_uno.sampling_params import SamplingParams


class TreeDecoderIntegrationTests(unittest.TestCase):
    def test_cycle_builds_walks_and_compacts_a_branch(self):
        vocab_size = 32
        observed = {}

        def run_block(seqs, tokens_batch, lora_mask_batch=None):
            self.assertEqual(tokens_batch.shape, (1, 3))
            seqs[0].num_cached_tokens += 3
            logits = torch.full((1, 3, vocab_size), -100.0)
            logits[0, 0, 5] = 10.0
            logits[0, 1, 10] = 2.0
            logits[0, 1, 11] = 1.8
            logits[0, 2, 20] = 4.0
            logits[0, 2, 21] = 0.0
            return logits

        def run_tree(seqs, tree_tokens, depths, parents):
            observed["tokens"] = tree_tokens.clone()
            observed["depths"] = depths.clone()
            observed["parents"] = parents.clone()
            seqs[0].num_cached_tokens += tree_tokens.size(1)

            root_alt = (
                depths[0].eq(1)
                & tree_tokens[0].eq(11)
            ).nonzero(as_tuple=False).item()
            branch_child = (
                depths[0].eq(2)
                & tree_tokens[0].eq(20)
                & parents[0].eq(root_alt)
            ).nonzero(as_tuple=False).item()

            logits = torch.full(
                (1, tree_tokens.size(1), vocab_size),
                -100.0,
            )
            logits[:, :, 29] = 1.0
            logits[0, 0, 11] = 10.0
            logits[0, root_alt, 20] = 10.0
            logits[0, branch_child, 30] = 10.0
            return logits

        def compact_tree_kv(seqs, accepted_nodes, cache_lengths, tree_size):
            observed["accepted"] = accepted_nodes.clone()
            observed["cache_lengths"] = cache_lengths.clone()
            write_start = seqs[0].num_cached_tokens - tree_size
            seqs[0].num_cached_tokens = write_start + int(cache_lengths[0])

        decoder = TwoPassDecoder(
            run_block=run_block,
            run_tree=run_tree,
            compact_tree_kv=compact_tree_kv,
            eos_token_id=31,
            pad_token_id=0,
            vocab_size=vocab_size,
            device=torch.device("cpu"),
            tree_verify_size=6,
            tree_candidate_top_k=2,
        )
        seq = Sequence([1, 2])
        seq.num_cached_tokens = 1
        params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            mask_token_id=3,
            noise_mode="mask",
            diffusion_block_size=3,
        )

        result = decoder.run_cycle([seq], params)

        self.assertEqual(result, [[5, 11, 20, 30]])
        self.assertEqual(seq.token_ids, [1, 2, 5, 11, 20, 30])
        self.assertEqual(seq.num_cached_tokens, len(seq) - 1)
        self.assertEqual(observed["tokens"][0, :3].tolist(), [5, 10, 20])
        self.assertEqual(observed["depths"][0, :3].tolist(), [0, 1, 2])
        self.assertEqual(observed["accepted"][0, :3].tolist()[0], 0)
        self.assertEqual(observed["cache_lengths"].tolist(), [3])
        self.assertEqual(seq.stats["forwards"], 2)
        self.assertEqual(seq.stats["accepts"], 4)
        self.assertEqual(seq.stats["lookaheads"], 1)


if __name__ == "__main__":
    unittest.main()
