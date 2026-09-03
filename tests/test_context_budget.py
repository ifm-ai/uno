import unittest

from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.engine.two_pass_decoding import TwoPassDecoder
from nano_vllm_uno.utils.context_budget import (
    active_forward_reserve,
    resolve_completion_budget,
)


class ContextBudgetTests(unittest.TestCase):
    def test_linear_reserves_block_width(self):
        self.assertEqual(active_forward_reserve(16, None), 16)

    def test_tree_reserves_larger_verifier_width(self):
        self.assertEqual(active_forward_reserve(16, 60), 60)

    def test_budget_uses_remaining_context(self):
        self.assertEqual(
            resolve_completion_budget(
                prompt_tokens=1000,
                context_length=32768,
                reserve_tokens=16,
            ),
            31752,
        )

    def test_optional_global_cap(self):
        self.assertEqual(
            resolve_completion_budget(
                prompt_tokens=1000,
                context_length=32768,
                reserve_tokens=16,
                global_max_tokens=100,
            ),
            100,
        )

    def test_exhausted_context_returns_zero(self):
        self.assertEqual(
            resolve_completion_budget(
                prompt_tokens=32760,
                context_length=32768,
                reserve_tokens=16,
            ),
            0,
        )

    def test_block_commit_does_not_overshoot_per_request_limit(self):
        seq = Sequence([1, 2], max_completion_tokens=2)
        # Verification has staged KV for the entire proposed block.
        seq.num_cached_tokens = 6
        accepted = [[]]
        TwoPassDecoder._apply_committed_rows(
            [seq],
            accepted,
            [[10, 11, 12, 3]],
            block_len=4,
        )
        self.assertEqual(seq.completion_token_ids, [10, 11])
        self.assertEqual(accepted, [[10, 11]])
        self.assertEqual(seq.stats["accepts"], 2)
        self.assertEqual(seq.stats["forwards"], 2)

    def test_legacy_commit_without_request_limit_is_unchanged(self):
        seq = Sequence([1, 2])
        seq.num_cached_tokens = 6
        accepted = [[]]
        TwoPassDecoder._apply_committed_rows(
            [seq],
            accepted,
            [[10, 11, 12, 3]],
            block_len=4,
        )
        self.assertEqual(seq.completion_token_ids, [10, 11, 12])
        self.assertEqual(seq.stats["accepts"], 3)


if __name__ == "__main__":
    unittest.main()
