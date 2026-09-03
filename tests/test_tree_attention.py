import unittest

import torch

from nano_vllm_uno.layers.attention import (
    _torch_block_decode_attention,
    _torch_tree_decode_attention,
)


class TreeAttentionTests(unittest.TestCase):

    def test_chain_mask_matches_causal_block_attention(self):
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(4, 2, 3, generator=generator)
        k_cache = torch.randn(1, 8, 1, 3, generator=generator)
        v_cache = torch.randn(1, 8, 1, 3, generator=generator)
        block_tables = torch.tensor([[0]], dtype=torch.int32)
        kv_seqlens = torch.tensor([7], dtype=torch.int32)
        tree_mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))

        causal = _torch_block_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            kv_seqlens,
            seqlen_q=4,
            num_heads=2,
            head_dim=3,
            scale=3 ** -0.5,
        )
        chain = _torch_tree_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            tree_mask,
            (7,),
            seqlen_q=4,
            num_heads=2,
            head_dim=3,
            scale=3 ** -0.5,
        )
        torch.testing.assert_close(chain, causal)

    def test_siblings_do_not_attend_to_each_other(self):
        # Two prefix tokens followed by root, child A, and sibling B.
        q = torch.zeros(3, 1, 1)
        k_cache = torch.zeros(1, 8, 1, 1)
        v_cache = torch.zeros(1, 8, 1, 1)
        v_cache[0, :5, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        block_tables = torch.tensor([[0]], dtype=torch.int32)
        tree_mask = torch.tensor(
            [
                [
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 0, 1],
                ]
            ],
            dtype=torch.bool,
        )

        output = _torch_tree_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            tree_mask,
            (5,),
            seqlen_q=3,
            num_heads=1,
            head_dim=1,
            scale=1.0,
        )

        expected = torch.tensor([2.0, 2.5, 2.75]).view(3, 1, 1)
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
