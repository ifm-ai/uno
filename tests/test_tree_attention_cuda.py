import unittest

import torch

from nano_vllm_uno.engine.tree_page_table import build_tree_page_table
from nano_vllm_uno.layers.attention import (
    _call_with_kvcache,
    _flash_tree_decode_attention,
    _load_attention_backend,
    _torch_tree_decode_attention,
)


_FA3_WITH_KVCACHE = None
if torch.cuda.is_available():
    try:
        _, _, _FA3_WITH_KVCACHE = _load_attention_backend("fa3")
    except ImportError:
        pass


def _fa3_with_kvcache(
    q,
    k_cache,
    v_cache,
    *,
    block_table,
    **kwargs,
):
    kwargs["page_table"] = block_table
    return _call_with_kvcache(
        _FA3_WITH_KVCACHE,
        q,
        k_cache,
        v_cache,
        kwargs,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and _FA3_WITH_KVCACHE is not None,
    "requires CUDA and FlashAttention 3",
)
class TreeAttentionCudaTests(unittest.TestCase):

    def test_fa3_cascade_matches_dense_tree_mask(self):
        torch.manual_seed(11)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        batch_size = 2
        tree_size = 4
        block_size = 256
        num_heads = 4
        num_kv_heads = 2
        head_dim = 64
        prefix_lengths = torch.tensor([7, 11], dtype=torch.int32, device=device)
        total_lengths = tuple(
            int(length) + tree_size for length in prefix_lengths.cpu().tolist()
        )

        q = torch.randn(
            batch_size * tree_size,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        k_cache = torch.randn(
            batch_size,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        v_cache = torch.randn_like(k_cache)
        block_tables = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(1)
        tree_mask = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
            ],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        suffix_page_table = torch.zeros(
            batch_size * tree_size,
            tree_size,
            dtype=torch.int32,
            device=device,
        )
        parents = torch.tensor(
            [-1, 0, 0, 1],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)
        depths = torch.tensor(
            [0, 1, 1, 2],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)
        built_prefix_lengths = torch.empty_like(prefix_lengths)
        positions = torch.empty(
            batch_size * tree_size,
            dtype=torch.long,
            device=device,
        )
        node_slots = torch.empty(
            batch_size * tree_size,
            dtype=torch.int32,
            device=device,
        )
        suffix_cache_seqlens = torch.empty(
            batch_size * tree_size,
            dtype=torch.int32,
            device=device,
        )
        build_tree_page_table(
            parents,
            depths,
            block_tables,
            prefix_lengths + tree_size,
            built_prefix_lengths,
            positions,
            node_slots,
            suffix_page_table,
            suffix_cache_seqlens,
            block_size,
        )
        torch.testing.assert_close(built_prefix_lengths, prefix_lengths)

        scale = head_dim**-0.5
        actual = _flash_tree_decode_attention(
            _fa3_with_kvcache,
            q,
            k_cache,
            v_cache,
            block_tables,
            built_prefix_lengths,
            suffix_page_table,
            suffix_cache_seqlens,
            tree_size,
            scale,
        )
        expected = _torch_tree_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            tree_mask,
            total_lengths,
            tree_size,
            num_heads,
            head_dim,
            scale,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_fused_suffix_matches_dense_at_production_shape(self):
        torch.manual_seed(12)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        tree_size = 32
        block_size = 256
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        prefix_lengths = torch.tensor([71], dtype=torch.int32, device=device)
        parents_list = [-1] + [(node - 1) // 2 for node in range(1, tree_size)]
        depths_list = [0] * tree_size
        tree_mask = torch.zeros(tree_size, tree_size, dtype=torch.bool)
        for node in range(tree_size):
            current = node
            while current >= 0:
                tree_mask[node, current] = True
                depths_list[node] += current != 0
                current = parents_list[current]

        q = torch.randn(
            tree_size,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        k_cache = torch.randn(
            1,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        v_cache = torch.randn_like(k_cache)
        block_tables = torch.zeros((1, 1), dtype=torch.int32, device=device)
        parents = torch.tensor(
            parents_list,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        depths = torch.tensor(
            depths_list,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        suffix_page_table = torch.empty(
            tree_size,
            tree_size,
            dtype=torch.int32,
            device=device,
        )
        built_prefix_lengths = torch.empty_like(prefix_lengths)
        positions = torch.empty(tree_size, dtype=torch.long, device=device)
        node_slots = torch.empty(tree_size, dtype=torch.int32, device=device)
        suffix_cache_seqlens = torch.empty(
            tree_size,
            dtype=torch.int32,
            device=device,
        )
        build_tree_page_table(
            parents,
            depths,
            block_tables,
            prefix_lengths + tree_size,
            built_prefix_lengths,
            positions,
            node_slots,
            suffix_page_table,
            suffix_cache_seqlens,
            block_size,
        )

        scale = head_dim**-0.5
        actual = _flash_tree_decode_attention(
            _fa3_with_kvcache,
            q,
            k_cache,
            v_cache,
            block_tables,
            built_prefix_lengths,
            suffix_page_table,
            suffix_cache_seqlens,
            tree_size,
            scale,
        )
        expected = _torch_tree_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            tree_mask.unsqueeze(0).to(device),
            (int(prefix_lengths[0]) + tree_size,),
            tree_size,
            num_heads,
            head_dim,
            scale,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
