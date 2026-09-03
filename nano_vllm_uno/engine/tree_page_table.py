"""Build FA3 tree-suffix page tables directly from parent links."""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_tree_page_table_kernel(
    parents,
    depths,
    block_tables,
    kv_seqlens,
    prefix_seqlens,
    positions,
    slot_mapping,
    page_table,
    cache_seqlens,
    Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PARENT_BATCH_STRIDE: tl.constexpr,
    PARENT_NODE_STRIDE: tl.constexpr,
    DEPTH_BATCH_STRIDE: tl.constexpr,
    DEPTH_NODE_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    batch = tl.program_id(0)
    nodes = tl.arange(0, BLOCK_Q)
    valid = nodes < Q
    row = batch * Q + nodes
    depth = tl.load(
        depths + batch * DEPTH_BATCH_STRIDE + nodes * DEPTH_NODE_STRIDE,
        mask=valid,
        other=0,
    )
    current = nodes.to(tl.int64)
    prefix = tl.load(kv_seqlens + batch) - Q

    tl.store(prefix_seqlens + batch, prefix)
    tl.store(positions + row, prefix + depth, mask=valid)
    logical_slot = prefix + nodes
    block_index = logical_slot // BLOCK_SIZE
    block_offset = logical_slot % BLOCK_SIZE
    physical_block = tl.load(
        block_tables + batch * BLOCK_TABLE_STRIDE + block_index,
        mask=valid,
    )
    tl.store(
        slot_mapping + row,
        physical_block * BLOCK_SIZE + block_offset,
        mask=valid,
    )
    tl.store(cache_seqlens + row, depth + 1, mask=valid)

    # Write each path in root-to-self order. FA3 only reads the first
    # depth + 1 entries, so unused columns need no initialization.
    for step in range(Q):
        active = valid & (step <= depth)
        current_safe = tl.where(active, current, 0)
        ancestor_slot = prefix + current_safe
        ancestor_block_index = ancestor_slot // BLOCK_SIZE
        ancestor_block_offset = ancestor_slot % BLOCK_SIZE
        ancestor_block = tl.load(
            block_tables
            + batch * BLOCK_TABLE_STRIDE
            + ancestor_block_index,
            mask=active,
        )
        tl.store(
            page_table + row * PAGE_STRIDE + depth - step,
            ancestor_block * BLOCK_SIZE + ancestor_block_offset,
            mask=active,
        )
        current = tl.load(
            parents
            + batch * PARENT_BATCH_STRIDE
            + current_safe * PARENT_NODE_STRIDE,
            mask=active & (step < depth),
            other=0,
        )


def build_tree_page_table(
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    block_tables: torch.Tensor,
    kv_seqlens: torch.Tensor,
    prefix_seqlens: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_size: int,
) -> None:
    """Fill all tree position and FA3 suffix metadata with one kernel launch."""
    if parent_indices.shape != depths.shape:
        raise ValueError("tree parents and depths must share [B, Q]")
    batch_size, tree_size = parent_indices.shape
    if block_tables.size(0) != batch_size:
        raise ValueError("tree block-table batch must equal B")
    if positions.numel() != batch_size * tree_size:
        raise ValueError("tree position count must equal B * Q")
    if slot_mapping.numel() != batch_size * tree_size:
        raise ValueError("tree slot count must equal B * Q")
    if page_table.shape[0] != batch_size * tree_size:
        raise ValueError("tree page-table row count must equal B * Q")
    if page_table.size(1) < tree_size:
        raise ValueError("tree page table must have at least Q columns")

    if not parent_indices.is_cuda:
        prefix_seqlens.copy_(kv_seqlens[:batch_size] - tree_size)
        for batch in range(batch_size):
            prefix = int(prefix_seqlens[batch])
            node_slots = []
            for node in range(tree_size):
                logical = prefix + node
                physical = (
                    int(block_tables[batch, logical // block_size]) * block_size
                    + logical % block_size
                )
                node_slots.append(physical)
                positions[batch * tree_size + node] = (
                    prefix + int(depths[batch, node])
                )
                slot_mapping[batch * tree_size + node] = physical
            for node in range(tree_size):
                path = []
                current = node
                while current >= 0:
                    path.append(node_slots[current])
                    current = int(parent_indices[batch, current])
                path.reverse()
                cache_seqlens[batch * tree_size + node] = len(path)
                page_table[batch * tree_size + node, : len(path)] = torch.tensor(
                    path,
                    dtype=page_table.dtype,
                    device=page_table.device,
                )
        return

    _build_tree_page_table_kernel[(batch_size,)](
        parent_indices,
        depths,
        block_tables,
        kv_seqlens,
        prefix_seqlens,
        positions,
        slot_mapping,
        page_table,
        cache_seqlens,
        Q=tree_size,
        BLOCK_SIZE=block_size,
        BLOCK_TABLE_STRIDE=block_tables.stride(0),
        PAGE_STRIDE=page_table.stride(0),
        PARENT_BATCH_STRIDE=parent_indices.stride(0),
        PARENT_NODE_STRIDE=parent_indices.stride(1),
        DEPTH_BATCH_STRIDE=depths.stride(0),
        DEPTH_NODE_STRIDE=depths.stride(1),
        BLOCK_Q=triton.next_power_of_2(tree_size),
        num_warps=1,
        num_stages=1,
    )
