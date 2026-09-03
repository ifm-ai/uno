"""Fused traversal of target samples through a fixed draft tree."""

import torch
import triton
import triton.language as tl


@triton.jit
def _walk_tree_kernel(
    tree_tokens,
    parents,
    target_tokens,
    committed,
    accepted_nodes,
    lengths,
    pad_token_id,
    Q: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    MAX_CACHED: tl.constexpr,
    MAX_COMMITTED: tl.constexpr,
    TOKEN_BATCH_STRIDE: tl.constexpr,
    TOKEN_NODE_STRIDE: tl.constexpr,
    PARENT_BATCH_STRIDE: tl.constexpr,
    PARENT_NODE_STRIDE: tl.constexpr,
    TARGET_BATCH_STRIDE: tl.constexpr,
    TARGET_NODE_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    batch = tl.program_id(0)
    nodes = tl.arange(0, BLOCK_Q)
    valid_nodes = nodes < Q
    outputs = tl.arange(0, BLOCK_OUT)

    committed_row = committed + batch * MAX_COMMITTED
    accepted_row = accepted_nodes + batch * MAX_CACHED
    tl.store(
        committed_row + outputs,
        pad_token_id,
        mask=outputs < MAX_COMMITTED,
    )
    tl.store(accepted_row + outputs, -1, mask=outputs < MAX_CACHED)

    root = tl.load(tree_tokens + batch * TOKEN_BATCH_STRIDE)
    tl.store(committed_row, root)
    tl.store(accepted_row, 0)
    current = tl.cast(0, tl.int64)
    active = tl.cast(1, tl.int1)
    length = tl.cast(1, tl.int64)

    for step in range(1, MAX_COMMITTED):
        sampled = tl.load(
            target_tokens
            + batch * TARGET_BATCH_STRIDE
            + current * TARGET_NODE_STRIDE,
        )
        tl.store(committed_row + step, sampled, mask=active)
        length = tl.where(active, step + 1, length)

        candidate_parents = tl.load(
            parents
            + batch * PARENT_BATCH_STRIDE
            + nodes * PARENT_NODE_STRIDE,
            mask=valid_nodes,
            other=-2,
        )
        candidate_tokens = tl.load(
            tree_tokens
            + batch * TOKEN_BATCH_STRIDE
            + nodes * TOKEN_NODE_STRIDE,
            mask=valid_nodes,
            other=-1,
        )
        matches = (
            valid_nodes
            & active
            & (candidate_parents == current)
            & (candidate_tokens == sampled)
        )
        child = tl.min(tl.where(matches, nodes, Q), axis=0).to(tl.int64)
        if step <= MAX_DEPTH:
            can_descend = active & (child < Q)
            tl.store(accepted_row + step, child, mask=can_descend)
            current = tl.where(can_descend, child, current)
            active = can_descend
        else:
            active = tl.cast(0, tl.int1)

    tl.store(lengths + batch, length)


def walk_tree(
    tree_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    max_depth: int,
    pad_token_id: int,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk every request's accepted branch with one GPU kernel."""
    batch_size, tree_size = tree_token_ids.shape
    max_cached_tokens = max_depth + 1
    max_committed_tokens = max_cached_tokens + 1
    if out is None:
        committed = torch.empty(
            (batch_size, max_committed_tokens),
            dtype=tree_token_ids.dtype,
            device=tree_token_ids.device,
        )
        accepted_nodes = torch.empty(
            (batch_size, max_cached_tokens),
            dtype=torch.long,
            device=tree_token_ids.device,
        )
        lengths = torch.empty(
            batch_size,
            dtype=torch.long,
            device=tree_token_ids.device,
        )
    else:
        committed, accepted_nodes, lengths = out
    _walk_tree_kernel[(batch_size,)](
        tree_token_ids,
        parent_indices,
        target_tokens,
        committed,
        accepted_nodes,
        lengths,
        pad_token_id,
        Q=tree_size,
        MAX_DEPTH=max_depth,
        MAX_CACHED=max_cached_tokens,
        MAX_COMMITTED=max_committed_tokens,
        TOKEN_BATCH_STRIDE=tree_token_ids.stride(0),
        TOKEN_NODE_STRIDE=tree_token_ids.stride(1),
        PARENT_BATCH_STRIDE=parent_indices.stride(0),
        PARENT_NODE_STRIDE=parent_indices.stride(1),
        TARGET_BATCH_STRIDE=target_tokens.stride(0),
        TARGET_NODE_STRIDE=target_tokens.stride(1),
        BLOCK_Q=triton.next_power_of_2(tree_size),
        BLOCK_OUT=triton.next_power_of_2(max_committed_tokens),
        num_warps=1,
        num_stages=1,
    )
    return committed, accepted_nodes, lengths
