"""GPU-native fixed-budget best-first draft-tree construction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_lse_partials_kernel(
    logits,
    partial_max,
    partial_sum,
    inverse_temperature,
    BATCH_STRIDE: tl.constexpr,
    DEPTH_STRIDE: tl.constexpr,
    VOCAB_STRIDE: tl.constexpr,
    NUM_DEPTHS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
):
    batch = tl.program_id(0)
    depth = tl.program_id(1)
    block = tl.program_id(2)
    row = batch * NUM_DEPTHS + depth
    offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    values = tl.load(
        logits
        + batch * BATCH_STRIDE
        + depth * DEPTH_STRIDE
        + offsets * VOCAB_STRIDE,
        mask=offsets < VOCAB_SIZE,
        other=-float("inf"),
    ).to(tl.float32)
    values *= inverse_temperature
    maximum = tl.max(values, axis=0)
    total = tl.sum(tl.exp(values - maximum), axis=0)
    output = row * NUM_BLOCKS + block
    tl.store(partial_max + output, maximum)
    tl.store(partial_sum + output, total)


@triton.jit
def _candidate_lse_finalize_kernel(
    top_values,
    partial_max,
    partial_sum,
    top_log_probs,
    inverse_temperature,
    K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    row = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_PARTIALS)
    block_max = tl.load(
        partial_max + row * NUM_BLOCKS + blocks,
        mask=blocks < NUM_BLOCKS,
        other=-float("inf"),
    )
    maximum = tl.max(block_max, axis=0)
    block_sum = tl.load(
        partial_sum + row * NUM_BLOCKS + blocks,
        mask=blocks < NUM_BLOCKS,
        other=0.0,
    )
    total = tl.sum(block_sum * tl.exp(block_max - maximum), axis=0)
    normalizer = maximum + tl.log(total)

    ranks = tl.arange(0, BLOCK_K)
    candidates = tl.load(
        top_values + row * K + ranks,
        mask=ranks < K,
    ).to(tl.float32)
    tl.store(
        top_log_probs + row * K + ranks,
        candidates * inverse_temperature - normalizer,
        mask=ranks < K,
    )


def build_candidate_log_probs_cuda(
    logits: torch.Tensor,
    top_values: torch.Tensor,
    top_log_probs: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    temperature: float,
) -> None:
    """Normalize BF16 top-k values without materializing FP32 vocabulary logits."""
    batch_size, num_depths, vocab_size = logits.shape
    num_rows = batch_size * num_depths
    candidate_top_k = top_values.size(-1)
    block_vocab = 8192
    num_blocks = triton.cdiv(vocab_size, block_vocab)
    inverse_temperature = (
        1.0 / float(temperature) if temperature > 0.0 else 1.0
    )
    _candidate_lse_partials_kernel[(batch_size, num_depths, num_blocks)](
        logits,
        partial_max,
        partial_sum,
        inverse_temperature,
        BATCH_STRIDE=logits.stride(0),
        DEPTH_STRIDE=logits.stride(1),
        VOCAB_STRIDE=logits.stride(-1),
        NUM_DEPTHS=num_depths,
        VOCAB_SIZE=vocab_size,
        NUM_BLOCKS=num_blocks,
        BLOCK_VOCAB=block_vocab,
        num_warps=4,
        num_stages=1,
    )
    _candidate_lse_finalize_kernel[(num_rows,)](
        top_values,
        partial_max,
        partial_sum,
        top_log_probs,
        inverse_temperature,
        K=candidate_top_k,
        NUM_BLOCKS=num_blocks,
        BLOCK_K=triton.next_power_of_2(candidate_top_k),
        BLOCK_PARTIALS=triton.next_power_of_2(num_blocks),
        num_warps=1,
        num_stages=1,
    )


@triton.jit
def _build_tree_kernel(
    root_tokens,
    top_token_ids,
    top_log_probs,
    output_tokens,
    output_parents,
    output_depths,
    output_log_masses,
    ROOT_BATCH_STRIDE: tl.constexpr,
    TOKEN_BATCH_STRIDE: tl.constexpr,
    TOKEN_DEPTH_STRIDE: tl.constexpr,
    TOKEN_RANK_STRIDE: tl.constexpr,
    PROB_BATCH_STRIDE: tl.constexpr,
    PROB_DEPTH_STRIDE: tl.constexpr,
    PROB_RANK_STRIDE: tl.constexpr,
    NUM_DEPTHS: tl.constexpr,
    K: tl.constexpr,
    Q: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    batch = tl.program_id(0)
    output_offset = batch * Q
    candidate_slots = tl.arange(0, BLOCK_CANDIDATES)
    candidate_parents = candidate_slots // K
    candidate_ranks = candidate_slots % K
    candidate_in_bounds = candidate_slots < Q * K
    used = tl.zeros((BLOCK_CANDIDATES,), dtype=tl.int1)

    tl.store(
        output_tokens + output_offset,
        tl.load(root_tokens + batch * ROOT_BATCH_STRIDE),
    )
    tl.store(output_parents + output_offset, -1)
    tl.store(output_depths + output_offset, 0)
    tl.store(output_log_masses + output_offset, 0.0)
    tl.debug_barrier()

    # Every selected node exposes K children. Re-scanning the at-most Q*K
    # implicit frontier is cheaper here than maintaining a device heap, and
    # preserves the CPU builder's exact global best-first policy.
    for node_index in range(1, Q):
        parent_valid = candidate_in_bounds & (candidate_parents < node_index)
        safe_parent = tl.where(parent_valid, candidate_parents, 0)
        parent_depth = tl.load(
            output_depths + output_offset + safe_parent,
            mask=parent_valid,
            other=NUM_DEPTHS,
        )
        parent_mass = tl.load(
            output_log_masses + output_offset + safe_parent,
            mask=parent_valid,
            other=-float("inf"),
        ).to(tl.float32)
        valid = parent_valid & (parent_depth < NUM_DEPTHS) & ~used
        safe_depth = tl.where(valid, parent_depth, 0)
        token = tl.load(
            top_token_ids
            + batch * TOKEN_BATCH_STRIDE
            + safe_depth * TOKEN_DEPTH_STRIDE
            + candidate_ranks * TOKEN_RANK_STRIDE,
            mask=valid,
            other=0,
        )
        log_prob = tl.load(
            top_log_probs
            + batch * PROB_BATCH_STRIDE
            + safe_depth * PROB_DEPTH_STRIDE
            + candidate_ranks * PROB_RANK_STRIDE,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        mass = parent_mass + log_prob
        child_depth = parent_depth + 1

        # Match heapq's tuple ordering:
        # higher mass, lower depth, lower rank, lower token, lower parent.
        best_mass = tl.max(tl.where(valid, mass, -float("inf")), axis=0)
        winner = valid & (mass == best_mass)
        best_depth = tl.min(tl.where(winner, child_depth, 1 << 30), axis=0)
        winner &= child_depth == best_depth
        best_rank = tl.min(tl.where(winner, candidate_ranks, 1 << 30), axis=0)
        winner &= candidate_ranks == best_rank
        best_token = tl.min(tl.where(winner, token, 1 << 30), axis=0)
        winner &= token == best_token
        best_parent = tl.min(
            tl.where(winner, candidate_parents, 1 << 30),
            axis=0,
        )
        winner &= candidate_parents == best_parent
        best_slot = tl.min(tl.where(winner, candidate_slots, 1 << 30), axis=0)

        selected_parent = best_slot // K
        selected_rank = best_slot % K
        selected_depth = tl.load(
            output_depths + output_offset + selected_parent,
        )
        selected_mass = tl.load(
            output_log_masses + output_offset + selected_parent,
        ).to(tl.float32) + tl.load(
            top_log_probs
            + batch * PROB_BATCH_STRIDE
            + selected_depth * PROB_DEPTH_STRIDE
            + selected_rank * PROB_RANK_STRIDE,
        ).to(tl.float32)
        selected_token = tl.load(
            top_token_ids
            + batch * TOKEN_BATCH_STRIDE
            + selected_depth * TOKEN_DEPTH_STRIDE
            + selected_rank * TOKEN_RANK_STRIDE,
        )

        tl.store(output_tokens + output_offset + node_index, selected_token)
        tl.store(output_parents + output_offset + node_index, selected_parent)
        tl.store(output_depths + output_offset + node_index, selected_depth + 1)
        tl.store(
            output_log_masses + output_offset + node_index,
            selected_mass,
        )
        used |= candidate_slots == best_slot
        # The next frontier scan gathers this node through global memory.
        tl.debug_barrier()

def build_tree_from_candidates_cuda(
    root_tokens: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_log_probs: torch.Tensor,
    max_nodes: int,
    *,
    out: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build each globally best fixed-budget tree without leaving the GPU."""
    if top_token_ids.shape != top_log_probs.shape:
        raise ValueError("tree candidate tokens and log probabilities must match")
    batch_size, num_depths, candidate_top_k = top_token_ids.shape
    if root_tokens.shape != (batch_size,):
        raise ValueError("tree roots must have shape [B]")

    capacity = 1
    width = 1
    for _ in range(num_depths):
        width *= candidate_top_k
        capacity += width
        if capacity >= max_nodes:
            break
    if capacity < max_nodes:
        raise ValueError(
            f"candidate set can produce only {capacity} tree nodes, "
            f"but fixed tree verification requires {max_nodes}"
        )

    if out is None:
        token_ids = torch.empty(
            (batch_size, max_nodes),
            dtype=torch.long,
            device=root_tokens.device,
        )
        parent_indices = torch.empty_like(token_ids)
        depths = torch.empty_like(token_ids)
        log_masses = torch.empty(
            (batch_size, max_nodes),
            dtype=torch.float32,
            device=root_tokens.device,
        )
    else:
        token_ids, parent_indices, depths, log_masses = out
    if max_nodes == 1:
        token_ids[:, 0].copy_(root_tokens)
        parent_indices[:, 0].fill_(-1)
        depths[:, 0].zero_()
        log_masses[:, 0].zero_()
        return token_ids, parent_indices, depths, log_masses

    _build_tree_kernel[(batch_size,)](
        root_tokens,
        top_token_ids,
        top_log_probs,
        token_ids,
        parent_indices,
        depths,
        log_masses,
        ROOT_BATCH_STRIDE=root_tokens.stride(0),
        TOKEN_BATCH_STRIDE=top_token_ids.stride(0),
        TOKEN_DEPTH_STRIDE=top_token_ids.stride(1),
        TOKEN_RANK_STRIDE=top_token_ids.stride(2),
        PROB_BATCH_STRIDE=top_log_probs.stride(0),
        PROB_DEPTH_STRIDE=top_log_probs.stride(1),
        PROB_RANK_STRIDE=top_log_probs.stride(2),
        NUM_DEPTHS=num_depths,
        K=candidate_top_k,
        Q=max_nodes,
        BLOCK_CANDIDATES=triton.next_power_of_2(max_nodes * candidate_top_k),
        num_warps=8,
        num_stages=1,
    )
    return token_ids, parent_indices, depths, log_masses
