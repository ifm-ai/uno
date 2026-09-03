"""Fused in-place Q/K RMSNorm implemented independently in Triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rms_norm_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_token_stride,
    k_token_stride,
    eps: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    NUM_WORKS: tl.constexpr,
):
    head_slots = tl.arange(0, HEADS_PER_PROGRAM)[:, None]
    work_ids = tl.program_id(0) * HEADS_PER_PROGRAM + head_slots
    heads_per_token = NUM_Q_HEADS + NUM_KV_HEADS
    token_id = work_ids // heads_per_token
    combined_head_id = work_ids % heads_per_token
    is_query = combined_head_id < NUM_Q_HEADS

    offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = (work_ids < NUM_WORKS) & (offsets < HEAD_DIM)
    q_offsets = token_id * q_token_stride + combined_head_id * HEAD_DIM + offsets
    k_offsets = (
        token_id * k_token_stride
        + (combined_head_id - NUM_Q_HEADS) * HEAD_DIM
        + offsets
    )
    input_ptrs = tl.where(is_query, q_ptr + q_offsets, k_ptr + k_offsets)
    weight_ptrs = tl.where(
        is_query,
        q_weight_ptr + offsets,
        k_weight_ptr + offsets,
    )

    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1)[:, None] / HEAD_DIM
    inverse_rms = tl.rsqrt(mean_square + eps)
    weight = tl.load(weight_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(input_ptrs, x * inverse_rms * weight, mask=mask)


@torch.compiler.disable
def fused_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize packed Q and K views in place with one fused launch."""
    head_dim = q.shape[2]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_tokens = q.shape[0]
    block_size = triton.next_power_of_2(head_dim)
    heads_per_program = 4
    num_works = num_tokens * (num_q_heads + num_kv_heads)
    _qk_rms_norm_kernel[(triton.cdiv(num_works, heads_per_program),)](
        q,
        k,
        q_weight,
        k_weight,
        q.stride(0),
        k.stride(0),
        eps=eps,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        HEADS_PER_PROGRAM=heads_per_program,
        NUM_WORKS=num_works,
        num_warps=4,
    )
    return q, k
