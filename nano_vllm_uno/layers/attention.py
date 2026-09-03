import importlib
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from nano_vllm_uno.utils.context import ContextMode, get_context


@lru_cache(maxsize=3)
def _load_attention_backend(name: str):
    name = str(name).strip().lower()
    if name not in {"fa2", "fa3", "fa4"}:
        raise ValueError(
            f"Unsupported attention_backend={name!r}; expected 'fa2', 'fa3', or 'fa4'."
        )
    if name == "fa4":
        module_names = ("sglang.kernels.ops.attention.flash_attention_v4",)
    elif name == "fa3":
        module_names = ("flash_attn_interface", "sgl_kernel.flash_attn")
    else:
        module_names = ("flash_attn",)
    failures = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return (
                name,
                module.flash_attn_varlen_func,
                module.flash_attn_with_kvcache,
            )
        except Exception as exc:
            failures.append((module_name, exc))
    detail = "; ".join(f"{module}: {error}" for module, error in failures)
    raise ImportError(
        f"attention_backend={name!r} is unavailable; {detail}"
    ) from failures[-1][1]


@torch.compiler.disable
def _call_with_kvcache(func, q, k_cache, v_cache, kwargs):
    """Keep FlashAttention's fake-op edge cases outside optional Dynamo graphs."""
    return func(q, k_cache, v_cache, **kwargs)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@torch.compiler.disable
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def load_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Load KV from paged cache into contiguous tensors (inverse of store_kvcache_kernel)."""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    # Read from cache
    cache_offsets = slot * D + tl.arange(0, D)
    key = tl.load(k_cache_ptr + cache_offsets)
    value = tl.load(v_cache_ptr + cache_offsets)
    # Write to contiguous output
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    tl.store(key_ptr + key_offsets, key)
    tl.store(value_ptr + value_offsets, value)


def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, num_heads: int, head_dim: int):
    """Load KV from paged cache into contiguous tensors."""
    N = slot_mapping.numel()
    D = num_heads * head_dim
    key = torch.empty(N, num_heads, head_dim, device=k_cache.device, dtype=k_cache.dtype)
    value = torch.empty(N, num_heads, head_dim, device=v_cache.device, dtype=v_cache.dtype)
    load_kvcache_kernel[(N,)](k_cache, v_cache, key, key.stride(0), value, value.stride(0), slot_mapping, D)
    return key, value


@triton.jit
def _tree_suffix_merge_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    suffix_seqlens_ptr,
    prefix_ptr,
    prefix_lse_ptr,
    scale,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    page_stride_t,
    page_stride_s,
    prefix_stride_t,
    prefix_stride_h,
    prefix_stride_d,
    prefix_lse_stride_b,
    prefix_lse_stride_h,
    prefix_lse_stride_q,
    Q: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SUFFIX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    batch = token // Q
    query = token - batch * Q
    kv_head = head // (NUM_HEADS // NUM_KV_HEADS)

    suffix_offsets = tl.arange(0, BLOCK_SUFFIX)
    suffix_length = tl.load(suffix_seqlens_ptr + token)
    suffix_valid = suffix_offsets < suffix_length
    slots = tl.load(
        page_table_ptr
        + token * page_stride_t
        + suffix_offsets * page_stride_s,
        mask=suffix_valid,
        other=0,
    ).to(tl.int64)

    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    q = tl.load(
        q_ptr
        + token * q_stride_t
        + head * q_stride_h
        + dims * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_cache_ptr
        + slots[:, None] * k_stride_t
        + kv_head * k_stride_h
        + dims[None, :] * k_stride_d,
        mask=suffix_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * scale
    scores = tl.where(suffix_valid, scores, -float("inf"))
    suffix_max = tl.max(scores, axis=0)

    prefix_lse = tl.load(
        prefix_lse_ptr
        + batch * prefix_lse_stride_b
        + head * prefix_lse_stride_h
        + query * prefix_lse_stride_q
    ).to(tl.float32)
    global_max = tl.maximum(prefix_lse, suffix_max)
    prefix_weight = tl.exp(prefix_lse - global_max)
    suffix_weights = tl.exp(scores - global_max)
    denominator = prefix_weight + tl.sum(suffix_weights, axis=0)

    v = tl.load(
        v_cache_ptr
        + slots[:, None] * v_stride_t
        + kv_head * v_stride_h
        + dims[None, :] * v_stride_d,
        mask=suffix_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    suffix_numerator = tl.sum(suffix_weights[:, None] * v, axis=0)
    prefix = tl.load(
        prefix_ptr
        + token * prefix_stride_t
        + head * prefix_stride_h
        + dims * prefix_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    output = (prefix * prefix_weight + suffix_numerator) / denominator
    tl.store(
        prefix_ptr
        + token * prefix_stride_t
        + head * prefix_stride_h
        + dims * prefix_stride_d,
        output,
        mask=dim_valid,
    )


def merge_tree_suffix_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    suffix_page_table: torch.Tensor,
    suffix_cache_seqlens: torch.Tensor,
    prefix: torch.Tensor,
    prefix_lse: torch.Tensor,
    tree_size: int,
    scale: float,
) -> torch.Tensor:
    """Compute tiny ancestor attention and merge it into prefix output in place."""
    num_queries, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.size(-2)
    if num_queries % tree_size:
        raise ValueError(
            f"tree query count {num_queries} is not divisible by Q={tree_size}"
        )
    if num_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    prefix_flat = prefix.view(num_queries, num_heads, head_dim)
    _tree_suffix_merge_kernel[(num_queries, num_heads)](
        q,
        k_cache,
        v_cache,
        suffix_page_table,
        suffix_cache_seqlens,
        prefix_flat,
        prefix_lse,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        suffix_page_table.stride(0),
        suffix_page_table.stride(1),
        prefix_flat.stride(0),
        prefix_flat.stride(1),
        prefix_flat.stride(2),
        prefix_lse.stride(0),
        prefix_lse.stride(1),
        prefix_lse.stride(2),
        Q=tree_size,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SUFFIX=triton.next_power_of_2(tree_size),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )
    return prefix_flat


def _repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    repeat = num_heads // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


def _torch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None,
    causal: bool,
) -> torch.Tensor:
    k = _repeat_kv(k, q.size(1))
    v = _repeat_kv(v, q.size(1))
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


def _load_paged_prefix(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = k_cache.size(1)
    num_blocks = (int(seqlen) + block_size - 1) // block_size
    block_ids = block_table[:num_blocks].to(dtype=torch.long)
    k = k_cache.index_select(0, block_ids).reshape(-1, k_cache.size(2), k_cache.size(3))[:seqlen]
    v = v_cache.index_select(0, block_ids).reshape(-1, v_cache.size(2), v_cache.size(3))[:seqlen]
    return k, v


def _torch_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outs = []
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    batch_size = len(cu_q) - 1
    for bid in range(batch_size):
        q_start, q_end = cu_q[bid], cu_q[bid + 1]
        k_start, k_end = cu_k[bid], cu_k[bid + 1]
        q_i = q[q_start:q_end]
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
        prefix_len = seqlen_k - seqlen_q
        if block_tables is None:
            k_i = k[q_start:q_end]
            v_i = v[q_start:q_end]
            outs.append(_torch_sdpa(q_i, k_i, v_i, scale, attn_mask=None, causal=True))
        else:
            k_i, v_i = _load_paged_prefix(k_cache, v_cache, block_tables[bid], seqlen_k)
            q_pos = torch.arange(seqlen_q, device=q.device, dtype=torch.long).unsqueeze(1)
            k_pos = torch.arange(seqlen_k, device=q.device, dtype=torch.long).unsqueeze(0)
            mask = k_pos <= (prefix_len + q_pos)
            outs.append(_torch_sdpa(q_i, k_i, v_i, scale, attn_mask=mask, causal=False))
    return torch.cat(outs, dim=0)


def _torch_block_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_seqlens: torch.Tensor,
    seqlen_q: int,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    batch_size = kv_seqlens.numel()
    q_batched = q.view(batch_size, seqlen_q, num_heads, head_dim)
    outs = []
    for bid, total_len in enumerate(kv_seqlens.tolist()):
        total_len = int(total_len)
        prefix_len = total_len - seqlen_q
        k_i, v_i = _load_paged_prefix(k_cache, v_cache, block_tables[bid], total_len)
        q_pos = torch.arange(seqlen_q, device=q.device, dtype=torch.long).unsqueeze(1)
        k_pos = torch.arange(total_len, device=q.device, dtype=torch.long).unsqueeze(0)
        mask = k_pos <= (prefix_len + q_pos)
        outs.append(_torch_sdpa(q_batched[bid], k_i, v_i, scale, attn_mask=mask, causal=False))
    return torch.cat(outs, dim=0)


def _torch_tree_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_kv_seqlens: tuple[int, ...],
    seqlen_q: int,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    """Correctness reference: dense prefix plus an ancestor-only tree suffix."""
    batch_size = len(tree_kv_seqlens)
    q_batched = q.view(batch_size, seqlen_q, num_heads, head_dim)
    outs = []
    for bid, total_len in enumerate(tree_kv_seqlens):
        prefix_len = int(total_len) - seqlen_q
        if prefix_len < 0:
            raise ValueError(
                f"tree KV length {total_len} is shorter than Q={seqlen_q}"
            )
        k_i, v_i = _load_paged_prefix(
            k_cache,
            v_cache,
            block_tables[bid],
            int(total_len),
        )
        prefix_mask = torch.ones(
            (seqlen_q, prefix_len),
            dtype=torch.bool,
            device=q.device,
        )
        # PyTorch SDPA boolean masks use True for allowed attention entries.
        mask = torch.cat((prefix_mask, tree_mask[bid]), dim=1)
        outs.append(
            _torch_sdpa(
                q_batched[bid],
                k_i,
                v_i,
                scale,
                attn_mask=mask,
                causal=False,
            )
        )
    return torch.cat(outs, dim=0)


def _flash_tree_decode_attention(
    with_kvcache,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    prefix_cache_seqlens: torch.Tensor,
    suffix_page_table: torch.Tensor,
    suffix_cache_seqlens: torch.Tensor,
    tree_size: int,
    scale: float,
) -> torch.Tensor:
    """Exact FlashAttention cascade: shared prefix plus a tiny tree suffix."""
    num_queries, num_heads, head_dim = q.shape
    if num_queries % tree_size:
        raise ValueError(
            f"tree query count {num_queries} is not divisible by Q={tree_size}"
        )
    batch_size = num_queries // tree_size
    q_prefix = q.view(batch_size, tree_size, num_heads, head_dim)
    prefix_result = with_kvcache(
        q_prefix,
        k_cache,
        v_cache,
        cache_seqlens=prefix_cache_seqlens,
        block_table=block_tables,
        softmax_scale=scale,
        causal=False,
        return_softmax_lse=True,
    )
    prefix, prefix_lse, *_ = prefix_result

    return merge_tree_suffix_attention(
        q,
        k_cache,
        v_cache,
        suffix_page_table,
        suffix_cache_seqlens,
        prefix,
        prefix_lse,
        tree_size,
        scale,
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        attention_backend: str = "fa3",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        (
            self.attention_backend,
            self._varlen_func,
            self._with_kvcache_func,
        ) = _load_attention_backend(attention_backend)
        self.k_cache = self.v_cache = torch.tensor([])

    def _with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        *,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        cu_seqlens_q=None,
        cu_seqlens_k_new=None,
        max_seqlen_q=None,
        return_softmax_lse=False,
    ):
        kwargs = dict(
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        key = "block_table" if self.attention_backend == "fa2" else "page_table"
        kwargs[key] = block_table
        if cu_seqlens_q is not None:
            kwargs["cu_seqlens_q"] = cu_seqlens_q
        if cu_seqlens_k_new is not None:
            kwargs["cu_seqlens_k_new"] = cu_seqlens_k_new
        if max_seqlen_q is not None:
            kwargs["max_seqlen_q"] = max_seqlen_q
        if return_softmax_lse:
            kwargs["return_softmax_lse"] = True
        if self.attention_backend == "fa2":
            return self._with_kvcache_func(q, k_cache, v_cache, **kwargs)
        return _call_with_kvcache(
            self._with_kvcache_func,
            q,
            k_cache,
            v_cache,
            kwargs,
        )

    def _varlen(
        self,
        q,
        k,
        v,
        *,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale,
        causal,
        block_table=None,
    ):
        kwargs = dict(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        if block_table is not None:
            key = "block_table" if self.attention_backend == "fa2" else "page_table"
            kwargs[key] = block_table
        return self._varlen_func(q, k, v, **kwargs)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        use_torch_fp32_fallback = (
            os.environ.get("NANO_VLLM_FORCE_FLOAT32", "0") == "1"
            and q.dtype == torch.float32
        )
        
        if context.mode is ContextMode.TREE_VERIFY:
            if (
                context.block_tables is None
                or context.tree_mask is None
                or context.tree_kv_seqlens is None
            ):
                raise RuntimeError(
                    "tree verify requires block_tables, tree_mask, and "
                    "tree_kv_seqlens"
                )
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if (
                self.attention_backend != "fa2"
                and q.is_cuda
                and not use_torch_fp32_fallback
                and context.tree_prefix_seqlens is not None
                and context.tree_page_table is not None
                and context.tree_cache_seqlens is not None
            ):
                o = _flash_tree_decode_attention(
                    self._with_kvcache,
                    q,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context.tree_prefix_seqlens,
                    context.tree_page_table,
                    context.tree_cache_seqlens,
                    context.seqlen_q,
                    self.scale,
                )
            else:
                o = _torch_tree_decode_attention(
                    q,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context.tree_mask,
                    context.tree_kv_seqlens,
                    context.seqlen_q,
                    self.num_heads,
                    self.head_dim,
                    self.scale,
                )
        elif context.mode is ContextMode.BLOCK_DECODE:
            if context.block_tables is None or context.kv_seqlens is None:
                raise RuntimeError(
                    "block decode requires block_tables and kv_seqlens"
                )
            B = context.kv_seqlens.shape[0]
            L = context.seqlen_q
            
            # Step 1: Store KV using triton (same as prefill and decode)
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

            if use_torch_fp32_fallback:
                o = _torch_block_decode_attention(
                    q,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context.kv_seqlens,
                    L,
                    self.num_heads,
                    self.head_dim,
                    self.scale,
                )
                return o
            
            # Step 2: Reshape Q for flash_attn_with_kvcache: (B*L, heads, dim) -> (B, L, heads, dim)
            q_batched = q.view(B, L, self.num_heads, self.head_dim)
            
            # Step 3: Compute attention over the staged post-store KV lengths.
            o = self._with_kvcache(
                q_batched,
                k_cache,
                v_cache,
                cache_seqlens=context.kv_seqlens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            # Output shape: (B, L, num_heads, head_dim) -> flatten to (B*L, num_heads, head_dim)
            o = o.view(B * L, self.num_heads, self.head_dim)
        elif context.mode is ContextMode.PREFILL:
            # Regular prefill (no paged cache) or prefix cache prefill
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if use_torch_fp32_fallback:
                o = _torch_prefill_attention(
                    q,
                    k,
                    v,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    self.scale,
                )
                return o
            if context.block_tables is not None:    # prefix cache
                if self.attention_backend != "fa2":
                    cache_seqlens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
                    o = self._with_kvcache(
                        q,
                        k_cache,
                        v_cache,
                        cache_seqlens=cache_seqlens,
                        block_table=context.block_tables,
                        softmax_scale=self.scale,
                        causal=True,
                        cu_seqlens_q=context.cu_seqlens_q,
                        cu_seqlens_k_new=context.cu_seqlens_k,
                        max_seqlen_q=context.max_seqlen_q,
                    )
                else:
                    o = self._varlen(
                        q,
                        k_cache,
                        v_cache,
                        max_seqlen_q=context.max_seqlen_q,
                        cu_seqlens_q=context.cu_seqlens_q,
                        max_seqlen_k=context.max_seqlen_k,
                        cu_seqlens_k=context.cu_seqlens_k,
                        softmax_scale=self.scale,
                        causal=True,
                        block_table=context.block_tables,
                    )
            else:
                o = self._varlen(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=None,
                )
        else:
            raise RuntimeError(f"unsupported attention context mode: {context.mode!r}")
        return o
