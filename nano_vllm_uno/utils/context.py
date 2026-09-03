from dataclasses import dataclass
from enum import Enum, auto

import torch


class ContextMode(Enum):
    """Attention execution mode for the active model forward."""

    PREFILL = auto()
    BLOCK_DECODE = auto()
    TREE_VERIFY = auto()


@dataclass
class Context:
    mode: ContextMode | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # For block decode: total valid KV length after this block is stored.
    kv_seqlens: torch.Tensor | None = None
    # For block decode, every sequence has the same query length.
    seqlen_q: int = 0
    # Dense correctness fallback for tree verification. Prefix tokens are
    # implicitly visible to every query; the mask covers the tree suffix.
    tree_mask: torch.Tensor | None = None
    tree_kv_seqlens: tuple[int, ...] | None = None
    # FlashAttention tree verification. The shared prefix uses the block table;
    # the token-page table contains only each query's ancestor/self suffix.
    tree_prefix_seqlens: torch.Tensor | None = None
    tree_page_table: torch.Tensor | None = None
    tree_cache_seqlens: torch.Tensor | None = None
    # Required whenever gated LoRA is enabled. Shape follows the flattened input
    # rows seen by linear layers. Values are 1.0 for LoRA-on rows and 0.0 for
    # base-only rows.
    lora_mask: torch.Tensor | None = None
    lora_enabled: bool = False

_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    mode: ContextMode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    block_tables=None,
    kv_seqlens=None,
    seqlen_q=0,
    tree_mask=None,
    tree_kv_seqlens=None,
    tree_prefix_seqlens=None,
    tree_page_table=None,
    tree_cache_seqlens=None,
    lora_mask=None,
    lora_enabled=False,
):
    global _CONTEXT
    if not isinstance(mode, ContextMode):
        raise TypeError(f"mode must be a ContextMode, got {mode!r}")
    _CONTEXT = Context(
        mode=mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        kv_seqlens=kv_seqlens,
        seqlen_q=seqlen_q,
        tree_mask=tree_mask,
        tree_kv_seqlens=tree_kv_seqlens,
        tree_prefix_seqlens=tree_prefix_seqlens,
        tree_page_table=tree_page_table,
        tree_cache_seqlens=tree_cache_seqlens,
        lora_mask=lora_mask,
        lora_enabled=lora_enabled,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
