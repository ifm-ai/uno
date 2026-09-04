"""Per-request completion budgets for fixed-context evaluations."""

from __future__ import annotations


DEFAULT_CONTEXT_LENGTH = 32768


def active_forward_reserve(
    diffusion_block_size: int,
    tree_verify_size: int | None,
) -> int:
    """Return temporary token slots needed by the largest decode forward."""
    block_size = int(diffusion_block_size)
    tree_size = 0 if tree_verify_size is None else int(tree_verify_size)
    if block_size < 1 or tree_size < 0:
        raise ValueError("decode widths must be non-negative and block size positive")
    return max(block_size, tree_size)


def resolve_completion_budget(
    *,
    prompt_tokens: int,
    context_length: int,
    reserve_tokens: int,
    global_max_tokens: int | None = None,
) -> int:
    """Return completion capacity after prompt and active-forward scratch space."""
    prompt_tokens = int(prompt_tokens)
    context_length = int(context_length)
    reserve_tokens = int(reserve_tokens)
    if prompt_tokens < 0 or context_length < 1 or reserve_tokens < 0:
        raise ValueError("invalid context-budget inputs")
    available = max(0, context_length - prompt_tokens - reserve_tokens)
    if global_max_tokens is None:
        return available
    global_max_tokens = int(global_max_tokens)
    if global_max_tokens < 1:
        raise ValueError("global_max_tokens must be positive when provided")
    return min(available, global_max_tokens)
