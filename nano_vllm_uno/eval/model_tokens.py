"""Resolve diffusion and stopping token IDs from Hugging Face models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from transformers import AutoConfig, GenerationConfig


def _as_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [int(item) for item in value if item is not None]
    return [int(value)]


def _dedupe(values: Iterable[int]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _read_generation_config(
    model_path: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> dict[str, Any]:
    try:
        config = GenerationConfig.from_pretrained(
            model_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except OSError:
        return {}
    value = config.to_dict()
    if not isinstance(value, dict):
        raise ValueError("Expected GenerationConfig.to_dict() to return a dictionary")
    return value


def resolve_model_token_ids(
    model_path: str,
    tokenizer: Any,
    *,
    mask_token_id: int | None = None,
    stop_token_ids: list[int] | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> tuple[int, list[int], int]:
    """Return ``(mask_id, stop_ids, vocab_size)`` with range validation.

    SGLang honors ``generation_config.json`` stopping IDs in addition
    to the tokenizer EOS. Nano-vLLM does not read that file itself, so the eval
    launcher resolves and forwards the same IDs explicitly.
    """

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    vocab_size = int(config.vocab_size)

    resolved_mask = mask_token_id
    if resolved_mask is None:
        resolved_mask = getattr(config, "mask_token_id", None)
    if resolved_mask is None:
        resolved_mask = getattr(tokenizer, "mask_token_id", None)
    if resolved_mask is None:
        candidate = tokenizer.convert_tokens_to_ids("<|MASK|>")
        unknown = getattr(tokenizer, "unk_token_id", None)
        if candidate is not None and candidate != unknown:
            resolved_mask = candidate
    if resolved_mask is None:
        raise ValueError(
            "Could not resolve the diffusion mask token. Set --mask-token-id "
            "or add mask_token_id to the model config."
        )
    resolved_mask = int(resolved_mask)

    if stop_token_ids is None:
        generation_config = _read_generation_config(
            model_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        resolved_stops = _dedupe(
            [
                *_as_token_ids(generation_config.get("eos_token_id")),
                *_as_token_ids(getattr(config, "eos_token_id", None)),
                *_as_token_ids(getattr(tokenizer, "eos_token_id", None)),
            ]
        )
    else:
        resolved_stops = _dedupe(stop_token_ids)

    invalid = [
        token_id
        for token_id in [resolved_mask, *resolved_stops]
        if not 0 <= token_id < vocab_size
    ]
    if invalid:
        raise ValueError(
            f"Token IDs {invalid} are outside model vocabulary [0, {vocab_size}). "
            "Do not reuse token IDs from a different checkpoint family."
        )
    return resolved_mask, resolved_stops, vocab_size
