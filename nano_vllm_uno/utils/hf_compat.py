"""Small Hugging Face compatibility helpers for native Nano models."""

from __future__ import annotations

import json
import os
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


NATIVE_K2_MODEL_TYPES = {"xllm", "k2_aurora", "k2_horizon"}


def is_native_k2_model(model_path: str) -> bool:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    return str(config.get("model_type", "")).lower() in NATIVE_K2_MODEL_TYPES


def _config_path(
    source: str,
    *,
    resolved_path: str | None,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> str:
    local_path = resolved_path or os.path.abspath(os.path.expanduser(source))
    path = os.path.join(local_path, "config.json")
    if os.path.isfile(path):
        return path
    return hf_hub_download(
        repo_id=source,
        filename="config.json",
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_model_config(
    source: str,
    *,
    resolved_path: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load K2 JSON without importing its newer Transformers implementation."""
    config_path = _config_path(
        source,
        resolved_path=resolved_path,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    with open(config_path, encoding="utf-8") as handle:
        raw_config = json.load(handle)
    if str(raw_config.get("model_type", "")).lower() in NATIVE_K2_MODEL_TYPES:
        return PretrainedConfig.from_dict(raw_config)

    source_is_local = os.path.isdir(os.path.abspath(os.path.expanduser(source)))
    config_source = resolved_path if source_is_local and resolved_path else source
    return AutoConfig.from_pretrained(
        config_source,
        trust_remote_code=True,
        revision=None if source_is_local else revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_tokenizer(source: str, **kwargs):
    """Fall back for K2 snapshots that declare TokenizersBackend."""
    try:
        return AutoTokenizer.from_pretrained(source, **kwargs)
    except ValueError as error:
        if "Tokenizer class TokenizersBackend does not exist" not in str(error):
            raise
        return PreTrainedTokenizerFast.from_pretrained(source, **kwargs)
