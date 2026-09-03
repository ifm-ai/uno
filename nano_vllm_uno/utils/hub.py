"""Resolve local paths and Hugging Face repositories to local snapshots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download


ADAPTER_ALLOW_PATTERNS = (
    "adapter_config.json",
    "adapter_model*.safetensors",
    "adapter_model*.bin",
)


def resolve_hf_snapshot(
    source: str | os.PathLike[str],
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
    allow_patterns: Sequence[str] | None = None,
    artifact_name: str = "artifact",
) -> str:
    """Return a local directory for a path or Hugging Face repository ID.

    Authentication follows the standard Hugging Face mechanisms, including
    ``HF_TOKEN`` and tokens stored by ``hf auth login``.
    """

    source = os.fspath(source)
    expanded = os.path.abspath(os.path.expanduser(source))
    if os.path.isdir(expanded):
        return expanded
    if os.path.exists(expanded):
        raise ValueError(f"{artifact_name} must be a directory: {expanded}")

    try:
        return snapshot_download(
            repo_id=source,
            revision=revision,
            cache_dir=os.fspath(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
    except Exception as exc:
        mode = "the local Hugging Face cache" if local_files_only else "Hugging Face"
        revision_text = f" at revision {revision!r}" if revision else ""
        raise ValueError(
            f"Could not resolve {artifact_name} {source!r}{revision_text} from "
            f"a local directory or {mode}."
        ) from exc

