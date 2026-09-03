from __future__ import annotations

import hashlib
import json
import math
import netrc
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .constants import (
    DEFAULT_CE_ALPHA,
    DEFAULT_KL_BETA,
    DEFAULT_LORA_ALPHA_RATIO,
    DEFAULT_TV_GAMMA,
)


CeTarget = Literal["teacher", "ground_truth"]


@dataclass(frozen=True)
class UnoObjectiveConfig:
    """The intentionally narrow public Uno objective."""

    ce_target: CeTarget = "teacher"
    noise: str = "uniform"
    reverse_kl: bool = True
    teacher_base_only: bool = True
    ce_weight: float = DEFAULT_CE_ALPHA
    kl_weight: float = DEFAULT_KL_BETA
    tv_weight: float = DEFAULT_TV_GAMMA

    def validate(self) -> None:
        if self.ce_target not in ("teacher", "ground_truth"):
            raise ValueError(
                "ce_target must be either 'teacher' or 'ground_truth'."
            )
        if self.noise != "uniform":
            raise ValueError("Public Uno training supports uniform noise only.")
        if not self.reverse_kl:
            raise ValueError("Public Uno training supports reverse KL only.")
        if not self.teacher_base_only:
            raise ValueError("The Uno teacher must use frozen base weights only.")
        weights = (self.ce_weight, self.kl_weight, self.tv_weight)
        if any(not isinstance(weight, (int, float)) for weight in weights):
            raise ValueError("Uno loss weights must be numeric.")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Uno loss weights must be finite and non-negative.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one Uno loss weight must be positive.")

    @property
    def internal_ce_target(self) -> str:
        self.validate()
        return "clean_argmax" if self.ce_target == "teacher" else "data"


@dataclass(frozen=True)
class LoraSettings:
    target: str
    rank: int
    alpha: int | None
    dropout: float

    @property
    def resolved_alpha(self) -> int:
        return self.alpha if self.alpha is not None else DEFAULT_LORA_ALPHA_RATIO * self.rank

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def manifest_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_resume_manifest(output_dir: Path, expected: dict[str, Any]) -> None:
    manifest_path = output_dir / "uno_training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume without the original run manifest: {manifest_path}"
        )
    actual = json.loads(manifest_path.read_text())
    if actual.get("compatibility_sha256") != manifest_digest(expected):
        raise ValueError(
            "Resume configuration does not match the original training run. "
            f"Expected compatibility hash {manifest_digest(expected)}, got "
            f"{actual.get('compatibility_sha256')}."
        )


def has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    host = base_url.split("://", 1)[-1].split("/", 1)[0]
    try:
        auth = netrc.netrc().authenticators(host)
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(auth and auth[2])


def validate_wandb_auth(mode: str) -> None:
    if mode != "online":
        return
    if not has_wandb_credentials():
        raise RuntimeError(
            "W&B online logging is enabled, but no credentials were found. "
            "Run `wandb login`, set WANDB_API_KEY, or set WANDB_MODE=disabled."
        )


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
