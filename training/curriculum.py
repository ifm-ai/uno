from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CurriculumStage:
    block_size: int
    token_budget: int
    start_step: int
    end_step: int


class BlockCurriculumPlan:
    """Deterministic, replay-free block-size curriculum."""

    def __init__(
        self,
        *,
        source_path: Path,
        tokens_per_step: int,
        sequence_length: int,
        global_batch_size: int,
        seed: int,
        stages: list[CurriculumStage],
    ) -> None:
        self.source_path = source_path
        self.tokens_per_step = tokens_per_step
        self.sequence_length = sequence_length
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.stages = tuple(stages)
        self._stage_end_steps = tuple(stage.end_step for stage in stages)
        self.checkpoint_reasons = {
            stage.end_step: f"completed_block_size_{stage.block_size}"
            for stage in stages
        }

    @property
    def max_steps(self) -> int:
        return self.stages[-1].end_step

    @property
    def total_tokens(self) -> int:
        return sum(stage.token_budget for stage in self.stages)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BlockCurriculumPlan":
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Curriculum file not found: {source_path}")
        raw = yaml.safe_load(source_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Curriculum YAML must contain a mapping.")
        if raw.get("replay") not in (None, {}, False):
            raise ValueError("Smaller-block replay is not supported by the public trainer.")

        tokens_per_step = _positive_int(raw, "tokens_per_step")
        sequence_length = _positive_int(raw, "sequence_length")
        global_batch_size = _positive_int(raw, "global_batch_size")
        seed = _integer(raw, "seed")
        if tokens_per_step != sequence_length * global_batch_size:
            raise ValueError(
                "tokens_per_step must equal sequence_length * global_batch_size."
            )

        raw_stages = raw.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Curriculum stages must be a non-empty list.")
        stages: list[CurriculumStage] = []
        cumulative_tokens = 0
        previous_end = 0
        previous_block = 0
        for index, item in enumerate(raw_stages):
            if not isinstance(item, dict):
                raise ValueError(f"Curriculum stage {index} must be a mapping.")
            block_size = _positive_int(item, "block_size")
            token_budget = _positive_int(item, "tokens")
            if block_size <= previous_block:
                raise ValueError("Curriculum block sizes must be strictly increasing.")
            cumulative_tokens += token_budget
            end_step = cumulative_tokens // tokens_per_step
            if end_step <= previous_end:
                raise ValueError(f"Curriculum stage B={block_size} is shorter than one step.")
            stages.append(
                CurriculumStage(block_size, token_budget, previous_end, end_step)
            )
            previous_end = end_step
            previous_block = block_size

        declared_total = raw.get("total_tokens")
        if declared_total is not None and declared_total != cumulative_tokens:
            raise ValueError("total_tokens does not match the sum of stage token budgets.")
        return cls(
            source_path=source_path,
            tokens_per_step=tokens_per_step,
            sequence_length=sequence_length,
            global_batch_size=global_batch_size,
            seed=seed,
            stages=stages,
        )

    def stage_for_step(self, step: int) -> CurriculumStage:
        if not 0 <= step < self.max_steps:
            raise ValueError(f"Optimizer step {step} is outside [0, {self.max_steps}).")
        return self.stages[bisect.bisect_right(self._stage_end_steps, step)]

    def validate_runtime(self, *, world_size: int, per_device_batch: int, accumulation: int) -> None:
        runtime_global_batch = world_size * per_device_batch * accumulation
        if runtime_global_batch != self.global_batch_size:
            raise ValueError(
                "Curriculum global batch mismatch: "
                f"runtime={runtime_global_batch}, expected={self.global_batch_size}."
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "tokens_per_step": self.tokens_per_step,
            "sequence_length": self.sequence_length,
            "global_batch_size": self.global_batch_size,
            "seed": self.seed,
            "total_tokens": self.total_tokens,
            "max_steps": self.max_steps,
            "replay_smaller_blocks": False,
            "stages": [
                {
                    "block_size": stage.block_size,
                    "token_budget": stage.token_budget,
                    "start_step": stage.start_step,
                    "end_step": stage.end_step,
                }
                for stage in self.stages
            ],
        }

    def sha256(self) -> str:
        import json

        encoded = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def set_model_block_size(model, block_size: int) -> None:
    while hasattr(model, "module"):
        model = model.module
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base_model.config, "block_size"):
        raise RuntimeError("Uno training requires model.config.block_size.")
    base_model.config.block_size = block_size


def _integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    return value


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = _integer(mapping, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive.")
    return value

