from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .config import LoraSettings
from .constants import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, SUPPORTED_LORA_MODULES


_COMPACT_TARGETS = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
}


def resolve_lora_targets(specification: str) -> tuple[str, ...]:
    """Resolve a portable LoRA target preset into projection suffixes."""
    normalized = specification.strip().lower()
    if normalized == "all":
        return SUPPORTED_LORA_MODULES
    if normalized and all(character in _COMPACT_TARGETS for character in normalized):
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"LoRA target shorthand contains duplicates: {specification!r}.")
        return tuple(_COMPACT_TARGETS[character] for character in normalized)

    targets = tuple(part.strip() for part in specification.split(",") if part.strip())
    if not targets:
        raise ValueError("LoRA targets cannot be empty.")
    unsupported = sorted(set(targets).difference(SUPPORTED_LORA_MODULES))
    if unsupported:
        raise ValueError(
            "Unsupported LoRA target(s): "
            f"{unsupported}. Supported projection names are {list(SUPPORTED_LORA_MODULES)}."
        )
    if len(set(targets)) != len(targets):
        raise ValueError(f"LoRA target list contains duplicates: {specification!r}.")
    return targets


def validate_target_modules(model, targets: Iterable[str]) -> dict[str, int]:
    """Require exactly one requested projection per transformer layer."""
    targets = tuple(targets)
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise RuntimeError("The base model must declare config.num_hidden_layers.")

    counts = {
        target: sum(name.endswith(f".{target}") for name, _ in model.named_modules())
        for target in targets
    }
    invalid = {target: count for target, count in counts.items() if count != num_layers}
    if invalid:
        raise RuntimeError(
            "Requested LoRA projections are not present exactly once per transformer "
            f"layer (expected {num_layers} each): {invalid}."
        )
    return counts


def create_lora_model(model, settings: LoraSettings):
    """Attach the only trainable weights used by public Uno training."""
    from peft import LoraConfig, TaskType, get_peft_model

    settings.validate()
    targets = resolve_lora_targets(settings.target)
    validate_target_modules(model, targets)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=settings.rank,
        lora_alpha=settings.resolved_alpha,
        lora_dropout=settings.dropout,
        target_modules=list(targets),
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    for name, parameter in peft_model.named_parameters():
        if parameter.requires_grad:
            if "lora_A" not in name and "lora_B" not in name:
                raise RuntimeError(
                    "Uno is LoRA-only, but PEFT exposed a non-LoRA trainable tensor: "
                    f"{name}."
                )
            parameter.data = parameter.data.float()
    validate_only_lora_trainable(peft_model)
    return peft_model, targets


def validate_only_lora_trainable(model) -> tuple[int, int]:
    trainable = []
    total_trainable = 0
    total_parameters = 0
    for name, parameter in model.named_parameters():
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable.append(name)
            total_trainable += parameter.numel()
    if not trainable:
        raise RuntimeError("No trainable LoRA tensors were found.")
    invalid = [
        name for name in trainable if "lora_A" not in name and "lora_B" not in name
    ]
    if invalid:
        raise RuntimeError(
            "Only LoRA A/B tensors may be trainable; found " f"{invalid[:5]}."
        )
    return total_trainable, total_parameters


def make_saved_adapter_portable(adapter_dir: str | Path) -> None:
    """Replace resolved cache paths in PEFT metadata with the pinned Hub source."""
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = DEFAULT_MODEL_ID
    config["revision"] = DEFAULT_MODEL_REVISION
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    readme_path = adapter_dir / "README.md"
    if not readme_path.is_file():
        return
    lines = readme_path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("base_model:"):
            lines[index] = f"base_model: {DEFAULT_MODEL_ID}"
            readme_path.write_text("\n".join(lines) + "\n")
            break


@dataclass
class TokenwiseLoraRouter:
    """Mask PEFT's low-rank update on clean teacher token rows."""

    model: object

    def __post_init__(self) -> None:
        self.token_mask: torch.Tensor | None = None
        self.handles = []
        hooked_modules: set[int] = set()
        validate_only_lora_trainable(self.model)
        for module in self.model.modules():
            lora_a_layers = getattr(module, "lora_A", None)
            if lora_a_layers is None:
                continue
            if any(getattr(module, "use_dora", {}).values()):
                raise RuntimeError("Token-conditional Uno routing does not support DoRA.")
            for lora_a in lora_a_layers.values():
                if id(lora_a) in hooked_modules:
                    continue
                hooked_modules.add(id(lora_a))
                self.handles.append(lora_a.register_forward_hook(self._mask_output))
        if not self.handles:
            raise RuntimeError("No PEFT LoRA-A layers were found for conditional routing.")

    def set_token_mask(self, token_mask: torch.Tensor) -> None:
        self.token_mask = token_mask

    def _mask_output(self, module, inputs, output):
        del module, inputs
        if self.token_mask is None:
            raise RuntimeError("Set the Uno LoRA token mask before the model forward.")
        if output.shape[:-1] != self.token_mask.shape:
            raise RuntimeError(
                "Uno LoRA mask and activation shapes differ: "
                f"mask={tuple(self.token_mask.shape)}, output={tuple(output.shape)}."
            )
        mask = self.token_mask.to(device=output.device, dtype=output.dtype)
        return output * mask.unsqueeze(-1)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
