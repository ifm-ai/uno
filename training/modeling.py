from __future__ import annotations

import sys
from types import MethodType

import torch


def unwrap_base_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def validate_uno_model(model) -> None:
    base_model = unwrap_base_model(model)
    required = ("prepare_for_bd_training", "model", "lm_head")
    missing = [name for name in required if not hasattr(base_model, name)]
    if missing:
        raise RuntimeError(
            "The selected HF model is not the Uno-prepared Qwen3 model. Missing "
            f"required interface member(s): {missing}."
        )
    _get_sdar_helpers(base_model)
    if not hasattr(base_model.config, "block_size"):
        raise RuntimeError("The Uno model config must define block_size.")


def configure_uniform_noise_config(config):
    configured_noise = getattr(config, "noise", None)
    if configured_noise not in (None, "uniform"):
        raise RuntimeError(
            "Public Uno training requires uniform noise, but the model config "
            f"declares noise={configured_noise!r}."
        )
    config.noise = "uniform"
    return config


def configure_uniform_noise(model) -> None:
    base_model = unwrap_base_model(model)
    validate_uno_model(base_model)
    base_model.noise = "uniform"
    configure_uniform_noise_config(base_model.config)


def _get_sdar_helpers(base_model):
    modeling_module = sys.modules.get(base_model.__class__.__module__)
    if modeling_module is None:
        raise RuntimeError(
            f"Could not resolve modeling module for {base_model.__class__.__name__}."
        )
    modify_position_ids = getattr(
        modeling_module, "modify_padded_position_ids_2d", None
    )
    calculate_token_nums = getattr(modeling_module, "calculate_token_nums", None)
    if modify_position_ids is None or calculate_token_nums is None:
        raise RuntimeError(
            "The Uno model module must provide modify_padded_position_ids_2d and "
            "calculate_token_nums."
        )
    return modify_position_ids, calculate_token_nums


def _build_noisy_region_mask(
    num_tokens: list[torch.Tensor], sequence_length: int
) -> torch.Tensor:
    rows = []
    for lengths in num_tokens:
        parts = []
        for length in lengths.tolist():
            parts.append(torch.ones(length, dtype=torch.bool, device=lengths.device))
            parts.append(torch.zeros(length, dtype=torch.bool, device=lengths.device))
        row = torch.cat(parts)
        if row.numel() != 2 * sequence_length:
            raise RuntimeError(
                "Uno noisy/clean layout has an invalid length: "
                f"{row.numel()} for sequence_length={sequence_length}."
            )
        rows.append(row)
    return torch.stack(rows)


def prepare_uno_layout(
    base_model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length = input_ids.shape
    if position_ids is None:
        position_ids = torch.arange(
            sequence_length, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
    modify_position_ids, calculate_token_nums = _get_sdar_helpers(base_model)
    model_position_ids = modify_position_ids(position_ids)
    token_counts = calculate_token_nums(model_position_ids)
    noisy_region_mask = _build_noisy_region_mask(token_counts, sequence_length)
    return position_ids, model_position_ids, noisy_region_mask


def extract_uno_regions(
    hidden_states: torch.Tensor,
    noisy_region_mask: torch.Tensor,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, hidden_size = hidden_states.shape
    noisy = hidden_states[noisy_region_mask].reshape(
        batch_size, sequence_length, hidden_size
    )
    clean = hidden_states[~noisy_region_mask].reshape(
        batch_size, sequence_length, hidden_size
    )
    return noisy, clean


def run_prepared_uno_forward(
    model,
    base_model,
    inputs: dict[str, torch.Tensor],
    concatenated_input_ids: torch.Tensor,
    concatenated_position_ids: torch.Tensor,
    flex_attention_mask,
) -> torch.Tensor:
    """Run the paired noisy/clean layout through wrappers in one transformer pass."""
    decoder = getattr(base_model, "model", None)
    if decoder is None:
        raise RuntimeError("The Uno model must expose its decoder as .model.")
    original_forward = base_model.forward

    def prepared_forward(self, *args, **kwargs):
        del self, args, kwargs
        outputs = decoder(
            input_ids=concatenated_input_ids,
            attention_mask=flex_attention_mask,
            position_ids=concatenated_position_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        return {"hidden_states": (outputs.last_hidden_state,)}

    base_model.forward = MethodType(prepared_forward, base_model)
    try:
        outputs = model(**inputs, output_hidden_states=True)
    finally:
        base_model.forward = original_forward
    return outputs["hidden_states"][-1]
