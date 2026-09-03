import json
import os
from glob import glob
from typing import Iterable

import torch
from torch import nn
from safetensors import safe_open

from nano_vllm_uno.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)


QKV_OUTPUT_STARTS = {
    "q_proj": lambda module: 0,
    "k_proj": lambda module: module.num_heads * module.head_size,
    "v_proj": lambda module: (module.num_heads + module.num_kv_heads) * module.head_size,
}

GATE_UP_SHARD_IDS = {
    "gate_proj": 0,
    "up_proj": 1,
}


def _adapter_model_files(path: str) -> list[str]:
    files = sorted(glob(os.path.join(path, "adapter_model*.safetensors")))
    if files:
        return files
    files = sorted(glob(os.path.join(path, "adapter_model*.bin")))
    if files:
        return files
    raise FileNotFoundError(f"No adapter_model safetensors/bin file found in {path}")


def _load_adapter_tensors(path: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for file in _adapter_model_files(path):
        if file.endswith(".safetensors"):
            with safe_open(file, "pt", "cpu") as handle:
                for key in handle.keys():
                    tensors[key] = handle.get_tensor(key)
        else:
            tensors.update(torch.load(file, map_location="cpu"))
    return tensors


def _iter_lora_pairs(tensors: dict[str, torch.Tensor]) -> Iterable[tuple[str, torch.Tensor, torch.Tensor]]:
    for key, lora_A in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        lora_B_key = prefix + ".lora_B.weight"
        if lora_B_key not in tensors:
            raise KeyError(f"Missing LoRA B tensor for {key}: expected {lora_B_key}")
        # PEFT checkpoints wrap the underlying model with `base_model.model`.
        prefix = prefix.removeprefix("base_model.model.")
        yield prefix, lora_A, tensors[lora_B_key]


def _module_for_prefix(model: nn.Module, prefix: str) -> tuple[nn.Module, int]:
    parts = prefix.split(".")
    if len(parts) != 5 or parts[0] != "model" or parts[1] != "layers":
        raise ValueError(f"Unsupported LoRA tensor prefix: {prefix}")

    layer_idx = int(parts[2])
    scope = parts[3]
    proj = parts[4]
    layer = model.model.layers[layer_idx]

    if scope == "self_attn" and proj in QKV_OUTPUT_STARTS:
        module = layer.self_attn.qkv_proj
        if not isinstance(module, QKVParallelLinear):
            raise TypeError(f"Expected QKVParallelLinear for {prefix}, got {type(module)}")
        return module, int(QKV_OUTPUT_STARTS[proj](module))

    if scope == "self_attn" and proj == "o_proj":
        module = layer.self_attn.o_proj
        if not isinstance(module, RowParallelLinear):
            raise TypeError(f"Expected RowParallelLinear for {prefix}, got {type(module)}")
        return module, 0

    if scope == "mlp" and proj in GATE_UP_SHARD_IDS:
        module = layer.mlp.gate_up_proj
        if not isinstance(module, MergedColumnParallelLinear):
            raise TypeError(f"Expected MergedColumnParallelLinear for {prefix}, got {type(module)}")
        shard_id = GATE_UP_SHARD_IDS[proj]
        output_start = sum(module.output_sizes[:shard_id]) // module.tp_size
        return module, int(output_start)

    if scope == "mlp" and proj == "down_proj":
        module = layer.mlp.down_proj
        if not isinstance(module, RowParallelLinear):
            raise TypeError(f"Expected RowParallelLinear for {prefix}, got {type(module)}")
        return module, 0

    raise ValueError(f"Unsupported LoRA target prefix: {prefix}")


def _shard_lora_for_module(
    module: nn.Module,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_rank = getattr(module, "tp_rank", 0)
    tp_size = getattr(module, "tp_size", 1)
    tp_dim = getattr(module, "tp_dim", None)

    if tp_size == 1:
        return lora_A, lora_B
    if tp_dim == 0:
        return lora_A, lora_B.chunk(tp_size, dim=0)[tp_rank]
    if tp_dim == 1:
        return lora_A.chunk(tp_size, dim=1)[tp_rank], lora_B
    raise ValueError(f"Cannot shard LoRA for module {module} with tp_dim={tp_dim}")


def load_lora_adapter(
    model: nn.Module,
    path: str,
) -> dict[str, object]:
    config_path = os.path.join(path, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    scaling = alpha / rank
    tensors = _load_adapter_tensors(path)

    applied = 0
    touched_modules: set[nn.Module] = set()
    targets = set()
    for prefix, lora_A, lora_B in _iter_lora_pairs(tensors):
        module, output_start = _module_for_prefix(model, prefix)
        lora_A, lora_B = _shard_lora_for_module(module, lora_A, lora_B)
        module.add_lora_slice(lora_A, lora_B, scaling, output_start=output_start)
        applied += 1
        touched_modules.add(module)
        targets.add(prefix.split(".")[-1])

    expected_targets = set(adapter_config.get("target_modules") or [])
    if expected_targets and not expected_targets.issubset(targets):
        missing = sorted(expected_targets - targets)
        raise RuntimeError(f"LoRA adapter missing expected target modules: {missing}")

    packed_groups = 0
    packed_slices = 0
    for module in touched_modules:
        pack_lora_slices = getattr(module, "pack_lora_slices", None)
        if not callable(pack_lora_slices):
            continue
        stats = pack_lora_slices()
        packed_groups += int(stats["packed_groups"])
        packed_slices += int(stats["packed_slices"])

    # TP=1 gated LoRA always overlaps its A projection with the base GEMM.
    # Sharing one stream across modules keeps multi-stream CUDA graphs compact.
    overlap_streams = {}
    for module in touched_modules:
        device = module.base_weight_device
        if getattr(module, "tp_size", 1) != 1 or device.type != "cuda":
            continue
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        stream = overlap_streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=device_index)
            overlap_streams[device_index] = stream
        module.set_lora_overlap_stream(stream)

    return {
        "path": path,
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "num_slices": applied,
        "packed_groups": packed_groups,
        "packed_slices": packed_slices,
        "num_overlap_streams": len(overlap_streams),
        "target_modules": sorted(targets),
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
    }
