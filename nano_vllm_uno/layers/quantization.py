"""Minimal quantized-linear lifecycle for native serialized checkpoints."""

from __future__ import annotations

import functools
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def _parameter(
    tensor: torch.Tensor,
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
    output_partition_sizes: list[int] | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> nn.Parameter:
    """Attach the sharding metadata consumed by nano's weight loader."""
    parameter = nn.Parameter(tensor, requires_grad=False)
    parameter.input_dim = input_dim
    parameter.output_dim = output_dim
    parameter.output_partition_sizes = output_partition_sizes
    parameter.weight_loader = functools.partial(
        _load_quantized_parameter,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    return parameter


def _narrow_for_tp(
    loaded: torch.Tensor,
    dim: int,
    target_size: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    if loaded.size(dim) == target_size:
        return loaded
    if loaded.size(dim) != target_size * tp_size:
        raise ValueError(
            f"Cannot shard dimension {dim}: loaded={loaded.size(dim)}, "
            f"target={target_size}, tp_size={tp_size}"
        )
    return loaded.narrow(dim, tp_rank * target_size, target_size)


def _load_quantized_parameter(
    parameter: nn.Parameter,
    loaded: torch.Tensor,
    shard_index: int | str | None = None,
    *,
    tp_rank: int,
    tp_size: int,
) -> None:
    """Load one serialized tensor using metadata supplied by its adapter."""
    input_dim = getattr(parameter, "input_dim", None)
    output_dim = getattr(parameter, "output_dim", None)

    if shard_index is not None and output_dim is not None:
        shard_index = (
            {"q": 0, "k": 1, "v": 2}[shard_index]
            if isinstance(shard_index, str)
            else int(shard_index)
        )
        partitions = parameter.output_partition_sizes
        target_size = partitions[shard_index]
        loaded = _narrow_for_tp(
            loaded, output_dim, target_size, tp_rank, tp_size
        )
        target = parameter.data.narrow(
            output_dim,
            sum(partitions[:shard_index]),
            target_size,
        )
    else:
        target = parameter.data

    if input_dim is not None and loaded.size(input_dim) != target.size(input_dim):
        loaded = _narrow_for_tp(
            loaded, input_dim, target.size(input_dim), tp_rank, tp_size
        )
    if (
        shard_index is None
        and output_dim is not None
        and loaded.size(output_dim) != target.size(output_dim)
    ):
        loaded = _narrow_for_tp(
            loaded, output_dim, target.size(output_dim), tp_rank, tp_size
        )
    if target.shape != loaded.shape:
        raise ValueError(
            f"Checkpoint shape {tuple(loaded.shape)} does not match "
            f"parameter shape {tuple(target.shape)}"
        )
    target.copy_(loaded)


class LinearQuantizationMethod(ABC):
    """Format adapter used by every tensor-parallel linear implementation."""

    kernel_name: str

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_partition_sizes: list[int],
    ) -> None:
        """Allocate parameters in the checkpoint's serialized representation."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Optionally prepare a loaded representation for its runtime kernel."""

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the quantized projection and return a 16-bit floating output."""


@functools.cache
def _block_fp8_runner():
    # Nano captures a small fixed set of M shapes. Avoid SGLang's eager
    # compilation of every M from 1..16384 and keep its cache isolated.
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    os.environ.setdefault(
        "SGLANG_DG_CACHE_DIR",
        os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "nano_vllm_uno",
            "deep_gemm",
        ),
    )
    try:
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
    except ImportError as error:
        raise ImportError(
            "Block FP8 checkpoints require SGLang's DeepGEMM kernels"
        ) from error
    return deepgemm_w8a8_block_fp8_linear_with_fallback


class BlockFP8LinearMethod(LinearQuantizationMethod):
    """Dynamic-activation FP8 with serialized 128x128 weight blocks."""

    kernel_name = "sglang_fp8_block_deep_gemm"
    block_size = (128, 128)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BlockFP8LinearMethod:
        if tuple(config.get("weight_block_size") or ()) != cls.block_size:
            raise ValueError("Block FP8 requires weight_block_size=[128, 128]")
        if config.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError("Block FP8 requires dynamic activations")
        return cls()

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_partition_sizes: list[int],
    ) -> None:
        block_n, block_k = self.block_size
        if input_size % block_k:
            raise ValueError("FP8 input partition must be divisible by 128")
        if any(size % block_n for size in output_partition_sizes):
            raise ValueError("Every fused FP8 output partition must be divisible by 128")

        output_size = sum(output_partition_sizes)
        layer.weight = _parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            input_dim=1,
            output_dim=0,
            output_partition_sizes=output_partition_sizes,
            tp_rank=layer.tp_rank,
            tp_size=layer.tp_size,
        )
        layer.weight_scale_inv = _parameter(
            torch.empty(
                output_size // block_n,
                input_size // block_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            output_partition_sizes=[
                size // block_n for size in output_partition_sizes
            ],
            tp_rank=layer.tp_rank,
            tp_size=layer.tp_size,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return _block_fp8_runner()(
            input=x,
            weight=layer.weight,
            block_size=list(self.block_size),
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )


_METHOD_REGISTRY: dict[
    str,
    Callable[[dict[str, Any]], LinearQuantizationMethod],
] = {
    "fp8": BlockFP8LinearMethod.from_config,
}


def get_linear_quantization_method(
    hf_config: Any,
) -> LinearQuantizationMethod | None:
    """Build the registered linear adapter selected by an HF config."""
    raw = getattr(hf_config, "quantization_config", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raw = raw.to_dict()

    method = raw.get("quant_method")
    factory = _METHOD_REGISTRY.get(method)
    if factory is None:
        supported = ", ".join(sorted(_METHOD_REGISTRY))
        raise ValueError(
            f"Unsupported quantization method {method!r}; supported: {supported}"
        )
    return factory(raw)
