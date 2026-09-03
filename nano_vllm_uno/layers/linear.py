import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from collections.abc import Callable

from nano_vllm_uno.utils.context import get_context
from nano_vllm_uno.layers.quantization import LinearQuantizationMethod


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LoRALinearSlice(nn.Module):

    def __init__(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        output_start: int,
    ) -> None:
        super().__init__()
        self.register_buffer("lora_A", lora_A.contiguous(), persistent=False)
        self.register_buffer("lora_B", lora_B.contiguous(), persistent=False)
        self.scaling = float(scaling)
        self.output_start = int(output_start)
        self.output_end = int(output_start + lora_B.size(0))


class LoRAPackedBSlice(nn.Module):

    def __init__(
        self,
        lora_B: torch.Tensor,
        scaling: float,
        output_start: int,
    ) -> None:
        super().__init__()
        self.register_buffer("lora_B", lora_B.contiguous(), persistent=False)
        self.scaling = float(scaling)
        self.output_start = int(output_start)
        self.output_end = int(output_start + lora_B.size(0))


class LoRAPackedGroup(nn.Module):

    def __init__(self, slices: list[LoRALinearSlice]) -> None:
        super().__init__()
        self.register_buffer(
            "lora_A",
            torch.cat([s.lora_A for s in slices], dim=0).contiguous(),
            persistent=False,
        )
        self.b_slices = nn.ModuleList(
            [
                LoRAPackedBSlice(
                    s.lora_B,
                    s.scaling,
                    s.output_start,
                )
                for s in slices
            ]
        )
        self.ranks = [s.lora_A.size(0) for s in slices]


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_method: LinearQuantizationMethod | None = None,
        output_partition_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.input_size = input_size
        self.output_size = output_size
        self.output_partition_sizes = output_partition_sizes or [output_size]
        # Quantized weights may be FP8 or packed INT32, but LoRA operates in the
        # model activation dtype selected while the module is constructed.
        self.activation_dtype = torch.get_default_dtype()
        self.quant_method = quant_method
        if self.quant_method is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
        else:
            self.quant_method.create_weights(
                self,
                input_size,
                self.output_partition_sizes,
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        self.lora_slices = nn.ModuleList()
        self.lora_packed_groups = nn.ModuleList()
        self._lora_slices_packed = False
        self._lora_overlap_stream = None

    @property
    def base_weight_device(self) -> torch.device:
        """Device shared by the dense or serialized-quantized base tensors."""
        try:
            return next(self.parameters(recurse=False)).device
        except StopIteration as error:
            raise RuntimeError("Linear layer has no base parameters") from error

    def add_lora_slice(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        output_start: int = 0,
    ) -> None:
        self.lora_slices.append(
            LoRALinearSlice(
                lora_A.to(
                    device=self.base_weight_device,
                    dtype=self.activation_dtype,
                ),
                lora_B.to(
                    device=self.base_weight_device,
                    dtype=self.activation_dtype,
                ),
                scaling,
                output_start,
            )
        )

    def pack_lora_slices(self) -> dict[str, int]:
        if self._lora_slices_packed or len(self.lora_slices) <= 1:
            return {"packed_groups": 0, "packed_slices": 0}

        slices = sorted(self.lora_slices, key=lambda s: s.output_start)
        input_dim = slices[0].lora_A.size(1)
        rank = slices[0].lora_A.size(0)
        dtype = slices[0].lora_A.dtype
        device = slices[0].lora_A.device
        for lora_slice in slices:
            if (
                lora_slice.lora_A.size(1) != input_dim
                or lora_slice.lora_A.size(0) != rank
                or lora_slice.lora_B.size(1) != rank
                or lora_slice.lora_A.dtype != dtype
                or lora_slice.lora_B.dtype != dtype
                or lora_slice.lora_A.device != device
                or lora_slice.lora_B.device != device
            ):
                return {"packed_groups": 0, "packed_slices": 0}

        self.lora_packed_groups.append(LoRAPackedGroup(slices))
        self._lora_slices_packed = True
        return {"packed_groups": 1, "packed_slices": len(slices)}

    def _accumulate_lora_hidden(
        self,
        y_2d: torch.Tensor,
        lora_hidden: torch.Tensor,
        lora_slice: LoRALinearSlice | LoRAPackedBSlice,
    ) -> None:
        # Fuse the LoRA-B projection, adapter scaling, and accumulation into one
        # cuBLAS operation. This avoids materializing a full-size delta tensor.
        y_2d[:, lora_slice.output_start : lora_slice.output_end].addmm_(
            lora_hidden,
            lora_slice.lora_B.t(),
            beta=1.0,
            alpha=lora_slice.scaling,
        )

    def _lora_row_mask(
        self,
        num_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row_mask = getattr(get_context(), "lora_mask", None)
        if row_mask is None:
            raise RuntimeError("Gated LoRA requires an explicit row mask")
        row_mask = row_mask.reshape(-1).to(device=device, dtype=dtype)
        if row_mask.numel() != num_rows:
            raise RuntimeError(
                f"LoRA row mask has {row_mask.numel()} rows, "
                f"but linear input has {num_rows} rows"
            )
        return row_mask

    def set_lora_overlap_stream(self, stream: torch.cuda.Stream) -> None:
        self._lora_overlap_stream = stream

    def _get_lora_overlap_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._lora_overlap_stream is None:
            device_index = (
                torch.cuda.current_device() if device.index is None else device.index
            )
            self._lora_overlap_stream = torch.cuda.Stream(device=device_index)
        return self._lora_overlap_stream

    def _compute_lora_hidden(
        self,
        x_2d: torch.Tensor,
        row_mask: torch.Tensor,
    ):
        if self._lora_slices_packed:
            hidden_groups = []
            for packed_group in self.lora_packed_groups:
                lora_hidden = F.linear(x_2d, packed_group.lora_A)
                lora_hidden.mul_(row_mask[:, None])
                hidden_groups.append((packed_group, lora_hidden))
            return "packed", hidden_groups

        hidden_slices = []
        for lora_slice in self.lora_slices:
            lora_hidden = F.linear(x_2d, lora_slice.lora_A)
            lora_hidden.mul_(row_mask[:, None])
            hidden_slices.append((lora_slice, lora_hidden))
        return "slices", hidden_slices

    def _apply_lora_hidden(self, y_2d: torch.Tensor, hidden_result) -> None:
        result_kind, result_items = hidden_result
        if result_kind == "packed":
            for packed_group, lora_hidden in result_items:
                hidden_chunks = torch.split(lora_hidden, packed_group.ranks, dim=-1)
                for hidden, b_slice in zip(hidden_chunks, packed_group.b_slices):
                    self._accumulate_lora_hidden(y_2d, hidden, b_slice)
            return

        for lora_slice, lora_hidden in result_items:
            self._accumulate_lora_hidden(y_2d, lora_hidden, lora_slice)

    def _linear_with_lora(
        self,
        x: torch.Tensor,
        base_linear: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if not self.lora_slices:
            return base_linear()
        context = get_context()
        if not getattr(context, "lora_enabled", False):
            return base_linear()

        if (
            self.tp_size != 1
            or not x.is_cuda
            or not torch.cuda.is_available()
        ):
            return self._apply_lora(x, base_linear())

        x_2d = x.reshape(-1, x.shape[-1])
        row_mask = self._lora_row_mask(x_2d.size(0), x_2d.device, x_2d.dtype)

        main_stream = torch.cuda.current_stream(x_2d.device)
        lora_stream = self._get_lora_overlap_stream(x_2d.device)

        lora_stream.wait_stream(main_stream)
        with torch.cuda.stream(lora_stream):
            hidden_result = self._compute_lora_hidden(x_2d, row_mask)

        y = base_linear()
        main_stream.wait_stream(lora_stream)
        orig_y_shape = y.shape
        y_2d = y.reshape(-1, y.shape[-1])
        self._apply_lora_hidden(y_2d, hidden_result)
        return y_2d.reshape(orig_y_shape)

    def _apply_lora(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.lora_slices:
            return y
        context = get_context()
        if not getattr(context, "lora_enabled", False):
            return y

        orig_y_shape = y.shape
        x_2d = x.reshape(-1, x.shape[-1])
        y_2d = y.reshape(-1, y.shape[-1])
        row_mask = self._lora_row_mask(y_2d.size(0), y_2d.device, y_2d.dtype)

        x_lora = x_2d

        if self._lora_slices_packed:
            for packed_group in self.lora_packed_groups:
                lora_hidden = F.linear(x_lora, packed_group.lora_A)
                if self.tp_size > 1 and self.tp_dim == 1:
                    dist.all_reduce(lora_hidden)
                lora_hidden.mul_(row_mask[:, None])
                hidden_chunks = torch.split(lora_hidden, packed_group.ranks, dim=-1)
                for hidden, b_slice in zip(hidden_chunks, packed_group.b_slices):
                    self._accumulate_lora_hidden(y_2d, hidden, b_slice)
            return y_2d.reshape(orig_y_shape)

        for lora_slice in self.lora_slices:
            lora_hidden = F.linear(x_lora, lora_slice.lora_A)
            if self.tp_size > 1 and self.tp_dim == 1:
                dist.all_reduce(lora_hidden)
            lora_hidden.mul_(row_mask[:, None])
            self._accumulate_lora_hidden(y_2d, lora_hidden, lora_slice)
        return y_2d.reshape(orig_y_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _base_linear(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, bias)
        return F.linear(x, self.weight, bias)

    def process_weights_after_loading(self) -> None:
        if self.quant_method is not None:
            self.quant_method.process_weights_after_loading(self)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_method: LinearQuantizationMethod | None = None,
    ):
        super().__init__(
            input_size,
            output_size,
            bias,
            quant_method=quant_method,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_with_lora(
            x,
            lambda: self._base_linear(x, self.bias),
        )


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_method: LinearQuantizationMethod | None = None,
        output_partition_sizes: list[int] | None = None,
    ):
        tp_size = dist.get_world_size()
        local_partitions = (
            [divide(size, tp_size) for size in output_partition_sizes]
            if output_partition_sizes is not None
            else None
        )
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            0,
            quant_method=quant_method,
            output_partition_sizes=local_partitions,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_with_lora(
            x,
            lambda: self._base_linear(x, self.bias),
        )


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_method: LinearQuantizationMethod | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            quant_method=quant_method,
            output_partition_sizes=output_sizes,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_method: LinearQuantizationMethod | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(
            hidden_size,
            output_size,
            bias,
            quant_method=quant_method,
            output_partition_sizes=[
                total_num_heads * self.head_size,
                total_num_kv_heads * self.head_size,
                total_num_kv_heads * self.head_size,
            ],
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_method: LinearQuantizationMethod | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            1,
            quant_method=quant_method,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def base_linear() -> torch.Tensor:
            y = self._base_linear(
                x,
                self.bias if self.tp_rank == 0 else None,
            )
            if self.tp_size > 1:
                dist.all_reduce(y)
            return y

        return self._linear_with_lora(x, base_linear)
