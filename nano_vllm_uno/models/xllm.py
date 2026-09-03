import torch
import torch.distributed as dist
from torch import nn

from nano_vllm_uno.layers.activation import SiluAndMul
from nano_vllm_uno.layers.attention import Attention
from nano_vllm_uno.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nano_vllm_uno.layers.layernorm import RMSNorm
from nano_vllm_uno.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nano_vllm_uno.layers.quantization import (
    LinearQuantizationMethod,
    get_linear_quantization_method,
)
from nano_vllm_uno.layers.rotary_embedding import get_rope

try:
    from xllm_extension.ops import group_rms_norm_fwd_affine
except ImportError:
    group_rms_norm_fwd_affine = None


class XllmGroupedRMSNorm(RMSNorm):

    def __init__(
        self,
        hidden_size: int,
        n_groups: int,
        eps: float = 1e-6,
    ) -> None:
        if hidden_size % n_groups != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_groups={n_groups}"
            )
        super().__init__(hidden_size, eps=eps)
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.group_size = hidden_size // n_groups

    def _fused_grouped_rms_forward(self, x: torch.Tensor) -> torch.Tensor | None:
        if group_rms_norm_fwd_affine is None or not x.is_cuda:
            return None
        original_shape = x.shape
        x_3d = x.reshape(1, -1, self.hidden_size) if x.dim() == 2 else x
        output, _ = group_rms_norm_fwd_affine(
            x_3d,
            self.hidden_size,
            self.n_groups,
            self.weight,
            self.eps,
        )
        return output.reshape(original_shape) if x.dim() == 2 else output

    def _grouped_rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._fused_grouped_rms_forward(x)
        if output is not None:
            return output
        original_shape = x.shape
        original_dtype = x.dtype
        x = x.float().reshape(
            *original_shape[:-1], self.n_groups, self.group_size
        )
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(variance + self.eps))
        return x.reshape(original_shape).to(original_dtype).mul_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._grouped_rms_forward(x)
        x = x.float().add_(residual.float())
        residual = x.to(residual.dtype)
        return self._grouped_rms_forward(residual), residual


class XllmAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        qkv_bias: bool,
        rope_theta: float,
        rope_scaling: dict | None,
        rope_head_dim: int,
        query_key_norm: bool,
        rms_norm_eps: float,
        quant_method: LinearQuantizationMethod | None = None,
        attention_backend: str = "fa3",
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={num_heads} must be divisible by tp_size={tp_size}"
            )
        if self.total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={num_kv_heads} must be divisible by tp_size={tp_size}"
            )
        if rope_head_dim != head_dim:
            raise NotImplementedError(
                "Nano XLLM currently requires rope_head_dim == head_dim"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_method=quant_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_method=quant_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            attention_backend,
        )
        self.query_key_norm = query_key_norm
        if self.query_key_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if self.query_key_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
        return self.o_proj(output.flatten(1, -1))


class XllmMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_method: LinearQuantizationMethod | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_method=quant_method,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_method=quant_method,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


def _rope_config(config) -> tuple[float, dict | None]:
    rope_parameters = getattr(config, "rope_parameters", None)
    rope_scaling = getattr(config, "rope_scaling", None)
    descriptor = rope_scaling if rope_scaling is not None else rope_parameters
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None and isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get("rope_theta")
    return float(rope_theta if rope_theta is not None else 10000.0), descriptor


class XllmDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        quant_method: LinearQuantizationMethod | None = None,
        attention_backend: str = "fa3",
    ) -> None:
        super().__init__()
        if getattr(config, "num_experts", 0) != 0:
            raise NotImplementedError("Nano XLLM supports dense checkpoints only")
        if getattr(config, "attention_gate_func", None) is not None:
            raise NotImplementedError("Nano XLLM does not support attention gating")
        if getattr(config, "use_sliding_window", False):
            raise NotImplementedError("Nano XLLM does not support sliding-window attention")
        rope_theta, rope_scaling = _rope_config(config)
        self.self_attn = XllmAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_head_dim=getattr(config, "rope_head_dim", config.head_dim),
            query_key_norm=getattr(config, "query_key_norm", False),
            rms_norm_eps=config.rms_norm_eps,
            quant_method=quant_method,
            attention_backend=attention_backend,
        )
        self.mlp = XllmMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_method=quant_method,
        )
        n_groups = getattr(config, "layernorm_num_groups", 1)
        self.input_layernorm = XllmGroupedRMSNorm(
            config.hidden_size, n_groups, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = XllmGroupedRMSNorm(
            config.hidden_size, n_groups, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        return self.mlp(hidden_states), residual


class XllmModel(nn.Module):

    def __init__(self, config, attention_backend: str = "fa3") -> None:
        super().__init__()
        if getattr(config, "num_experts", 0) != 0:
            raise NotImplementedError("Nano XLLM supports dense checkpoints only")
        quant_method = get_linear_quantization_method(config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                XllmDecoderLayer(config, quant_method, attention_backend)
                for _ in range(config.num_hidden_layers)
            ]
        )
        n_groups = getattr(config, "layernorm_num_groups", 1)
        self.norm = XllmGroupedRMSNorm(
            config.hidden_size, n_groups, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class XllmForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, attention_backend: str = "fa3") -> None:
        super().__init__()
        self.model = XllmModel(config, attention_backend)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
