from functools import lru_cache
import math

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        inv_freq: torch.Tensor | None = None,
        attention_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        if inv_freq is None:
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, rotary_dim, 2, dtype=torch.float)
                    / rotary_dim
                )
            )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


YaRNKey = tuple[float, int, float, float, float, bool]


def _normalize_rope_scaling(
    rope_scaling: dict | None,
    base: float,
    max_position: int,
) -> YaRNKey | None:
    if rope_scaling is None:
        return None
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type in (None, "default"):
        rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is not None and float(rope_theta) != float(base):
            raise NotImplementedError(
                f"Default RoPE descriptor has rope_theta={rope_theta}, but base={base}"
            )
        return None
    if rope_type == "yarn":
        factor = float(rope_scaling["factor"])
        if factor <= 0:
            raise ValueError(f"YaRN factor must be positive, got {factor}")
        original_max_position = int(
            rope_scaling.get("original_max_position_embeddings") or max_position
        )
        if original_max_position <= 0:
            raise ValueError(
                "YaRN original_max_position_embeddings must be positive, "
                f"got {original_max_position}"
            )
        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is None:
            attention_factor = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
        return (
            factor,
            original_max_position,
            float(attention_factor),
            float(rope_scaling.get("beta_fast") or 32.0),
            float(rope_scaling.get("beta_slow") or 1.0),
            bool(rope_scaling.get("truncate", True)),
        )
    raise NotImplementedError(f"Unsupported rope_scaling={rope_scaling!r}")


def _yarn_inv_freq(
    rotary_dim: int,
    base: float,
    yarn: YaRNKey,
) -> tuple[torch.Tensor, float]:
    (
        factor,
        original_max_position,
        attention_factor,
        beta_fast,
        beta_slow,
        truncate,
    ) = yarn

    def correction_dim(rotations: float) -> float:
        return rotary_dim * math.log(
            original_max_position / (rotations * 2 * math.pi)
        ) / (2 * math.log(base))

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rotary_dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
    )
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    ramp = torch.clamp(
        (torch.arange(rotary_dim // 2, dtype=torch.float) - low) / (high - low),
        0,
        1,
    )
    extrapolation_weight = 1 - ramp
    inv_freq = (
        interpolation * (1 - extrapolation_weight)
        + extrapolation * extrapolation_weight
    )
    return inv_freq, attention_factor


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling_key: YaRNKey | None = None,
):
    inv_freq = None
    attention_factor = 1.0
    if rope_scaling_key is not None:
        inv_freq, attention_factor = _yarn_inv_freq(
            rotary_dim,
            base,
            rope_scaling_key,
        )
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        inv_freq=inv_freq,
        attention_factor=attention_factor,
    )
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    rope_scaling_key = _normalize_rope_scaling(
        rope_scaling,
        base,
        max_position,
    )
    return _get_rope_cached(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_scaling_key,
    )
