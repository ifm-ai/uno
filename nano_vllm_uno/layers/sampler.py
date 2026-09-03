from __future__ import annotations

import torch
from torch import Tensor, nn

from nano_vllm_uno.sampling_params import SamplingParams

try:
    from flashinfer import top_k as _flashinfer_top_k
except ImportError:
    _flashinfer_top_k = None


@torch.inference_mode()
def sample_from_probs(probs: Tensor) -> Tensor:
    """Draw one token per row by Gumbel-max."""
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div(noise).argmax(dim=-1)


@torch.inference_mode()
@torch.compile(dynamic=True)
def build_sparse_top_k_probs(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float | None,
) -> tuple[Tensor, Tensor]:
    """Return sparse token IDs and probabilities after top-k/top-p filtering."""
    nucleus = 1.0 if top_p is None else float(top_p)
    if logits.is_cuda and _flashinfer_top_k is not None:
        values, indices = _flashinfer_top_k(
            logits.contiguous(),
            int(top_k),
            sorted=True,
            deterministic=False,
        )
    else:
        # Select before FP32 promotion so only [rows, top_k] is widened.
        values, indices = torch.topk(logits, k=int(top_k), dim=-1)
    probs = torch.softmax(values.float() / float(temperature), dim=-1)
    if nucleus < 1.0:
        cdf = torch.cumsum(probs, dim=-1)
        probs.masked_fill_((cdf - probs) > nucleus, 0.0)
        probs /= probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return indices, probs


class Sampler(nn.Module):
    """Sample a logits batch using one configuration shared by every row."""

    @torch.inference_mode()
    def forward(
        self,
        logits: Tensor,
        sampling_params: SamplingParams,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Sample tokens and retain q when it has compact top-k support."""
        temperature = sampling_params.temperature
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1), None
        if top_p is not None and top_p < 1.0 and (
            top_k is None or not 0 < int(top_k) < logits.size(-1)
        ):
            raise ValueError("top_p requires top_k smaller than the model vocabulary")
        if top_k is not None and 0 < int(top_k) < logits.size(-1):
            indices, probs = build_sparse_top_k_probs(
                logits,
                temperature,
                int(top_k),
                top_p,
            )
            offsets = sample_from_probs(probs)
            tokens = indices.gather(1, offsets.unsqueeze(1)).view(-1)
            return tokens, (indices, probs)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        tokens = sample_from_probs(probs)
        return tokens, None
