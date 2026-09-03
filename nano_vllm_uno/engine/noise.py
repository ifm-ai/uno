from __future__ import annotations

import torch
from torch import Tensor

from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.sampling_params import SamplingParams


def _noise_bounds(
    sampling_params: SamplingParams,
    vocab_size: int,
) -> tuple[int, int]:
    """Return the half-open token range used for draft noise sampling."""
    if sampling_params.noise_mode in {
        "random_uniform",
        "deterministic_uniform",
    }:
        mask_token_id = sampling_params.mask_token_id
        if mask_token_id is None or int(mask_token_id) <= 1:
            raise ValueError(
                "uniform diffusion noise requires a valid mask_token_id"
            )
        # Match the I-DLM block sampler and Uno training convention:
        # random replacement IDs are sampled from [1, mask_token_id).
        return 1, min(int(mask_token_id), int(vocab_size))

    mask_token_id = sampling_params.mask_token_id
    if mask_token_id is not None and int(mask_token_id) > 1:
        return 1, min(int(mask_token_id), int(vocab_size))
    raise ValueError(
        "two-pass decoding requires mask_token_id on SamplingParams; "
        "noise tokens must be bounded by the diffusion-training mask token."
    )


def _noise_salt(seq: Sequence, sampling_params: SamplingParams) -> int:
    """Return the per-sequence salt for deterministic draft noise."""
    salt = sampling_params.noise_salt
    return int(seq.seq_id if salt is None else salt)


def _mix_u64(value: int) -> int:
    """Mix one integer into a stable 64-bit pseudo-random value."""
    value &= 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    value &= 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB
    value &= 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def _sequence_noise_seed(seq: Sequence) -> int:
    """Hash the prompt once to seed deterministic noise for this request."""
    seed = getattr(seq, "_two_pass_noise_seed", None)
    if seed is not None:
        return int(seed)
    seed = 0xD6E8FEB86659FD93
    for token in seq.prompt_token_ids:
        seed = _mix_u64(seed ^ int(token))
    seq._two_pass_noise_seed = int(seed)
    return int(seed)


def _make_deterministic_noise(
    seq: Sequence,
    count: int,
    low: int,
    high: int,
    seed_token: int,
    sampling_params: SamplingParams,
    device: torch.device,
) -> Tensor:
    """Create reproducible uniform-looking noise tokens without RNG state."""
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    span = max(1, int(high) - int(low))
    base = (
        _sequence_noise_seed(seq) * 0x9E3779B185EBCA87
        + _noise_salt(seq, sampling_params) * 0xD1B54A32D192ED03
        + int(seq.num_completion_tokens) * 0xC2B2AE3D27D4EB4F
        + int(seed_token) * 0x165667B19E3779F9
        + int(len(seq)) * 0x85EBCA77C2B2AE63
    )
    tokens = [
        int(low) + (_mix_u64(base + slot * 0x27D4EB2F165667C5) % span)
        for slot in range(count)
    ]
    return torch.tensor(tokens, dtype=torch.long, device=device)


def _make_noise(
    seq: Sequence,
    count: int,
    seed_token: int,
    sampling_params: SamplingParams,
    *,
    vocab_size: int,
    device: torch.device,
) -> Tensor:
    """Create draft noise tokens according to the configured noise mode."""
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    low, high = _noise_bounds(sampling_params, vocab_size)
    mode = str(sampling_params.noise_mode)
    if mode == "deterministic_uniform":
        return _make_deterministic_noise(
            seq,
            count,
            low,
            high,
            seed_token,
            sampling_params,
            device,
        )
    if mode == "random_uniform":
        return torch.randint(
            low,
            high,
            (count,),
            dtype=torch.long,
            device=device,
        )
    if mode == "mask":
        return torch.full(
            (count,),
            high,
            dtype=torch.long,
            device=device,
        )
    raise ValueError(f"Unsupported noise_mode={mode!r}")


def build_draft_batch(
    seqs: list[Sequence],
    block_len: int,
    sampling_params: SamplingParams,
    *,
    vocab_size: int,
    device: torch.device,
) -> Tensor:
    """Build the two-pass draft inputs as ``[seed, noise...]`` rows."""
    batch_size = len(seqs)
    draft = torch.empty(
        (batch_size, block_len),
        dtype=torch.long,
        device=device,
    )
    draft[:, 0] = torch.tensor(
        [int(seq.token_ids[-1]) for seq in seqs],
        dtype=torch.long,
        device=device,
    )
    if block_len <= 1:
        return draft

    mode = str(sampling_params.noise_mode)
    low, high = _noise_bounds(sampling_params, vocab_size)
    if mode == "random_uniform":
        draft[:, 1:] = torch.randint(
            low,
            high,
            (batch_size, block_len - 1),
            dtype=torch.long,
            device=device,
        )
    elif mode == "mask":
        draft[:, 1:] = high
    else:
        for row, seq in enumerate(seqs):
            draft[row, 1:] = _make_noise(
                seq,
                block_len - 1,
                seed_token=int(seq.token_ids[-1]),
                sampling_params=sampling_params,
                vocab_size=vocab_size,
                device=device,
            )
    return draft
