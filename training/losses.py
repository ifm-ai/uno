from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import IGNORE_INDEX


def shift_labels_for_next_token(
    labels: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    """Shift causal labels and mask packed-sequence boundaries."""
    shifted = torch.full_like(labels, IGNORE_INDEX)
    shifted[:, :-1] = labels[:, 1:]
    shifted[:, :-1].masked_fill_(
        position_ids[:, :-1] >= position_ids[:, 1:], IGNORE_INDEX
    )
    return shifted


def project_hidden_states(
    lm_head, hidden_states: torch.Tensor, masks: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    union_mask = torch.zeros_like(next(iter(masks.values())), dtype=torch.bool)
    for mask in masks.values():
        union_mask |= mask
    if not union_mask.any():
        empty = hidden_states.new_empty((0, lm_head.weight.shape[0]))
        return {name: empty for name in masks}
    union_logits = lm_head(hidden_states[union_mask])
    union_indices = union_mask.reshape(-1).long().cumsum(0) - 1
    return {
        name: union_logits[union_indices[mask.reshape(-1)]]
        for name, mask in masks.items()
    }


def resolve_ce_targets(
    target_source: str,
    shifted_labels: torch.Tensor,
    ce_mask: torch.Tensor,
    clean_teacher_logits: torch.Tensor | None,
) -> torch.Tensor:
    if target_source == "data":
        return shifted_labels[ce_mask]
    if target_source != "clean_argmax":
        raise ValueError(f"Unsupported CE target source: {target_source}.")
    if clean_teacher_logits is None:
        raise RuntimeError("Teacher CE requires clean teacher logits.")
    expected = int(ce_mask.sum().item())
    if clean_teacher_logits.size(0) != expected:
        raise RuntimeError(
            "Teacher logits do not align with noised CE positions: "
            f"{clean_teacher_logits.size(0)} rows for {expected} positions."
        )
    return clean_teacher_logits.detach().argmax(dim=-1)


def token_normalized_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Compute KL(student || detached base teacher) per supervised token."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    value = F.kl_div(
        teacher_log_probs,
        student_log_probs,
        log_target=True,
        reduction="sum",
    )
    return value / max(num_tokens, 1)


class _ChunkedTotalVariation(torch.autograd.Function):
    """Exact full-vocabulary L1 distance without dense FP32 probabilities."""

    @staticmethod
    def _normalizer(logits, row_max, chunk_size):
        denominator = torch.zeros_like(row_max, dtype=torch.float32)
        for start in range(0, logits.size(-1), chunk_size):
            chunk = logits[:, start : start + chunk_size].float()
            denominator.add_(torch.exp(chunk - row_max[:, None]).sum(dim=-1))
        return denominator.clamp_min_(torch.finfo(torch.float32).tiny)

    @staticmethod
    def forward(ctx, student_logits, teacher_logits, num_tokens, chunk_size):
        student_row_max = student_logits.amax(dim=-1).float()
        teacher_row_max = teacher_logits.amax(dim=-1).float()
        student_denominator = _ChunkedTotalVariation._normalizer(
            student_logits, student_row_max, chunk_size
        )
        teacher_denominator = _ChunkedTotalVariation._normalizer(
            teacher_logits, teacher_row_max, chunk_size
        )
        l1_sum = student_logits.new_zeros((), dtype=torch.float32)
        for start in range(0, student_logits.size(-1), chunk_size):
            student_chunk = student_logits[:, start : start + chunk_size].float()
            teacher_chunk = teacher_logits[:, start : start + chunk_size].float()
            student_probs = torch.exp(
                student_chunk - student_row_max[:, None]
            ) / student_denominator[:, None]
            teacher_probs = torch.exp(
                teacher_chunk - teacher_row_max[:, None]
            ) / teacher_denominator[:, None]
            l1_sum.add_((student_probs - teacher_probs).abs().sum())
        ctx.save_for_backward(
            student_logits,
            teacher_logits,
            student_row_max,
            teacher_row_max,
            student_denominator,
            teacher_denominator,
        )
        ctx.chunk_size = chunk_size
        ctx.normalizer = max(num_tokens, 1)
        return l1_sum / ctx.normalizer

    @staticmethod
    def backward(ctx, grad_output):
        (
            student_logits,
            teacher_logits,
            student_row_max,
            teacher_row_max,
            student_denominator,
            teacher_denominator,
        ) = ctx.saved_tensors
        signed_probability_sum = torch.zeros_like(student_row_max, dtype=torch.float32)
        for start in range(0, student_logits.size(-1), ctx.chunk_size):
            student_chunk = student_logits[:, start : start + ctx.chunk_size].float()
            teacher_chunk = teacher_logits[:, start : start + ctx.chunk_size].float()
            student_probs = torch.exp(
                student_chunk - student_row_max[:, None]
            ) / student_denominator[:, None]
            teacher_probs = torch.exp(
                teacher_chunk - teacher_row_max[:, None]
            ) / teacher_denominator[:, None]
            signs = torch.sign(student_probs - teacher_probs)
            signed_probability_sum.add_((signs * student_probs).sum(dim=-1))
        gradient = torch.empty_like(student_logits)
        scale = grad_output.float() / ctx.normalizer
        for start in range(0, student_logits.size(-1), ctx.chunk_size):
            student_chunk = student_logits[:, start : start + ctx.chunk_size].float()
            teacher_chunk = teacher_logits[:, start : start + ctx.chunk_size].float()
            student_probs = torch.exp(
                student_chunk - student_row_max[:, None]
            ) / student_denominator[:, None]
            teacher_probs = torch.exp(
                teacher_chunk - teacher_row_max[:, None]
            ) / teacher_denominator[:, None]
            signs = torch.sign(student_probs - teacher_probs)
            chunk_gradient = student_probs * (
                signs - signed_probability_sum[:, None]
            )
            gradient[:, start : start + ctx.chunk_size].copy_(
                (chunk_gradient * scale).to(student_logits.dtype)
            )
        return gradient, None, None, None


def token_normalized_total_variation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    num_tokens: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes.")
    if chunk_size <= 0:
        raise ValueError("TV chunk_size must be positive.")
    vocabulary_size = student_logits.size(-1)
    student_flat = student_logits.reshape(-1, vocabulary_size)
    teacher_flat = teacher_logits.detach().reshape(-1, vocabulary_size)
    if not student_flat.size(0):
        return student_flat.sum()
    return _ChunkedTotalVariation.apply(
        student_flat,
        teacher_flat,
        num_tokens,
        min(chunk_size, vocabulary_size),
    )
