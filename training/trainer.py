from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

from .config import UnoObjectiveConfig, write_json_atomic
from .constants import IGNORE_INDEX
from .curriculum import BlockCurriculumPlan, set_model_block_size
from .lora import TokenwiseLoraRouter, make_saved_adapter_portable
from .losses import (
    project_hidden_states,
    resolve_ce_targets,
    shift_labels_for_next_token,
    token_normalized_reverse_kl,
    token_normalized_total_variation,
)
from .modeling import (
    extract_uno_regions,
    prepare_uno_layout,
    run_prepared_uno_forward,
    unwrap_base_model,
    validate_uno_model,
)


LOGGER = logging.getLogger(__name__)


def combine_objective_losses(
    ce_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    tv_loss: torch.Tensor,
    objective: UnoObjectiveConfig,
) -> torch.Tensor:
    """Combine active Uno losses using the configured alpha, beta, and gamma."""
    objective.validate()
    return (
        objective.ce_weight * ce_loss
        + objective.kl_weight * kl_loss
        + objective.tv_weight * tv_loss
    )


def sync_scheduler_param_groups(lr_scheduler) -> None:
    """Repair the scheduler metadata if DeepSpeed consolidates optimizer groups."""
    if lr_scheduler is None:
        return
    optimizer = getattr(lr_scheduler, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    base_lrs = getattr(lr_scheduler, "base_lrs", None)
    if param_groups is None or base_lrs is None:
        return
    target_count = len(param_groups)
    if len(base_lrs) == target_count and (
        not hasattr(lr_scheduler, "lr_lambdas")
        or len(lr_scheduler.lr_lambdas) == target_count
    ):
        return
    if len({float(value) for value in base_lrs}) != 1:
        raise RuntimeError(
            "DeepSpeed changed optimizer groups, but the LR scheduler has distinct "
            f"base learning rates: {base_lrs}."
        )
    lr_scheduler.base_lrs = [
        group.get("initial_lr", group.get("lr", 0.0)) for group in param_groups
    ]
    if hasattr(lr_scheduler, "lr_lambdas"):
        lambdas = list(lr_scheduler.lr_lambdas)
        if not lambdas:
            raise RuntimeError("Cannot synchronize an empty LR-lambda list.")
        lambdas.extend([lambdas[-1]] * max(0, target_count - len(lambdas)))
        lr_scheduler.lr_lambdas = lambdas[:target_count]
    if hasattr(lr_scheduler, "_last_lr"):
        lr_scheduler._last_lr = [group["lr"] for group in param_groups]


class SchedulerSyncCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        del args, state
        sync_scheduler_param_groups(kwargs.get("lr_scheduler"))
        return control


class CurriculumCheckpointCallback(TrainerCallback):
    """Select block sizes and save complete state at every stage boundary."""

    def __init__(self, curriculum: BlockCurriculumPlan) -> None:
        self.curriculum = curriculum

    def on_step_begin(self, args, state, control, **kwargs):
        del args
        if state.global_step < self.curriculum.max_steps:
            set_model_block_size(
                kwargs["model"],
                self.curriculum.stage_for_step(state.global_step).block_size,
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step in self.curriculum.checkpoint_reasons:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        if not state.is_world_process_zero:
            return control

        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        make_saved_adapter_portable(checkpoint)
        reason = self.curriculum.checkpoint_reasons.get(state.global_step)
        if reason:
            write_json_atomic(
                checkpoint / "uno_curriculum_state.json",
                {
                    "version": 1,
                    "global_step": state.global_step,
                    "reason": reason,
                    "curriculum_sha256": self.curriculum.sha256(),
                },
            )
        return control


def validate_curriculum_resume(
    checkpoint: str | Path, curriculum: BlockCurriculumPlan
) -> None:
    checkpoint = Path(checkpoint)
    state_path = checkpoint / "uno_curriculum_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            "Uno resumes are supported only from curriculum-boundary checkpoints; "
            f"missing {state_path}."
        )
    state = json.loads(state_path.read_text())
    if state.get("curriculum_sha256") != curriculum.sha256():
        raise ValueError("Checkpoint curriculum does not match the current curriculum.")
    step = state.get("global_step")
    if step not in curriculum.checkpoint_reasons:
        raise ValueError(f"Checkpoint step {step} is not a curriculum boundary.")


class UnoTrainer(Trainer):
    """LoRA-only, single-forward Uno trainer."""

    def __init__(
        self,
        *,
        objective: UnoObjectiveConfig,
        curriculum: BlockCurriculumPlan,
        **kwargs,
    ) -> None:
        objective.validate()
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False
        self.objective = objective
        self.curriculum = curriculum
        validate_uno_model(self.model)
        self._lora_router = TokenwiseLoraRouter(self.model)
        self.add_callback(SchedulerSyncCallback())
        self.add_callback(CurriculumCheckpointCallback(curriculum))

    def create_scheduler(self, num_training_steps, optimizer=None):
        scheduler = super().create_scheduler(num_training_steps, optimizer)
        sync_scheduler_param_groups(scheduler)
        return scheduler

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        step = min(self.state.global_step, self.curriculum.max_steps - 1)
        block_size = self.curriculum.stage_for_step(step).block_size
        set_model_block_size(model, block_size)

        labels = inputs.get("labels")
        if labels is None:
            raise RuntimeError("Uno training batches must contain labels.")
        unwrapped = self.accelerator.unwrap_model(model)
        base_model = unwrap_base_model(unwrapped)
        sequence_length = inputs["input_ids"].size(-1)
        original_positions, model_positions, noisy_region_mask = prepare_uno_layout(
            base_model,
            inputs["input_ids"],
            inputs.get("position_ids"),
        )
        shifted_labels = shift_labels_for_next_token(labels, original_positions)
        (
            concatenated_ids,
            concatenated_positions,
            flex_attention_mask,
            noised_positions,
            _,
            p_mask,
        ) = base_model.prepare_for_bd_training(
            inputs["input_ids"],
            model_positions,
            labels == IGNORE_INDEX,
        )
        self._lora_router.set_token_mask(noisy_region_mask)
        hidden_states = run_prepared_uno_forward(
            model,
            base_model,
            inputs,
            concatenated_ids,
            concatenated_positions,
            flex_attention_mask,
        )
        noisy_hidden, clean_hidden = extract_uno_regions(
            hidden_states, noisy_region_mask, sequence_length
        )

        supervised_mask = labels != IGNORE_INDEX
        valid_next_token = shifted_labels != IGNORE_INDEX
        ce_mask = noised_positions & valid_next_token
        distribution_active = (
            self.objective.kl_weight > 0 or self.objective.tv_weight > 0
        )
        student_masks = {}
        teacher_masks = {}
        if distribution_active:
            student_masks["distribution"] = supervised_mask
            teacher_masks["distribution"] = supervised_mask
        if self.objective.ce_weight > 0:
            student_masks["ce"] = ce_mask
            if self.objective.ce_target == "teacher":
                teacher_masks["ce"] = ce_mask
        student_logits = project_hidden_states(
            unwrapped.lm_head,
            noisy_hidden,
            student_masks,
        )
        teacher_logits = (
            project_hidden_states(unwrapped.lm_head, clean_hidden, teacher_masks)
            if teacher_masks
            else {}
        )
        token_count = int(supervised_mask.sum().item())
        zero_loss = noisy_hidden.sum() * 0.0
        if self.objective.kl_weight > 0:
            kl_loss = token_normalized_reverse_kl(
                student_logits["distribution"],
                teacher_logits["distribution"],
                token_count,
            )
        else:
            kl_loss = zero_loss
        if self.objective.tv_weight > 0:
            tv_loss = token_normalized_total_variation(
                student_logits["distribution"],
                teacher_logits["distribution"],
                token_count,
            )
        else:
            tv_loss = zero_loss

        ce_count = int(ce_mask.sum().item())
        if self.objective.ce_weight > 0 and ce_count:
            targets = resolve_ce_targets(
                self.objective.internal_ce_target,
                shifted_labels,
                ce_mask,
                teacher_logits["ce"],
            )
            per_token_ce = F.cross_entropy(
                student_logits["ce"], targets, reduction="none"
            )
            selected_probabilities = p_mask[valid_next_token[noised_positions]]
            if selected_probabilities.numel() != per_token_ce.numel():
                raise RuntimeError("Noise probabilities do not align with Uno CE rows.")
            ce_loss = (per_token_ce / selected_probabilities).sum() / max(token_count, 1)
        else:
            ce_loss = zero_loss

        combined_loss = combine_objective_losses(
            ce_loss, kl_loss, tv_loss, self.objective
        )
        self.log(
            {
                "train/ce_loss": ce_loss.detach().item(),
                "train/reverse_kl_loss": kl_loss.detach().item(),
                "train/tv_loss": tv_loss.detach().item(),
                "train/combined_loss": combined_loss.detach().item(),
                "train/alpha": self.objective.ce_weight,
                "train/beta": self.objective.kl_weight,
                "train/gamma": self.objective.tv_weight,
                "train/block_size": block_size,
                "train/transformer_forwards": 1,
            }
        )
        if return_outputs:
            return combined_loss, {
                "loss": combined_loss,
                "ce_loss": ce_loss,
                "reverse_kl_loss": kl_loss,
                "tv_loss": tv_loss,
            }
        return combined_loss
