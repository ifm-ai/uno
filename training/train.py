from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from .checkpointing import configure_gradient_checkpointing
from .config import (
    LoraSettings,
    UnoObjectiveConfig,
    dataclass_dict,
    manifest_digest,
    validate_resume_manifest,
    validate_wandb_auth,
    write_json_atomic,
)
from .constants import (
    DEFAULT_CE_ALPHA,
    DEFAULT_KL_BETA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SEED,
    DEFAULT_TV_GAMMA,
    DEFAULT_WARMUP_STEPS,
)
from .curriculum import BlockCurriculumPlan
from .data import UnoDataCollator, read_dataset_manifest, validate_dataset_manifest
from .lora import (
    create_lora_model,
    make_saved_adapter_portable,
    validate_only_lora_trainable,
)
from .modeling import (
    configure_uniform_noise,
    configure_uniform_noise_config,
    validate_uno_model,
)
from .trainer import UnoTrainer, validate_curriculum_resume


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRICULUM = REPOSITORY_ROOT / "training/configs/uno_3epoch_curriculum.yaml"
DEFAULT_DEEPSPEED = REPOSITORY_ROOT / "training/configs/deepspeed_zero2.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA-only Uno adapter.")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--deepspeed", type=Path, default=DEFAULT_DEEPSPEED)
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--ce-target", choices=("teacher", "ground_truth"), default="teacher"
    )
    parser.add_argument("--ce-alpha", type=float, default=DEFAULT_CE_ALPHA)
    parser.add_argument("--kl-beta", type=float, default=DEFAULT_KL_BETA)
    parser.add_argument("--tv-gamma", type=float, default=DEFAULT_TV_GAMMA)
    parser.add_argument("--lora-target", default="all")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--run-name", default="uno-qwen3-8b-3epoch")
    parser.add_argument("--wandb-project", default="uno-training")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    return parser


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _resolve_model_snapshot(
    model_cache_dir: Path | None,
    *,
    local_files_only: bool,
) -> Path:
    return Path(
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            revision=DEFAULT_MODEL_REVISION,
            cache_dir=str(model_cache_dir) if model_cache_dir else None,
            local_files_only=local_files_only,
        )
    )


def _load_base_model(model_snapshot: Path):
    load_source = str(model_snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        load_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(
        load_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    configure_uniform_noise_config(model_config)
    model = AutoModelForCausalLM.from_pretrained(
        load_source,
        config=model_config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    )
    return tokenizer, model


def _compatibility_manifest(
    *,
    dataset_manifest: dict,
    curriculum: BlockCurriculumPlan,
    objective: UnoObjectiveConfig,
    lora: LoraSettings,
    resolved_targets: tuple[str, ...],
    args: argparse.Namespace,
) -> dict:
    return {
        "version": 1,
        "model": {
            "id": DEFAULT_MODEL_ID,
            "revision": DEFAULT_MODEL_REVISION,
            "trust_remote_code": True,
        },
        "dataset_hash": dataset_manifest["hash"],
        "curriculum": curriculum.manifest(),
        "curriculum_sha256": curriculum.sha256(),
        "objective": dataclass_dict(objective),
        "lora": {
            **dataclass_dict(lora),
            "alpha": lora.resolved_alpha,
            "resolved_targets": list(resolved_targets),
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "stable_steps": curriculum.max_steps - args.warmup_steps,
            "decay_steps": 0,
            "max_steps": curriculum.max_steps,
            "global_batch_size": curriculum.global_batch_size,
            "seed": DEFAULT_SEED,
            "precision": "bf16",
            "deepspeed": "zero2",
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    objective = UnoObjectiveConfig(
        ce_target=args.ce_target,
        ce_weight=args.ce_alpha,
        kl_weight=args.kl_beta,
        tv_weight=args.tv_gamma,
    )
    objective.validate()
    lora = LoraSettings(
        target=args.lora_target,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    lora.validate()
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive.")
    if args.learning_rate <= 0 or args.warmup_steps < 0:
        raise ValueError("Learning rate must be positive and warmup steps non-negative.")
    validate_wandb_auth(args.wandb_mode)

    dataset_manifest = read_dataset_manifest(args.dataset_path)
    validate_dataset_manifest(dataset_manifest)
    curriculum = BlockCurriculumPlan.from_yaml(args.curriculum)
    if args.warmup_steps >= curriculum.max_steps:
        raise ValueError("Warmup steps must be smaller than total curriculum steps.")
    curriculum.validate_runtime(
        world_size=_world_size(),
        per_device_batch=args.per_device_batch_size,
        accumulation=args.gradient_accumulation_steps,
    )
    if not args.deepspeed.is_file():
        raise FileNotFoundError(f"DeepSpeed configuration not found: {args.deepspeed}")

    model_snapshot = _resolve_model_snapshot(
        args.model_cache_dir,
        local_files_only=args.local_files_only,
    )
    tokenizer, model = _load_base_model(model_snapshot)
    validate_uno_model(model)
    configure_uniform_noise(model)
    configure_gradient_checkpointing(model)
    model, resolved_targets = create_lora_model(model, lora)
    trainable_parameters, total_parameters = validate_only_lora_trainable(model)

    compatibility = _compatibility_manifest(
        dataset_manifest=dataset_manifest,
        curriculum=curriculum,
        objective=objective,
        lora=lora,
        resolved_targets=resolved_targets,
        args=args,
    )
    manifest_path = args.output_dir / "uno_training_manifest.json"
    if args.resume_from_checkpoint:
        validate_resume_manifest(args.output_dir, compatibility)
        validate_curriculum_resume(args.resume_from_checkpoint, curriculum)
    elif _rank() == 0 and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; use --resume-from-checkpoint: {args.output_dir}"
        )
    if _rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            manifest_path,
            {
                **compatibility,
                "compatibility_sha256": manifest_digest(compatibility),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_name": args.run_name,
                "wandb": {
                    "project": args.wandb_project,
                    "mode": args.wandb_mode,
                },
                "parameters": {
                    "trainable": trainable_parameters,
                    "total_with_adapter": total_parameters,
                },
                "runtime_batch": {
                    "world_size": _world_size(),
                    "per_device_batch_size": args.per_device_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                },
            },
        )

    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_PROJECT"] = args.wandb_project
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        max_steps=curriculum.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": 0},
        warmup_steps=args.warmup_steps,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        deepspeed=str(args.deepspeed),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        save_strategy="no",
        save_total_limit=8,
        save_only_model=False,
        save_safetensors=True,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        report_to=[] if args.wandb_mode == "disabled" else ["wandb"],
        seed=curriculum.seed,
        data_seed=curriculum.seed,
        ddp_find_unused_parameters=False,
        ddp_timeout=180_000_000,
    )
    dataset = load_from_disk(str(args.dataset_path))
    trainer = UnoTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=UnoDataCollator(tokenizer),
        processing_class=tokenizer,
        objective=objective,
        curriculum=curriculum,
    )
    result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    if trainer.is_world_process_zero():
        final_adapter = args.output_dir / "final_adapter"
        trainer.save_model(str(final_adapter))
        tokenizer.save_pretrained(final_adapter)
        make_saved_adapter_portable(final_adapter)
        trainer.save_metrics("train", result.metrics)
        manifest = json.loads(manifest_path.read_text())
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["final_adapter"] = "final_adapter"
        manifest["final_global_step"] = trainer.state.global_step
        write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
