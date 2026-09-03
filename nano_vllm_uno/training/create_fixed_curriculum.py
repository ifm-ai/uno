from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import yaml

from .constants import (
    COLLATED_SEQUENCE_LENGTH,
    DEFAULT_GLOBAL_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_STEPS_PER_EPOCH,
)


def parse_epochs(value: str | Decimal) -> Decimal:
    try:
        epochs = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid epoch count: {value!r}.") from error
    if not epochs.is_finite() or epochs <= 0:
        raise ValueError("Epoch count must be a finite positive number.")
    return epochs


def build_fixed_curriculum(
    epochs: str | Decimal,
    *,
    block_size: int = 8,
    sequence_length: int = COLLATED_SEQUENCE_LENGTH,
    global_batch_size: int = DEFAULT_GLOBAL_BATCH_SIZE,
    steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH,
    seed: int = DEFAULT_SEED,
) -> dict:
    epochs = parse_epochs(epochs)
    if min(block_size, sequence_length, global_batch_size, steps_per_epoch) <= 0:
        raise ValueError("Fixed-curriculum integer settings must be positive.")
    exact_steps = epochs * steps_per_epoch
    max_steps = max(1, int(exact_steps.to_integral_value(rounding=ROUND_HALF_UP)))
    tokens_per_step = sequence_length * global_batch_size
    total_tokens = max_steps * tokens_per_step
    return {
        "version": 1,
        "requested_epochs": str(epochs),
        "resolved_epochs": str(Decimal(max_steps) / Decimal(steps_per_epoch)),
        "steps_per_epoch": steps_per_epoch,
        "tokens_per_step": tokens_per_step,
        "sequence_length": sequence_length,
        "global_batch_size": global_batch_size,
        "total_tokens": total_tokens,
        "seed": seed,
        "stages": [{"block_size": block_size, "tokens": total_tokens}],
    }


def write_fixed_curriculum(path: Path, payload: dict) -> None:
    rendered = yaml.safe_dump(payload, sort_keys=False)
    if path.exists():
        if path.read_text() != rendered:
            raise FileExistsError(
                f"Refusing to replace a different fixed curriculum: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic fixed-block Uno curriculum."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", default="1")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=COLLATED_SEQUENCE_LENGTH)
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = build_fixed_curriculum(
        args.epochs,
        block_size=args.block_size,
        sequence_length=args.sequence_length,
        global_batch_size=args.global_batch_size,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )
    write_fixed_curriculum(args.output, payload)
    print(yaml.safe_dump(payload, sort_keys=False), end="")


if __name__ == "__main__":
    main()
