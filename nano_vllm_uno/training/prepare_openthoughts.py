from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from .constants import (
    CUTOFF_LENGTH,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    DEFAULT_DATASET_ROWS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    PREPROCESS_TOKEN_ALIGNMENT,
    PREPROCESSED_MAX_LENGTH,
)
from .data import hash_hf_dataset, preprocess_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the pinned OpenThoughts Uno corpus.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--writer-batch-size", type=int, default=100)
    parser.add_argument("--max-samples", type=int, help="Smoke testing only.")
    parser.add_argument(
        "--tokenizer-local-files-only",
        action="store_true",
        help="Resolve the pinned tokenizer only from --cache-dir.",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="Reference prepared dataset or its uno_dataset_manifest.json.",
    )
    return parser


def _reference_hash(path: Path) -> dict:
    manifest_path = path if path.is_file() else path / "uno_dataset_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())["hash"]
    from datasets import load_from_disk

    reference = load_from_disk(str(path))
    if isinstance(reference, DatasetDict):
        if set(reference) != {"train"}:
            raise ValueError(
                "Reference DatasetDict must contain only the train split."
            )
        reference = reference["train"]
    return hash_hf_dataset(reference)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be positive.")
    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            raise FileExistsError(f"Output path must be an empty directory: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=args.tokenizer_local_files_only,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_samples is not None:
        source = load_dataset(
            DEFAULT_DATASET_ID,
            revision=DEFAULT_DATASET_REVISION,
            split="train",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            streaming=True,
        )
        dataset = Dataset.from_list(list(source.take(args.max_samples)))
    else:
        dataset = load_dataset(
            DEFAULT_DATASET_ID,
            revision=DEFAULT_DATASET_REVISION,
            split="train",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
    if args.max_samples is None and len(dataset) != DEFAULT_DATASET_ROWS:
        raise RuntimeError(
            f"Pinned OpenThoughts revision has {len(dataset)} rows; expected {DEFAULT_DATASET_ROWS}."
        )

    prepared = dataset.map(
        preprocess_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        writer_batch_size=args.writer_batch_size,
        remove_columns=dataset.column_names,
        desc="Tokenizing OpenThoughts for Uno",
    )
    content_hash = hash_hf_dataset(prepared)
    if content_hash["row_count"] != len(dataset):
        raise RuntimeError("Preprocessing unexpectedly changed the dataset row count.")

    if args.verify_against is not None:
        reference_hash = _reference_hash(args.verify_against)
        for key in ("row_count", "content_sha256"):
            if content_hash.get(key) != reference_hash.get(key):
                raise RuntimeError(
                    f"Prepared corpus does not match reference {key}: "
                    f"{content_hash.get(key)} != {reference_hash.get(key)}"
                )

    args.output.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(args.output))
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": DEFAULT_DATASET_ID,
            "revision": DEFAULT_DATASET_REVISION,
            "split": "train",
            "source_order_preserved": True,
            "license": "Apache-2.0",
        },
        "tokenizer": {
            "model_id": DEFAULT_MODEL_ID,
            "revision": DEFAULT_MODEL_REVISION,
            "class": tokenizer.__class__.__name__,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "preprocessing": {
            "template": "qwen3-sharegpt-reasoning",
            "cutoff_length": CUTOFF_LENGTH,
            "token_alignment": PREPROCESS_TOKEN_ALIGNMENT,
            "max_preprocessed_length": PREPROCESSED_MAX_LENGTH,
            "prompt_label_id": -100,
            "truncate_mode": "cut",
        },
        "smoke_max_samples": args.max_samples,
        "hash": content_hash,
    }
    (args.output / "uno_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
