from __future__ import annotations

import hashlib
import json
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch

from .constants import (
    COLLATED_SEQUENCE_LENGTH,
    CUTOFF_LENGTH,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    IGNORE_INDEX,
    PREPROCESS_TOKEN_ALIGNMENT,
    PREPROCESSED_MAX_LENGTH,
)

THINK_START = "<think>\n"
THINK_END = "\n</think>\n\n"
EMPTY_THINK = THINK_START + THINK_END


def infer_sequence_lengths(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Match the source/target truncation used for the released checkpoint."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    new_source_len = min(max(cutoff_len - new_target_len, 0), source_len)
    return new_source_len, new_target_len


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _assistant_content(content: str) -> str:
    # This deliberately mirrors ReasoningTemplate's exact substring test.
    if THINK_START not in content and THINK_END not in content:
        return EMPTY_THINK + content
    return content


def encode_qwen3_sharegpt_example(
    conversations: Sequence[dict[str, str]],
    tokenizer,
    *,
    cutoff_length: int = CUTOFF_LENGTH,
    token_alignment: int = PREPROCESS_TOKEN_ALIGNMENT,
) -> dict[str, list[int]]:
    """Encode one OpenThoughts row using the released Uno preprocessing recipe."""
    if cutoff_length <= 0 or token_alignment <= 0:
        raise ValueError("cutoff_length and token_alignment must be positive.")
    effective_cutoff = cutoff_length - cutoff_length % token_alignment
    if effective_cutoff <= 0:
        raise ValueError("The rounded preprocessing cutoff must be positive.")
    if not conversations or len(conversations) % 2:
        raise ValueError("ShareGPT conversations must contain complete user/assistant pairs.")

    input_ids: list[int] = []
    labels: list[int] = []
    total_length = 0
    for index in range(0, len(conversations), 2):
        user = conversations[index]
        assistant = conversations[index + 1]
        if user.get("from") != "human" or assistant.get("from") != "gpt":
            raise ValueError(
                "OpenThoughts conversations must alternate 'human' and 'gpt' roles."
            )
        user_text = (
            "<|im_start|>user\n"
            + str(user.get("value", ""))
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        assistant_text = _assistant_content(str(assistant.get("value", ""))) + "<|im_end|>\n"
        source_ids = _encode(tokenizer, user_text)
        target_ids = _encode(tokenizer, assistant_text)
        if total_length >= effective_cutoff:
            break
        source_len, target_len = infer_sequence_lengths(
            len(source_ids), len(target_ids), effective_cutoff - total_length
        )
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        input_ids.extend(source_ids)
        input_ids.extend(target_ids)
        labels.extend([IGNORE_INDEX] * source_len)
        labels.extend(target_ids)
        total_length += source_len + target_len

    if not input_ids:
        raise ValueError("Conversation produced no tokens.")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id.")
    pad_len = (-len(input_ids)) % token_alignment
    # The released recipe treats alignment padding as data; batch collation
    # adds the final 4096 padding position with IGNORE_INDEX.
    input_ids.extend([pad_token_id] * pad_len)
    labels.extend([pad_token_id] * pad_len)
    if len(input_ids) > effective_cutoff:
        raise AssertionError("Encoded sequence exceeded its rounded cutoff.")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def preprocess_batch(examples: dict[str, list[Any]], *, tokenizer) -> dict[str, list[Any]]:
    output = {"input_ids": [], "attention_mask": [], "labels": []}
    for conversations in examples["conversations"]:
        encoded = encode_qwen3_sharegpt_example(conversations, tokenizer)
        for key in output:
            output[key].append(encoded[key])
    return output


def _int32_bytes(values: Sequence[int]) -> bytes:
    encoded = array("i", values)
    if encoded.itemsize != 4:
        raise RuntimeError("Dataset hashing requires a 32-bit C integer type.")
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def content_hash_rows(rows: Iterable[dict[str, Sequence[int]]]) -> dict[str, int | str]:
    """Hash row boundaries, input IDs, and labels in source order."""
    digest = hashlib.sha256()
    row_count = 0
    token_count = 0
    supervised_count = 0
    for row in rows:
        input_ids = row["input_ids"]
        labels = row["labels"]
        if len(input_ids) != len(labels):
            raise ValueError(f"Row {row_count} has mismatched input and label lengths.")
        digest.update(struct.pack("<Q", len(input_ids)))
        digest.update(_int32_bytes(input_ids))
        digest.update(_int32_bytes(labels))
        row_count += 1
        token_count += len(input_ids)
        supervised_count += sum(label != IGNORE_INDEX for label in labels)
    return {
        "algorithm": "sha256-rowlen-u64le-input-i32le-label-i32le-v1",
        "content_sha256": digest.hexdigest(),
        "row_count": row_count,
        "token_count": token_count,
        "supervised_token_count": supervised_count,
    }


def iter_hf_dataset(dataset, batch_size: int = 256) -> Iterator[dict[str, Sequence[int]]]:
    for batch in dataset.iter(batch_size=batch_size):
        for input_ids, labels in zip(batch["input_ids"], batch["labels"]):
            yield {"input_ids": input_ids, "labels": labels}


def hash_hf_dataset(dataset) -> dict[str, int | str]:
    return content_hash_rows(iter_hf_dataset(dataset))


def read_dataset_manifest(dataset_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(dataset_path) / "uno_dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset manifest not found: {manifest_path}. "
            "Run `python -m training.prepare_openthoughts` first."
        )
    return json.loads(manifest_path.read_text())


def validate_dataset_manifest(manifest: dict[str, Any]) -> None:
    source = manifest.get("source", {})
    tokenizer = manifest.get("tokenizer", {})
    if (source.get("dataset_id"), source.get("revision")) != (
        DEFAULT_DATASET_ID,
        DEFAULT_DATASET_REVISION,
    ):
        raise ValueError("Dataset manifest does not use the pinned OpenThoughts revision.")
    if (tokenizer.get("model_id"), tokenizer.get("revision")) != (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
    ):
        raise ValueError("Dataset manifest does not use the pinned Uno tokenizer revision.")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("cutoff_length") != CUTOFF_LENGTH:
        raise ValueError("Dataset cutoff does not match the Uno training recipe.")
    if preprocessing.get("token_alignment") != PREPROCESS_TOKEN_ALIGNMENT:
        raise ValueError("Dataset token alignment must be 3.")
    if preprocessing.get("max_preprocessed_length") != PREPROCESSED_MAX_LENGTH:
        raise ValueError("Dataset maximum row length must be 4095.")
    if manifest.get("hash", {}).get("row_count", 0) <= 0:
        raise ValueError("Dataset manifest has no hashed rows.")


@dataclass
class UnoDataCollator:
    tokenizer: Any
    sequence_length: int = COLLATED_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if self.tokenizer.padding_side != "right":
            raise ValueError("Uno training requires right padding.")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Uno training requires a tokenizer pad token.")

    def __call__(self, features: list[dict[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            length = len(feature["input_ids"])
            if length > self.sequence_length:
                raise ValueError(
                    f"Prepared row length {length} exceeds sequence length {self.sequence_length}."
                )
            if len(feature["labels"]) != length:
                raise ValueError("Prepared input IDs and labels must have equal lengths.")
            pad_len = self.sequence_length - length
            result["input_ids"].append(
                list(feature["input_ids"]) + [self.tokenizer.pad_token_id] * pad_len
            )
            result["attention_mask"].append(
                list(feature.get("attention_mask", [1] * length)) + [0] * pad_len
            )
            result["labels"].append(list(feature["labels"]) + [IGNORE_INDEX] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
