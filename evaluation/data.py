"""Pinned data preparation for the canonical 13-benchmark Uno suite."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import zlib

from huggingface_hub import hf_hub_download

from .benchmarks import (
    BENCHMARKS,
    DEFAULT_DATA_ROOT,
    get_benchmark,
    normalize_benchmark_name,
)


PROTOCOL_SOURCE_REPO = "LLM360/eval-360-sources"
PROTOCOL_SOURCE_REVISION = "f1d3a57020a52de762e73faba260ed15d9a8b6d7"
PROTOCOL_SOURCE_DIR_ENV = "UNO_EVAL_PROTOCOL_SOURCE_DIR"

LCB_SOURCE_REPO = "livecodebench/code_generation_lite"
LCB_SOURCE_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "hypaai/Hypa_AIME2024": "11ab79f0eed5f4fdf3d469b466663ab86bbd77c8",
    "math-ai/aime25": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
    "math-ai/aime26": "79037aebdb6580008fb960d17cb21fd3099083e3",
    "openai/openai_humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "google-research-datasets/mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "Idavidrein/gpqa": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
}

HUMANEVAL_INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function."
)
ANSWER_INSTRUCTION = (
    'Think step by step, then give your answer as "The answer is (X)".'
)
MBPP_TEXT_OVERRIDES = {
    388: "Write a python function to find the highest power of 2 that is less "
    "than or equal to n.",
}
LCB_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests. You will NOT "
    "return anything except for the program."
)
LCB_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
LCB_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows."
)

class _NoGlobalsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"LiveCodeBench test payload attempted to load {module}.{name}"
        )


def _load_dataset(
    repo_id: str,
    config_name: str | None = None,
    *,
    split: str,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dataset preparation requires the optional eval dependencies. "
            "Install with: pip install -e '.[eval]'"
        ) from exc
    try:
        dataset = load_dataset(
            repo_id,
            config_name,
            split=split,
            revision=DATASET_REVISIONS[repo_id],
        )
    except Exception as exc:
        if repo_id == "Idavidrein/gpqa":
            raise RuntimeError(
                "GPQA is gated. Accept its terms at "
                "https://huggingface.co/datasets/Idavidrein/gpqa and "
                "authenticate with `hf auth login` or HF_TOKEN."
            ) from exc
        raise
    return [dict(row) for row in dataset]


def _write_jsonl(
    records: Iterable[dict[str, Any]],
    path: Path,
    *,
    expected: int,
) -> None:
    records = list(records)
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} rows for {path.name}, got {len(records)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_artifact_fingerprint(path: Path) -> tuple[int, str]:
    """Return the row count and SHA256 used by the canonical protocol."""

    return _row_count(path), _sha256(path)


def _validate_protocol_artifact(path: Path, *, expected_rows: int, expected_sha256: str) -> None:
    count = _row_count(path)
    if count != expected_rows:
        raise ValueError(f"{path} has {count} rows; expected {expected_rows}")
    digest = _sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            f"{path} has sha256 {digest}; expected pinned protocol artifact "
            f"{expected_sha256}"
        )


def _protocol_source_path(
    source_dir: Path,
    *,
    benchmark_name: str,
    source_filename: str,
) -> Path:
    """Resolve either a flat or repository-shaped exact protocol bundle."""

    candidates = (
        source_dir / f"{benchmark_name}.jsonl",
        source_dir / source_filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No exact protocol artifact for {benchmark_name}; checked {rendered}"
    )


def prepare_gsm8k() -> list[dict[str, Any]]:
    rows = _load_dataset("openai/gsm8k", "main", split="test")
    records = []
    for index, row in enumerate(rows):
        prompt = f"Q: {row['question']}\nA: Let's think step by step."
        records.append(
            {
                "row": index,
                "ground_truth": row["answer"],
                "completion_input": prompt,
                "chat_input": [{"role": "user", "content": prompt}],
            }
        )
    return records


def prepare_math500() -> list[dict[str, Any]]:
    rows = _load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {
            "row": index,
            "ground_truth": row["answer"],
            "completion_input": row["problem"],
            "chat_input": [{"role": "user", "content": row["problem"]}],
        }
        for index, row in enumerate(rows)
    ]


def _aime_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        problem = str(row["problem"])
        answer = str(row.get("answer", row.get("solution"))).strip()
        prompt = f"Question: {problem}\nAnswer:"
        records.append(
            {
                "row": index,
                "ground_truth": answer,
                "completion_input": prompt,
                "chat_input": [{"role": "user", "content": prompt}],
            }
        )
    return records


def prepare_aime24() -> list[dict[str, Any]]:
    return _aime_records(
        _load_dataset("hypaai/Hypa_AIME2024", split="english")
    )


def prepare_aime25() -> list[dict[str, Any]]:
    return _aime_records(_load_dataset("math-ai/aime25", split="test"))


def prepare_aime26() -> list[dict[str, Any]]:
    return _aime_records(_load_dataset("math-ai/aime26", split="test"))


def prepare_humaneval() -> list[dict[str, Any]]:
    """Build chat prompts while retaining completion-form grader metadata.

    HumanEval supplies a partial function in ``prompt``. The model receives a
    user instruction plus that prompt, while the grader retains the unmodified
    prompt, tests, and entry point so it can execute the generated function.
    """

    rows = _load_dataset("openai/openai_humaneval", split="test")
    records = []
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        records.append(
            {
                "row": index,
                "completion_input": prompt,
                "chat_input": [
                    {
                        "role": "user",
                        "content": f"{HUMANEVAL_INSTRUCTION}\n{prompt}",
                    }
                ],
                "ground_truth": {
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            }
        )
    return records


def _mbpp_prompt(row: Mapping[str, Any]) -> str:
    task_id = int(row["task_id"])
    text = MBPP_TEXT_OVERRIDES.get(task_id, str(row["text"]))
    tests = "\n".join(str(test) for test in row["test_list"])
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{text} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
    )


def prepare_mbpp() -> list[dict[str, Any]]:
    prompt_rows = _load_dataset(
        "google-research-datasets/mbpp",
        "full",
        split="prompt",
    )
    examples = {
        int(row["task_id"]): row
        for row in prompt_rows
    }
    prefix = "\n[DONE]\n\n".join(
        _mbpp_prompt(examples[task_id]) + str(examples[task_id]["code"])
        for task_id in (2, 3, 4)
    ) + "\n[DONE]\n\n"

    rows = _load_dataset(
        "google-research-datasets/mbpp",
        "full",
        split="test",
    )
    records = []
    for index, row in enumerate(rows):
        prompt = prefix + _mbpp_prompt(row)
        records.append(
            {
                "row": index,
                "ground_truth": "\n".join(row["test_list"]),
                "completion_input": prompt,
                "chat_input": [{"role": "user", "content": prompt}],
            }
        )
    return records


def _decode_lcb_private_tests(value: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        payload = zlib.decompress(base64.b64decode(value.encode("utf-8")))
        serialized_json = _NoGlobalsUnpickler(io.BytesIO(payload)).load()
        decoded = json.loads(serialized_json)
    if not isinstance(decoded, list):
        raise ValueError("LiveCodeBench private tests must decode to a list")
    return decoded


def _lcb_prompt(row: Mapping[str, Any]) -> tuple[str, str]:
    question = str(row["question_content"])
    starter_code = str(row.get("starter_code", "")).strip()
    if starter_code:
        format_prompt = (
            f"### Format: {LCB_FORMAT_WITH_STARTER}\n"
            f"```python\n{starter_code}\n```\n\n"
        )
    else:
        format_prompt = (
            f"### Format: {LCB_FORMAT_WITHOUT_STARTER}\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    user_prompt = (
        f"### Question:\n{question}\n\n"
        f"{format_prompt}"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    return user_prompt, f"{LCB_SYSTEM_MESSAGE}\n\n{user_prompt}"


def prepare_lcbv6() -> list[dict[str, Any]]:
    """Build the 175-problem v6 shard used by the reported Uno results."""

    source = Path(
        hf_hub_download(
            repo_id=LCB_SOURCE_REPO,
            repo_type="dataset",
            filename="test6.jsonl",
            revision=LCB_SOURCE_REVISION,
        )
    )
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    records = []
    for index, row in enumerate(rows):
        public_tests = json.loads(row["public_test_cases"])
        private_tests = _decode_lcb_private_tests(row["private_test_cases"])
        metadata = json.loads(row["metadata"])
        method_name = metadata.get("func_name")
        tests = public_tests + private_tests
        if method_name:
            tests = [dict(test, method_name=method_name) for test in tests]
        user_prompt, completion_prompt = _lcb_prompt(row)
        records.append(
            {
                "row": str(index),
                "completion_input": completion_prompt,
                "chat_input": [
                    {"role": "system", "content": LCB_SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt},
                ],
                "ground_truth": json.dumps(tests),
            }
        )
    return records


def _gpqa_permutation(question: str) -> list[int]:
    digest = hashlib.md5(
        question.encode(),
        usedforsecurity=False,
    ).hexdigest()
    seed = int(digest[:8], 16)
    indices = list(range(4))
    random.Random(seed).shuffle(indices)
    return indices


def _gpqa_record(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    question = str(row["Question"])
    choices = [
        str(row["Correct Answer"]),
        str(row["Incorrect Answer 1"]),
        str(row["Incorrect Answer 2"]),
        str(row["Incorrect Answer 3"]),
    ]
    indices = _gpqa_permutation(question)
    shuffled = [choices[index] for index in indices]
    answer = "ABCD"[indices.index(0)]
    choices_text = "\n".join(
        f"{'ABCD'[index]}) {choice}"
        for index, choice in enumerate(shuffled)
    )
    prompt = (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n"
        f"{question}\n\n{choices_text}"
    )
    return {
        "row": row_index,
        "ground_truth": answer,
        "completion_input": prompt,
        "chat_input": [{"role": "user", "content": prompt}],
    }


def prepare_gpqa(input_csv: Path | None = None) -> list[dict[str, Any]]:
    if input_csv is None:
        rows = _load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    else:
        with input_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    return [_gpqa_record(row, index) for index, row in enumerate(rows)]


def prepare_gpqa_diamond(input_csv: Path | None = None) -> list[dict[str, Any]]:
    if input_csv is None:
        rows = _load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    else:
        with input_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    return [_gpqa_record(row, index) for index, row in enumerate(rows)]


def prepare_ifeval() -> list[dict[str, Any]]:
    rows = _load_dataset("google/IFEval", split="train")
    records = []
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        kwargs = [
            {key: value for key, value in values.items() if value is not None}
            for values in row["kwargs"]
        ]
        records.append(
            {
                "row": index,
                "ground_truth": {
                    "instruction_id_list": list(row["instruction_id_list"]),
                    "kwargs": kwargs,
                },
                "completion_input": prompt,
                "chat_input": [{"role": "user", "content": prompt}],
            }
        )
    return records


def _knowledge_prompt(question: str, choices: str) -> str:
    return f"{question}\n\n{choices}\n\n{ANSWER_INSTRUCTION}"


def prepare_mmlu_pro() -> list[dict[str, Any]]:
    rows = _load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    records = []
    letters = "ABCDEFGHIJ"
    for index, row in enumerate(rows):
        choices = "\n".join(
            f"{letters[choice_index]}. {choice}"
            for choice_index, choice in enumerate(row["options"][: len(letters)])
        )
        prompt = _knowledge_prompt(str(row["question"]), choices)
        answer_index = row["answer_index"]
        answer = (
            letters[answer_index]
            if isinstance(answer_index, int)
            else str(row["answer"])
        )
        records.append(
            {
                "row": index,
                "ground_truth": answer,
                "completion_input": prompt,
                "chat_input": [{"role": "user", "content": prompt}],
            }
        )
    return records


BUILDERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "gsm8k": prepare_gsm8k,
    "math500": prepare_math500,
    "aime24": prepare_aime24,
    "aime25": prepare_aime25,
    "aime26": prepare_aime26,
    "humaneval": prepare_humaneval,
    "mbpp": prepare_mbpp,
    "lcbv6": prepare_lcbv6,
    "gpqa": prepare_gpqa,
    "gpqa_diamond": prepare_gpqa_diamond,
    "mmlu_pro": prepare_mmlu_pro,
    "ifeval": prepare_ifeval,
}


def prepare_benchmark_data(
    name: str,
    *,
    output_dir: Path = DEFAULT_DATA_ROOT,
    source_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    name = normalize_benchmark_name(name)
    benchmark = get_benchmark(name)
    output = output_dir / benchmark.data_path.name
    if output.is_file() and not overwrite:
        if benchmark.data_sha256 is None:
            count = _row_count(output)
            if count == benchmark.expected_rows:
                return output
            raise ValueError(
                f"{output} has {count} rows; expected {benchmark.expected_rows}. "
                "Rebuild it with overwrite=True."
            )
        _validate_protocol_artifact(
            output,
            expected_rows=benchmark.expected_rows,
            expected_sha256=benchmark.data_sha256,
        )
        return output

    if source_dir is None:
        configured_source_dir = os.environ.get(PROTOCOL_SOURCE_DIR_ENV)
        source_dir = Path(configured_source_dir) if configured_source_dir else None

    if benchmark.source_filename and benchmark.data_sha256:
        if source_dir is not None:
            source = _protocol_source_path(
                source_dir,
                benchmark_name=benchmark.name,
                source_filename=benchmark.source_filename,
            )
            _validate_protocol_artifact(
                source,
                expected_rows=benchmark.expected_rows,
                expected_sha256=benchmark.data_sha256,
            )
        else:
            try:
                source = Path(
                    hf_hub_download(
                        repo_id=PROTOCOL_SOURCE_REPO,
                        repo_type="dataset",
                        filename=benchmark.source_filename,
                        revision=PROTOCOL_SOURCE_REVISION,
                    )
                )
                _validate_protocol_artifact(
                    source,
                    expected_rows=benchmark.expected_rows,
                    expected_sha256=benchmark.data_sha256,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not obtain the exact pinned {benchmark.name} input. "
                    f"Set {PROTOCOL_SOURCE_DIR_ENV} or pass --source-dir with a "
                    "flat or eval-360-sources-shaped protocol bundle. The public "
                    "source repository may require authentication and some current "
                    "files do not byte-match the historical protocol artifact."
                ) from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(output)
        return output

    try:
        builder = BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"No public data builder registered for {name}") from exc
    _write_jsonl(
        builder(),
        output,
        expected=benchmark.expected_rows,
    )
    return output


def prepare_all_benchmark_data(
    names: Iterable[str] | None = None,
    *,
    output_dir: Path = DEFAULT_DATA_ROOT,
    source_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    selected = list(names) if names else list(BENCHMARKS)
    return {
        normalize_benchmark_name(name): prepare_benchmark_data(
            name,
            output_dir=output_dir,
            source_dir=source_dir,
            overwrite=overwrite,
        )
        for name in selected
    }
