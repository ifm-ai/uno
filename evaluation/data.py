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
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import zipfile
import zlib

from huggingface_hub import hf_hub_download

from .benchmarks import (
    BENCHMARKS,
    BenchmarkConfig,
    DEFAULT_DATA_ROOT,
    get_benchmark,
    normalize_benchmark_name,
)


PROTOCOL_SOURCE_DIR_ENV = "UNO_EVAL_PROTOCOL_SOURCE_DIR"

LCB_SOURCE_REPO = "livecodebench/code_generation_lite"
LCB_SOURCE_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

DATASET_REVISIONS = {
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
}

HUMANEVAL_INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function."
)
ANSWER_INSTRUCTION = (
    'Think step by step, then give your answer as "The answer is (X)".'
)
MATH500_FEWSHOT_PREAMBLE = """\
Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.
Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
Solution: We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. Equating $2\\cdot15\\cdot n=480$, we get $n=\\boxed{16}$.
Final Answer: The final answer is $16$. I hope it is correct.

Problem: If the system of equations $6x-4y=a$ and $6y-9x=b$ has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.
Solution: Multiplying the first equation by $-\\frac{3}{2}$ gives $-\\frac{3}{2}a=b$, so $\\frac{a}{b}=\\boxed{-\\frac{2}{3}}$.
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.

"""
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


def _download_sources(benchmark: BenchmarkConfig) -> tuple[Path, ...]:
    if not (
        benchmark.source_repo_id
        and benchmark.source_revision
        and benchmark.source_files
    ):
        raise ValueError(f"No pinned official source configured for {benchmark.name}")

    paths = []
    for filename in benchmark.source_files:
        try:
            paths.append(
                Path(
                    hf_hub_download(
                        repo_id=benchmark.source_repo_id,
                        repo_type="dataset",
                        filename=filename,
                        revision=benchmark.source_revision,
                    )
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {benchmark.source_repo_id}@"
                f"{benchmark.source_revision}:{filename}. Gated datasets "
                "require Hugging Face access and authentication."
            ) from exc
    return tuple(paths)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Dataset preparation requires the eval dependencies. "
            "Install with: pip install -e '.[eval]'"
        ) from exc
    return parquet.read_table(path).to_pylist()


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
) -> Path:
    """Resolve an optional offline bundle of final protocol artifacts."""

    candidates = (source_dir / f"{benchmark_name}.jsonl",)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No exact protocol artifact for {benchmark_name}; checked {rendered}"
    )


def prepare_gsm8k(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    rows = _read_parquet(_download_sources(benchmark)[0])
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


def prepare_math500(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    rows = _read_jsonl(_download_sources(benchmark)[0])
    return [
        {
            "row": index,
            "completion_input": (
                MATH500_FEWSHOT_PREAMBLE
                + f"Problem: {row['problem']}\nSolution:"
            ),
            "chat_input": [{"role": "user", "content": row["problem"]}],
            "ground_truth": row["answer"],
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


def prepare_aime24(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    rows = _read_parquet(_download_sources(benchmark)[0])
    return _aime_records(
        [{"problem": row["Problem"], "answer": row["Answer"]} for row in rows]
    )


def prepare_aime25(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    return _aime_records(_read_jsonl(_download_sources(benchmark)[0]))


def prepare_aime26(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    return _aime_records(_read_jsonl(_download_sources(benchmark)[0]))


def prepare_humaneval(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    """Build chat prompts while retaining completion-form grader metadata.

    HumanEval supplies a partial function in ``prompt``. The model receives a
    user instruction plus that prompt, while the grader retains the unmodified
    prompt, tests, and entry point so it can execute the generated function.
    """

    rows = _read_parquet(_download_sources(benchmark)[0])
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


def _mbpp_zeroshot_prompt(row: Mapping[str, Any]) -> str:
    tests = "\n".join(str(test) for test in row["test_list"]).rstrip()
    return (
        "You are an expert Python programmer, and here is your task:\n"
        f"{str(row['text']).strip()}\n"
        "Your code should pass these tests:\n"
        f"{tests}\n\n"
        "Write your solution as a complete Python function wrapped in a "
        "```python ... ``` code block."
    )


def prepare_mbpp(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    """Build the historical 500-item ``mbpp_zeroshot`` protocol input."""

    rows = _read_parquet(_download_sources(benchmark)[0])
    records = []
    for index, row in enumerate(rows):
        prompt = _mbpp_zeroshot_prompt(row)
        records.append(
            {
                "row": index,
                "completion_input": prompt,
                "chat_input": [{"role": "user", "content": prompt}],
                "ground_truth": "\n".join(row["test_list"]),
                "meta": {"original_row": index},
            }
        )
    return records


ARC_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "The following are multiple choice science questions. "
    "Choose the correct answer from the options and answer with the correct letter.\n\n"
)


def prepare_arc_challenge(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    records = []
    letters = "ABCDE"
    for index, row in enumerate(_read_parquet(_download_sources(benchmark)[0])):
        labels = [str(value).strip().upper() for value in row["choices"]["label"]]
        texts = [str(value).strip() for value in row["choices"]["text"]]
        normalized = [
            letters[int(label) - 1] if label.isdigit() else label
            for label in labels
        ]
        answer_key = str(row["answerKey"]).strip().upper()
        if answer_key.isdigit():
            answer_key = letters[int(answer_key) - 1]
        answer = letters[normalized.index(answer_key)]
        choices = "\n".join(
            f"{letters[choice_index]}. {text}"
            for choice_index, text in enumerate(texts)
        )
        user_prompt = (
            f"Question:\n{str(row['question']).strip()}\n"
            f"Choices:\n{choices}\nAnswer:"
        )
        scoring_labels = list(letters[: len(texts)])
        records.append(
            {
                "row": index,
                "ground_truth": answer,
                "completion_input": ARC_SYSTEM_PROMPT + user_prompt,
                "chat_input": [
                    {"role": "system", "content": ARC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "scoring_mode": "choice_scoring",
                "scoring_completions": scoring_labels,
                "scoring_completion_labels": scoring_labels,
            }
        )
    return records


HLE_MC_SYSTEM = (
    "You are a helpful assistant.\n\n"
    "The following are multiple choice questions (with answers). "
    "Choose the correct answer from the options."
)
HLE_EXACT_SYSTEM = "You are a helpful assistant."
HLE_EXACT_INSTRUCTION = "Provide only your final answer."


def prepare_hle(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    records = []
    for row in _read_parquet(_download_sources(benchmark)[0]):
        if row.get("image"):
            continue
        question = str(row["question"]).strip()
        answer_type = row["answer_type"]
        if answer_type == "multipleChoice":
            user_prompt = f"{question}\nAnswer:"
            completion_input = f"{HLE_MC_SYSTEM}\n\n{user_prompt}"
            chat_input = [
                {"role": "system", "content": HLE_MC_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        else:
            system = f"{HLE_EXACT_SYSTEM}\n\n{HLE_EXACT_INSTRUCTION}"
            completion_input = f"{system}\n\n{question}"
            chat_input = [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ]
        records.append(
            {
                "row": len(records),
                "completion_input": completion_input,
                "chat_input": chat_input,
                "ground_truth": row["answer"],
                "answer_type": answer_type,
                "category": row.get("category", ""),
                "raw_subject": row.get("raw_subject", ""),
                "id": row.get("id", ""),
            }
        )
    return records


LCR_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Use ONLY the provided reference "
    "documents to answer the question. Do not rely on prior knowledge."
)


def prepare_lcr(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    csv_path, archive_path = _download_sources(benchmark)
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_names = {}
        for archive_name in archive.namelist():
            try:
                decoded_name = archive_name.encode("cp437").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                decoded_name = archive_name
            archive_names[unicodedata.normalize("NFC", decoded_name)] = archive_name
        for index, row in enumerate(_read_csv(csv_path)):
            filenames = row["data_source_filenames"].split(";")
            documents = []
            for filename in filenames:
                archive_name = (
                    f"lcr/{row['document_category']}/"
                    f"{row['document_set_id']}/{filename}"
                )
                stored_name = archive_names[
                    unicodedata.normalize("NFC", archive_name)
                ]
                document = archive.read(stored_name).decode("utf-8").strip()
                documents.append(f"[Document: {filename}]\n{document}")
            user_prompt = (
                "=== REFERENCE DOCUMENTS ===\n\n"
                + "\n\n---\n\n".join(documents)
                + f"\n\n=== QUESTION ===\n\n{row['question']}\n\nAnswer:"
            )
            records.append(
                {
                    "row": index,
                    "completion_input": f"{LCR_SYSTEM_PROMPT}\n\n{user_prompt}",
                    "chat_input": [
                        {"role": "system", "content": LCR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "ground_truth": row["answer"],
                    "document_category": row["document_category"],
                    "document_set_id": row["document_set_id"],
                    "question_id": int(row["question_id"]),
                    "data_source_filenames": filenames,
                    "data_source_urls": row["data_source_urls"].split(";"),
                    "input_tokens": int(row["input_tokens"]),
                }
            )
    return records


def prepare_omniscience(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(_read_csv(_download_sources(benchmark)[0])):
        system = (
            f"You are answering questions about {row['domain']}, and in "
            f"particular {row['topic']}.\n"
            "You will be given a question, answer with JUST the answer (no explanation).\n"
            "If you do not know the answer, or you need more context or tools to answer the question,\n"
            "be clear about this - it is better that you say this than get the wrong answer."
        )
        question = row["question"]
        records.append(
            {
                "row": index,
                "completion_input": f"{system}\n\n{question}",
                "chat_input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
                "ground_truth": row["answer"],
                "answer_type": "exactMatch",
                "domain": row["domain"],
                "topic": row["topic"],
                "question_id": int(row["question_id"]),
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


GPQA_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "The following are multiple choice questions (with answers). "
    "Choose the correct answer from the options.\n\n"
)


def prepare_gpqa_diamond(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    """Reproduce the historical GPQA choice order (global seed 42)."""

    rows = _read_csv(_download_sources(benchmark)[0])
    rng = random.Random(42)
    records = []
    for index, row in enumerate(rows):
        choices = [(row["Correct Answer"].strip(), True)] + [
            (row[f"Incorrect Answer {choice_index}"].strip(), False)
            for choice_index in (1, 2, 3)
        ]
        rng.shuffle(choices)
        answer = "ABCD"[
            next(i for i, (_, is_correct) in enumerate(choices) if is_correct)
        ]
        question_block = row["Question"].strip() + "\n" + "\n".join(
            f"{letter}. {choice}"
            for letter, (choice, _) in zip("ABCD", choices)
        )
        completion_input = f"{GPQA_SYSTEM_PROMPT}{question_block}\nAnswer:"
        chat_prompt = (
            f"{question_block}\n\n"
            "Answer with only the correct option letter (A, B, C, or D)."
        )
        records.append(
            {
                "row": index,
                "ground_truth": answer,
                "completion_input": completion_input,
                "chat_input": [{"role": "user", "content": chat_prompt}],
            }
        )
    return records


def prepare_ifeval(benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
    rows = _read_jsonl(_download_sources(benchmark)[0])
    records = []
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        instruction_ids = list(row["instruction_id_list"])
        kwargs = [
            {
                key: value for key, value in values.items() if value is not None
            }
            for values in row["kwargs"]
        ]
        records.append(
            {
                "row": index,
                "ground_truth": {
                    "instruction_id_list": instruction_ids,
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


BUILDERS: dict[str, Callable[[BenchmarkConfig], list[dict[str, Any]]]] = {
    "gsm8k": prepare_gsm8k,
    "math500": prepare_math500,
    "aime24": prepare_aime24,
    "aime25": prepare_aime25,
    "aime26": prepare_aime26,
    "arc_challenge": prepare_arc_challenge,
    "gpqa_diamond": prepare_gpqa_diamond,
    "hle": prepare_hle,
    "humaneval": prepare_humaneval,
    "mbpp": prepare_mbpp,
    "ifeval": prepare_ifeval,
    "lcr": prepare_lcr,
    "omniscience": prepare_omniscience,
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

    if source_dir is not None:
        if benchmark.data_sha256 is None:
            raise ValueError(f"No protocol SHA configured for {benchmark.name}")
        source = _protocol_source_path(
            source_dir,
            benchmark_name=benchmark.name,
        )
        _validate_protocol_artifact(
            source,
            expected_rows=benchmark.expected_rows,
            expected_sha256=benchmark.data_sha256,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(output)
        return output

    try:
        builder = BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"No public data builder registered for {name}") from exc
    temporary = output.with_suffix(output.suffix + ".building")
    try:
        _write_jsonl(
            builder(benchmark),
            temporary,
            expected=benchmark.expected_rows,
        )
        if benchmark.data_sha256 is not None:
            _validate_protocol_artifact(
                temporary,
                expected_rows=benchmark.expected_rows,
                expected_sha256=benchmark.data_sha256,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
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
