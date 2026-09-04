"""Local HumanEval and MBPP scorers."""

from __future__ import annotations

import copy
import json
from math import comb
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

from ..parsers import generation_list, get_accuracy


_PYTHON_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE
)
_CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\s*\n(.*?)\n?```", re.DOTALL)
_PYTHON_START_RE = re.compile(r"(?m)^\s*(?:from\s+\S+\s+import\s+|import\s+|def\s+|class\s+)")
_ASSERT_FUNCTION_RE = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")
_DEF_FUNCTION_RE = re.compile(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(")


def _pick_grading_python() -> str:
    env = os.environ.get("NANO_VLLM_UNO_GRADING_PYTHON")
    if env:
        return env
    for candidate in ("/usr/bin/python3", "/usr/bin/python"):
        if os.path.exists(candidate):
            return candidate
    return sys.executable


_GRADING_PYTHON = _pick_grading_python()


def _strip_bare_code(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("thought\n"):
        candidate = candidate[len("thought\n") :].lstrip()
    if candidate.endswith("```"):
        candidate = candidate[:-3].rstrip()
    lines = candidate.splitlines()
    while lines and lines[0].strip().lower() in {"python", "py"}:
        lines.pop(0)
    while lines and lines[0].strip() in {"[BEGIN]", "BEGIN"}:
        lines.pop(0)
    while lines and lines[-1].strip() in {"[DONE]", "DONE", "[END]", "END"}:
        lines.pop()
    candidate = "\n".join(lines).strip()
    start = _PYTHON_START_RE.search(candidate)
    if start and start.start() > 0:
        candidate = candidate[start.start() :].lstrip()
    return candidate


def _looks_like_python(text: str) -> bool:
    return bool(_PYTHON_START_RE.search(text))


def _expected_function_names(test_code: str | None) -> set[str]:
    return set(_ASSERT_FUNCTION_RE.findall(test_code or ""))


def _defined_function_names(code: str) -> set[str]:
    return set(_DEF_FUNCTION_RE.findall(code or ""))


def _dedupe_preserving_order(blocks: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for block in blocks:
        block = _strip_bare_code(block)
        if not block or block in seen:
            continue
        deduped.append(block)
        seen.add(block)
    return deduped


def extract_mbpp_code(text: str, test_code: str | None = None) -> str:
    if not text:
        return ""
    matches = _dedupe_preserving_order(
        _PYTHON_CODE_BLOCK_RE.findall(text) + _CODE_BLOCK_RE.findall(text)
    )
    expected_functions = _expected_function_names(test_code)
    for match in matches:
        if expected_functions & _defined_function_names(match):
            return match
    for match in matches:
        if _looks_like_python(match):
            return match
    open_match = re.search(r"```(?:[A-Za-z0-9_+-]*)\s*\n", text)
    if open_match:
        return _strip_bare_code(text[open_match.end() :])
    candidate = _strip_bare_code(text)
    if _looks_like_python(candidate):
        return candidate
    return ""


def _run_python(program: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory(prefix="nano_eval_code_") as tmp:
        try:
            result = subprocess.run(
                [_GRADING_PYTHON, "-c", program],
                capture_output=True,
                timeout=timeout,
                cwd=tmp,
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "PYTHONUNBUFFERED": "1",
                    "LANG": "C.UTF-8",
                },
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def grade_mbpp_row(row: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    test_code = row.get("ground_truth")
    if isinstance(test_code, list):
        test_code = "\n".join(str(item) for item in test_code)
    test_code = str(test_code or "")
    picked: list[str] = []
    correct: list[bool] = []
    for generation in generation_list(row):
        code = extract_mbpp_code(generation, test_code)
        picked.append(code[:400])
        correct.append(bool(code and _run_python(code + "\n" + test_code + "\n", timeout)))
    graded = dict(row)
    graded["picked"] = picked
    graded["correct"] = correct
    graded["accuracy"] = get_accuracy(correct)
    graded["grader"] = "mbpp_local"
    return graded


def _source_group_key(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("source_row", row.get("row", fallback)))


def _pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0:
        return 0.0
    if k > n:
        k = n
    wrong = n - c
    if wrong < k:
        return 1.0
    return 1.0 - (comb(wrong, k) / comb(n, k))


def _summarize_grouped_correct(graded: list[dict[str, Any]]) -> dict[str, Any]:
    by_problem: dict[str, list[bool]] = {}
    for index, row in enumerate(graded):
        by_problem.setdefault(_source_group_key(row, index), []).extend(
            bool(value) for value in row.get("correct", [])
        )

    max_samples = max((len(values) for values in by_problem.values()), default=0)
    summary: dict[str, Any] = {
        "num_problems": len(by_problem),
        "max_samples_per_problem": max_samples,
    }
    for k in (1, 5, 10):
        if max_samples < k:
            continue
        avg_values = [values[:k] for values in by_problem.values()]
        flat = [value for values in avg_values for value in values]
        pass_values = [
            _pass_at_k(len(values), sum(values), k)
            for values in by_problem.values()
        ]
        summary[f"avg_at_{k}"] = get_accuracy(flat)
        summary[f"pass_at_{k}"] = sum(pass_values) / len(pass_values) if pass_values else float("nan")
    return summary


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _extract_fenced_code(text: str) -> str:
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    open_match = re.search(r"```(?:python|py)?\s*\n", text)
    if open_match:
        return text[open_match.end() :]
    return text


def _strip_function_signature(generation: str, entry_point: str) -> str:
    pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\(.*?\)\s*(?:->.*?)?:\s*\n"
    match = re.match(pattern, generation, flags=re.DOTALL)
    if not match:
        return generation
    body = generation[match.end() :]
    doc_match = re.match(r'\s*""".*?"""\s*\n', body, flags=re.DOTALL)
    if doc_match:
        body = body[doc_match.end() :]
    return body


def _prepare_completion_problem(
    problem: dict[str, Any],
    completion: str,
) -> tuple[dict[str, Any], str]:
    """Preserve a complete re-emitted function and its parameter names.

    Removing only a generated signature is incorrect when the model renames a
    parameter and then uses that name in the body. Remove the original prompt
    signature and execute the complete generated function instead.
    """

    entry_point = str(problem["entry_point"])
    generated_def = re.match(
        rf"^[ \t]*(?:async[ \t]+)?def[ \t]+{re.escape(entry_point)}[ \t]*\(",
        completion,
    )
    if generated_def is None:
        return problem, _strip_function_signature(completion, entry_point)

    prompt_def = re.search(
        rf"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+{re.escape(entry_point)}[ \t]*\(",
        str(problem["prompt"]),
    )
    if prompt_def is None:
        return problem, _strip_function_signature(completion, entry_point)

    generated_problem = copy.copy(problem)
    generated_problem["prompt"] = str(problem["prompt"])[: prompt_def.start()]
    return generated_problem, completion.lstrip()


def _human_eval_program(problem: dict[str, Any], completion: str) -> str:
    return (
        str(problem["prompt"])
        + completion
        + "\n"
        + str(problem["test"])
        + "\n"
        + f"check({problem['entry_point']})"
    )


def grade_humaneval_row(row: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    gt = row.get("ground_truth")
    if isinstance(gt, str):
        gt = json.loads(gt)
    if not isinstance(gt, dict):
        raise ValueError("HumanEval ground_truth must be a dict or JSON object string")
    prompt = str(row.get("completion_input") or "")
    entry_point = str(gt["entry_point"])
    problem = {
        "prompt": prompt,
        "test": gt["test"],
        "entry_point": entry_point,
    }
    picked: list[str] = []
    correct: list[bool] = []
    for generation in generation_list(row):
        candidate = _extract_fenced_code(generation) if "```" in generation else generation
        if candidate.startswith(prompt):
            candidate = candidate[len(prompt) :]
        generation_problem, candidate = _prepare_completion_problem(
            problem,
            candidate,
        )
        picked.append(candidate[:400])
        correct.append(
            _run_python(
                _human_eval_program(generation_problem, candidate),
                timeout,
            )
        )
    graded = dict(row)
    graded["picked"] = picked
    graded["correct"] = correct
    graded["accuracy"] = get_accuracy(correct)
    graded["grader"] = "humaneval_local"
    return graded


def score_code(
    rows: list[dict[str, Any]],
    *,
    task: str,
    timeout: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if task == "humaneval":
        graded = [grade_humaneval_row(row, timeout=timeout) for row in rows]
        grader = "humaneval_local"
    elif task == "mbpp":
        graded = [grade_mbpp_row(row, timeout=timeout) for row in rows]
        grader = "mbpp_local"
    else:
        raise ValueError(f"Unsupported code task: {task}")
    correct = [bool(row["correct"][0]) for row in graded if row.get("correct")]
    summary = {
        "grader": grader,
        "num_rows": len(graded),
        "num_correct": sum(int(value) for value in correct),
        "accuracy": get_accuracy(correct),
        "timeout": timeout,
        "grading_python": _GRADING_PYTHON,
    }
    if task in {"humaneval", "mbpp"}:
        summary.update(_summarize_grouped_correct(graded))
    return graded, summary
