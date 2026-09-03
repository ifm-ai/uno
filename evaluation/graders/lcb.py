"""LiveCodeBench v6 scorer using the local CodeGrader implementation."""

from __future__ import annotations

import json
import multiprocessing
import re
from typing import Any

from ..lcb_utils import codegen_metrics
from ..parsers import raw_generation_list


_FENCED_BLOCK_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)


def _looks_like_fence_label(line: str) -> bool:
    label = line.strip()
    return (
        bool(label)
        and len(label) < 32
        and re.fullmatch(r"[A-Za-z0-9_.+-]+", label) is not None
    )


def extract_python_code(completion: str) -> str:
    if not completion or not str(completion).strip():
        return ""
    matches = _FENCED_BLOCK_RE.findall(str(completion))
    if not matches:
        return str(completion).strip()
    block = matches[-1].strip()
    if "\n" in block:
        first, rest = block.split("\n", 1)
        if _looks_like_fence_label(first):
            return rest.strip()
    return block


def _parse_ground_truth(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(
            "LCB ground_truth must be a JSON list of test-case objects"
        )
    return value


def _codegrader_input_output(
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    if not test_cases:
        raise ValueError("LCB row has no test cases")
    test_types = {test.get("testtype", "stdin") for test in test_cases}
    if len(test_types) != 1:
        raise ValueError(
            f"Mixed LCB test types are unsupported: {sorted(test_types)}"
        )
    converted: dict[str, Any] = {
        "inputs": [test["input"] for test in test_cases],
        "outputs": [test["output"] for test in test_cases],
    }
    test_type = next(iter(test_types))
    if test_type == "functional":
        method_name = test_cases[0].get("method_name")
        if not method_name:
            raise ValueError("Functional LCB row is missing method_name")
        converted["fn_name"] = method_name
    elif test_type != "stdin":
        raise ValueError(f"Unsupported LCB test type: {test_type}")
    return converted


def score_lcbv6(
    rows: list[dict[str, Any]],
    *,
    timeout: int = 6,
    num_processes: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute all generated programs and return benchmark metrics."""

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    samples = []
    generations = []
    for row in rows:
        tests = _parse_ground_truth(row.get("ground_truth"))
        samples.append(
            {
                "input_output": json.dumps(
                    _codegrader_input_output(tests)
                )
            }
        )
        generations.append(
            [
                extract_python_code(generation)
                for generation in raw_generation_list(row)
            ]
        )

    k_values = [1]
    if generations and min(len(values) for values in generations) >= 5:
        k_values.append(5)
    metrics, results = codegen_metrics(
        samples=samples,
        generations=generations,
        k_list=k_values,
        num_process_evaluate=num_processes,
        timeout=timeout,
    )

    graded = []
    all_correct: list[bool] = []
    for index, row in enumerate(rows):
        generation_results = results[index]
        correct = [
            bool(test_results)
            and all(int(value) > 0 for value in test_results)
            for test_results in generation_results
        ]
        all_correct.extend(correct)
        output = dict(row)
        output["correct"] = correct
        output["accuracy"] = (
            sum(correct) / len(correct) if correct else float("nan")
        )
        output["grader"] = "lcbv6_codegrader"
        graded.append(output)

    summary = {
        "grader": "lcbv6_codegrader",
        "num_rows": len(rows),
        "num_generations": sum(len(values) for values in generations),
        "num_correct": sum(all_correct),
        "accuracy": (
            sum(all_correct) / len(all_correct)
            if all_correct
            else float("nan")
        ),
        "timeout": timeout,
        "num_processes": num_processes,
    }
    for key, value in metrics.items():
        if key != "detail":
            summary[key] = float(value)
    return graded, summary
