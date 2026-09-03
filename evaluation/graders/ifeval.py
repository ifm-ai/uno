"""IFEval scorer implementing strict and loose instruction metrics."""

from __future__ import annotations

import random
import re
from typing import Any

from langdetect import DetectorFactory

from ..ifeval_lib.evaluation_lib import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from ..parsers import generation_list


_THINK_CLOSE_RE = re.compile(r"</think[^>]*>", flags=re.IGNORECASE)


def _strip_thinking(text: str | None) -> str:
    if not text:
        return ""
    last_close = None
    for match in _THINK_CLOSE_RE.finditer(text):
        last_close = match
    if last_close is None:
        return text
    return text[last_close.end() :].lstrip()


def grade_ifeval_row(row: dict[str, Any]) -> dict[str, Any]:
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError("IFEval ground_truth must be an object")

    example = InputExample(
        key=row.get("row", row.get("source_row")),
        instruction_id_list=ground_truth["instruction_id_list"],
        prompt=str(row.get("completion_input") or ""),
        kwargs=ground_truth["kwargs"],
    )
    strict_prompt_follow = []
    strict_inst_follow = []
    loose_prompt_follow = []
    loose_inst_follow = []
    for generation in generation_list(row):
        prompt_to_response = {
            example.prompt: _strip_thinking(generation),
        }
        strict = test_instruction_following_strict(
            example,
            prompt_to_response,
        )
        loose = test_instruction_following_loose(
            example,
            prompt_to_response,
        )
        strict_prompt_follow.append(strict.follow_all_instructions)
        strict_inst_follow.append(strict.follow_instruction_list)
        loose_prompt_follow.append(loose.follow_all_instructions)
        loose_inst_follow.append(loose.follow_instruction_list)

    graded = dict(row)
    if len(strict_prompt_follow) == 1:
        graded["strict_prompt_follow"] = strict_prompt_follow[0]
        graded["strict_inst_follow"] = strict_inst_follow[0]
        graded["loose_prompt_follow"] = loose_prompt_follow[0]
        graded["loose_inst_follow"] = loose_inst_follow[0]
    else:
        graded["strict_prompt_follow"] = strict_prompt_follow
        graded["strict_inst_follow"] = strict_inst_follow
        graded["loose_prompt_follow"] = loose_prompt_follow
        graded["loose_inst_follow"] = loose_inst_follow
    graded["correct"] = [bool(value) for value in strict_prompt_follow]
    graded["accuracy"] = (
        sum(strict_prompt_follow) / len(strict_prompt_follow)
        if strict_prompt_follow
        else float("nan")
    )
    graded["grader"] = "ifeval"
    return graded


def _as_generation_values(
    row: dict[str, Any],
    key: str,
) -> list[Any]:
    value = row[key]
    if isinstance(value, list) and key.endswith("_prompt_follow"):
        return value
    return [value]


def _as_instruction_values(
    row: dict[str, Any],
    key: str,
) -> list[bool]:
    value = row[key]
    if not isinstance(value, list):
        return [bool(value)]
    if value and isinstance(value[0], list):
        return [
            bool(item)
            for generation in value
            for item in generation
        ]
    return [bool(item) for item in value]


def score_ifeval(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The upstream rubric uses random fallbacks for malformed letter-frequency
    # constraints, and langdetect is stochastic unless its factory is seeded.
    # Isolate both RNGs so identical generations always receive identical grades.
    random_state = random.getstate()
    detector_seed = DetectorFactory.seed
    try:
        random.seed(0)
        DetectorFactory.seed = 0
        graded = [grade_ifeval_row(row) for row in rows]
    finally:
        random.setstate(random_state)
        DetectorFactory.seed = detector_seed

    strict_prompt = [
        bool(value)
        for row in graded
        for value in _as_generation_values(row, "strict_prompt_follow")
    ]
    loose_prompt = [
        bool(value)
        for row in graded
        for value in _as_generation_values(row, "loose_prompt_follow")
    ]
    strict_instruction = [
        value
        for row in graded
        for value in _as_instruction_values(row, "strict_inst_follow")
    ]
    loose_instruction = [
        value
        for row in graded
        for value in _as_instruction_values(row, "loose_inst_follow")
    ]

    def average(values: list[bool]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return graded, {
        "grader": "ifeval",
        "num_rows": len(graded),
        "num_generations": len(strict_prompt),
        "strict_prompt_accuracy": average(strict_prompt),
        "strict_instruction_accuracy": average(strict_instruction),
        "loose_prompt_accuracy": average(loose_prompt),
        "loose_instruction_accuracy": average(loose_instruction),
        # The suite's headline IFEval number is strict prompt accuracy.
        "accuracy": average(strict_prompt),
    }
