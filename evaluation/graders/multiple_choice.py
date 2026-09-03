"""Multiple-choice benchmark scorer."""

from __future__ import annotations

from typing import Any

from ..parsers import extract_mc_answer, generation_list, get_accuracy, normalize_mc_match


def grade_multiple_choice_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("ground_truth")
    expected_values = expected if isinstance(expected, list) else [expected]
    generations = generation_list(row)
    picked: list[str | None] = []
    correct: list[bool] = []
    for index, generation in enumerate(generations):
        choice = extract_mc_answer(generation)
        gold = expected_values[index] if len(expected_values) == len(generations) else expected_values[0]
        expected_choice = normalize_mc_match(str(gold)[0] if gold else None)
        picked.append(choice)
        correct.append(choice is not None and choice == expected_choice)
    graded = dict(row)
    graded["picked"] = picked
    graded["correct"] = correct
    graded["accuracy"] = get_accuracy(correct)
    graded["grader"] = "multiple_choice"
    return graded


def score_multiple_choice(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graded = [grade_multiple_choice_row(row) for row in rows]
    correct = [
        bool(value)
        for row in graded
        for value in row.get("correct", [])
    ]
    return graded, {
        "grader": "multiple_choice",
        "num_rows": len(graded),
        "num_generations": len(correct),
        "num_correct": sum(int(value) for value in correct),
        "accuracy": get_accuracy(correct),
    }
