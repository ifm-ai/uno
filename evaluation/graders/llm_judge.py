"""OpenAI-compatible HLE, AA-LCR, and AA-Omniscience judges."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from .multiple_choice import grade_multiple_choice_row
from ..parsers import generation_list, get_accuracy


LCR_PROMPT = """
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}

Output:
{{"GRADE": "CORRECT" or "INCORRECT"}}
""".strip()

HLE_PROMPT = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()

OMNI_PROMPT = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "PARTIAL_ANSWER", "NOT_ATTEMPTED"].

CORRECT: The answer fully matches or is equivalent to the gold target
INCORRECT: The answer contradicts or differs in meaning from the gold target
PARTIAL_ANSWER: The answer is accurate and nearly complete but not at the correct level of detail
NOT_ATTEMPTED: Used only when the model refuses, omits, or explicitly states it does not know the answer, or needs more context or tools to answer the question

Here is a new example.

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: PARTIAL_ANSWER
D: NOT_ATTEMPTED

Just return the letters "A", "B", "C", or "D", with no text around it.
""".strip()

_THINK_CLOSE_RE = re.compile(r"</(?:ifm\|)?think[^>]*>", re.IGNORECASE)


def _visible_answer(text: str | None) -> str:
    value = text or ""
    matches = list(_THINK_CLOSE_RE.finditer(value))
    return value[matches[-1].end():].strip() if matches else value.strip()


def parse_correct_incorrect_response(response: str | None) -> int:
    text = _visible_answer(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            grade = payload.get("GRADE", payload.get("grade"))
            if isinstance(payload.get("answer"), dict):
                grade = payload["answer"].get("GRADE", payload["answer"].get("grade", grade))
            if isinstance(grade, str):
                return int(grade.upper() == "CORRECT")
        except json.JSONDecodeError:
            pass
    match = re.search(r"\b(CORRECT|INCORRECT)\b", text, re.IGNORECASE)
    return int(bool(match) and match.group(1).upper() == "CORRECT")


def parse_hle_response(response: str | None) -> int:
    text = _visible_answer(response)
    match = re.search(r'"?correct"?\s*:\s*"?(yes|no)"?', text, re.IGNORECASE)
    return int(bool(match) and match.group(1).lower() == "yes")


def parse_omniscience_response(response: str | None) -> int:
    match = re.search(r"\b([ABCD])\b", _visible_answer(response).upper())
    return {"A": 1, "B": -1, "C": 0, "D": 0}.get(match.group(1), 0) if match else 0


def build_judge_messages(task: str, row: dict[str, Any], response: str) -> list[dict[str, str]]:
    question = str(row.get("completion_input") or row.get("problem") or "")
    answer = row.get("ground_truth")
    if task == "aa_lcr":
        if "\n=== QUESTION ===\n" in question:
            question = question.split("\n=== QUESTION ===\n")[-1]
        if question.endswith("\n\nAnswer:"):
            question = question[:-len("\n\nAnswer:")]
        return [
            {"role": "system", "content": "You are a teacher grading a quiz, and output JSON."},
            {"role": "user", "content": LCR_PROMPT.format(query=question.strip(), result=response or "No answer", answer=answer)},
        ]
    if task == "hle":
        return [{"role": "user", "content": HLE_PROMPT.format(question=question, response=response or "No answer", correct_answer=answer)}]
    if task == "aa_omniscience":
        return [{"role": "user", "content": OMNI_PROMPT.format(question=question, target=answer, predicted_answer=response or "No answer")}]
    raise ValueError(f"Unsupported judge task: {task}")


def _parse(task: str, response: str | None) -> int:
    if task == "hle":
        return parse_hle_response(response)
    if task == "aa_omniscience":
        return parse_omniscience_response(response)
    return parse_correct_incorrect_response(response)


async def score_llm_judge_async(
    rows: list[dict[str, Any]],
    *,
    task: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    max_concurrency: int = 8,
    request_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("LLM judge scoring requires the optional 'openai' package") from exc

    client = AsyncOpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    kwargs = dict(request_kwargs or {})

    async def grade_row(row: dict[str, Any]) -> dict[str, Any]:
        if task == "hle" and row.get("answer_type") == "multipleChoice":
            return grade_multiple_choice_row(row)
        values: list[int] = []
        responses: list[str | None] = []
        for generation in generation_list(row):
            try:
                async with semaphore:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=build_judge_messages(task, row, generation),
                        **kwargs,
                    )
                response = completion.choices[0].message.content
                responses.append(response)
                values.append(_parse(task, response))
            except Exception as exc:
                responses.append(None)
                values.append(0)
                row = {**row, "judge_error": str(exc)}
        return {
            **row,
            "correct": values,
            "accuracy": sum(value == 1 for value in values) / len(values),
            "judge_responses": responses,
            "grader": f"{task}_llm_judge",
        }

    graded = await asyncio.gather(*(grade_row(row) for row in rows))
    values = [int(value) for row in graded for value in row.get("correct", [])]
    if task == "aa_omniscience":
        denominator = 600
        correct = sum(value == 1 for value in values)
        incorrect = sum(value == -1 for value in values)
        return graded, {
            "grader": "aa_omniscience_llm_judge",
            "num_rows": len(graded),
            "num_graded": len(values),
            "denominator": denominator,
            "num_correct": correct,
            "num_incorrect": incorrect,
            "accuracy": correct / denominator,
            "non_hallucination_rate": 1.0 - incorrect / denominator,
            "omniscience_index": (correct - incorrect) / denominator,
        }
    bool_values = [value == 1 for value in values]
    return graded, {
        "grader": f"{task}_llm_judge",
        "num_rows": len(graded),
        "num_generations": len(values),
        "num_correct": sum(bool_values),
        "accuracy": get_accuracy(bool_values),
    }


def score_llm_judge(rows: list[dict[str, Any]], **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return asyncio.run(score_llm_judge_async(rows, **kwargs))
