"""AA-LCR grader using an OpenAI-compatible judge endpoint."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.request import Request, urlopen

from ..parsers import generation_list, get_accuracy


GRADE_RE = re.compile(r'"GRADE"\s*:\s*"(CORRECT|INCORRECT)"', re.IGNORECASE)


def _judge_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return (
        f"{endpoint}/chat/completions"
        if endpoint.endswith("/v1")
        else f"{endpoint}/v1/chat/completions"
    )


def _judge(question: str, answer: str, reference: str) -> tuple[bool, str]:
    endpoint = os.environ.get("UNO_LCR_JUDGE_URL")
    model = os.environ.get("UNO_LCR_JUDGE_MODEL")
    if not endpoint or not model:
        raise RuntimeError(
            "AA-LCR scoring requires UNO_LCR_JUDGE_URL and UNO_LCR_JUDGE_MODEL"
        )
    prompt = (
        "Judge whether the candidate answer is semantically correct given the "
        "question and reference answer. Return only "
        '{"GRADE":"CORRECT"} or {"GRADE":"INCORRECT"}.\n\n'
        f"Question:\n{question}\n\nCandidate answer:\n{answer}\n\n"
        f"Reference answer:\n{reference}"
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("UNO_LCR_JUDGE_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(_judge_url(endpoint), payload, headers), timeout=300) as response:
        body = json.load(response)
    content = body["choices"][0]["message"]["content"]
    match = GRADE_RE.search(content)
    if match is None:
        return False, content
    return match.group(1).upper() == "CORRECT", content


def score_lcr(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graded = []
    all_correct: list[bool] = []
    for row in rows:
        question = str(row.get("completion_input") or row.get("problem") or "")
        reference = str(row.get("ground_truth") or "")
        correct = []
        responses = []
        for answer in generation_list(row):
            is_correct, response = _judge(question, answer, reference)
            correct.append(is_correct)
            responses.append(response)
        result = dict(row)
        result.update(
            {
                "correct": correct,
                "accuracy": get_accuracy(correct),
                "judge_responses": responses,
                "grader": "openai_compatible_lcr_judge",
            }
        )
        graded.append(result)
        all_correct.extend(correct)
    return graded, {
        "accuracy": get_accuracy(all_correct),
        "num_graded": len(all_correct),
        "grader": "openai_compatible_lcr_judge",
    }
