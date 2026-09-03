"""Math benchmark answer scorer."""

from __future__ import annotations

import re
from typing import Any

from .parsers import extract_last_boxed_content, generation_list, get_accuracy, source_key

try:
    from math_verify import parse, verify
except ModuleNotFoundError:  # pragma: no cover - exercised in envs without optional dep
    parse = None
    verify = None


def _require_math_verify() -> None:
    if parse is None or verify is None:
        raise RuntimeError(
            "math scoring requires the optional 'math_verify' package. "
            "Install it or run in an environment that provides it."
        )


def normalize_answer_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().strip("$").strip()
    gsm8k_answer = re.search(r"####\s*([^\n]+)", text)
    if gsm8k_answer:
        text = gsm8k_answer.group(1).strip()
    text = re.sub(r"\\(?:boxed|fbox)\s*\{(.+)\}\s*$", r"\1", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)", "", text)
    text = text.replace("\\displaystyle", "")
    text = re.sub(r"\\mathbf\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\s+([A-Za-z])", r"\1", text)
    text = re.sub(r"\\\\\s*\[[^\]]+\]", r"\\\\", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\frac\s*([+-]?\d+)\s*([+-]?\d+)", r"\1/\2", text)
    text = text.replace("{,}", "")
    text = re.sub(r"\\(?:,|;|!|:)", "", text)
    text = text.replace("\\%", "%")
    text = text.replace("\\$", "").replace("$", "")
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^\{([^{}])\}", r"^\1", text)
    text = re.sub(r"_\{([^{}])\}", r"_\1", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"^[A-Za-z]+=(?=\\begin\{pmatrix\})", "", text)
    return text


def has_plain_variable(text: str) -> bool:
    text = re.sub(r"\\(?:text|mathrm)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    return bool(re.search(r"[A-Za-z]", text))


def parse_boxed_content(boxed: str) -> Any:
    _require_math_verify()
    boxed = normalize_answer_text(boxed)
    leading_number = re.match(r"\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", boxed)
    if leading_number and re.fullmatch(r"[A-Za-z]+", boxed[leading_number.end() :]):
        ans = leading_number.group(1).replace(",", "")
        try:
            return parse(ans)
        except Exception:
            return ans
    if has_plain_variable(boxed):
        return boxed
    try:
        parsed = parse(f"${boxed}$")
        if parsed:
            return parsed
        parsed = parse(boxed)
        if parsed:
            return parsed
    except Exception:
        pass
    if leading_number:
        ans = leading_number.group(1).replace(",", "")
        try:
            return parse(ans)
        except Exception:
            return ans
    return boxed


def parse_unboxed_answer_with_verify(text: str) -> Any:
    _require_math_verify()
    try:
        parsed = parse(f"${text}$")
        if parsed:
            return parsed
        parsed = parse(text)
        if parsed:
            return parsed
    except Exception:
        pass

    answer_patterns = [
        r"The answer is:?\s*\$?([\-0-9\.,]+)",
        r"#### ?\$?([\-0-9\.,]+)",
        r"Therefore,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"So,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Thus,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Hence,? the answer is:?\s*\$?([\-0-9\.,]+)",
        r"Final answer:?\s*\$?([\-0-9\.,]+)",
        r"The final answer is:?\s*\$?([\-0-9\.,]+)",
        r"The answer is:?\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?",
        r"=\s*\$?([\-0-9\.,]+)\s*(?:miles?|minutes?|hours?|dollars?|GB)?\.?\s*(?:The answer|$)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            ans = matches[-1].replace(",", "").strip().rstrip(".")
            if ans:
                try:
                    return parse(ans)
                except Exception:
                    return ans

    sentence_end_pattern = r"(?:is|are|equals?|makes?|has|have|gets?|arrives?|covers?|travels?)\s+\$?([\-0-9\.,]+)(?:\s*(?:miles?|minutes?|hours?|dollars?|GB))?\.?\s*$"
    match = re.search(sentence_end_pattern, text, re.MULTILINE | re.IGNORECASE)
    if match:
        ans = match.group(1).replace(",", "").strip().rstrip(".")
        if ans:
            try:
                return parse(ans)
            except Exception:
                return ans

    for sentence in reversed(text.split(".")):
        if "Human:" in sentence or "Assistant:" in sentence:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sentence)
        if numbers:
            num = numbers[-1].lstrip("0") or "0"
            try:
                return parse(num)
            except Exception:
                return num
    return None


def parse_answer_with_verify(text: str) -> Any:
    boxed = extract_last_boxed_content(text)
    if boxed:
        return parse_boxed_content(boxed)
    return parse_unboxed_answer_with_verify(text)


def parse_answer_candidates_with_verify(text: str) -> list[Any]:
    boxed = extract_last_boxed_content(text)
    if boxed:
        candidates = [parse_boxed_content(boxed), normalize_answer_text(boxed), boxed]
    else:
        candidates = [parse_unboxed_answer_with_verify(text)]
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        key = repr(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def vector_components(text: str) -> tuple[str, ...] | None:
    matrix = re.fullmatch(r"\\begin\{pmatrix\}(.+)\\end\{pmatrix\}", text)
    if matrix:
        return tuple(part for part in matrix.group(1).split(r"\\") if part)
    tuple_match = re.fullmatch(r"\(([^()]+)\)", text)
    if tuple_match and "," in tuple_match.group(1):
        return tuple(part for part in tuple_match.group(1).split(",") if part)
    return None


def text_answers_match(answer: str, gold_raw: str) -> bool:
    answer_norm = normalize_answer_text(answer)
    gold_norm = normalize_answer_text(gold_raw)
    if answer_norm == gold_norm:
        return True
    try:
        if float(answer_norm.rstrip("%")) == float(gold_norm.rstrip("%")):
            return True
    except ValueError:
        pass
    if re.fullmatch(r"\([A-Za-z]\)", gold_norm) and answer_norm == gold_norm[1:-1]:
        return True
    answer_components = vector_components(answer_norm)
    gold_components = vector_components(gold_norm)
    return bool(answer_components and answer_components == gold_components)


def compare_answers(answer: Any, gold_raw: str | None) -> bool:
    _require_math_verify()
    if not answer or gold_raw is None:
        return False
    if isinstance(answer, str) and text_answers_match(answer, gold_raw):
        return True
    try:
        if verify(answer, gold_raw):
            return True
        gold_parsed = parse_answer_with_verify(gold_raw)
        if gold_parsed:
            return bool(verify(gold_parsed, answer))
    except Exception:
        pass
    return isinstance(answer, str) and text_answers_match(answer, gold_raw)


def grade_math_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("ground_truth")
    expected_values = expected if isinstance(expected, list) else [expected]
    correct = []
    parsed_generations = []
    for generation in generation_list(row):
        candidates = parse_answer_candidates_with_verify(generation)
        parsed_generations.append([str(candidate) for candidate in candidates])
        matched = any(
            compare_answers(candidate, str(gold))
            for candidate in candidates
            for gold in expected_values
            if gold is not None
        )
        correct.append(matched)
    graded = dict(row)
    graded["parsed_generations"] = parsed_generations
    graded["correct"] = correct
    graded["accuracy"] = get_accuracy(correct)
    graded["grader"] = "math"
    return graded


def score_math(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graded = [grade_math_row(row) for row in rows]
    correct = [bool(value) for row in graded for value in row.get("correct", [])]

    by_source: dict[str, list[tuple[int, bool]]] = {}
    for index, row in enumerate(graded):
        sample_index = row.get("sample_index", 0)
        try:
            sample_base = int(sample_index)
        except (TypeError, ValueError):
            sample_base = 0
        key = source_key(row, index)
        by_source.setdefault(key, [])
        for offset, value in enumerate(row.get("correct", [])):
            by_source[key].append((sample_base + offset, bool(value)))

    per_problem = [
        [value for _, value in sorted(samples, key=lambda item: item[0])]
        for samples in by_source.values()
        if samples
    ]
    samples_per_problem = max((len(samples) for samples in per_problem), default=0)
    sample0_correct = [samples[0] for samples in per_problem if samples]
    pass_correct = [any(samples) for samples in per_problem]

    summary = {
        "grader": "math",
        "num_rows": len(graded),
        "num_problems": len(per_problem),
        "samples_per_problem": samples_per_problem,
        "num_correct": sum(int(value) for value in correct),
        "accuracy": get_accuracy(correct),
    }
    if per_problem:
        summary["avg_at_1"] = get_accuracy(sample0_correct)
        summary["pass_at_1"] = get_accuracy(sample0_correct)
        summary["num_correct_at_1"] = sum(int(value) for value in sample0_correct)
        if samples_per_problem > 1:
            summary[f"avg_at_{samples_per_problem}"] = get_accuracy(correct)
            summary[f"pass_at_{samples_per_problem}"] = get_accuracy(pass_correct)
            summary[f"num_pass_at_{samples_per_problem}"] = sum(
                int(value) for value in pass_correct
            )
    return graded, summary
