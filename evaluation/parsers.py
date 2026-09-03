"""Common benchmark-output parsers."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def get_accuracy(correct: list[bool] | list[int]) -> float:
    if not correct:
        return float("nan")
    return sum(int(x) for x in correct) / len(correct)


def raw_generation_list(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("generations"), list):
        return [str(g or "") for g in row["generations"]]
    return [str(row.get("generation") or "")]


def generation_list(row: dict[str, Any]) -> list[str]:
    raw = raw_generation_list(row)
    parsed = row.get("parsed_generations")
    if not isinstance(parsed, list) or len(parsed) != len(raw):
        return raw
    return [
        str(parsed_value) if parsed_value is not None else raw_value
        for parsed_value, raw_value in zip(parsed, raw, strict=True)
    ]


def source_key(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("source_row", row.get("row", fallback)))


def load_source_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records = read_jsonl(path)
    merged: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        keys = {
            str(index),
            str(record.get("row", "")),
            str(record.get("id", "")),
            str(record.get("question_id", "")),
            str(record.get("problem_id", "")),
        }
        for key in keys:
            if key:
                merged[key] = record
    return merged


def merge_source_metadata(
    rows: list[dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not source_records:
        return rows
    merged_rows = []
    for index, row in enumerate(rows):
        merged = dict(source_records.get(source_key(row, index), {}))
        merged.update(row)
        merged_rows.append(merged)
    return merged_rows


def extract_last_boxed_content(text: str | None) -> str | None:
    if not text:
        return None
    pattern = r"\\(?:boxed|fbox)\s*\{"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    for match in reversed(matches):
        start = match.end()
        brace_count = 1
        i = start
        while i < len(text) and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            i += 1
        if brace_count == 0:
            return text[start : i - 1].strip().strip("$").strip()
    return None


def parse_think_suffix(generation: str | None) -> str | None:
    if generation is None:
        return None
    parts = re.split(
        r"</(?:think[^>]*|ifm\|think)>",
        generation,
        flags=re.IGNORECASE | re.DOTALL,
    )
    tail = parts[-1].strip() if parts else ""
    return tail or None


def parse_code_completion(generation: str | None) -> str | None:
    if generation is None:
        return None
    # Strip an optional terminating Qwen chat token.
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", generation)
    think_match = re.search(r"</think>\n*", text)
    if think_match:
        text = text[think_match.end() :]
    elif re.match(r"\s*<think>", text):
        fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1)
        else:
            parts = re.split(r"\n\n", text)
            text = parts[-1] if parts else text
    fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return text if text.strip() else None


def normalize_mc_match(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip().upper()
    if candidate in {"1", "2", "3", "4", "5"}:
        return "ABCDE"[int(candidate) - 1]
    return candidate or None


_MC_EMPHASIS = r"(?:\*\*|__|`)?"
_MC_CHOICE = r"([A-J1-5])"
_MC_LINE_SCAN_LIMIT = 96
_MC_TAIL_SCAN_CHARS = 4096
_THINK_CLOSE_RE = re.compile(r"</think[^>]*>", flags=re.IGNORECASE)

_MC_TAG_PATTERNS = (
    re.compile(
        rf"<(?:final\s*)?answer>\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*</(?:final\s*)?answer>",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"<answer>\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*</answer>",
        flags=re.IGNORECASE,
    ),
    re.compile(rf"\\(?:boxed|fbox)\s*\{{\s*{_MC_CHOICE}\s*\}}", flags=re.IGNORECASE),
    re.compile(
        rf"\\(?:boxed|fbox)\s*\{{\s*\\text\s*\{{\s*{_MC_CHOICE}\s*\}}\s*\}}",
        flags=re.IGNORECASE,
    ),
)

_MC_FINAL_OPTION_LINE_PATTERN = re.compile(
    rf"^\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?"
    rf"\s*[.):-]\s+.+$",
    flags=re.IGNORECASE,
)

_MC_STRONG_LINE_PATTERNS = (
    re.compile(
        rf"^\s*(?:final[.:]?\s*)?answer\b\s*[:：-]?\s*(?:is\s*)?{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*(?:the\s+)?choice\s+is\s+{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s*(?:[.!?])?\s*$",
        flags=re.IGNORECASE,
    ),
)

_MC_WEAK_LINE_PATTERNS = (
    re.compile(
        rf"\b(?:final[.:]?\s*)?answer\b\s*[:：-]?\s*(?:is\s*)?{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:the\s+)?choice\s+is\s+{_MC_EMPHASIS}\s*(?:option\s*)?[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:match(?:es|ed)?|coincides?\s+with|corresponds?\s+to|"
        rf"gives?|yields?|confirms?)\s+(?:option|choice)\s*{_MC_EMPHASIS}"
        rf"\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\bonly\s+(?:option\s*)?{_MC_EMPHASIS}\s*[\(\[]?\s*"
        rf"{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:option|choice)\s*{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?\s*{_MC_EMPHASIS}\s+(?:is|would\s+be|remains?|works?|fits?|correct|best|right|compatible|consistent|larger)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_MC_EMPHASIS}\s*[\(\[]?\s*{_MC_CHOICE}\s*[\)\]]?"
        rf"\s*{_MC_EMPHASIS}\s+is\s+"
        rf"(?:correct|best|right|compatible|consistent)\b",
        flags=re.IGNORECASE,
    ),
)


def _mc_match_last(pattern: re.Pattern[str], text: str) -> str | None:
    last = None
    for match in pattern.finditer(text):
        last = match
    return normalize_mc_match(last.group(1)) if last is not None else None


def _collect_mc_scan_lines(text: str) -> tuple[str | None, list[str]]:
    first = None
    trailing = deque(maxlen=_MC_LINE_SCAN_LIMIT)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if first is None:
            first = line
        trailing.append(line)
    return first, list(trailing)


def _extract_mc_from_line(line: str, *, allow_weak: bool) -> str | None:
    for pattern in _MC_TAG_PATTERNS:
        match = pattern.search(line)
        if match:
            return normalize_mc_match(match.group(1))
    for pattern in _MC_STRONG_LINE_PATTERNS:
        match = pattern.search(line)
        if match:
            return normalize_mc_match(match.group(1))
    if allow_weak:
        for pattern in _MC_WEAK_LINE_PATTERNS:
            match = pattern.search(line)
            if match:
                return normalize_mc_match(match.group(1))
    return None


def _extract_mc_candidate(candidate: str) -> str | None:
    first_line, trailing_lines = _collect_mc_scan_lines(candidate)
    if first_line is not None:
        first_line_match = _extract_mc_from_line(first_line, allow_weak=False)
        if first_line_match is not None:
            return first_line_match
    if trailing_lines:
        final_option = _MC_FINAL_OPTION_LINE_PATTERN.search(trailing_lines[-1])
        if final_option is not None:
            return normalize_mc_match(final_option.group(1))
    for line in reversed(trailing_lines):
        line_match = _extract_mc_from_line(line, allow_weak=True)
        if line_match is not None:
            return line_match
    tail = candidate[-_MC_TAIL_SCAN_CHARS:]
    for pattern in _MC_TAG_PATTERNS + _MC_WEAK_LINE_PATTERNS:
        tail_match = _mc_match_last(pattern, tail)
        if tail_match is not None:
            return tail_match
    return None


def extract_mc_answer(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    candidate = text
    last_close = None
    for match in _THINK_CLOSE_RE.finditer(text):
        last_close = match
    if last_close is not None:
        candidate = text[last_close.end() :].strip()

    extracted = _extract_mc_candidate(candidate)
    if extracted is not None:
        return extracted
    if candidate != text:
        return _extract_mc_candidate(text)
    return None


def _answer_tail(generation: str) -> str:
    if "</think>" in generation:
        return re.sub(
            r"^.*?</think>\s*",
            "",
            generation,
            count=1,
            flags=re.DOTALL,
        )
    return generation[-500:]


def parse_gpqa_answer(generation: str | None) -> str:
    """Extract a terminal A-D answer from a reasoning response."""
    if not generation:
        return "?"
    candidate = _answer_tail(generation)
    match = re.search(r"ANSWER:\s*([A-Da-d])", candidate)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"[Aa]nswer\s*(?:is\s*:?\s*|:\s*)\*{0,2}\s*\(?([A-Da-d])\)?\*{0,2}",
        candidate,
    )
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{[^}]*?([A-Da-d])[^A-Da-d}]*\}", candidate)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"(?:\*\*\(?([A-Da-d])\)?\*\*|\(([A-Da-d])\))\s*[.\s]*$",
        candidate.strip(),
    )
    return (match.group(1) or match.group(2)).upper() if match else "?"


def parse_mmlu_pro_answer(generation: str | None) -> str:
    """Extract a terminal A-J answer from a reasoning response."""
    if not generation:
        return "?"
    candidate = _answer_tail(generation)
    match = re.search(
        r"[Aa]nswer is:?\s*\*{0,2}\(?([A-Ja-j])\)?\*{0,2}",
        candidate,
    )
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{[^}]*?([A-Ja-j])[^A-Ja-j}]*\}", candidate)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"(?:\*\*\(?([A-Ja-j])\)?\*\*|\(([A-Ja-j])\))\s*[.\s]*$",
        candidate.strip(),
    )
    if match:
        return (match.group(1) or match.group(2)).upper()
    matches = re.findall(
        r"(?:^|\n)\s*\(?([A-Ja-j])[.)]\s*$",
        candidate,
        re.MULTILINE,
    )
    if matches:
        return matches[-1].upper()
    match = re.search(r"[Aa]nswer:?\s*\(?([A-Ja-j])\)?", candidate)
    return match.group(1).upper() if match else "?"


def _strip_example_code(code: str) -> str:
    lines = code.split("\n")
    kept: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.lstrip()
        if (
            re.match(
                r"^(def |async def |class |import |from \S+ import )",
                stripped,
            )
            and not line[0:1].isspace()
        ):
            in_block = True
            kept.append(line)
            continue
        if in_block:
            if stripped == "" or line[0:1].isspace():
                kept.append(line)
            else:
                in_block = False
        elif not stripped:
            if kept:
                kept.append(line)
        elif re.match(r"^(import |from \S+ import )", stripped):
            kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) if kept else code


def _best_code_block(matches: list[str]) -> str:
    for block in reversed(matches):
        if re.search(
            r"^(def |async def |class )",
            block,
            re.MULTILINE,
        ):
            return _strip_example_code(block)
    return _strip_example_code(matches[-1])


_CHANNEL_RE = re.compile(
    r"<\|channel\|>\s*(?P<channel>[^<\n]+)\s*<\|message\|>"
    r"(?P<body>.*?)(?=(?:<\|channel\|>|<\|end\|>|$))",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_lcb_code(generation: str | None) -> str | None:
    if generation is None:
        return None
    channel_matches = list(_CHANNEL_RE.finditer(generation))
    if channel_matches:
        preferred = None
        for match in channel_matches:
            if match.group("channel").strip().lower() in {"final", "answer"}:
                preferred = match
        text = (preferred or channel_matches[-1]).group("body")
    else:
        text = generation
    text = re.sub(r"<\|[^|]+(?:\|[^|]+)*\|>", "", text).strip()
    matches = re.findall(
        r"```(?:python3?|py)?\s*\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        return _best_code_block(matches)
    candidate = _strip_example_code(text)
    if re.search(
        r"^(def |async def |class |import |from \S+ import )",
        candidate,
        flags=re.MULTILINE,
    ):
        return candidate
    return None


def parse_generation(parser_name: str, generation: str) -> str | None:
    parsers = {
        "passthrough": lambda value: value,
        "noop": lambda value: value,
        "identity": lambda value: value,
        "code_completion": parse_code_completion,
        "humaneval": parse_code_completion,
        "mc_answer": extract_mc_answer,
        "multiple_choice_answer": extract_mc_answer,
        "gpqa_answer": parse_gpqa_answer,
        "mmlu_pro_answer": parse_mmlu_pro_answer,
        "lcb_code": parse_lcb_code,
    }
    try:
        parser = parsers[parser_name]
    except KeyError as exc:
        available = ", ".join(sorted(parsers))
        raise ValueError(
            f"Unknown parser {parser_name!r}. Available: {available}"
        ) from exc
    return parser(generation)


def apply_parser(
    rows: list[dict[str, Any]],
    parser_name: str,
) -> list[dict[str, Any]]:
    parsed_rows = []
    for row in rows:
        parsed = dict(row)
        parsed["parsed_generations"] = [
            parse_generation(parser_name, generation)
            for generation in raw_generation_list(row)
        ]
        parsed_rows.append(parsed)
    return parsed_rows
