"""Canonical benchmark settings for the Uno evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "UNO_EVAL_DATA_DIR",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "nano-vllm-uno"
        / "benchmarks",
    )
)


MATH_INSTRUCTION = (
    "Please reason step by step and put your final answer in \\boxed{}.\n"
)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    task: str
    expected_rows: int
    data_path: Path
    instruction: str = ""
    parser: str = "passthrough"
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    grader_timeout: float = 10.0


def _data(filename: str) -> Path:
    return DEFAULT_DATA_ROOT / filename


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "gsm8k": BenchmarkConfig(
        name="gsm8k",
        task="gsm8k",
        expected_rows=1319,
        data_path=_data("gsm8k.jsonl"),
        instruction=MATH_INSTRUCTION,
        chat_template_kwargs={"reasoning_effort": "high"},
    ),
    "math500": BenchmarkConfig(
        name="math500",
        task="math500",
        expected_rows=500,
        data_path=_data("math500.jsonl"),
        instruction=MATH_INSTRUCTION,
        chat_template_kwargs={"reasoning_effort": "high"},
    ),
    "aime24": BenchmarkConfig(
        name="aime24",
        task="aime24",
        expected_rows=30,
        data_path=_data("aime24.jsonl"),
        instruction=MATH_INSTRUCTION,
        chat_template_kwargs={"reasoning_effort": "high"},
    ),
    "aime25": BenchmarkConfig(
        name="aime25",
        task="aime25",
        expected_rows=30,
        data_path=_data("aime25.jsonl"),
        instruction=MATH_INSTRUCTION,
        chat_template_kwargs={"reasoning_effort": "high"},
    ),
    "aime26": BenchmarkConfig(
        name="aime26",
        task="aime26",
        expected_rows=30,
        data_path=_data("aime26.jsonl"),
        instruction=MATH_INSTRUCTION,
        chat_template_kwargs={"reasoning_effort": "high"},
    ),
    "humaneval": BenchmarkConfig(
        name="humaneval",
        task="humaneval",
        expected_rows=164,
        data_path=_data("humaneval.jsonl"),
        parser="code_completion",
        grader_timeout=10.0,
    ),
    "mbpp": BenchmarkConfig(
        name="mbpp",
        task="mbpp",
        expected_rows=500,
        data_path=_data("mbpp.jsonl"),
        chat_template_kwargs={"reasoning_effort": "high"},
        grader_timeout=10.0,
    ),
    "lcbv6": BenchmarkConfig(
        name="lcbv6",
        task="lcbv6",
        expected_rows=175,
        data_path=_data("lcbv6.jsonl"),
        parser="lcb_code",
        chat_template_kwargs={"reasoning_effort": "high"},
        grader_timeout=6.0,
    ),
    "gpqa": BenchmarkConfig(
        name="gpqa",
        task="gpqa",
        expected_rows=448,
        data_path=_data("gpqa_main.jsonl"),
        parser="gpqa_answer",
    ),
    "gpqa_diamond": BenchmarkConfig(
        name="gpqa_diamond",
        task="gpqa_diamond",
        expected_rows=198,
        data_path=_data("gpqa_diamond.jsonl"),
        parser="mc_answer",
    ),
    "mmlu_pro": BenchmarkConfig(
        name="mmlu_pro",
        task="mmlu_pro",
        expected_rows=12032,
        data_path=_data("mmlu_pro.jsonl"),
        parser="mmlu_pro_answer",
    ),
    "ifeval": BenchmarkConfig(
        name="ifeval",
        task="ifeval",
        expected_rows=541,
        data_path=_data("ifeval.jsonl"),
    ),
}


ALIASES = {
    "gpqa-main": "gpqa",
    "gpqa_main": "gpqa",
    "gpqa-diamond": "gpqa_diamond",
    "human-eval": "humaneval",
    "lcb": "lcbv6",
    "livecodebench": "lcbv6",
    "mmlu-pro": "mmlu_pro",
}


def normalize_benchmark_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    return ALIASES.get(normalized, normalized)


def get_benchmark(name: str) -> BenchmarkConfig:
    normalized = normalize_benchmark_name(name)
    try:
        return BENCHMARKS[normalized]
    except KeyError as exc:
        available = ", ".join(BENCHMARKS)
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {available}"
        ) from exc


def list_benchmarks() -> tuple[str, ...]:
    return tuple(BENCHMARKS)
