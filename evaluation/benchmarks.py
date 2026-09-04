"""Canonical K2V3 evaluation protocol for the supported 13 benchmarks."""

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

PROTOCOL_NAME = "K2V3 Eval Protocol"
PROTOCOL_DATE = "2026-09-01"


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    task: str
    expected_rows: int
    data_path: Path
    num_samples: int = 1
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 131_072
    max_model_len: int = 262_144
    top_k: int | None = None
    max_num_seqs: int = 4
    instruction: str = ""
    parser: str = "passthrough"
    chat_template_kwargs: dict[str, object] = field(
        default_factory=lambda: {"reasoning_effort": "high"}
    )
    grader_timeout: float = 10.0
    source_filename: str | None = None
    data_sha256: str | None = None
    judge: str | None = None


def _data(filename: str) -> Path:
    return DEFAULT_DATA_ROOT / filename


def _benchmark(
    name: str,
    *,
    task: str | None = None,
    rows: int,
    samples: int,
    max_tokens: int,
    context: int,
    temperature: float = 1.0,
    parser: str = "passthrough",
    max_num_seqs: int = 4,
    grader_timeout: float = 10.0,
    source_filename: str,
    data_sha256: str,
    judge: str | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=name,
        task=task or name,
        expected_rows=rows,
        data_path=_data(f"{name}.jsonl"),
        num_samples=samples,
        temperature=temperature,
        top_p=0.95,
        top_k=None,
        max_tokens=max_tokens,
        max_model_len=context,
        max_num_seqs=max_num_seqs,
        parser=parser,
        grader_timeout=grader_timeout,
        source_filename=source_filename,
        data_sha256=data_sha256,
        judge=judge,
    )


# The protocol document is authoritative for pinned values. For fields absent
# from that document, these settings retain the latest nano-vllm-uno protocol.
BENCHMARKS: dict[str, BenchmarkConfig] = {
    "aime24": _benchmark(
        "aime24", rows=30, samples=10, max_tokens=500_000, context=500_000,
        source_filename="aime-2024/aime-2024.jsonl",
        data_sha256="266dcb058c1983ff3c7520a74c32962a7cb5ee2386c9f274aa9406e9909af9e5",
    ),
    "aime25": _benchmark(
        "aime25", rows=30, samples=32, max_tokens=500_000, context=500_000,
        source_filename="aime-2025/aime-2025.jsonl",
        data_sha256="25673735af85a06ec49de6c74a412819ad1e4f6b2deef2e50004dc78d15a09aa",
    ),
    "aime26": _benchmark(
        "aime26", rows=30, samples=32, max_tokens=500_000, context=500_000,
        source_filename="aime-2026/aime-2026.jsonl",
        data_sha256="476869564d174db87072caad1bcf4bb3b2d3baf811c1352a4887d5e1b0f1709c",
    ),
    "arc_challenge": _benchmark(
        "arc_challenge", task="arc_challenge", rows=1172, samples=1,
        max_tokens=131_072, context=262_144, parser="mc_answer",
        source_filename="arc-challenge/arc_challenge.jsonl",
        data_sha256="11199fe212915b7bfd3fc166c9452f2006d974300d2432c141f63a99b4f9fec1",
    ),
    "gpqa_diamond": _benchmark(
        "gpqa_diamond", rows=198, samples=5, temperature=0.6,
        max_tokens=262_144, context=262_144, parser="mc_answer",
        source_filename="gpqa-diamond/gpqa_diamond.jsonl",
        data_sha256="66f387c172aabe10089de0d97dbe1d120fff4b55ab371746a633ca37ef088f59",
    ),
    "gsm8k": _benchmark(
        "gsm8k", rows=1319, samples=2, max_tokens=131_072, context=262_144,
        source_filename="gsm8k_cot_zeroshot/gsm8k_cot_zeroshot.jsonl",
        data_sha256="94d4be9a732880b5518d1c1ab5385cf2e9f1375653ce2561ba02d539c1065c51",
    ),
    "hle": _benchmark(
        "hle", rows=2158, samples=1, max_tokens=262_144, context=262_144,
        source_filename="hle/hle.jsonl",
        data_sha256="58b8356aff4324d808a9001dfa6e89f2c93b70d3c949ee1be55bafbd7aabd2ec",
        judge="gpt-5.5-no-tools-medium",
    ),
    "humaneval": _benchmark(
        "humaneval", rows=164, samples=2, max_tokens=131_072,
        context=262_144, parser="code_completion", grader_timeout=10.0,
        source_filename="humaneval/humaneval.jsonl",
        data_sha256="2910cac0699d8179df4e357ce017b6cd7a6c6653f22e55c5da7512f49f4fd595",
    ),
    "ifeval": _benchmark(
        "ifeval", rows=541, samples=1, max_tokens=500_000, context=500_000,
        max_num_seqs=64, source_filename="ifeval/ifeval.jsonl",
        data_sha256="b58c072afb150ba249ccb36577001d52f8f872ce06c4b3877e01d35e730db03f",
    ),
    "lcr": _benchmark(
        "lcr", task="aa_lcr", rows=100, samples=1, max_tokens=32_768,
        context=262_144, source_filename="lcr/lcr.jsonl",
        data_sha256="3a7271a1dc30cd65c06b6bb2bde18172f15f72f45d15ce7d173fbd47f121ef2d",
        judge="GLM-5.2-FP8",
    ),
    "math500": _benchmark(
        "math500", rows=500, samples=2, max_tokens=131_072,
        context=262_144, source_filename="math500/math500.jsonl",
        data_sha256="e9db693a52216152e9a18ca70110d393fcaa5dd04178693008dbfd20f1a233c8",
    ),
    "mbpp": _benchmark(
        "mbpp", rows=500, samples=2, max_tokens=131_072, context=262_144,
        grader_timeout=10.0, source_filename="mbpp_zeroshot/mbpp_zeroshot.jsonl",
        data_sha256="3b14ca75991841aba4b9276333e2303c6edb28fe2447c6597682044fb392671e",
    ),
    "omniscience": _benchmark(
        "omniscience", task="aa_omniscience", rows=600, samples=1,
        max_tokens=131_072, context=262_144,
        source_filename="omniscience/omniscience.jsonl",
        data_sha256="e654f02c2981e3f0e0ba1290fb00f1ba4cd2a0d086bb1213307d3a2f5ef6b5ac",
        judge="GLM-5.2-FP8",
    ),
}


ALIASES = {
    "aime-24": "aime24",
    "aime-25": "aime25",
    "aime-26": "aime26",
    "arc": "arc_challenge",
    "arc-c": "arc_challenge",
    "arc-challenge": "arc_challenge",
    "gpqa-diamond": "gpqa_diamond",
    "human-eval": "humaneval",
    "aa-lcr": "lcr",
    "aa_lcr": "lcr",
    "aa-omniscience": "omniscience",
    "aa_omniscience": "omniscience",
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
