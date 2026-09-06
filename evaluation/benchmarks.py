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
    source_repo_id: str | None = None
    source_revision: str | None = None
    source_files: tuple[str, ...] = ()
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
    source_repo_id: str,
    source_revision: str,
    source_files: tuple[str, ...],
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
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        source_files=source_files,
        data_sha256=data_sha256,
        judge=judge,
    )


# The protocol document is authoritative for pinned values. For fields absent
# from that document, these settings retain the latest nano-vllm-uno protocol.
BENCHMARKS: dict[str, BenchmarkConfig] = {
    "aime24": _benchmark(
        "aime24", rows=30, samples=10, max_tokens=500_000, context=500_000,
        source_repo_id="Maxwell-Jia/AIME_2024",
        source_revision="8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        source_files=("aime_2024_problems.parquet",),
        data_sha256="266dcb058c1983ff3c7520a74c32962a7cb5ee2386c9f274aa9406e9909af9e5",
    ),
    "aime25": _benchmark(
        "aime25", rows=30, samples=32, max_tokens=500_000, context=500_000,
        source_repo_id="math-ai/aime25",
        source_revision="563bb8404243c5f09de6ec262f2db674fe5bce9b",
        source_files=("test.jsonl",),
        data_sha256="25673735af85a06ec49de6c74a412819ad1e4f6b2deef2e50004dc78d15a09aa",
    ),
    "aime26": _benchmark(
        "aime26", rows=30, samples=32, max_tokens=500_000, context=500_000,
        source_repo_id="math-ai/aime26",
        source_revision="79037aebdb6580008fb960d17cb21fd3099083e3",
        source_files=("aime2026.jsonl",),
        data_sha256="476869564d174db87072caad1bcf4bb3b2d3baf811c1352a4887d5e1b0f1709c",
    ),
    "arc_challenge": _benchmark(
        "arc_challenge", task="arc_challenge", rows=1172, samples=1,
        max_tokens=131_072, context=262_144, parser="mc_answer",
        source_repo_id="allenai/ai2_arc",
        source_revision="210d026faf9955653af8916fad021475a3f00453",
        source_files=("ARC-Challenge/test-00000-of-00001.parquet",),
        data_sha256="11199fe212915b7bfd3fc166c9452f2006d974300d2432c141f63a99b4f9fec1",
    ),
    "gpqa_diamond": _benchmark(
        "gpqa_diamond", rows=198, samples=5, temperature=0.6,
        max_tokens=262_144, context=262_144, parser="mc_answer",
        source_repo_id="Idavidrein/gpqa",
        source_revision="633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        source_files=("gpqa_diamond.csv",),
        data_sha256="28359e3c903c834ef383b54c9cb9c1497d2574a5e8c15ecf20b22b24f8bdf8fc",
    ),
    "gsm8k": _benchmark(
        "gsm8k", rows=1319, samples=2, max_tokens=131_072, context=262_144,
        source_repo_id="openai/gsm8k",
        source_revision="740312add88f781978c0658806c59bc2815b9866",
        source_files=("main/test-00000-of-00001.parquet",),
        data_sha256="94d4be9a732880b5518d1c1ab5385cf2e9f1375653ce2561ba02d539c1065c51",
    ),
    "hle": _benchmark(
        "hle", rows=2158, samples=1, max_tokens=262_144, context=262_144,
        source_repo_id="cais/hle",
        source_revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        source_files=("data/test-00000-of-00001.parquet",),
        data_sha256="58b8356aff4324d808a9001dfa6e89f2c93b70d3c949ee1be55bafbd7aabd2ec",
        judge="gpt-5.5-no-tools-medium",
    ),
    "humaneval": _benchmark(
        "humaneval", rows=164, samples=2, max_tokens=131_072,
        context=262_144, parser="code_completion", grader_timeout=10.0,
        source_repo_id="openai/openai_humaneval",
        source_revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        source_files=("openai_humaneval/test-00000-of-00001.parquet",),
        data_sha256="2910cac0699d8179df4e357ce017b6cd7a6c6653f22e55c5da7512f49f4fd595",
    ),
    "ifeval": _benchmark(
        "ifeval", rows=541, samples=1, max_tokens=500_000, context=500_000,
        max_num_seqs=64, source_repo_id="google/IFEval",
        source_revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
        source_files=("ifeval_input_data.jsonl",),
        data_sha256="3dbcbe04d15abd2c5e41046f96b8c6852dc83c57e2694b364be1b9675f404be2",
    ),
    "lcr": _benchmark(
        "lcr", task="aa_lcr", rows=100, samples=1, max_tokens=32_768,
        context=262_144, source_repo_id="ArtificialAnalysis/AA-LCR",
        source_revision="bdae010bbce259820c0e34c1d7cce210d966fb75",
        source_files=(
            "AA-LCR_Dataset.csv",
            "extracted_text/AA-LCR_extracted-text.zip",
        ),
        data_sha256="3a7271a1dc30cd65c06b6bb2bde18172f15f72f45d15ce7d173fbd47f121ef2d",
        judge="GLM-5.2-FP8",
    ),
    "math500": _benchmark(
        "math500", rows=500, samples=2, max_tokens=131_072,
        context=262_144, source_repo_id="HuggingFaceH4/MATH-500",
        source_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        source_files=("test.jsonl",),
        data_sha256="e9db693a52216152e9a18ca70110d393fcaa5dd04178693008dbfd20f1a233c8",
    ),
    "mbpp": _benchmark(
        "mbpp", rows=500, samples=2, max_tokens=131_072, context=262_144,
        grader_timeout=10.0, source_repo_id="google-research-datasets/mbpp",
        source_revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        source_files=("full/test-00000-of-00001.parquet",),
        data_sha256="3347bb07cdf003935fdc5d4bb0feee104ed14f5e90a8ef7384752134106824f2",
    ),
    "omniscience": _benchmark(
        "omniscience", task="aa_omniscience", rows=600, samples=1,
        max_tokens=131_072, context=262_144,
        source_repo_id="ArtificialAnalysis/AA-Omniscience-Public",
        source_revision="4a8ffc87c4650054825fb767fe0da4a4fc97ff32",
        source_files=("AA-Omniscience_dataset_public.csv",),
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
