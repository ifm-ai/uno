"""Score nano-vllm-uno generation JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .benchmarks import get_benchmark, list_benchmarks
from .graders.code import score_code
from .graders.math import score_math
from .graders.multiple_choice import score_multiple_choice
from .parsers import (
    apply_parser,
    load_source_records,
    merge_source_metadata,
    read_jsonl,
    write_json,
    write_jsonl,
)


MATH_TASKS = {"gsm8k", "math500", "aime24", "aime25", "aime26"}
MC_TASKS = {
    "gpqa",
    "gpqa_diamond",
    "mmlu_pro",
}
CODE_TASKS = {"humaneval", "mbpp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        help=(
            "Use canonical task, parser, data, and timeout defaults. Available: "
            + ", ".join(list_benchmarks())
        ),
    )
    parser.add_argument("--task")
    parser.add_argument("--parser")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--generation-summary",
        type=Path,
        help=(
            "Optional summary JSON emitted by "
            "evaluation.run."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Original dataset JSONL used to merge grading metadata.",
    )
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--grader-num-processes",
        type=int,
        default=min(64, os.cpu_count() or 1),
        help="Worker count for aggregate LCB code grading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    benchmark = get_benchmark(args.benchmark) if args.benchmark else None
    if args.task is None and benchmark is None:
        raise ValueError("Set --benchmark or --task")
    task = (
        benchmark.task
        if args.task is None
        else args.task.lower().replace("-", "_")
    )
    parser_name = args.parser or (
        benchmark.parser if benchmark is not None else "passthrough"
    )
    if args.data is None and benchmark is not None:
        from .data import prepare_benchmark_data

        args.data = prepare_benchmark_data(benchmark.name)
    if args.timeout is None:
        args.timeout = (
            benchmark.grader_timeout if benchmark is not None else 10.0
        )

    rows = read_jsonl(args.generations)
    rows = merge_source_metadata(rows, load_source_records(args.data))
    rows = apply_parser(rows, parser_name)

    if task in MATH_TASKS:
        graded, summary = score_math(rows, task=task)
    elif task in MC_TASKS:
        graded, summary = score_multiple_choice(rows)
    elif task in CODE_TASKS:
        graded, summary = score_code(rows, task=task, timeout=args.timeout)
    elif task == "lcbv6":
        from .graders.lcb import score_lcbv6

        graded, summary = score_lcbv6(
            rows,
            timeout=int(args.timeout),
            num_processes=args.grader_num_processes,
        )
    elif task == "ifeval":
        from .graders.ifeval import score_ifeval

        graded, summary = score_ifeval(rows)
    elif task == "aa_lcr":
        from .graders.lcr import score_lcr

        graded, summary = score_lcr(rows)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    summary.setdefault("num_generations", len(rows))
    summary.update(
        {
            "task": task,
            "benchmark": benchmark.name if benchmark is not None else None,
            "parser": parser_name,
            "source_generations": str(args.generations),
            "grades_path": str(args.grades),
        }
    )
    if args.generation_summary is not None:
        generation_summary = json.loads(
            args.generation_summary.read_text(encoding="utf-8")
        )
        summary["generation_metrics"] = generation_summary
        for key in (
            "total_output_tokens",
            "elapsed_seconds",
            "output_tokens_per_second",
            "decoder_stats",
            "decoder_tokens_per_sequence_forward",
            "resolved_settings",
        ):
            if key in generation_summary:
                summary[key] = generation_summary[key]
    write_jsonl(args.grades, graded)
    write_json(args.scores, summary)
    print(summary)


if __name__ == "__main__":
    main()
