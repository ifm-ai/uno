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
MC_TASKS = {"arc_challenge", "gpqa_diamond"}
CODE_TASKS = {"humaneval", "mbpp"}
JUDGE_TASKS = {"aa_lcr", "aa_omniscience"}


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
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-max-concurrency", type=int, default=8)
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument("--judge-reasoning-effort")
    return parser.parse_args(argv)


def _judge_request_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if args.judge_temperature is not None:
        kwargs["temperature"] = args.judge_temperature
    if args.judge_max_tokens is not None:
        kwargs["max_tokens"] = args.judge_max_tokens
    if args.judge_reasoning_effort:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "reasoning_effort": args.judge_reasoning_effort,
            }
        }
    return kwargs


def _intended_samples(
    benchmark: object,
    generation_summary: dict[str, object] | None,
) -> int:
    samples = int(getattr(benchmark, "num_samples"))
    if generation_summary is not None:
        samples = int(
            generation_summary.get("num_samples_per_problem", samples)
        )
    return samples


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

    generation_summary = None
    if args.generation_summary is not None:
        generation_summary = json.loads(
            args.generation_summary.read_text(encoding="utf-8")
        )

    rows = read_jsonl(args.generations)
    rows = merge_source_metadata(rows, load_source_records(args.data))
    rows = apply_parser(rows, parser_name)
    judge_kwargs = _judge_request_kwargs(args)

    if task in MATH_TASKS:
        graded, summary = score_math(rows, task=task)
    elif task in MC_TASKS:
        graded, summary = score_multiple_choice(rows)
        if task == "gpqa_diamond" and benchmark is not None:
            expected = benchmark.expected_rows * _intended_samples(
                benchmark,
                generation_summary,
            )
            summary["fixed_denominator"] = expected
            summary["accuracy"] = summary["num_correct"] / expected
    elif task in CODE_TASKS:
        graded, summary = score_code(rows, task=task, timeout=args.timeout)
    elif task == "ifeval":
        from .graders.ifeval import score_ifeval

        graded, summary = score_ifeval(rows)
    elif task == "hle" or task in JUDGE_TASKS:
        from .graders.llm_judge import score_llm_judge

        if task == "hle":
            exact_rows = [
                row for row in rows
                if row.get("answer_type") != "multipleChoice"
            ]
            if exact_rows and args.judge_model != "gpt-5.5":
                raise ValueError(
                    "HLE exact-answer scoring requires --judge-model gpt-5.5"
                )
        elif not args.judge_model or "GLM-5.2-FP8" not in args.judge_model:
            raise ValueError(
                f"{task} scoring requires a GLM-5.2-FP8 --judge-model"
            )
        if not args.judge_model:
            # This is valid only for an HLE file containing MC rows alone.
            graded, summary = score_multiple_choice(rows)
            summary["grader"] = "hle_multiple_choice"
        else:
            graded, summary = score_llm_judge(
                rows,
                task=task,
                model=args.judge_model,
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
                max_concurrency=args.judge_max_concurrency,
                request_kwargs=judge_kwargs,
            )
        if benchmark is not None and task in {"hle", "aa_lcr"}:
            expected = benchmark.expected_rows * _intended_samples(
                benchmark,
                generation_summary,
            )
            summary["fixed_denominator"] = expected
            summary["accuracy"] = summary["num_correct"] / expected
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
    if generation_summary is not None:
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
