#!/usr/bin/env python3
"""Download and prepare public data for the 12-benchmark Uno suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from nano_vllm_uno.eval.benchmarks import (
    DEFAULT_DATA_ROOT,
    list_benchmarks,
    normalize_benchmark_name,
)
from nano_vllm_uno.eval.data import prepare_all_benchmark_data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmarks",
        nargs="*",
        metavar="BENCHMARK",
        help="Benchmarks to prepare; omitted means all 12.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    available = set(list_benchmarks())
    normalized = [normalize_benchmark_name(name) for name in args.benchmarks]
    unknown = [name for name in normalized if name not in available]
    if unknown:
        parser.error(
            f"unknown benchmark(s): {', '.join(unknown)}; "
            f"choose from {', '.join(list_benchmarks())}"
        )
    args.benchmarks = normalized
    return args


def main() -> None:
    args = parse_args()
    prepared = prepare_all_benchmark_data(
        args.benchmarks,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    for name, path in prepared.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
