#!/usr/bin/env python3
"""Download and prepare public data for the Uno evaluation suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from evaluation.benchmarks import (
    DEFAULT_DATA_ROOT,
    normalize_benchmark_name,
)
from evaluation.data import BUILDERS, prepare_all_benchmark_data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmarks",
        nargs="*",
        metavar="BENCHMARK",
        help="Public benchmarks to prepare; omitted means all available builders.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    available = set(BUILDERS)
    normalized = [normalize_benchmark_name(name) for name in args.benchmarks]
    unknown = [name for name in normalized if name not in available]
    if unknown:
        parser.error(
            f"unknown benchmark(s): {', '.join(unknown)}; "
            f"choose from {', '.join(BUILDERS)}"
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
