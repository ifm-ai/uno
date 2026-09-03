#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the LiveCodeBench v6 evaluation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_benchmark.sh" lcbv6
