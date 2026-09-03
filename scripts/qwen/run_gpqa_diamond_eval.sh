#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the GPQA-Diamond evaluation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_benchmark.sh" gpqa_diamond
