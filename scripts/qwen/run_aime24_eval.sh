#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the AIME 2024 evaluation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_benchmark.sh" aime24
