#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the MBPP evaluation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_benchmark.sh" mbpp
