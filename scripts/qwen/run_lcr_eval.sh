#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for AA-LCR generation. Set SKIP_GRADING=1 when no
# OpenAI-compatible judge is configured.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_benchmark.sh" lcr
