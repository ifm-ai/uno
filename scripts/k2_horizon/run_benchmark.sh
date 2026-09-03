#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UNO_MODEL_DEFAULTS="${SCRIPT_DIR}/release_model_defaults.sh"
export UNO_RESULTS_NAMESPACE=k2_horizon
exec "${SCRIPT_DIR}/../common/run_benchmark.sh" "$@"
