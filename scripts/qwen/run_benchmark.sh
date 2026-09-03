#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${UNO_BUNDLE_REPO:-}" ]]; then
  if [[ -z "${UNO_BUNDLE_REVISION:-}" ]]; then
    echo "UNO_BUNDLE_REVISION is required with UNO_BUNDLE_REPO" >&2
    exit 2
  fi

  BUNDLE_DIR="$(
    "${PYTHON:-python}" - "${UNO_BUNDLE_REPO}" "${UNO_BUNDLE_REVISION}" <<'PY'
import sys

from huggingface_hub import snapshot_download


repo, revision = sys.argv[1:]
print(
    snapshot_download(
        repo_id=repo,
        revision=revision,
    )
)
PY
  )"

  if [[ ! -f "${BUNDLE_DIR}/config.json" ]]; then
    echo "Bundle base config is missing: ${BUNDLE_DIR}/config.json" >&2
    exit 2
  fi
  if [[ ! -f "${BUNDLE_DIR}/model.safetensors" \
        && ! -f "${BUNDLE_DIR}/model.safetensors.index.json" ]]; then
    echo "Bundle base weights are missing under ${BUNDLE_DIR}" >&2
    exit 2
  fi
  if [[ ! -f "${BUNDLE_DIR}/adapter/adapter_config.json" ]]; then
    echo "Bundle adapter config is missing: ${BUNDLE_DIR}/adapter/adapter_config.json" >&2
    exit 2
  fi
  if [[ ! -f "${BUNDLE_DIR}/adapter/adapter_model.safetensors" \
        && ! -f "${BUNDLE_DIR}/adapter/adapter_model.bin" ]]; then
    echo "Bundle adapter weights are missing under ${BUNDLE_DIR}/adapter" >&2
    exit 2
  fi

  export MODEL="${BUNDLE_DIR}"
  export TOKENIZER_PATH="${BUNDLE_DIR}"
  export GATED_LORA_PATH="${BUNDLE_DIR}/adapter"
  export RUN_NAME="${RUN_NAME:-$(basename "${UNO_BUNDLE_REPO}")}"
  unset MODEL_REVISION GATED_LORA_REVISION
fi

export UNO_MODEL_DEFAULTS="${SCRIPT_DIR}/release_model_defaults.sh"
export UNO_RESULTS_NAMESPACE=qwen
exec "${SCRIPT_DIR}/../common/run_benchmark.sh" "$@"
