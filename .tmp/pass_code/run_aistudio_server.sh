#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ONLINE_SERVICE_SCRIPT="$REPO_ROOT/deployment_scripts/ant/online_inference_service.py"

usage() {
  cat <<'EOF'
Usage:
  MODEL_DIR=/abs/model_dir deployment_scripts/ant/run_aistudio_server.sh

Environment variables:
  MODEL_DIR                  Required. Both left/right model paths default to this value.
  LEFT_MODEL_PATH            Optional. Overrides left model path.
  RIGHT_MODEL_PATH           Optional. Overrides right model path.
  BACKEND                    Optional. pytorch (default) or tensorrt.
  TRT_ENGINE_DIR             Optional shared TensorRT engine directory.
  LEFT_TRT_ENGINE_PATH       Optional left TensorRT engine directory.
  RIGHT_TRT_ENGINE_PATH      Optional right TensorRT engine directory.
  AISTUDIO_DATA_CONFIG       Optional. Defaults to deployment_scripts.ant.aistudio_test_config:AistudioCompatDataConfig
  AISTUDIO_EMBODIMENT_TAG    Optional. Defaults to new_embodiment.
  HOST                       Optional. Defaults to 0.0.0.0.
  PORT                       Optional. Defaults to 8001.
  API_TOKEN                  Optional.
  PYTHON_BIN                 Optional. Defaults to python.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_DIR=${MODEL_DIR:-}
if [[ -z "$MODEL_DIR" ]]; then
  echo "MODEL_DIR is required." >&2
  usage >&2
  exit 1
fi

BACKEND=${BACKEND:-pytorch}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}
AISTUDIO_DATA_CONFIG=${AISTUDIO_DATA_CONFIG:-deployment_scripts.ant.aistudio_test_config:AistudioCompatDataConfig}
AISTUDIO_EMBODIMENT_TAG=${AISTUDIO_EMBODIMENT_TAG:-new_embodiment}
LEFT_MODEL_PATH=${LEFT_MODEL_PATH:-$MODEL_DIR}
RIGHT_MODEL_PATH=${RIGHT_MODEL_PATH:-$MODEL_DIR}

CMD=(
  "$PYTHON_BIN" "$ONLINE_SERVICE_SCRIPT"
  --role server
  --service-mode aistudio
  --transport http
  --backend "$BACKEND"
  --left-model-path "$LEFT_MODEL_PATH"
  --right-model-path "$RIGHT_MODEL_PATH"
  --aistudio-data-config "$AISTUDIO_DATA_CONFIG"
  --aistudio-embodiment-tag "$AISTUDIO_EMBODIMENT_TAG"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "${API_TOKEN:-}" ]]; then
  CMD+=(--api-token "$API_TOKEN")
fi

if [[ "$BACKEND" == "tensorrt" ]]; then
  if [[ -n "${TRT_ENGINE_DIR:-}" ]]; then
    CMD+=(--trt-engine-path "$TRT_ENGINE_DIR")
  else
    if [[ -z "${LEFT_TRT_ENGINE_PATH:-}" || -z "${RIGHT_TRT_ENGINE_PATH:-}" ]]; then
      echo "For TensorRT, set TRT_ENGINE_DIR or both LEFT_TRT_ENGINE_PATH and RIGHT_TRT_ENGINE_PATH." >&2
      exit 1
    fi
    CMD+=(--left-trt-engine-path "$LEFT_TRT_ENGINE_PATH" --right-trt-engine-path "$RIGHT_TRT_ENGINE_PATH")
  fi
fi

cd "$REPO_ROOT"
printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
