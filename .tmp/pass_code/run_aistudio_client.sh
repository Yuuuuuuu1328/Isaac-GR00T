#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ONLINE_SERVICE_SCRIPT="$REPO_ROOT/deployment_scripts/ant/online_inference_service.py"

usage() {
  cat <<'EOF'
Usage:
  REQUEST_FILE=/tmp/aistudio_left.json deployment_scripts/ant/run_aistudio_client.sh

Environment variables:
  REQUEST_FILE       Optional for act/echo. Defaults to /tmp/aistudio_left.json.
  HOST               Optional. Defaults to 127.0.0.1.
  PORT               Optional. Defaults to 8001.
  LATENCY_ENDPOINT   Optional. act (default), ping, or echo.
  API_TOKEN          Optional.
  PYTHON_BIN         Optional. Defaults to python.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
LATENCY_ENDPOINT=${LATENCY_ENDPOINT:-act}
REQUEST_FILE=${REQUEST_FILE:-/tmp/aistudio_left.json}

CMD=(
  "$PYTHON_BIN" "$ONLINE_SERVICE_SCRIPT"
  --role client
  --service-mode aistudio
  --transport http
  --host "$HOST"
  --port "$PORT"
  --latency-endpoint "$LATENCY_ENDPOINT"
)

if [[ -n "${API_TOKEN:-}" ]]; then
  CMD+=(--api-token "$API_TOKEN")
fi

if [[ "$LATENCY_ENDPOINT" != "ping" ]]; then
  CMD+=(--request-file "$REQUEST_FILE")
fi

cd "$REPO_ROOT"
printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
