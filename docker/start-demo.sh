#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
  if [[ -n "${VLM_PID:-}" ]]; then
    kill "${VLM_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CORE_PID:-}" ]]; then
    kill "${CORE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

APP_ROOT="${APP_ROOT:-/app}"
CORE_PYTHON="${CORE_PYTHON:-/opt/core-venv/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-/opt/vlm-venv/bin/python}"
VLM_HOST="${VLM_HOST:-127.0.0.1}"
VLM_PORT="${VLM_PORT:-8001}"
CORE_HOST="${CORE_HOST:-0.0.0.0}"
CORE_PORT="${CORE_PORT:-8000}"
LOG_DIR="${LOG_DIR:-/tmp}"
VLM_LOG="${VLM_LOG:-${LOG_DIR}/outfit-vlm.log}"
CORE_LOG="${CORE_LOG:-/dev/stderr}"

cd "${APP_ROOT}"
mkdir -p "${LOG_DIR}"

"${VLM_PYTHON}" -m uvicorn src.inference.vlm_http_api:app \
  --host "${VLM_HOST}" --port "${VLM_PORT}" \
  > "${VLM_LOG}" 2>&1 &
VLM_PID=$!

for _ in $(seq 1 60); do
  if "${CORE_PYTHON}" -c \
    "import urllib.request; urllib.request.urlopen('http://${VLM_HOST}:${VLM_PORT}/healthz', timeout=1).read()" \
    >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! "${CORE_PYTHON}" -c \
  "import urllib.request; urllib.request.urlopen('http://${VLM_HOST}:${VLM_PORT}/healthz', timeout=2).read()" \
  >/dev/null 2>&1; then
  echo "VLM service failed to start" >&2
  sed -n '1,160p' "${VLM_LOG}" >&2 || true
  exit 1
fi

"${CORE_PYTHON}" -m uvicorn src.inference.http_api:app \
  --host "${CORE_HOST}" --port "${CORE_PORT}" \
  --workers 1 \
  > "${CORE_LOG}" 2>&1 &
CORE_PID=$!
wait "${CORE_PID}"
