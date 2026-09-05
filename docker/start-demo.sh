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

cd /app

/opt/vlm-venv/bin/python -m uvicorn src.inference.vlm_http_api:app \
  --host 127.0.0.1 --port 8001 \
  > /tmp/outfit-vlm.log 2>&1 &
VLM_PID=$!

for _ in $(seq 1 60); do
  if /opt/core-venv/bin/python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz', timeout=1).read()" \
    >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! /opt/core-venv/bin/python -c \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz', timeout=2).read()" \
  >/dev/null 2>&1; then
  echo "VLM service failed to start" >&2
  sed -n '1,160p' /tmp/outfit-vlm.log >&2 || true
  exit 1
fi

/opt/core-venv/bin/python -m uvicorn src.inference.http_api:app \
  --host 0.0.0.0 --port 8000 \
  --workers 1 &
CORE_PID=$!
wait "${CORE_PID}"
