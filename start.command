#!/bin/zsh
set -e

cd "$(dirname "$0")"

echo "Starting 拼豆管理 page: http://127.0.0.1:8000/index_v2.html"
echo "Starting local recognition service: http://127.0.0.1:5055"
echo

if [[ -f "local_ai_server/vlm.env" ]]; then
  source "local_ai_server/vlm.env"
fi

cleanup() {
  if [[ -n "${PAGE_PID:-}" ]]; then kill "$PAGE_PID" 2>/dev/null || true; fi
  if [[ -n "${AI_PID:-}" ]]; then kill "$AI_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

/usr/bin/python3 local_ai_server/server.py &
AI_PID=$!

python3 -m http.server 8000 &
PAGE_PID=$!

wait
