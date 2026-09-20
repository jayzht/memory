#!/usr/bin/env bash
# check_webui.sh -- is the inspector up?  restart it if not.
set -uo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8010}"
TOKEN="${TOKEN:-mem2024}"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/?token=${TOKEN}" || true)
PID=$(cat .webui.pid 2>/dev/null || echo "-")
if [[ "${CODE}" == "200" ]]; then
  echo "OK   服务在跑 (pid ${PID})   打开: http://127.0.0.1:${PORT}/?token=${TOKEN}"
else
  echo "DOWN 未响应 (HTTP ${CODE})，重启中…"
  ./start_webui.sh
fi
