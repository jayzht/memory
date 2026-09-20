#!/usr/bin/env bash
# stop_webui.sh -- stop the detached memory inspector.
#
# NOTE: match only the python process, never "web_ui.py" loosely -- the start
# script's own command line contains the same string, and a loose pkill would
# kill the caller too.
set -uo pipefail
cd "$(dirname "$0")"

stopped=0
if [[ -f .webui.pid ]]; then
  PID=$(cat .webui.pid 2>/dev/null || true)
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" 2>/dev/null && stopped=1
    sleep 1
    kill -0 "${PID}" 2>/dev/null && kill -9 "${PID}" 2>/dev/null
  fi
  rm -f .webui.pid
fi
# Fallback: match the interpreter + script, not the wrapper shell.
for PID in $(pgrep -f "^python3 .*web_ui\.py" 2>/dev/null); do
  kill "${PID}" 2>/dev/null && stopped=1
done
sleep 1
[[ "${stopped}" == "1" ]] && echo "stopped" || echo "nothing was running"
