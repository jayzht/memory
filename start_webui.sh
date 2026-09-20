#!/usr/bin/env bash
# start_webui.sh -- bring the memory inspector up, verify it, print the URL.
#
# Usage:
#   ./start_webui.sh                     # DeepSeek backend, synthetic episodes
#   ./start_webui.sh --llm-backend heuristic     # offline, no API cost
#   ./start_webui.sh --dataset data/sample_episodes.json
#   ./start_webui.sh --port 8011 --token ""      # no token required
#
# Ctrl-C does NOT stop the server (it runs detached); use ./stop_webui.sh.

set -euo pipefail
cd "$(dirname "$0")"

PORT=8010
TOKEN="mem2024"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

# Stop any previous instance so the port is free.
if [[ -f .webui.pid ]]; then
  OLD=$(cat .webui.pid 2>/dev/null || true)
  if [[ -n "${OLD}" ]] && kill -0 "${OLD}" 2>/dev/null; then
    kill "${OLD}" 2>/dev/null || true
    sleep 1
  fi
fi

CMD=(python3 web_ui.py --host 0.0.0.0 --port "${PORT}"
     --token "${TOKEN}"
     --llm-backend "${LLM_BACKEND:-deepseek}" --model "${MODEL:-deepseek-flash}"
     --num-episodes "${NUM_EPISODES:-3}" --synthetic-turns "${SYNTHETIC_TURNS:-30}"
     --seed "${SEED:-777}" --log-level WARNING)
if [[ ${#ARGS[@]} -gt 0 ]]; then CMD+=("${ARGS[@]}"); fi

echo "启动: ${CMD[*]}"
echo "（前台运行；要停就 Ctrl-C。若在本沙箱环境，请让它在后台任务/独立终端里跑，"
echo "  因为每个 shell 有独立 PID namespace，自 detach 的子进程会随 shell 退出而消失。）"
"${CMD[@]}" &
SERVER_PID=$!
sleep 2

echo
echo "--- 自检 ---"
if [[ -n "${TOKEN}" ]]; then
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/?token=${TOKEN}" || true)
else
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)
fi
LISTEN=$(ss -ltn 2>/dev/null | grep -c ":${PORT}" || true)
echo "监听 0.0.0.0:${PORT}  ->  ${LISTEN} (1 = OK)   页面自检 HTTP ${CODE}"

echo
echo "================================================================"
echo " 本机/隧道打开:   http://127.0.0.1:${PORT}/?token=${TOKEN}"
echo " 若端口已转发，直接在浏览器打开上面这条即可。"
echo " 需要在你自己主机上做一次端口转发（二选一）:"
echo "   a) VS Code: Ctrl+Shift+P -> 'Forward a Port' -> ${PORT}（命令面板里叫 转发端口）"
echo "   b) 终端:    ssh -N -L ${PORT}:127.0.0.1:${PORT} \$(whoami)@\$(hostname -I | awk '{print \$1}')"
echo " 停止: kill ${SERVER_PID}   或 ./stop_webui.sh"
echo "================================================================"
