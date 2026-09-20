#!/usr/bin/env bash
# 在【宿主机终端】运行：启动 vLLM（OpenAI 兼容），供本项目的 --llm-backend vllm 使用。
#
# 参数来自你机器上 vLLM 的历史启动记录（~/.config/vllm/usage_stats.json）：
#   双卡 RTX 4090 D 24GB ×2 | tensor_parallel_size=2 | gpu_memory_utilization=0.9
#   fp16 | enable_prefix_caching=True | block_size=16
#
#   bash start_vllm.sh                                  # Qwen2.5-7B-Instruct, 端口 8100
#   MODEL=/home/ubuntu/models/Qwen2.5-7B-Instruct-IDS2017FT PORT=8101 bash start_vllm.sh
set -euo pipefail

MODEL="${MODEL:-/home/ubuntu/models/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8100}"
TP="${TP:-2}"                       # 双 4090 D：2 卡张量并行（与你历史配置一致）
GPU_UTIL="${GPU_UTIL:-0.9}"
DTYPE="${DTYPE:-float16}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/miniconda3/bin/vllm}"

echo "vLLM 启动参数："
echo "  model=${MODEL}"
echo "  port=${PORT}  tensor_parallel=${TP}  gpu_util=${GPU_UTIL}  dtype=${DTYPE}"
echo "  提示：若 Ollama 正驻留大模型，先释放显存：ollama stop deepseek-r1:14b"
echo
exec "${VLLM_BIN}" serve "${MODEL}" \
  --served-model-name "$(basename "${MODEL}")" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --dtype "${DTYPE}" \
  --enable-prefix-caching \
  --max-model-len 16384
