#!/bin/bash
# 3.5 代表性算子案例：提取 T_base / T_before / T_after / S 与 Profiling 前后值
#
#   bash experiments/run_3_5.sh mean_kernel _rms_norm_kernel _act_quant_kernel
set -u
cd /workspace/Agent

PY=/usr/local/python3.11.15/bin/python3.11
export PYTHONUNBUFFERED=1   # 确保日志实时落盘，便于 tail -f 观察进度
METHOD="${METHOD:-full}"
RUN_ID="${RUN_ID:-0}"

if [ $# -eq 0 ]; then
  echo "用法: bash experiments/run_3_5.sh <kernel1> [kernel2] [kernel3]"
  echo "建议从 3.1 结果中挑选加速比差异明显的算子作为案例"
  exit 1
fi

echo "===== [$(date +%H:%M:%S)] 采集案例数据 (method=$METHOD) ====="
$PY experiments/case_study.py -k "$@" -m "$METHOD" --run-id "$RUN_ID"
