#!/bin/bash
# 3.1 主实验 · 3 算子验证（修复版：AST 拒绝后重新生成补位，保证 N_eval 对齐）
# 复用已有 manifest（T_base / T_seed 为纯测量，不受代码修复影响）
set -u
cd /workspace/Agent

source set_env/set_api_huoshan.sh
export ENGINE_FLASH=deepseek-v4-flash-ga-260731
export ENGINE_PRO=deepseek-v4-pro-ga-260813

PY=/usr/local/python3.11.15/bin/python3.11
KERNELS="eye_kernel matmul_kernel_simplified _act_quant_kernel"
METHODS="b1 b2 full"

for k in $KERNELS; do
  for m in $METHODS; do
    echo "===== [$(date +%H:%M:%S)] Search: $k / $m ====="
    $PY experiments/pipeline.py search -k "$k" -m "$m" --run-id 0 --seed 0
    echo "===== [$(date +%H:%M:%S)] Remeasure: $k / $m ====="
    $PY experiments/pipeline.py remeasure -k "$k" -m "$m" --run-id 0 --repeats 5 --device 0
  done
done

echo "===== [$(date +%H:%M:%S)] ALL DONE ====="
$PY experiments/summarize.py
