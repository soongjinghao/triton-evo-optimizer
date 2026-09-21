#!/bin/bash
# 3.1 主实验 · 5 算子试点
# 阶段 A(manifest) -> 阶段 B(search: b1/b2/full) -> 阶段 C(remeasure)
set -u
cd /workspace/Agent

source set_env/set_api_huoshan.sh
export ENGINE_FLASH=deepseek-v4-flash-ga-260731
export ENGINE_PRO=deepseek-v4-pro-ga-260813

PY=/usr/local/python3.11.15/bin/python3.11
KERNELS="eye_kernel matmul_kernel_simplified _act_quant_kernel _rms_norm_kernel mean_kernel"
METHODS="b1 b2 full"

for k in $KERNELS; do
  echo "===== [$(date +%H:%M:%S)] Manifest: $k ====="
  $PY experiments/pipeline.py manifest -k "$k" --repeats 5 --device 0
  for m in $METHODS; do
    echo "===== [$(date +%H:%M:%S)] Search: $k / $m ====="
    $PY experiments/pipeline.py search -k "$k" -m "$m" --run-id 0 --seed 0
    echo "===== [$(date +%H:%M:%S)] Remeasure: $k / $m ====="
    $PY experiments/pipeline.py remeasure -k "$k" -m "$m" --run-id 0 --repeats 5 --device 0
  done
done

echo "===== [$(date +%H:%M:%S)] ALL DONE ====="
$PY experiments/summarize.py
