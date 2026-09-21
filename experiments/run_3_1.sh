#!/bin/bash
# 3.1 主实验：B0(重测) + B1/B2/FULL(搜索+复测) + 汇总表4
#
#   bash experiments/run_3_1.sh                      # 全部 50 个算子
#   bash experiments/run_3_1.sh eye_kernel mean_kernel
#   RUN_ID=1 SEED=7 bash experiments/run_3_1.sh ...  # 第 2 次独立重复
#
# 已完成的算子自动跳过，可反复执行实现断点续跑。
set -u
cd /workspace/Agent

source set_env/set_api_huoshan.sh
export ENGINE_FLASH=deepseek-v4-flash-ga-260731
export ENGINE_PRO=deepseek-v4-pro-ga-260813

PY=/usr/local/python3.11.15/bin/python3.11
export PYTHONUNBUFFERED=1   # 确保日志实时落盘，便于 tail -f 观察进度
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-5}"

if [ $# -gt 0 ]; then KERNELS="$*"; else KERNELS=$(ls datasets); fi
echo "[3.1] kernels: $KERNELS | RUN_ID=$RUN_ID SEED=$SEED"

for k in $KERNELS; do
  # 阶段 A：B0 重测参考实现与各初始种子
  if [ ! -f "experiments/manifest/$k.json" ]; then
    echo "===== [$(date +%H:%M:%S)] B0 重测: $k ====="
    $PY experiments/pipeline.py manifest -k "$k" --repeats "$REPEATS" --device 0
  fi

  for m in b1 b2 full; do
    if [ -f "experiments/results/$m/${k}__r${RUN_ID}.remeasure.json" ]; then
      echo "[skip] $m/$k/r$RUN_ID 已完成"
      continue
    fi
    echo "===== [$(date +%H:%M:%S)] 搜索: $k / $m ====="
    $PY experiments/pipeline.py search -k "$k" -m "$m" --run-id "$RUN_ID" --seed "$SEED"
    echo "===== [$(date +%H:%M:%S)] 复测: $k / $m ====="
    $PY experiments/pipeline.py remeasure -k "$k" -m "$m" --run-id "$RUN_ID" \
        --repeats "$REPEATS" --device 0
  done
done

echo "===== [$(date +%H:%M:%S)] 汇总表4 ====="
$PY experiments/summarize.py
