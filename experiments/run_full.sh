#!/bin/bash
# 3.1 主实验统一启动脚本（阶段 A -> B -> C -> 汇总）
#
# 用法：
#   bash experiments/run_full.sh                       # 跑全部 50 个算子
#   bash experiments/run_full.sh eye_kernel mean_kernel # 只跑指定算子
#   SKIP_MANIFEST=1 bash experiments/run_full.sh ...   # 已测过 T_base/T_seed 时跳过阶段 A
#   RUN_ID=1 SEED=7 bash experiments/run_full.sh ...   # 独立重复（第 2 次，随机种子 7）
#
# 注意：脚本会占用 NPU，同一时刻只允许一个实例运行。
set -u
cd /workspace/Agent

source set_env/set_api_huoshan.sh
export ENGINE_FLASH=deepseek-v4-flash-ga-260731
export ENGINE_PRO=deepseek-v4-pro-ga-260813

PY=/usr/local/python3.11.15/bin/python3.11

if [ $# -gt 0 ]; then
  KERNELS="$*"
else
  KERNELS=$(ls datasets)
fi

RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
SKIP_MANIFEST="${SKIP_MANIFEST:-0}"
REPEATS="${REPEATS:-5}"

echo "[run_full] kernels: $KERNELS"
echo "[run_full] RUN_ID=$RUN_ID SEED=$SEED SKIP_MANIFEST=$SKIP_MANIFEST"

for k in $KERNELS; do
  if [ "$SKIP_MANIFEST" != "1" ]; then
    echo "===== [$(date +%H:%M:%S)] B0/Manifest: $k ====="
    $PY experiments/pipeline.py manifest -k "$k" --repeats "$REPEATS" --device 0
  fi

  for m in b1 b2 full; do
    echo "===== [$(date +%H:%M:%S)] Search: $k / $m ====="
    $PY experiments/pipeline.py search -k "$k" -m "$m" --run-id "$RUN_ID" --seed "$SEED"
    echo "===== [$(date +%H:%M:%S)] Remeasure: $k / $m ====="
    $PY experiments/pipeline.py remeasure -k "$k" -m "$m" --run-id "$RUN_ID" \
        --repeats "$REPEATS" --device 0
  done
done

echo "===== [$(date +%H:%M:%S)] ALL DONE ====="
$PY experiments/summarize.py
