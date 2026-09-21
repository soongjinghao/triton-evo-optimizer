#!/bin/bash
# 3.2 种子相似度阈值敏感性
#
#   bash experiments/run_3_2.sh                 # 默认跑 6 种设置
#   TAUS="0 0.85 0.90" bash experiments/run_3_2.sh   # 只跑部分设置
#
# 默认算子集为对 tau 敏感的 12 个算子（见 tau_survey.json）。
# 已完成的自动跳过，可反复执行断点续跑。
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
TAUS="${TAUS:-0 0.70 0.80 0.85 0.90 0.95}"

# 先做零成本普查，生成敏感算子子集
$PY experiments/ablation_tau.py survey

if [ $# -gt 0 ]; then
  KERNELS="$*"
else
  # 只取对 tau 真正敏感的 Kernel（相似度落在 [0.70, 0.95)）
  KERNELS=$($PY -c "
import json, sys
sys.path.insert(0, '/workspace/Agent')
from experiments import common
d = json.loads((common.RESULTS_DIR / 'tau_survey.json').read_text(encoding='utf-8'))
print(' '.join(d['sensitive_kernels']))
")
fi
echo "[3.2] kernels=$KERNELS"

# GM(S) 依赖 manifest 中的 T_base，先确保 B0 重测已完成
for k in $KERNELS; do
  if [ ! -f "experiments/manifest/$k.json" ]; then
    echo "===== [$(date +%H:%M:%S)] B0 重测: $k ====="
    $PY experiments/pipeline.py manifest -k "$k" --repeats "$REPEATS" --device 0
  fi
done

for tau in $TAUS; do
  echo "===== [$(date +%H:%M:%S)] tau=$tau ====="
  $PY experiments/ablation_tau.py run --tau "$tau" -k $KERNELS \
      --run-id "$RUN_ID" --seed "$SEED" --repeats "$REPEATS"
done

echo "===== [$(date +%H:%M:%S)] 汇总表5 ====="
$PY experiments/ablation_tau.py table
