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

# 默认本地 vLLM；ENV=cloud 时切回火山方舟
if [ "${ENV:-local}" = "cloud" ]; then
  source set_env/set_api_huoshan.sh
  export ENGINE_FLASH=deepseek-v4-flash-ga-260731
  export ENGINE_PRO=deepseek-v4-pro-ga-260813
else
  source set_env/set_api_local.sh
fi

PY=/usr/local/python3.11.15/bin/python3.11

if [ $# -gt 0 ]; then
  KERNELS="$*"
else
  # 跟随 common.DATASETS_DIR（默认 datasets2），不要硬编码 datasets。
  # 同时跳过 manifest 中没有任何可用种子的 Kernel（否则搜索阶段必然失败）。
  KERNELS=$($PY - <<'PYEOF'
import sys, json
sys.path.insert(0, '.')
from experiments import common

avail = []
for k in common.list_kernels():
    mf = common.MANIFEST_DIR / f"{k}.json"
    if mf.exists():
        d = json.load(open(mf, encoding='utf-8'))
        if not any(v.get('success') for v in d.get('seeds', {}).values()):
            continue          # 无任何可用种子，排除
        if d.get('t_base_us') is None:
            continue          # 主种子失败 -> T_base 缺失 -> S 无法计算，排除
    avail.append(k)
print(' '.join(avail))
PYEOF
)
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
    # 断点续跑：已有「搜索有效 + 复测成功」的结果则跳过，避免重复消耗 NPU 与 LLM。
    # 断连导致的假成功（search_ok=False）不会被跳过，会自动重跑。
    _sj="experiments/results/$m/${k}__r${RUN_ID}.json"
    _rj="experiments/results/$m/${k}__r${RUN_ID}.remeasure.json"
    if [ -f "$_sj" ] && [ -f "$_rj" ]; then
      _ok=$($PY -c "
import json
try:
    s = json.load(open('$_sj', encoding='utf-8'))
    r = json.load(open('$_rj', encoding='utf-8'))
    print(1 if (s.get('search_ok', True) and r.get('success')) else 0)
except Exception:
    print(0)")
      if [ "$_ok" = "1" ]; then
        echo "===== [$(date +%H:%M:%S)] SKIP: $k / $m 已有有效结果 ====="
        continue
      fi
    fi

    echo "===== [$(date +%H:%M:%S)] Search: $k / $m ====="
    $PY experiments/pipeline.py search -k "$k" -m "$m" --run-id "$RUN_ID" --seed "$SEED"
    echo "===== [$(date +%H:%M:%S)] Remeasure: $k / $m ====="
    $PY experiments/pipeline.py remeasure -k "$k" -m "$m" --run-id "$RUN_ID" \
        --repeats "$REPEATS" --device 0
  done
done

echo "===== [$(date +%H:%M:%S)] ALL DONE ====="
$PY experiments/summarize.py
