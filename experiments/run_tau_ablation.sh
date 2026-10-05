#!/bin/bash
# ============================================================
# 3.2 种子相似度阈值敏感性（基于 datasets2 官方原始种子）
#
# 只跑「tau 真正生效」的 Kernel = 相似度落在敏感区 且 两个种子都评测通过。
# （单种子或无可用种子的 Kernel 会被自动排除，避免白跑）
#
# 用法：
#   bash experiments/run_tau_ablation.sh              # 便宜层，只跑到第 0 代，出 D_G0/有效候选率，约 2.5 h
#   BUDGET=13 bash experiments/run_tau_ablation.sh    # 完整预算，额外产出 GM(S)，约 7 h
#   TAUS="0.85 0.90" bash experiments/run_tau_ablation.sh   # 只跑指定档位
# ============================================================
set -u
cd /workspace/Agent

source set_env/set_api_local.sh

PY=/usr/local/python3.11.15/bin/python3.11
BUDGET="${BUDGET:-5}"
TAUS="${TAUS:-0.70 0.80 0.85 0.90 0.95}"

# 有效样本 = 敏感区算子 ∩ 两种子均可用
KERNELS=$($PY - <<'PYEOF'
import json, glob

sur = json.load(open('experiments/results/tau_survey.json'))
sens = set(sur['sensitive_kernels'])
both = set()
for f in glob.glob('experiments/manifest/*.json'):
    d = json.load(open(f, encoding='utf-8'))
    if sum(1 for v in d.get('seeds', {}).values() if v.get('success')) >= 2:
        both.add(d['kernel'])
print(' '.join(sorted(sens & both)))
PYEOF
)

N=$(echo "$KERNELS" | wc -w)
echo "[3.2] 数据集   = $($PY -c "import sys;sys.path.insert(0,'.');from experiments import common;print(common.DATASETS_DIR)")"
echo "[3.2] 有效样本 = $N 个"
echo "[3.2] 档位     = $TAUS 以及 0(不筛选)"
echo "[3.2] 预算     = $BUDGET 次评测/Kernel"
echo

for t in $TAUS 0; do
  echo "===== [$(date +%H:%M:%S)] tau=$t ====="
  $PY experiments/ablation_tau.py run --tau "$t" --budget "$BUDGET" -k $KERNELS
done

echo
echo "===== [$(date +%H:%M:%S)] 输出表5 ====="
$PY experiments/ablation_tau.py table -k $KERNELS
