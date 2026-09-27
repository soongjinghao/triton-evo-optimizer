#!/bin/bash
# ============================================================
# 阶段 A / B0：重测全部 Kernel 的 T_base 与 T_seed
#
# 数据集：datasets2（官方原始种子，未经人工优化）
#        如需切回 datasets：DATASETS_DIR=/workspace/Agent/datasets bash 本脚本
#
# 用法：
#   bash experiments/run_manifest_all.sh          # 全部 Kernel，repeats=5
#   REPEATS=1 bash experiments/run_manifest_all.sh   # 快速筛查（约 50 分钟）
#   FORCE=1   bash experiments/run_manifest_all.sh   # 忽略已有结果，全部重测
# ============================================================
set -u
cd /workspace/Agent

source set_env/set_api_local.sh

PY=/usr/local/python3.11.15/bin/python3.11
REPEATS="${REPEATS:-5}"
DEVICE="${DEVICE:-0}"
# 默认跳过已完成的 Kernel，便于中断后续跑
SKIP="${FORCE:-0}"   # FORCE=1 时不跳过

DS=$($PY -c "import sys;sys.path.insert(0,'.');from experiments import common;print(common.DATASETS_DIR)")
KERNELS=$($PY -c "import sys;sys.path.insert(0,'.');from experiments import common;print(' '.join(common.list_kernels()))")
TOTAL=$(echo "$KERNELS" | wc -w)

echo "[manifest] 数据集 = $DS"
echo "[manifest] Kernel = $TOTAL 个   repeats=$REPEATS   device=$DEVICE"
echo "[manifest] 预计 $(( TOTAL * 2 * REPEATS )) 次 NPU 评测"
echo

n=0
for k in $KERNELS; do
  n=$((n + 1))
  mf="experiments/manifest/$k.json"
  if [ "$SKIP" = "0" ] && [ -f "$mf" ]; then
    # 仅当已完成的 repeats 与本次一致时才跳过；否则按新 repeats 重测
    prev=$($PY -c "import json;print(json.load(open('$mf')).get('repeats',0))" 2>/dev/null || echo 0)
    if [ "$prev" = "$REPEATS" ]; then
      echo "[$n/$TOTAL] SKIP 已完成 (repeats=$prev): $k"
      continue
    fi
    echo "[$n/$TOTAL] 重测 (旧 repeats=$prev -> $REPEATS): $k"
  fi
  echo "===== [$(date +%H:%M:%S)] ($n/$TOTAL) $k ====="
  $PY experiments/pipeline.py manifest -k "$k" --repeats "$REPEATS" --device "$DEVICE"
done

echo
echo "===== [$(date +%H:%M:%S)] 全部完成 ====="

# ---------------- 可用性统计 ----------------
$PY - <<'PYEOF'
import json, glob

rows = []
for f in sorted(glob.glob('experiments/manifest/*.json')):
    d = json.load(open(f, encoding='utf-8'))
    seeds = d.get('seeds', {})
    ok = [k for k, v in seeds.items() if v.get('success')]
    rows.append((d['kernel'], len(seeds), len(ok), d.get('t_base_us'), d.get('t_seed_us')))

both = [r for r in rows if r[2] >= 2]
one = [r for r in rows if r[2] == 1]
none = [r for r in rows if r[2] == 0]

print()
print("========== 可用性统计 ==========")
print(f"总 Kernel            : {len(rows)}")
print(f"  两种子均可用      : {len(both)}   <- tau 只对这些生效（3.2 样本）")
print(f"  仅主种子可用      : {len(one)}    <- 单种子，tau 不生效，仍进 3.1")
print(f"  无可用种子        : {len(none)}    <- 必须从实验中排除")
if none:
    print()
    print("排除清单（两种子均不可用）:")
    for r in none:
        print(f"    {r[0]}")
if one:
    print()
    print("单种子 Kernel（第二种子不可用）:")
    for r in one:
        print(f"    {r[0]}")

tb = [r[3] for r in rows if r[3]]
ts = [r[4] for r in rows if r[4]]
if tb:
    print()
    print(f"T_base 范围: {min(tb):.2f} ~ {max(tb):.2f} us")
    print(f"T_seed 范围: {min(ts):.2f} ~ {max(ts):.2f} us")
PYEOF
