#!/usr/bin/env bash
# ============================================================
# v8.2 冻结后的 V1/V2 小规模正式回归
#
#   3 算子 × 3 独立随机种子 × 2 个版本 = 18 次搜索，每次预算 13 次评测
#   （合计 234 次评测名额）
#
# 只有 1 张 NPU，两个版本**串行**执行。
# 用法：
#   nohup bash experiments/run_regression_v1_v2.sh > experiments/results/regression_v1_v2.log 2>&1 &
#   tail -f experiments/results/regression_v1_v2.log
# ============================================================
set -u
cd /workspace/Agent
source set_env/set_api_local.sh
export EA_NUM_DEVICES=1

PY=python
KERNELS="eye_kernel compute_identity_kernel moe_align_block_size_stage"
SEEDS="0 1 2"
R=experiments/results

gate() {
  local desc="$1"; shift
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[GATE-FAILED] $desc （退出码 $rc）—— 回归中止"
    exit $rc
  fi
}

echo "########## V1/V2 回归开始 $(date '+%F %T') ##########"
echo "算子: $KERNELS"
echo "种子: $SEEDS   预算: 每次 13 次评测"

for CFG in full_v1 full_v2; do
  echo
  echo "=============== 版本 $CFG  $(date '+%F %T') ==============="
  gate "$CFG 三次独立搜索" $PY experiments/filter_online_exp.py \
      --config $CFG --kernels $KERNELS --seeds $SEEDS \
      > $R/collect_${CFG}.log 2>&1
  echo "✅ $CFG 完成 $(date '+%F %T')"
done

echo
echo "=============== 对照分析 $(date '+%F %T') ==============="
for S in $SEEDS; do
  gate "seed $S 对照" $PY experiments/compare_v1_v2.py \
      --v1-config full_v1 --v1-run-id $((900 + S)) --v1-seed $S \
      --v2-config full_v2 --v2-run-id $((900 + S)) --v2-seed $S \
      --pool-source filter \
      --v1-results $R/filter_online/full_v1 \
      --v2-results $R/filter_online/full_v2 \
      --kernels $KERNELS \
      --json-out $R/compare_v1_v2_seed${S}.json \
      > $R/compare_v1_v2_seed${S}.txt 2>&1
done

echo
echo "=============== 汇总 $(date '+%F %T') ==============="
$PY - <<'PYEOF'
import json, glob, collections
rows = []
for p in sorted(glob.glob("experiments/results/filter_online/full_v*/*__r*.json")):
    if p.endswith(".old"):
        continue
    d = json.loads(open(p, encoding="utf-8").read())
    a = d.get("audit", {}) or {}
    rows.append((p.split("full_v")[1].split("/")[0], d.get("kernel"), d.get("seed"),
                 d.get("success"), d.get("S"), d.get("t_best_us"),
                 a.get("npu_calls"), a.get("eval_budget"), a.get("budget_ok"),
                 (d.get("acceptance") or {}).get("passed")))
print(f"{'ver':>6s} {'kernel':34s} {'seed':>4s} {'ok':>5s} {'S':>9s} "
      f"{'npu':>5s} {'budget_ok':>9s} {'accept':>6s}")
for v, k, s, ok, S, t, n, b, bk, ac in sorted(rows):
    f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
    print(f"{('v'+v):>6s} {str(k)[:34]:34s} {str(s):>4s} {str(ok):>5s} "
          f"{f(S):>9s} {str(n):>5s} {str(bk):>9s} {str(ac):>6s}")
bad = [r for r in rows if r[3] is not True or r[8] is not True or r[9] is not True]
print(f"\n合计 {len(rows)} 次搜索，未通过硬性验收 {len(bad)} 次")
for r in bad:
    print("  ❌", r[:3], "success=", r[3], "budget_ok=", r[8], "accept=", r[9])
PYEOF

echo
echo "########## 回归结束 $(date '+%F %T') ##########"
