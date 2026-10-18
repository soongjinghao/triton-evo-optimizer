#!/usr/bin/env bash
# ============================================================
# v8.2 流水线：5 算子收尾 → 重复诊断 → 同码复测 → V1/V2 同预算对照
#
# 只有 1 张 NPU，所有需要真机的步骤必须**串行**。
# 用法：
#   nohup bash experiments/run_v82_pipeline.sh > experiments/results/v82_pipeline.log 2>&1 &
#   # 进度：
#   tail -f experiments/results/v82_pipeline.log
# ============================================================
set -u
# 【修复】评审：关键步骤不能再"失败后打印错误继续执行"。
# 这里用 gate 包裹：任何关键步骤失败即整条流水线以非零状态退出。
gate() {   # gate <说明> <命令...>
  local desc="$1"; shift
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[GATE-FAILED] $desc （退出码 $rc）—— 流水线中止"
    exit $rc
  fi
}
cd /workspace/Agent
source set_env/set_api_local.sh
export EA_NUM_DEVICES=1

# ⚠️ 本脚本是**一次性诊断流水线**（v8.2 的采集收尾用），不是正式实验入口。
# 正式实验请直接用：
#   python experiments/filter_online_exp.py --config full_v1|full_v2|full_v2_p \
#       --kernels ... --seeds 0 1 2
WAIT_PID=${WAIT_PID:-}    # 可选：等待某个采集进程结束后再继续（不再硬编码 PID）
PY=python
V1_KERNELS="eye_kernel compute_identity_kernel moe_align_block_size_stage"
DEV15=$(python -c "import json;print(' '.join(json.load(open('experiments/splits/dev15.json'))['test_kernels']))")

step() { echo; echo "########## [$1] $2  $(date '+%F %T') ##########"; }

# ---------------------------------------------------------------- 0) 等采集结束
step 0 "等待采集进程结束"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "采集进程 $WAIT_PID 已结束"
else
  echo "未指定 WAIT_PID，直接进入下一步"
fi

# ---------------------------------------------------------------- 1) 合并 5 算子样本
step 1 "合并 run_id=905 的新样本"
gate "合并 5 算子样本" $PY experiments/merge_new_samples.py --run-ids 905 \
    --out experiments/dataset/samples_new_5op.jsonl

# ---------------------------------------------------------------- 2) 验收
step 2 "全池验收（数据链路口径）"
# 【修复】评审指出这一步原本只打印退出码、失败仍会继续，现改为 gate。
gate "5 算子全池验收" $PY experiments/verify_new_samples.py \
    --samples experiments/dataset/samples_new_5op.jsonl \
    --expect-pool 20 --expected-full-pools 10 --strict \
    --json-out experiments/results/verify_new_samples_5op.json \
    > experiments/results/verify_new_samples_5op.txt 2>&1

step 2b "划分隔离检查（dev15 留出名单）"
$PY experiments/verify_new_samples.py \
    --samples experiments/dataset/samples_new_5op.jsonl \
    --expect-pool 20 --exclude-kernels $DEV15 \
    --json-out experiments/results/verify_new_samples_5op_leak.json \
    > experiments/results/verify_new_samples_5op_leak.txt 2>&1
_leak_rc=$?
echo "退出码 $_leak_rc（非 0 是**预期**的：算子落在 dev15，说明这批只能作诊断基线）"
[ $_leak_rc -ne 0 ] || echo "[GATE-WARN] 隔离检查竟然通过了，请核对留出名单"

# ---------------------------------------------------------------- 3) 重复与计时分析
step 3 "候选池重复率与计时稳定性（5 算子）"
gate "5 算子重复分析" $PY experiments/analyze_pool_duplicates.py \
    --samples experiments/dataset/samples_new_5op.jsonl \
    --json-out experiments/results/pool_duplicates_5op.json \
    > experiments/results/pool_duplicates_5op.txt 2>&1

# ---------------------------------------------------------------- 4) 正式 P@4
step 4 "5 算子完整池上的正式 P@4"
gate "5 算子正式 P@4" $PY experiments/eval_real_pools.py --config fulleval \
    --kernels eye_kernel compute_identity_kernel moe_align_block_size_stage \
              fused_gdn_gating_kernel reshape_and_cache_kernel_flash \
    --run-ids 905 \
    --old experiments/dataset/candidate_filter_model__dev15.joblib \
    --new experiments/dataset/candidate_filter_model__v7.joblib \
    --extra gen12=experiments/dataset/candidate_filter_model__v7gen12.joblib \
    --json-out experiments/results/real_pools_new_5op.json \
    > experiments/results/real_pools_new_5op.txt 2>&1

# ---------------------------------------------------------------- 5) 同码复测（NPU，少量）
step 5 "同一源码复测诊断（固定输入与启动配置）"
gate "同码复测诊断" $PY experiments/remeasure_same_code.py \
    --samples experiments/dataset/samples_new_5op.jsonl \
    --top 4 --repeats 10 \
    --json-out experiments/results/remeasure_same_code.json \
    > experiments/results/remeasure_same_code.txt 2>&1

# ---------------------------------------------------------------- 6) V2 同预算采集
step 6 "V2 同预算采集（3 算子 × 45 次评测，run_id=906）"
gate "V2 同预算采集" $PY experiments/filter_online_exp.py --config fulleval_v2 \
    --kernels $V1_KERNELS --seeds 6 \
    --model-path experiments/dataset/candidate_filter_model__dev15.joblib \
    > experiments/results/collect_v2.log 2>&1

# ---------------------------------------------------------------- 7) V2 分析
step 7 "V2 样本合并与重复分析"
gate "V2 样本合并" $PY experiments/merge_new_samples.py --run-ids 906 \
    --out experiments/dataset/samples_new_v2.jsonl
gate "V2 重复分析" $PY experiments/analyze_pool_duplicates.py \
    --samples experiments/dataset/samples_new_v2.jsonl \
    --json-out experiments/results/pool_duplicates_v2.json \
    > experiments/results/pool_duplicates_v2.txt 2>&1

step 8 "V1 / V2 同预算对照（分母 = 生成候选总数，显式 run_id）"
gate "V1/V2 对照" $PY experiments/compare_v1_v2.py \
    --v1-config fulleval --v1-run-id 905 --v1-seed 5 \
    --v2-config fulleval_v2 --v2-run-id 906 --v2-seed 6 \
    --kernels $V1_KERNELS \
    --json-out experiments/results/compare_v1_v2.json \
    > experiments/results/compare_v1_v2.txt 2>&1

step 9 "全部完成"
echo "结果目录：experiments/results/"
