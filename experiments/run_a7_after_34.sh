#!/usr/bin/env bash
# 等 3.4 跑完后，补跑 3.3 的双消融配置 a7（同时关闭 Profiling 与 RAG），
# 再汇总表6 并把结果写入第三章修订版。
#
# 用法（后台运行）：
#   nohup bash experiments/run_a7_after_34.sh > /tmp/run_a7.log 2>&1 &
set -u
cd /workspace/Agent

PY="${PY:-/usr/local/python3.11.15/bin/python3.11}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-5}"

# 3.3 与 3.4 共用同一批 15 个算子
KERNELS15="_set_k_and_s_triton_kernel eye_kernel l2norm_fwd_kernel2 assign_extend_cache_locs \
_pack_seq_kernel reshape_and_cache_kernel_flash _dequantize_k_cache_fast_kernel \
_selective_scan_update_kernel _fwd_kernel_ep_gather moe_align_block_size_stage \
recompute_w_u_fwd_kernel chunk_local_cumsum_scalar_kernel _act_quant_kernel \
fused_gdn_gating_kernel compute_identity_kernel"

echo "===== [$(date +%H:%M:%S)] 等待 3.4 进程结束 ====="
while pgrep -f "ablation_components.py run --group 3.4" > /dev/null 2>&1; do
    sleep 60
done
echo "===== [$(date +%H:%M:%S)] 3.4 已结束，开始补跑 a7（无 Profiling + 无 RAG）====="

$PY experiments/ablation_components.py run --group 3.3 --config a7 \
    -k $KERNELS15 --run-id "$RUN_ID" --seed "$SEED" --repeats "$REPEATS"
rc=$?
echo "[a7] exit=$rc"

if [ "$rc" -ne 0 ]; then
    echo "===== [$(date +%H:%M:%S)] a7 未成功完成，跳过汇总 ====="
    exit "$rc"
fi

echo "===== [$(date +%H:%M:%S)] 汇总表6（含 a7）====="
$PY experiments/analyze_logs.py 3.3

echo "===== [$(date +%H:%M:%S)] 更新第三章 3.3 分析（含 a7 叠加效应）====="
$PY experiments/update_ch3_33.py

echo "===== [$(date +%H:%M:%S)] 全部完成 ====="
