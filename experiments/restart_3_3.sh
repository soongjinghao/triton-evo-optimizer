#!/bin/bash
# ============================================================
# 重启 3.3 Profiling 证据与知识库检索策略消融
#
# 用法（隧道恢复后）：
#   bash experiments/restart_3_3.sh
#
# 说明：
#   - 已完成的算子/配置会自动跳过，不会重复跑
#   - 断连产生的假成功会被清掉并自动重跑
# ============================================================
set -u
cd /workspace/Agent

PY=/usr/local/python3.11.15/bin/python3.11

echo "=== 1/3 停止旧任务 ==="
pkill -f "run_3_3.sh" 2>/dev/null
pkill -f "ablation_components" 2>/dev/null
pkill -f "pipeline.py" 2>/dev/null
sleep 2
echo "  完成"

echo
echo "=== 2/3 清理断连废数据 ==="
$PY experiments/clean_invalid.py

echo
echo "=== 3/3 重启 3.3（自动续跑）==="
# 与首次启动相同的 15 个算子（覆盖短/中/长三类规模）
KERNELS="_set_k_and_s_triton_kernel eye_kernel l2norm_fwd_kernel2 assign_extend_cache_locs _pack_seq_kernel reshape_and_cache_kernel_flash _dequantize_k_cache_fast_kernel _selective_scan_update_kernel _fwd_kernel_ep_gather moe_align_block_size_stage recompute_w_u_fwd_kernel chunk_local_cumsum_scalar_kernel _act_quant_kernel fused_gdn_gating_kernel compute_identity_kernel"

LOG="/tmp/ab33_$(date +%m%d_%H%M).log"
nohup bash experiments/run_3_3.sh $KERNELS > "$LOG" 2>&1 &
sleep 6

echo
if pgrep -f "ablation_components" > /dev/null; then
  echo "  ✅ 已启动，日志: $LOG"
  echo
  echo "查看进度:"
  echo "  cd /workspace/Agent && for c in a6 b3 b4; do echo \"\$c: \$(ls experiments/results/\$c/*.remeasure.json 2>/dev/null | wc -l)/15\"; done"
  echo "  grep -E '^===== \\[' $LOG | tail -1"
else
  echo "  ❌ 未启动（LLM 很可能仍不可用）"
  echo "     检查: $LOG"
  echo "     确认隧道恢复后重新运行本脚本"
fi
