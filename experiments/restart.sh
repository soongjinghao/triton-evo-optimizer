#!/bin/bash
# ============================================================
# 一键重启实验：停止旧任务 -> 清理断连废数据 -> 重启续跑
#
# 用法（隧道恢复后直接跑这条即可）：
#   bash experiments/restart.sh
#
# 说明：
#   - 已完成的算子会自动跳过，不会重复跑
#   - 断连产生的假成功会被清掉并自动重跑
#   - 自带 preflight 自检：LLM 不通时会拒绝启动并提示
# ============================================================
set -u
cd /workspace/Agent

PY=/usr/local/python3.11.15/bin/python3.11

echo "=== 1/3 停止旧任务 ==="
pkill -f "run_all.sh" 2>/dev/null
pkill -f "run_full.sh" 2>/dev/null
pkill -f "run_tau_ablation.sh" 2>/dev/null
pkill -f "pipeline.py" 2>/dev/null
sleep 2
echo "  完成"

echo
echo "=== 2/3 清理断连废数据 ==="
$PY experiments/clean_invalid.py

echo
echo "=== 3/3 重启（自动续跑）==="
LOG="/tmp/all_$(date +%m%d_%H%M).log"
nohup bash experiments/run_all.sh > "$LOG" 2>&1 &
sleep 5

if pgrep -f "run_all.sh" > /dev/null; then
  echo "  ✅ 已启动，日志: $LOG"
  echo
  echo "查看进度:"
  echo "  grep 'Search:' $LOG | tail -1"
  echo "  cd /workspace/Agent && for m in b1 b2 full; do echo \"\$m 有效: \$(grep -l '\\\"success\\\": true' experiments/results/\$m/*.remeasure.json 2>/dev/null | wc -l)/50\"; done"
else
  echo "  ❌ 未启动（多半是 LLM 仍不可用，preflight 拒绝了启动）"
  echo "     检查日志: $LOG"
fi
