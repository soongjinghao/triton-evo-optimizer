#!/bin/bash
# ============================================================
# 自动衔接：等待 3.3 结束后自动启动 3.4
#
# 用法（后台运行）：
#   nohup bash experiments/chain_3_3_then_3_4.sh > /tmp/chain.log 2>&1 &
#
# 说明：3.4 使用与 3.3 相同的 15 个算子，便于横向对比。
# ============================================================
set -u
cd /workspace/Agent

echo "[chain $(date '+%m-%d %H:%M:%S')] 等待 3.3 结束..."

while pgrep -f "ablation_components.py run --group 3.3" > /dev/null; do
  sleep 60
done

echo "[chain $(date '+%m-%d %H:%M:%S')] 3.3 已结束，开始运行 3.4"
bash experiments/run_3_4.sh
echo "[chain $(date '+%m-%d %H:%M:%S')] 3.4 结束"
