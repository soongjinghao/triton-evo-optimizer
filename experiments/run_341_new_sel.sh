#!/usr/bin/env bash
# a7（3.3 双消融）跑完后，自动补跑 3.4.1 新增的两个父代选择策略：
#   sel_uniform —— 均匀随机，无选择压力（下界基线，验证“选择本身是否有效”）
#   sel_ucb     —— UCB 式探索-利用平衡选择（本文提出）
#
# 已完成的 sel_roulette / sel_tournament 会被 skip_done 自动跳过，
# 因此实际只跑 15 算子 × 2 配置 = 30 次，约 150 分钟。
#
# 为什么要等 a7：两者都要占用 NPU，并行会互相干扰评测结果。
#
# 用法（后台运行）：
#   nohup bash experiments/run_341_new_sel.sh > /tmp/run_341_new_sel.log 2>&1 &
set -u
cd /workspace/Agent

PY="${PY:-/usr/local/python3.11.15/bin/python3.11}"

echo "===== [$(date +%H:%M:%S)] 等待 a7 结束（避免 NPU 争用）====="
while pgrep -f "ablation_components.py run" > /dev/null 2>&1; do
    sleep 60
done
echo "===== [$(date +%H:%M:%S)] a7 已结束，开始补跑 3.4.1 新选择策略 ====="

set -a
. set_env/set_api_local.sh
set +a

$PY experiments/ablation_components.py run --group 3.4.1 \
    --run-id 0 --seed 0 --repeats 5
rc=$?
echo "[3.4.1] exit=$rc"

echo "===== [$(date +%H:%M:%S)] 汇总表7（含两个新策略）====="
$PY experiments/analyze_logs.py 3.4.1

echo "===== [$(date +%H:%M:%S)] Friedman + Nemenyi 多策略检验 ====="
$PY experiments/stats_341.py

echo "===== [$(date +%H:%M:%S)] 全部完成 ====="
