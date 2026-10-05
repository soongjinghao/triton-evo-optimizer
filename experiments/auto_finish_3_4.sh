#!/bin/bash
# 等待当前 3.4 进程结束后，自动补跑缺失项并生成表7/8/9。
#
# 背景：隧道抖动会在部分算子上留下"预算没跑满"的残次结果（n_eval < 13）。
# 这些残次已被清理，但当前进程已经跑过对应算子、不会回头重跑，
# 因此本脚本在当前进程退出后重启一轮，利用脚本自身的 skip 逻辑
# 只补跑缺失项，然后汇总并生成 Word 表。
#
# 用法：nohup bash experiments/auto_finish_3_4.sh > /tmp/auto_finish.log 2>&1 &
set -u
cd /workspace/Agent

PY=/usr/local/python3.11.15/bin/python3.11

echo "[$(date +%H:%M:%S)] 等待当前 3.4 进程结束..."
while pgrep -f "experiments/ablation_components.py" > /dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] 当前进程已结束，等待 30s 确保文件落盘"
sleep 30

echo "[$(date +%H:%M:%S)] 统计缺失项"
$PY - <<'PYEOF'
import glob, json
cfgs = ['sel_roulette','sel_tournament','cross_unconstrained','cross_protected',
        'mut_adaptive','mut_uniform','mut_aggressive']
total = 0
for c in cfgs:
    fs = [f for f in glob.glob(f'experiments/results/{c}/*__r0.json') if 'remeasure' not in f]
    good = sum(1 for f in fs if json.load(open(f)).get('n_eval', 0) >= 13)
    total += good
    print(f"  {c:<22} {good}/15")
print(f"  健康合计 {total}/105")
PYEOF

echo "[$(date +%H:%M:%S)] 开始补跑（已有结果会被自动 skip）"
bash experiments/restart_3_3_3_4.sh
rc=$?
echo "[$(date +%H:%M:%S)] 补跑结束 exit=$rc"

echo "[$(date +%H:%M:%S)] 生成表7 / 表8 / 表9"
$PY experiments/gen_table789_docx.py

echo "[$(date +%H:%M:%S)] 全部完成"
