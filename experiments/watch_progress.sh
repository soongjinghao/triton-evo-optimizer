#!/bin/bash
# 3.4 进化策略消融 —— 一键进度查看
#
# 用法：
#   bash experiments/watch_progress.sh          # 看一次
#   watch -n 60 bash experiments/watch_progress.sh   # 每 60 秒自动刷新
#
set -u
cd /workspace/Agent
PY=/usr/local/python3.11.15/bin/python3.11
LOG=$(ls -t /tmp/restart33_34*.log 2>/dev/null | head -1)
AUTO_LOG=/tmp/auto_finish.log

echo "════════ 3.4 进度 $(date +%H:%M:%S) ════════"

# 1. 进程与 LLM
if pgrep -f "experiments/ablation_components.py" > /dev/null; then
  echo "进程: ✅ 运行中"
else
  echo "进程: ⏹ 未运行"
fi
if curl -s --max-time 5 http://127.0.0.1:13000/v1/models > /dev/null 2>&1; then
  echo "LLM : ✅ 在线"
else
  echo "LLM : ❌ 离线（隧道需重连）"
fi

# 2. 各配置进度
echo
echo "── 各配置（目标 15/组，共 105）──"
$PY - <<'PYEOF'
import glob, json
cfgs = ['sel_roulette','sel_tournament','cross_unconstrained','cross_protected',
        'mut_adaptive','mut_uniform','mut_aggressive']
total = 0
for c in cfgs:
    fs = [f for f in glob.glob(f'experiments/results/{c}/*__r0.json') if 'remeasure' not in f]
    good = sum(1 for f in fs if json.load(open(f)).get('n_eval', 0) >= 13)
    total += good
    bar = '█' * good + '·' * (15 - good)
    print(f"  {c:<22} {bar} {good:>2}/15")
print(f"\n  健康合计: {total}/105  ({total/105*100:.0f}%)")
remain = 105 - total
if remain > 0:
    print(f"  预计剩余: 约 {remain*5/60:.1f} 小时（按 5 分钟/算子估算）")
PYEOF

# 3. 当前任务与错误
if [ -n "$LOG" ]; then
  echo
  echo "── 当前任务 ──"
  echo "  $(grep -E '===== \[.*\] (sel|cross|mut)_' "$LOG" | tail -1 | sed 's/===== //')"
  echo
  echo "── 健康度 ──"
  echo "  Connection error: $(grep -c 'Connection error' "$LOG")"
  echo "  Traceback       : $(grep -c 'Traceback' "$LOG")"
  echo "  重试次数        : $(grep -c '⚠️ 连接失败' "$LOG")"
  echo "  日志: $LOG"
fi

# 4. 自动补跑守护
echo
if pgrep -f "auto_finish_3_4.sh" > /dev/null; then
  echo "补跑守护: ✅ 已挂起（主进程结束后自动补缺失项并生成表7/8/9）"
else
  echo "补跑守护: ⏹ 未运行"
fi
if [ -f "$AUTO_LOG" ]; then
  echo "  守护日志: $AUTO_LOG"
fi

# 5. Word 产出
echo
echo "── Word 产出 ──"
for f in doc/表4_3.1主实验结果.docx doc/表5_3.2阈值敏感性实验结果.docx doc/表6_3.3组件消融实验结果.docx doc/表7_3.4选择策略消融.docx doc/表8_3.4交叉策略消融.docx doc/表9_3.4变异策略消融.docx; do
  if [ -f "$f" ]; then
    echo "  ✅ $(basename "$f")"
  else
    echo "  ⏳ $(basename "$f")（待生成）"
  fi
done
echo "═══════════════════════════════"
