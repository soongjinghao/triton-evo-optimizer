#!/usr/bin/env bash
# 查看 3.4.1 实验进度（sel_uniform / sel_ucb）
# 用法：cd /workspace/Agent && bash experiments/watch_341.sh

cd /workspace/Agent || exit 1

LOG=experiments/logs/_341_run2.log
PID=$(pgrep -f "ablation_components.py run --group 3.4.1" | head -1)

echo "时间: $(date +%H:%M:%S)"

if [ -z "$PID" ]; then
    echo "进程: ❌ 未运行"
else
    echo "进程: ✅ 运行中  PID=$PID  已跑 $(ps -o etime= -p "$PID" | tr -d ' ')"
    CPU=$(ps -o %cpu= -p "$PID" | tr -d ' ')
    BAD=$(ss -tanp 2>/dev/null | grep "$PID" | grep -c CLOSE-WAIT)
    echo "健康: CPU ${CPU}%   坏连接 ${BAD} 个"

    # 日志是否在增长（最可靠的存活判据）
    # 注意：算子之间有写结果 / 启动下一个 / 等 LLM 的空档，窗口需足够长才不会误报
    A=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    sleep 25
    B=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$B" -gt "$A" ]; then
        echo "日志: ✅ 增长中 (+$((B - A)) 字节/25秒)"
    else
        # 停滞且 CPU 极低才算卡死；否则只是算子间的正常空档
        if [ "$(echo "$CPU < 5" | bc 2>/dev/null || echo 0)" = "1" ]; then
            echo "日志: ❌ 停滞且 CPU 极低 —— 可能已卡死"
        else
            echo "日志: ⏸ 25秒内无新增（CPU ${CPU}% 非空闲，可能处于算子间空档）"
        fi
    fi
fi

echo "--- 进度 ---"
for c in sel_uniform sel_ucb; do
    echo "  $c: $(ls experiments/results/$c/*.py 2>/dev/null | wc -l)/15"
done

echo "--- 当前算子 ---"
grep -E "===== \[" "$LOG" 2>/dev/null | tail -1 | sed 's/^/  /'

echo "--- 判读 ---"
echo "  正常: 日志增长 + CPU 波动 + 坏连接 0~3"
echo "  危险: 日志停滞 + CPU 连续持平在 0~3% + 坏连接持续上涨"
