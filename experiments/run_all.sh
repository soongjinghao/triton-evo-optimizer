#!/bin/bash
# ============================================================
# 串行执行：3.2 阈值敏感性 -> 3.1 主实验全量
#
# 两个脚本都会占用 NPU，必须串行，不能并行。
# 预计总耗时约 21 小时（3.2 约 3.4h + 3.1 约 17.5h）。
#
# 用法：
#   nohup bash experiments/run_all.sh > /tmp/all.log 2>&1 &
#
# 查看进度：
#   grep -E "开始|结束|===== \[" /tmp/all.log | tail -10
# ============================================================
set -u
cd /workspace/Agent

PY=/usr/local/python3.11.15/bin/python3.11

# ---------------- 启动前自检（关键防护）----------------
# 教训：LLM 不可达时搜索会静默降级——候选生成失败后回退到种子，
# 产出 fitness=1.0 的"假成功"结果，表面 50 个 Kernel 全部完成，
# 实则整轮作废且极难察觉。这里先行拦截。
source set_env/set_api_local.sh
echo "=== 启动前自检（LLM / NPU / 数据）==="
if ! $PY experiments/preflight.py; then
  echo
  echo "❌ 自检未通过，终止启动——请先恢复 LLM 服务再运行"
  exit 1
fi
echo

echo "=========================================="
echo " 步骤 1/2：3.2 种子相似度阈值敏感性"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
bash experiments/run_tau_ablation.sh
RC1=$?

echo
echo "===== [$(date '+%H:%M:%S')] 3.2 结束 (exit=$RC1) ====="
echo

echo "=========================================="
echo " 步骤 2/2：3.1 主实验全量 (50 Kernel x 3 方法)"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
bash experiments/run_full.sh
RC2=$?

echo
echo "===== [$(date '+%H:%M:%S')] 3.1 结束 (exit=$RC2) ====="
echo "===== 全部完成 $(date '+%Y-%m-%d %H:%M:%S') ====="
