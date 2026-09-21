#!/bin/bash
# 3.3 Profiling 证据与知识库检索策略消融
#   配置：a6(关Profiling) / b3(无RAG) / b4(普通语义Top-k) / full(混合重排+Guard)
#
#   bash experiments/run_3_3.sh                      # 默认 5 个试点算子
#   bash experiments/run_3_3.sh eye_kernel mean_kernel
set -u
cd /workspace/Agent

source set_env/set_api_huoshan.sh
export ENGINE_FLASH=deepseek-v4-flash-ga-260731
export ENGINE_PRO=deepseek-v4-pro-ga-260813

PY=/usr/local/python3.11.15/bin/python3.11
export PYTHONUNBUFFERED=1   # 确保日志实时落盘，便于 tail -f 观察进度
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-5}"

DEFAULT_KERNELS="eye_kernel matmul_kernel_simplified _act_quant_kernel _rms_norm_kernel mean_kernel"
if [ $# -gt 0 ]; then KERNELS="$*"; else KERNELS="$DEFAULT_KERNELS"; fi
echo "[3.3] RUN_ID=$RUN_ID SEED=$SEED kernels=$KERNELS"

# 3.3 的 GM(S) 依赖 manifest 中的 T_base，先确保 B0 重测已完成
for k in $KERNELS; do
  if [ ! -f "experiments/manifest/$k.json" ]; then
    echo "===== [$(date +%H:%M:%S)] B0 重测: $k ====="
    $PY experiments/pipeline.py manifest -k "$k" --repeats "$REPEATS" --device 0
  fi
done

echo "===== 可用配置 ====="
$PY experiments/ablation_components.py list

echo "===== [$(date +%H:%M:%S)] 运行 3.3 四配置 ====="
$PY experiments/ablation_components.py run --group 3.3 -k $KERNELS \
    --run-id "$RUN_ID" --seed "$SEED" --repeats "$REPEATS"

echo "===== [$(date +%H:%M:%S)] 汇总表6 ====="
$PY experiments/analyze_logs.py 3.3
