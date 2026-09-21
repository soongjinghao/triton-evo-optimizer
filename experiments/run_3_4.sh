#!/bin/bash
# 3.4 进化策略消融（3.4.1 选择 / 3.4.2 交叉 / 3.4.3 变异，共 7 个配置）
#
#   bash experiments/run_3_4.sh                 # 三组全跑
#   GROUP=3.4.1 bash experiments/run_3_4.sh     # 只跑某一组
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
GROUP="${GROUP:-all}"

DEFAULT_KERNELS="eye_kernel matmul_kernel_simplified _act_quant_kernel _rms_norm_kernel mean_kernel"
if [ $# -gt 0 ]; then KERNELS="$*"; else KERNELS="$DEFAULT_KERNELS"; fi
echo "[3.4] RUN_ID=$RUN_ID SEED=$SEED kernels=$KERNELS"

# 3.4 的 GM(S) 依赖 manifest 中的 T_base，先确保 B0 重测已完成
for k in $KERNELS; do
  if [ ! -f "experiments/manifest/$k.json" ]; then
    echo "===== [$(date +%H:%M:%S)] B0 重测: $k ====="
    $PY experiments/pipeline.py manifest -k "$k" --repeats "$REPEATS" --device 0
  fi
done

run_group () {
  local g="$1"
  echo "===== [$(date +%H:%M:%S)] 运行 $g ====="
  $PY experiments/ablation_components.py run --group "$g" -k $KERNELS \
      --run-id "$RUN_ID" --seed "$SEED" --repeats "$REPEATS"
}

case "$GROUP" in
  all)
    run_group 3.4.1
    run_group 3.4.2
    run_group 3.4.3
    echo "===== 汇总表7 / 表8 / 表9 ====="
    $PY experiments/analyze_logs.py 3.4.1
    $PY experiments/analyze_logs.py 3.4.2
    $PY experiments/analyze_logs.py 3.4.3
    ;;
  3.4.1)
    run_group 3.4.1; $PY experiments/analyze_logs.py 3.4.1 ;;
  3.4.2)
    run_group 3.4.2; $PY experiments/analyze_logs.py 3.4.2 ;;
  3.4.3)
    run_group 3.4.3; $PY experiments/analyze_logs.py 3.4.3 ;;
  *)
    echo "未知 GROUP=$GROUP（可选 all / 3.4.1 / 3.4.2 / 3.4.3）"; exit 1 ;;
esac
