#!/bin/bash
# 隧道恢复后重启：
#   步骤1  补跑 3.3 中 b4 缺失的 4 个算子
#   步骤2  重跑 3.4（选择/交叉/变异，共 7 个配置 × 15 算子）
# 脚本默认跳过已完成的结果，中断后重跑会自动续跑。
set -u
cd /workspace/Agent

# 默认本地 vLLM；ENV=cloud 时切回火山方舟
if [ "${ENV:-local}" = "cloud" ]; then
  source set_env/set_api_huoshan.sh
  export ENGINE_FLASH=deepseek-v4-flash-ga-260731
  export ENGINE_PRO=deepseek-v4-pro-ga-260813
else
  source set_env/set_api_local.sh
fi

PY=/usr/local/python3.11.15/bin/python3.11
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-5}"

# 3.3 与 3.4 共用同一批 15 个算子（覆盖短/中/长三类规模）
KERNELS15="_set_k_and_s_triton_kernel eye_kernel l2norm_fwd_kernel2 assign_extend_cache_locs \
_pack_seq_kernel reshape_and_cache_kernel_flash _dequantize_k_cache_fast_kernel \
_selective_scan_update_kernel _fwd_kernel_ep_gather moe_align_block_size_stage \
recompute_w_u_fwd_kernel chunk_local_cumsum_scalar_kernel _act_quant_kernel \
fused_gdn_gating_kernel compute_identity_kernel"

# b4 相对 b3/a6 缺失的 4 个
MISS4="_set_k_and_s_triton_kernel assign_extend_cache_locs eye_kernel l2norm_fwd_kernel2"

echo "===== [$(date +%H:%M:%S)] 步骤1/2 补跑 b4 缺失的 4 个算子 ====="
$PY experiments/ablation_components.py run --group 3.3 --config b4 \
    -k $MISS4 --run-id "$RUN_ID" --seed "$SEED" --repeats "$REPEATS"
rc1=$?
echo "[步骤1] exit=$rc1"

echo "===== [$(date +%H:%M:%S)] 汇总表6（有效候选率）====="
$PY experiments/analyze_logs.py 3.3

echo "===== [$(date +%H:%M:%S)] 步骤2/2 重跑 3.4 全部 7 配置 × 15 算子 ====="
$PY experiments/ablation_components.py run --group 3.4 \
    -k $KERNELS15 --run-id "$RUN_ID" --seed "$SEED" --repeats "$REPEATS"
rc2=$?
echo "[步骤2] exit=$rc2"

echo "===== [$(date +%H:%M:%S)] 汇总表7 / 表8 / 表9 ====="
$PY experiments/analyze_logs.py 3.4.1
$PY experiments/analyze_logs.py 3.4.2
$PY experiments/analyze_logs.py 3.4.3

echo "===== [$(date +%H:%M:%S)] 全部结束 (步骤1 exit=$rc1, 步骤2 exit=$rc2) ====="
