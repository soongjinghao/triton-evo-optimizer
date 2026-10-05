#!/usr/bin/env bash
set -uo pipefail

cd /data/Agent
source set_env/set_npu_single.sh
source set_env/set_api_local.sh

export PYTHONUNBUFFERED=1
export EA_NPU_DEVICES=0
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_DEVICE_ID=0
export EA_MAX_WORKERS="${EA_MAX_WORKERS_OVERRIDE:-4}"

RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
PY=/data/Agent/.venv-npu/bin/python

CONFIGS=(
  sel_roulette sel_tournament
  cross_unconstrained cross_protected
  mut_adaptive mut_uniform mut_aggressive
)
KERNELS=(
  _set_k_and_s_triton_kernel eye_kernel l2norm_fwd_kernel2
  assign_extend_cache_locs _pack_seq_kernel reshape_and_cache_kernel_flash
  _dequantize_k_cache_fast_kernel _selective_scan_update_kernel
  _fwd_kernel_ep_gather moe_align_block_size_stage recompute_w_u_fwd_kernel
  chunk_local_cumsum_scalar_kernel _act_quant_kernel fused_gdn_gating_kernel
  compute_identity_kernel
)

health_count() {
  "$PY" - <<'PY'
import json
from pathlib import Path
configs = ['sel_roulette', 'sel_tournament', 'cross_unconstrained',
           'cross_protected', 'mut_adaptive', 'mut_uniform', 'mut_aggressive']
kernels = '''_set_k_and_s_triton_kernel eye_kernel l2norm_fwd_kernel2
assign_extend_cache_locs _pack_seq_kernel reshape_and_cache_kernel_flash
_dequantize_k_cache_fast_kernel _selective_scan_update_kernel
_fwd_kernel_ep_gather moe_align_block_size_stage recompute_w_u_fwd_kernel
chunk_local_cumsum_scalar_kernel _act_quant_kernel fused_gdn_gating_kernel
compute_identity_kernel'''.split()
root = Path('experiments/results')
good = 0
for config in configs:
    count = 0
    for kernel in kernels:
        base = root / config / f'{kernel}__r0'
        try:
            search = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
            remeasure = json.loads(Path(f'{base}.remeasure.json').read_text(encoding='utf-8'))
            healthy = (base.with_suffix('.py').is_file()
                       and search.get('n_eval', 0) >= 13
                       and search.get('search_ok', True) is not False
                       and remeasure.get('success') is True)
        except Exception:
            healthy = False
        count += int(healthy)
    good += count
    print(f'[progress] {config:<22} {count}/15', flush=True)
print(f'[progress] healthy_total={good}/105', flush=True)
print(good)
PY
}

archive_unhealthy() {
  local stamp="$1"
  STAMP="$stamp" "$PY" - <<'PY'
import json, os, shutil
from pathlib import Path
configs = ['sel_roulette', 'sel_tournament', 'cross_unconstrained',
           'cross_protected', 'mut_adaptive', 'mut_uniform', 'mut_aggressive']
kernels = '''_set_k_and_s_triton_kernel eye_kernel l2norm_fwd_kernel2
assign_extend_cache_locs _pack_seq_kernel reshape_and_cache_kernel_flash
_dequantize_k_cache_fast_kernel _selective_scan_update_kernel
_fwd_kernel_ep_gather moe_align_block_size_stage recompute_w_u_fwd_kernel
chunk_local_cumsum_scalar_kernel _act_quant_kernel fused_gdn_gating_kernel
compute_identity_kernel'''.split()
root = Path('experiments/results')
archive = Path('experiments/archive') / f"partial_3_4_{os.environ['STAMP']}"
moved = 0
for config in configs:
    for kernel in kernels:
        base = root / config / f'{kernel}__r0'
        try:
            search = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
            remeasure = json.loads(Path(f'{base}.remeasure.json').read_text(encoding='utf-8'))
            healthy = (base.with_suffix('.py').is_file()
                       and search.get('n_eval', 0) >= 13
                       and search.get('search_ok', True) is not False
                       and remeasure.get('success') is True)
        except Exception:
            healthy = False
        if healthy:
            continue
        targets = [Path('experiments/logs') / f'{config}__{kernel}__r0.jsonl']
        targets.extend((root / config).glob(f'{kernel}__r0*'))
        for src in targets:
            if not src.exists():
                continue
            rel_group = 'logs' if src.parent.name == 'logs' else config
            dst = archive / rel_group / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                suffix = 1
                while dst.with_name(f'{dst.name}.{suffix}').exists():
                    suffix += 1
                dst = dst.with_name(f'{dst.name}.{suffix}')
            shutil.move(str(src), str(dst))
            moved += 1
print(f'[resume] archived_partial_artifacts={moved} path={archive}', flush=True)
PY
}

echo "[$(date '+%F %T')] 3.4 单卡断点续跑开始"
echo "[env] python=$PY devices=$EA_NPU_DEVICES llm_workers=$EA_MAX_WORKERS repeats=$REPEATS"

for round in $(seq 1 "$MAX_ROUNDS"); do
  stamp="$(date '+%Y%m%d_%H%M%S')_round${round}"
  echo "[$(date '+%F %T')] round=$round 清理并归档残次任务"
  archive_unhealthy "$stamp"

  echo "[$(date '+%F %T')] round=$round 开始补跑，完整结果将自动跳过"
  "$PY" experiments/ablation_components.py run --group 3.4 \
    -k "${KERNELS[@]}" --run-id "$RUN_ID" --seed "$SEED" \
    --repeats "$REPEATS" --device 0

  echo "[$(date '+%F %T')] round=$round 完成，重新统计"
  count_output="$(health_count)"
  printf '%s\n' "$count_output"
  good="$(printf '%s\n' "$count_output" | tail -n 1)"
  if [ "$good" = "105" ]; then
    break
  fi
  echo "[$(date '+%F %T')] 仍有 $((105-good)) 项，30 秒后进入下一轮"
  sleep 30
done

"$PY" experiments/analyze_logs.py 3.4.1
"$PY" experiments/analyze_logs.py 3.4.2
"$PY" experiments/analyze_logs.py 3.4.3
"$PY" experiments/gen_table789_docx.py

final_output="$(health_count)"
printf '%s\n' "$final_output"
final_good="$(printf '%s\n' "$final_output" | tail -n 1)"
if [ "$final_good" != "105" ]; then
  echo "[$(date '+%F %T')] 3.4 尚未全部完成：$final_good/105"
  exit 1
fi

echo "[$(date '+%F %T')] 3.4 全部完成：105/105"
