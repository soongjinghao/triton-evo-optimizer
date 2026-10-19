#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Agent

REFRESH_PID=290247
REFRESH_LOG=experiments/results/paper_formal_logs_manifest_refresh_20261019.log
STAGING_DIR=experiments/manifest_refresh_20261019
BACKUP_DIR=experiments/manifest_backup_pre_refresh_20261019
STATUS_FILE=experiments/results/paper_formal_logs/stage2_stage3_status_20261019.txt
PIPELINE_LOG=experiments/results/paper_formal_logs/paper_formal_pipeline_20261019.log
KERNELS=(eye_kernel compute_identity_kernel moe_align_block_size_stage
         fused_gdn_gating_kernel reshape_and_cache_kernel_flash)
SEEDS=(11 12 13)
CONFIGS=(paper_b1 paper_b2 paper_full)
EXPECTED_COMMIT=68ab813757679dd1f3c9ca3c7122c1b78a5dd0df
EXPECTED_RAG=3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397

mkdir -p "$(dirname "$STATUS_FILE")"
printf 'WAITING_FOR_STAGE2 %s\n' "$(date -u +%FT%TZ)" > "$STATUS_FILE"

if [[ ! -r "/proc/$REFRESH_PID/stat" || ! -r "/proc/$REFRESH_PID/cmdline" ]]; then
  printf 'ERROR stage2 process %s is not running\n' "$REFRESH_PID" | tee "$STATUS_FILE"
  exit 1
fi
refresh_cmdline=$(tr '\0' ' ' < "/proc/$REFRESH_PID/cmdline")
if [[ "$refresh_cmdline" != *"/usr/local/python3.11.15/bin/python3.11 -u -"* ]]; then
  printf 'ERROR unexpected stage2 process: %s\n' "$refresh_cmdline" | tee "$STATUS_FILE"
  exit 1
fi
refresh_start=$(awk '{print $22}' "/proc/$REFRESH_PID/stat")

while :; do
  if [[ ! -r "/proc/$REFRESH_PID/stat" ]]; then
    break
  fi
  current_start=$(awk '{print $22}' "/proc/$REFRESH_PID/stat")
  if [[ "$current_start" != "$refresh_start" ]]; then
    break
  fi
  process_state=$(ps -p "$REFRESH_PID" -o stat= 2>/dev/null || true)
  if [[ "$process_state" == Z* ]]; then
    break
  fi
  sleep 30
done

if ! grep -q '^\[refresh\] completed=50/50 failures=0$' "$REFRESH_LOG"; then
  printf 'ERROR stage2 did not complete successfully; see %s\n' "$REFRESH_LOG" | tee "$STATUS_FILE"
  exit 1
fi

python - <<'PY'
import json
import math
from pathlib import Path
from experiments import common

staging = Path("experiments/manifest_refresh_20261019")
kernels = common.list_kernels()
files = sorted(staging.glob("*.json"))
if len(kernels) != 50 or len(files) != 50:
    raise SystemExit(f"Expected exactly 50 kernels/manifests; got {len(kernels)}/{len(files)}")
missing = []
for kernel in kernels:
    path = staging / f"{kernel}.json"
    if not path.is_file():
        missing.append(f"{kernel}: missing manifest")
        continue
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        missing.append(f"{kernel}: invalid JSON ({exc})")
        continue
    baseline = rec.get("t_base_us")
    if (rec.get("kernel") != kernel or rec.get("repeats") != 5
            or rec.get("device_id") != 0
            or not isinstance(baseline, (int, float))
            or not math.isfinite(baseline) or baseline <= 0):
        missing.append(f"{kernel}: invalid reference metadata")
        continue
    seed_paths = common.seed_code_paths(kernel)
    if not seed_paths:
        missing.append(f"{kernel}: no seed sources")
    for seed_path in seed_paths:
        seed_rec = (rec.get("seeds") or {}).get(seed_path.name) or {}
        latency = seed_rec.get("latency_us")
        if (seed_rec.get("success") is not True
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency <= 0):
            missing.append(f"{kernel}: invalid seed measurement {seed_path.name}")
if missing:
    raise SystemExit("\n".join(missing))
print(f"Stage 2 audit passed: {len(kernels)} refreshed manifests, 5 repeats, device 0")
PY

if [[ -e "$BACKUP_DIR" ]]; then
  printf 'ERROR backup directory already exists: %s\n' "$BACKUP_DIR" | tee "$STATUS_FILE"
  exit 1
fi
mkdir "$BACKUP_DIR"
while IFS= read -r -d '' manifest; do
  cp -p "$manifest" "$BACKUP_DIR/"
done < <(find experiments/manifest -maxdepth 1 -type f -name '*.json' -print0)
if [[ $(find "$BACKUP_DIR" -maxdepth 1 -type f -name '*.json' | wc -l) -ne 50 ]]; then
  printf 'ERROR expected 50 original manifests in backup\n' | tee "$STATUS_FILE"
  exit 1
fi

for manifest in "$STAGING_DIR"/*.json; do
  kernel=$(basename "$manifest")
  cp "$manifest" "experiments/manifest/.${kernel}.refresh.tmp"
  mv "experiments/manifest/.${kernel}.refresh.tmp" "experiments/manifest/$kernel"
done
printf 'STAGE2_DONE manifests=50 backup=%s %s\n' "$BACKUP_DIR" "$(date -u +%FT%TZ)" | tee "$STATUS_FILE"

source set_env/set_api_local.sh >/dev/null 2>&1
export EA_NUM_DEVICES=1
export EA_MAX_WORKERS=2
set -o pipefail

if [[ $(git rev-parse HEAD) != "$EXPECTED_COMMIT" ]]; then
  printf 'ERROR HEAD changed from frozen commit\n' | tee "$STATUS_FILE"
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  printf 'STAGE3_RUNNING config=%s %s\n' "$config" "$(date -u +%FT%TZ)" | tee "$STATUS_FILE"
  python experiments/filter_online_exp.py \
    --config "$config" \
    --kernels "${KERNELS[@]}" \
    --seeds "${SEEDS[@]}" \
    2>&1 | tee "experiments/results/paper_formal_logs/${config}_pilot_20261019.log"
done

python - <<'PY'
import hashlib
import json
import math
from pathlib import Path

root = Path("experiments/results/filter_online")
kernels = ("eye_kernel", "compute_identity_kernel", "moe_align_block_size_stage",
           "fused_gdn_gating_kernel", "reshape_and_cache_kernel_flash")
configs = ("paper_b1", "paper_b2", "paper_full")
seeds = (11, 12, 13)
expected_commit = "68ab813757679dd1f3c9ca3c7122c1b78a5dd0df"
expected_rag = "3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397"
errors = []
for config in configs:
    for kernel in kernels:
        manifest_path = Path("experiments/manifest") / f"{kernel}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        for seed in seeds:
            result_path = root / config / f"{kernel}__r{seed}.json"
            if not result_path.is_file():
                errors.append(f"missing {result_path}")
                continue
            rec = json.loads(result_path.read_text(encoding="utf-8"))
            audit = rec.get("audit") or {}
            prov = rec.get("provenance") or {}
            checks = {
                "success": rec.get("success") is True,
                "acceptance": (rec.get("acceptance") or {}).get("passed") is True,
                "budget": audit.get("budget_ok") is True
                    and isinstance(audit.get("npu_calls"), int)
                    and audit["npu_calls"] <= audit.get("eval_budget", 0),
                "remeasure": audit.get("remeasure_repeats") == 5,
                "model_id": bool(rec.get("llm_model_ids")),
                "rag": rec.get("rag_index_sha256") == expected_rag,
                "commit": prov.get("git_commit") == expected_commit,
                "manifest_hash": rec.get("reference_manifest_sha256") == manifest_hash,
                "baseline": rec.get("reference_baseline_us") == manifest.get("t_base_us")
                    and rec.get("reference_baseline_source") == "manifest",
                "speedup": isinstance(rec.get("S"), (int, float))
                    and isinstance(rec.get("t_best_us"), (int, float))
                    and math.isclose(rec["S"], rec["reference_baseline_us"] / rec["t_best_us"],
                                     rel_tol=1e-12),
            }
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                errors.append(f"{config}/{kernel}/seed{seed}: {','.join(failed)}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Stage 3 audit passed: {len(configs) * len(kernels) * len(seeds)} records")
PY

printf 'DONE stage2=50/50 stage3=45/45 %s\n' "$(date -u +%FT%TZ)" | tee "$STATUS_FILE"
