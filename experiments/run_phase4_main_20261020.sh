#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Agent

LOG_DIR=experiments/results/paper_formal_logs
STATUS_FILE="$LOG_DIR/phase4_main_status_20261020.txt"
RUN_LOG="$LOG_DIR/phase4_main_20261020.log"
CONFIGS=(paper_b1 paper_b2 paper_full)
SEEDS=(11 12 13)
EXPECTED_RAG=3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397

mkdir -p "$LOG_DIR"
source set_env/set_api_local.sh >/dev/null 2>&1
export EA_NUM_DEVICES=1
export EA_MAX_WORKERS=2
mapfile -t KERNELS < <(python -c 'from experiments import common; print("\n".join(common.list_kernels()))')
EXPECTED_COMMIT=$(git rev-parse HEAD)
export EXPECTED_COMMIT

python - <<'PY' | tee "$STATUS_FILE"
import math
import subprocess
from pathlib import Path
from experiments import common
from experiments.filter_online_exp import _rag_index_sha256

expected_rag = "3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397"
kernels = common.list_kernels()
if len(kernels) != 50:
    raise SystemExit(f"Expected 50 kernels, found {len(kernels)}")
if _rag_index_sha256(required=True) != expected_rag:
    raise SystemExit("RAG index fingerprint does not match the frozen protocol")
for kernel in kernels:
    path = common.MANIFEST_DIR / f"{kernel}.json"
    record = common.read_reference_manifest(kernel)
    if record is None or record["repeats"] != 5 or record["device_id"] != 0:
        raise SystemExit(f"Invalid formal reference manifest: {path}")
    baseline = record["baseline_us"] if "baseline_us" in record else record["t_base_us"]
    if not math.isfinite(float(baseline)) or float(baseline) <= 0:
        raise SystemExit(f"Invalid reference latency: {path}")
    if not common.find_test_file(kernel).is_file():
        raise SystemExit(f"Missing correctness test for {kernel}")
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if len(kernels) != 50:
    raise SystemExit("Expected exactly 50 kernels")
print(f"Preflight PASS: kernels={len(kernels)} seeds=11,12,13 configs=3 "
      f"head={head} rag_sha256={expected_rag}")
print("Kernel order: " + " ".join(kernels))
PY

for config in "${CONFIGS[@]}"; do
  if [[ $(git rev-parse HEAD) != "$EXPECTED_COMMIT" ]]; then
    printf 'ERROR HEAD changed during phase 4; stop before config=%s\n' "$config" \
      | tee -a "$STATUS_FILE"
    exit 1
  fi
  printf 'START config=%s time=%s\n' "$config" "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
  if ! python experiments/filter_online_exp.py \
      --config "$config" \
      --kernels "${KERNELS[@]}" \
      --seeds "${SEEDS[@]}" \
      2>&1 | tee -a "$RUN_LOG" | tee -a "$LOG_DIR/${config}_50kernels_20261020.log"; then
    printf 'FAILED config=%s time=%s; rerun this script to resume verified results\n' \
      "$config" "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
    exit 1
  fi
  printf 'DONE config=%s time=%s\n' "$config" "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
done

if [[ $(git rev-parse HEAD) != "$EXPECTED_COMMIT" ]]; then
  printf 'ERROR HEAD changed during phase 4; audit cannot treat mixed commits as one run\n' \
    | tee -a "$STATUS_FILE"
  exit 1
fi

python - <<'PY' | tee -a "$STATUS_FILE"
import hashlib
import json
import math
import os
from pathlib import Path
from experiments import common

configs = ("paper_b1", "paper_b2", "paper_full")
seeds = (11, 12, 13)
kernels = common.list_kernels()
expected_rag = "3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397"
expected_commit = os.environ["EXPECTED_COMMIT"]
errors = []
for config in configs:
    for kernel in kernels:
        manifest_path = common.MANIFEST_DIR / f"{kernel}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        for seed in seeds:
            result_path = Path("experiments/results/filter_online") / config / f"{kernel}__r{seed}.json"
            if not result_path.is_file():
                errors.append(f"missing {result_path}")
                continue
            record = json.loads(result_path.read_text(encoding="utf-8"))
            audit = record.get("audit") or {}
            provenance = record.get("provenance") or {}
            checks = (
                record.get("success") is True,
                (record.get("acceptance") or {}).get("passed") is True,
                audit.get("budget_ok") is True,
                isinstance(audit.get("npu_calls"), int)
                    and audit["npu_calls"] <= audit.get("eval_budget", 0),
                audit.get("remeasure_repeats") == 5,
                bool(record.get("llm_model_ids")),
                record.get("rag_index_sha256") == expected_rag,
                record.get("reference_baseline_source") == "manifest",
                record.get("reference_manifest_sha256") == manifest_hash,
                record.get("reference_baseline_us") == manifest.get("t_base_us"),
                provenance.get("git_commit") == expected_commit,
                isinstance(record.get("S"), (int, float))
                    and isinstance(record.get("t_best_us"), (int, float))
                    and math.isclose(record["S"],
                                     record["reference_baseline_us"] / record["t_best_us"],
                                     rel_tol=1e-12),
            )
            if not all(checks):
                errors.append(f"failed audit {config}/{kernel}/seed{seed}: {checks}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"AUDIT_PASS records={len(configs) * len(kernels) * len(seeds)}")
PY

printf 'DONE phase4=450/450 audited=%s time=%s\n' "$(date -u +%FT%TZ)" \
  "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
