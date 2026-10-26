#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Agent

LOG_DIR=experiments/results/paper_formal_logs
RUN_DIR=experiments/results/phase4_recovery_20261020
QUARANTINE_DIR=experiments/results/phase4_quarantine_20261020
STATUS_FILE="$LOG_DIR/phase4_recovery_status_20261020.txt"
CONFIGS=(paper_b1 paper_b2 paper_full)
SEEDS=(11 12 13)
EXPECTED_RAG=3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397

mkdir -p "$LOG_DIR" "$RUN_DIR"

source set_env/set_api_local.sh >/dev/null 2>&1
export EA_NUM_DEVICES=1
export EA_MAX_WORKERS=5
mapfile -t KERNELS < <(python -c 'from experiments import common; print("\n".join(common.list_kernels()))')
EXPECTED_COMMIT=$(git rev-parse HEAD)
export EXPECTED_COMMIT
mkdir -p "$QUARANTINE_DIR"

python - <<'PY'
import json
import math
import os
import subprocess
from pathlib import Path
from experiments import common
from experiments.filter_online_exp import _rag_index_sha256

freeze = "68ab813757679dd1f3c9ca3c7122c1b78a5dd0df"
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if head != os.environ["EXPECTED_COMMIT"]:
    raise SystemExit("HEAD changed during recovery startup")
core = ("config.py", "evolutionary_algorithm.py", "genetic_operators.py",
        "executor.py", "main.py", "RAG/knowledge_node.py",
        "experiments/common.py", "experiments/harness.py",
        "experiments/filter_online_exp.py", "experiments/paper_full_config.json")
for name in core:
    frozen = subprocess.check_output(["git", "show", f"{freeze}:{name}"])
    if frozen != Path(name).read_bytes():
        raise SystemExit(f"Frozen source differs from paper-full-v1: {name}")
kernels = common.list_kernels()
if len(kernels) != 50:
    raise SystemExit(f"Expected 50 kernels; got {len(kernels)}")
rag = _rag_index_sha256(required=True)
if rag != "3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397":
    raise SystemExit(f"Unexpected RAG hash: {rag}")
for kernel in kernels:
    manifest = common.read_reference_manifest(kernel)
    if (manifest is None or manifest["repeats"] != 5
            or manifest["device_id"] != 0
            or not math.isfinite(float(manifest["t_base_us"]))
            or float(manifest["t_base_us"]) <= 0
            or not common.find_test_file(kernel).is_file()):
        raise SystemExit(f"Invalid manifest or test: {kernel}")
print(f"Preflight PASS: HEAD={head}, kernels=50, RAG={rag}")
PY

printf 'LLM_HEALTHCHECK_START %s\n' "$(date -u +%FT%TZ)" | tee "$STATUS_FILE"
LLM_MAX_RETRIES=0 LLM_RETRY_WAIT=0 python - <<'PY' 2>&1 | tee "$RUN_DIR/llm_healthcheck.log"
from config import EAConfig
from llm_interface import LLMInterface

llm = LLMInterface(EAConfig())
response = llm.generate("Reply with exactly: OK", purpose="healthcheck")
if not response.strip():
    raise SystemExit("Empty LLM health-check response")
print(f"LLM_HEALTHCHECK=PASS model={llm.current_model} tokens={llm.get_stats()['total_tokens']}")
PY
printf 'LLM_HEALTHCHECK_PASS %s\n' "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
printf 'LLM_CONCURRENCY=5\n' | tee -a "$STATUS_FILE"

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path("experiments/results/filter_online")
quarantine = Path(os.environ.get("PHASE4_QUARANTINE_DIR",
                                 "experiments/results/phase4_quarantine_20261020"))
configs = ("paper_b1", "paper_b2", "paper_full")
seeds = (11, 12, 13)
pilot = {"eye_kernel", "compute_identity_kernel", "moe_align_block_size_stage",
         "fused_gdn_gating_kernel", "reshape_and_cache_kernel_flash"}
moved = []
for config in configs:
    for seed in seeds:
        run_id = 900 + seed
        for result in sorted((root / config).glob(f"*__r{seed}.json")):
            record = json.loads(result.read_text(encoding="utf-8"))
            if (record.get("config") != config or record.get("run_id") != run_id
                    or record.get("kernel") in pilot):
                continue
            audit = record.get("audit") or {}
            usable = (
                record.get("success") is True
                and (record.get("acceptance") or {}).get("passed") is True
                and isinstance(audit.get("npu_calls"), int)
                and audit["npu_calls"] > 0
                and isinstance(audit.get("llm_tokens"), int)
                and audit["llm_tokens"] > 0
                and bool(record.get("llm_model_ids"))
            )
            if usable:
                continue
            kernel = record.get("kernel")
            paths = [
                result,
                result.with_suffix(".py"),
                Path("experiments/logs") / f"filter_online__{config}__{kernel}__r{run_id}.jsonl",
                Path("experiments/logs") / f"filter_scores__{config}__{kernel}__r{run_id}.jsonl",
            ]
            for path in paths:
                if path.is_file():
                    destination = quarantine / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(destination))
                    moved.append(str(path))
print(f"Quarantined invalid non-pilot Phase 4 artifacts: {len(moved)} files")
for path in moved:
    print(path)
PY

for config in "${CONFIGS[@]}"; do
  printf 'CONFIG_START %s %s\n' "$config" "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
  for kernel in "${KERNELS[@]}"; do
    if [[ $(git rev-parse HEAD) != "$EXPECTED_COMMIT" ]]; then
      printf 'STOP HEAD_CHANGED config=%s kernel=%s\n' "$config" "$kernel" | tee -a "$STATUS_FILE"
      exit 1
    fi
    task_log="$RUN_DIR/${config}__${kernel}.log"
    printf 'TASK_START config=%s kernel=%s time=%s\n' \
      "$config" "$kernel" "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
    python experiments/filter_online_exp.py \
      --config "$config" --kernels "$kernel" --seeds "${SEEDS[@]}" \
      2>&1 | tee "$task_log"
    if grep -Eq '\\[LLM\\].*❌ Failed|UCB parent selection requires at least two distinct individuals' "$task_log"; then
      printf 'STOP task_log contains terminal LLM/UCB failure config=%s kernel=%s\n' \
        "$config" "$kernel" | tee -a "$STATUS_FILE"
      exit 1
    fi
    python - "$config" "$kernel" "${SEEDS[@]}" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

config, kernel = sys.argv[1:3]
seeds = [int(x) for x in sys.argv[3:]]
expected_commit = os.environ["EXPECTED_COMMIT"]
rag = "3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397"
manifest_path = Path("experiments/manifest") / f"{kernel}.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
for seed in seeds:
    path = Path("experiments/results/filter_online") / config / f"{kernel}__r{seed}.json"
    if not path.is_file():
        raise SystemExit(f"Missing result: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    audit = record.get("audit") or {}
    provenance = record.get("provenance") or {}
    checks = {
        "success": record.get("success") is True,
        "acceptance": (record.get("acceptance") or {}).get("passed") is True,
        "budget": audit.get("budget_ok") is True
            and isinstance(audit.get("npu_calls"), int)
            and 0 < audit["npu_calls"] <= audit.get("eval_budget", 0),
        "tokens": isinstance(audit.get("llm_tokens"), int) and audit["llm_tokens"] > 0,
        "model_id": bool(record.get("llm_model_ids")),
        "remeasure": audit.get("remeasure_repeats") == 5,
        "rag": record.get("rag_index_sha256") == rag,
        "manifest": record.get("reference_baseline_source") == "manifest"
            and record.get("reference_manifest_sha256") == manifest_hash
            and record.get("reference_baseline_us") == manifest.get("t_base_us"),
        "commit": provenance.get("git_commit") == expected_commit,
        "speedup": isinstance(record.get("S"), (int, float))
            and isinstance(record.get("t_best_us"), (int, float))
            and record["t_best_us"] > 0
            and math.isclose(record["S"],
                             record["reference_baseline_us"] / record["t_best_us"],
                             rel_tol=1e-12),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"Audit failed for {path}: {failed}")
print(f"TASK_AUDIT_PASS config={config} kernel={kernel} seeds={seeds}")
PY
    printf 'TASK_DONE config=%s kernel=%s time=%s\n' \
      "$config" "$kernel" "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
  done
  printf 'CONFIG_DONE %s %s\n' "$config" "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
done

printf 'DONE phase4=450/450 time=%s\n' "$(date -u +%FT%TZ)" | tee -a "$STATUS_FILE"
