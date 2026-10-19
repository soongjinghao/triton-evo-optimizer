#!/usr/bin/env bash
# Small real-NPU acceptance after freezing PAPER-FULL.
# Default: 3 kernels x 6 controlled configurations x seed 1 (preserves seed-0 records).
# Set PAPER_SEEDS="1 2 3" for repeated smoke runs; this remains non-final evidence.
set -euo pipefail
cd /workspace/Agent

source set_env/set_api_local.sh >/dev/null 2>&1
export EA_NUM_DEVICES=1
export EA_MAX_WORKERS=2

KERNELS=(eye_kernel compute_identity_kernel moe_align_block_size_stage)
read -r -a SEEDS <<< "${PAPER_SEEDS:-1}"
CONFIGS=(paper_full paper_tournament paper_full_v2 paper_full_v2_tournament paper_b1 paper_b2)
RESULTS=experiments/results/filter_online

python - <<'PY'
import json
import os
from pathlib import Path
from experiments.filter_online_exp import _rag_index_sha256

expected = json.loads(Path("experiments/paper_full_config.json").read_text())[
    "evidence_guidance"
]["rag_index_sha256"]
actual = _rag_index_sha256(required=True)
if not expected or actual != expected:
    raise SystemExit(f"RAG index fingerprint mismatch: expected={expected}, actual={actual}")
try:
    import torch
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit(f"NPU runtime unavailable: {exc}") from exc
available = torch.npu.device_count()
requested = int(os.environ["EA_NUM_DEVICES"])
if available < requested:
    raise SystemExit(
        f"Requested {requested} NPU devices, but runtime reports {available}")
print(f"PAPER-FULL RAG index verified: {actual}")
print(f"NPU devices: requested={requested}, available={available}")
PY

for config in "${CONFIGS[@]}"; do
  echo "=== $config seeds=${SEEDS[*]} ==="
  python experiments/filter_online_exp.py \
    --config "$config" --kernels "${KERNELS[@]}" --seeds "${SEEDS[@]}"
done

for seed in "${SEEDS[@]}"; do
  run_id=$((900 + seed))
  for pair in \
    "paper_full paper_tournament ucb_vs_tournament_v1" \
    "paper_full paper_full_v2 ucb_v1_vs_v2" \
    "paper_tournament paper_full_v2_tournament tournament_v1_vs_v2" \
    "paper_full_v2 paper_full_v2_tournament ucb_vs_tournament_v2"; do
    read -r left right label <<< "$pair"
    python experiments/compare_v1_v2.py \
      --v1-config "$left" --v1-run-id "$run_id" --v1-seed "$seed" \
      --v2-config "$right" --v2-run-id "$run_id" --v2-seed "$seed" \
      --v1-results "$RESULTS/$left" --v2-results "$RESULTS/$right" \
      --pool-source eval --kernels "${KERNELS[@]}" \
      --json-out "experiments/results/paper_acceptance_${label}_seed${seed}.json" \
      > "experiments/results/paper_acceptance_${label}_seed${seed}.txt"
  done
done

python - "${SEEDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

kernels = ("eye_kernel", "compute_identity_kernel", "moe_align_block_size_stage")
configs = ("paper_full", "paper_tournament",
           "paper_full_v2", "paper_full_v2_tournament", "paper_b1", "paper_b2")
expected_rag = json.loads(
    Path("experiments/paper_full_config.json").read_text(encoding="utf-8")
)["evidence_guidance"]["rag_index_sha256"]
seeds = [int(x) for x in sys.argv[1:]]
root = Path("experiments/results/filter_online")
bad = []
for config in configs:
    for seed in seeds:
        for kernel in kernels:
            path = root / config / f"{kernel}__r{seed}.json"
            if not path.is_file():
                bad.append(f"missing {path}")
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            audit = rec.get("audit") or {}
            acceptance = rec.get("acceptance") or {}
            provenance = rec.get("provenance") or {}
            if (rec.get("success") is not True
                    or acceptance.get("passed") is not True
                    or audit.get("budget_ok") is not True
                    or audit.get("npu_calls") is None
                    or audit.get("npu_calls") > audit.get("eval_budget", 0)
                    or audit.get("remeasure_repeats") != 5
                    or not rec.get("llm_model_ids")
                    or rec.get("rag_index_sha256") != expected_rag
                    or rec.get("reference_baseline_source") != "manifest"
                    or rec.get("reference_baseline_us")
                       != provenance.get("reference_baseline_us")
                    or not rec.get("reference_manifest_sha256")):
                bad.append(f"failed acceptance {path}")
if bad:
    raise SystemExit("\n".join(bad))
print(f"Acceptance records passed: {len(configs) * len(seeds) * len(kernels)}")
PY
