import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "costModel_v2/ir/compile_results.jsonl"
OUTPUT = ROOT / "costModel_v2/ir/failure_details.json"

TARGETS = {
    ("_silu_mul_fp8_quant_deep_gemm", "historical_p50"),
    ("assign_draft_cache_locs_page_size_1", "dataset_seed"),
    ("assign_extend_cache_locs", "historical_p90"),
    ("moe_align_block_size_stage", "historical_p10"),
    ("moe_align_block_size_stage", "historical_best"),
    ("reduce_segments", "historical_p90"),
}

FIELDS = [
    "operator_name",
    "candidate_type",
    "status",
    "compile_pass",
    "compile_status",
    "source_path",
    "test_path",
    "test_error",
    "test_traceback",
    "compile_errors",
    "worker_status",
    "process_return_code",
    "worker_log_tail",
    "ir_found",
    "ir_paths",
    "kernel_calls",
]

results = []

with INPUT.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue

        d = json.loads(line)

        key = (
            d.get("operator_name"),
            d.get("candidate_type"),
        )

        if key not in TARGETS:
            continue

        item = {
            "jsonl_line": line_no,
            "record": {
                k: d.get(k)
                for k in FIELDS
                if k in d
            }
        }

        results.append(item)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=" * 65)
print("CostModel V2 Failure Diagnostics")
print("=" * 65)
print("Matched records:", len(results))
print("Output:", OUTPUT)

for item in results:
    d = item["record"]

    print(
        f"[{d.get('operator_name')}] "
        f"{d.get('candidate_type')} | "
        f"{d.get('status')}"
    )

print("=" * 65)
