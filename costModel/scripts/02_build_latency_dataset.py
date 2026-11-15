#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build historical per-candidate runtime labels for the Triton Cost Model.

Inputs (relative to project root by default):
  experiments/dataset/samples.jsonl
  experiments/manifest/*.json
  costModel/data/manifests/candidate_pool.jsonl

Outputs:
  costModel/data/labels/candidate_latency_records.jsonl
      All logical candidates: historical candidates + Baseline + Seed.
  costModel/data/labels/historical_candidate_latency.jsonl
      One row per historical evolution candidate (child_latency is the target).
  costModel/data/labels/operator_reference_latency.jsonl
      Baseline and Seed reference latencies from per-operator manifests.
  costModel/data/labels/historical_source_latency.jsonl
      Historical candidates grouped by exact (operator, source SHA256);
      the target is the median of the available logical-candidate measurements.
  costModel/data/labels/candidate_latency_records.csv
      CSV view of all logical candidate latency records.
  costModel/data/labels/operator_latency_summary.csv
      One summary row per operator.
  costModel/data/labels/latency_dataset_summary.json
      Counts, coverage, and warnings.

Important:
  * Historical child_latency is an actual recorded runtime, but it is historical.
  * Baseline/Seed values come from experiments/manifest/*.json.
  * The script preserves measurement provenance and does NOT claim that these
    values were measured in one identical profiling batch/environment.
  * It does NOT run NPU profiling, compile kernels, or extract TTIR features.
  * The historical binary field `label` is renamed to
    `historical_improvement_label`; it is NOT the runtime regression target.
  * No baseline-relative speedup is computed for historical candidates because
    exact workload/environment equivalence is not established by these files.

Run from project root:
  python costModel/scripts/02_build_latency_dataset.py
  python costModel/scripts/02_build_latency_dataset.py --dry-run
  python costModel/scripts/02_build_latency_dataset.py --root /path/to/Agent
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "1.0"
LABEL_NAME = "candidate_runtime_us"


def project_root_from_script() -> Path:
    # Work both when this script is placed in costModel/scripts/ and when it
    # is temporarily run from the project root.
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (
            (parent / "experiments/dataset/samples.jsonl").is_file()
            and (parent / "costModel/data/manifests/candidate_pool.jsonl").is_file()
        ):
            return parent
    # Fallback for the intended location:
    # .../Agent/costModel/scripts/02_build_latency_dataset.py -> .../Agent
    return script_path.parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("Input JSONL not found: {}".format(path))

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{}:{} contains invalid JSON: {}".format(path, line_no, exc)
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    "{}:{} must contain a JSON object".format(path, line_no)
                )
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Input JSON not found: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in {}".format(path))
    return value


def positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def json_safe(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    temp.replace(path)
    return count


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                cooked[key] = value
            writer.writerow(cooked)
    temp.replace(path)
    return len(rows)


def source_info(
    root: Path,
    pool_row: Optional[Dict[str, Any]],
    fallback_path: Optional[str] = None,
) -> Dict[str, Any]:
    pool_row = pool_row or {}
    source_path = pool_row.get("source_path") or fallback_path
    source_file = resolve_path(root, source_path) if source_path else None
    exists = bool(source_file and source_file.is_file())

    # Do not trust only the old availability flag: verify the actual file too.
    pool_says_available = pool_row.get("source_available")
    if pool_says_available is False:
        exists = False

    return {
        "source_path": source_path,
        "source_sha256": pool_row.get("source_sha256"),
        "source_available": exists,
        "compile_eligible": bool(pool_row.get("compile_eligible", False)),
        "jit_functions": pool_row.get("jit_functions", []),
        "test_path": pool_row.get("test_path"),
        "test_sha256": pool_row.get("test_sha256"),
    }


def load_manifests(manifest_dir: Path) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    if not manifest_dir.is_dir():
        raise FileNotFoundError("Manifest directory not found: {}".format(manifest_dir))

    manifests: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for path in sorted(manifest_dir.glob("*.json")):
        data = read_json(path)
        operator = str(data.get("kernel") or path.stem).strip()
        if not operator:
            continue
        if operator in manifests:
            raise ValueError(
                "Duplicate manifest for operator '{}': {}".format(operator, path)
            )
        manifests[operator] = (path, data)
    return manifests


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    samples_path = resolve_path(root, args.samples)
    pool_path = resolve_path(root, args.pool)
    manifest_dir = resolve_path(root, args.manifest_dir)
    output_dir = resolve_path(root, args.output_dir)

    samples = read_jsonl(samples_path)
    pool_rows = read_jsonl(pool_path)
    manifests = load_manifests(manifest_dir)

    pool_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in pool_rows:
        operator = str(row.get("operator_name") or "").strip()
        candidate_id = str(row.get("candidate_id") or "").strip()
        if operator and candidate_id:
            key = (operator, candidate_id)
            if key in pool_index:
                raise ValueError("Duplicate candidate_pool key: {}".format(key))
            pool_index[key] = row

    historical_rows: List[Dict[str, Any]] = []
    invalid_latency_rows: List[Dict[str, Any]] = []
    missing_pool_keys: List[str] = []
    duplicate_sample_keys: List[str] = []
    seen_sample_keys = set()

    for sample in samples:
        operator = str(sample.get("kernel") or "").strip()
        candidate_id = str(sample.get("candidate_id") or "").strip()
        if not operator or not candidate_id:
            invalid_latency_rows.append({
                "source_line": sample.get("_source_line"),
                "reason": "missing kernel/operator or candidate_id",
            })
            continue

        key = (operator, candidate_id)
        if key in seen_sample_keys:
            duplicate_sample_keys.append("{}::{}".format(operator, candidate_id))
        seen_sample_keys.add(key)

        latency = positive_float(sample.get("child_latency"))
        if latency is None:
            invalid_latency_rows.append({
                "operator_name": operator,
                "candidate_id": candidate_id,
                "source_line": sample.get("_source_line"),
                "reason": "child_latency is not a positive finite number",
                "raw_child_latency": sample.get("child_latency"),
            })
            continue

        pool_row = pool_index.get(key)
        if pool_row is None:
            missing_pool_keys.append("{}::{}".format(operator, candidate_id))

        src = source_info(
            root,
            pool_row,
            fallback_path="experiments/dataset/codes/{}.py".format(candidate_id),
        )

        historical_rows.append({
            "schema_version": SCHEMA_VERSION,
            "operator_name": operator,
            "candidate_id": candidate_id,
            "candidate_type": "historical_evolution",
            "label_name": LABEL_NAME,
            "label_latency_us": latency,
            "latency_us": latency,
            "latency_unit": "us",
            "latency_source": "experiments/dataset/samples.jsonl:child_latency",
            "measurement_provenance": "historical_evolution_sample",
            "is_current_profile": False,
            "measurement_count": 1,
            "sample_source_line": sample.get("_source_line"),
            "run_id": sample.get("run_id"),
            "method": sample.get("method"),
            "generation": sample.get("gen"),
            "operation": sample.get("operation"),
            "mutation_type": sample.get("mutation_type"),
            "parent_ids": sample.get("parent_ids"),
            "parent_latency_us": positive_float(sample.get("parent_latency")),
            "seed_time_us": positive_float(sample.get("seed_time")),
            "g_mut": sample.get("g_mut"),
            "historical_improvement_label": sample.get("label"),
            "child_profile": sample.get("child_profile"),
            "timestamp": sample.get("timestamp"),
            "source_path": src["source_path"],
            "source_sha256": src["source_sha256"],
            "source_available": src["source_available"],
            "compile_eligible": src["compile_eligible"],
            "jit_functions": src["jit_functions"],
            "test_path": src["test_path"],
            "test_sha256": src["test_sha256"],
            "feature_join_key": src["source_sha256"],
            "feature_join_status": (
                "awaiting_ttir_features" if src["source_available"] else "source_missing"
            ),
            "workload_id": None,
            "environment_id": None,
        })

    reference_rows: List[Dict[str, Any]] = []
    missing_manifest_operators: List[str] = []

    operators = sorted(
        set(str(row.get("kernel") or "").strip() for row in samples if row.get("kernel"))
        | set(manifests.keys())
    )

    for operator in operators:
        manifest_entry = manifests.get(operator)
        if manifest_entry is None:
            missing_manifest_operators.append(operator)
            continue

        manifest_path, manifest = manifest_entry
        repeats = manifest.get("repeats")
        created_at = manifest.get("created_at")
        device_id = manifest.get("device_id")

        for candidate_type, candidate_id, field_name in (
            ("baseline", "baseline:{}".format(operator), "t_base_us"),
            ("seed", "seed:{}".format(operator), "t_seed_us"),
        ):
            latency = positive_float(manifest.get(field_name))
            if latency is None:
                continue

            pool_row = pool_index.get((operator, candidate_id))
            src = source_info(root, pool_row)
            reference_rows.append({
                "schema_version": SCHEMA_VERSION,
                "operator_name": operator,
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "label_name": LABEL_NAME,
                "label_latency_us": latency,
                "latency_us": latency,
                "latency_unit": "us",
                "latency_source": "experiments/manifest/{}:{}".format(
                    manifest_path.name, field_name
                ),
                "measurement_provenance": "operator_reference_manifest",
                "is_current_profile": None,
                "measurement_count": repeats,
                "measurement_timestamp": created_at,
                "device_id": device_id,
                "manifest_path": manifest_path.relative_to(root).as_posix()
                    if manifest_path.is_relative_to(root) else str(manifest_path),
                "source_path": src["source_path"],
                "source_sha256": src["source_sha256"],
                "source_available": src["source_available"],
                "compile_eligible": src["compile_eligible"],
                "jit_functions": src["jit_functions"],
                "test_path": src["test_path"],
                "test_sha256": src["test_sha256"],
                "feature_join_key": src["source_sha256"],
                "feature_join_status": (
                    "awaiting_ttir_features" if src["source_available"] else "source_missing"
                ),
                "workload_id": None,
                "environment_id": None,
            })

    all_rows = historical_rows + reference_rows
    type_order = {"baseline": 0, "seed": 1, "historical_evolution": 2}
    all_rows.sort(key=lambda row: (
        row["operator_name"],
        type_order.get(row["candidate_type"], 9),
        row["candidate_id"],
    ))
    historical_rows.sort(key=lambda row: (row["operator_name"], row["candidate_id"]))
    reference_rows.sort(key=lambda row: (row["operator_name"], row["candidate_type"]))

    # Aggregate repeated logical candidates that have exactly the same source
    # within the same operator. Raw candidate rows are kept unchanged above.
    source_buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in historical_rows:
        source_hash = row.get("source_sha256")
        if row.get("source_available") and source_hash:
            source_buckets[(row["operator_name"], str(source_hash))].append(row)

    source_rows: List[Dict[str, Any]] = []
    for (operator, source_hash), group in sorted(source_buckets.items()):
        latencies = [float(row["label_latency_us"]) for row in group]
        first = group[0]
        source_rows.append({
            "schema_version": SCHEMA_VERSION,
            "operator_name": operator,
            "candidate_type": "historical_source_group",
            "source_sha256": source_hash,
            "source_path": first.get("source_path"),
            "source_available": True,
            "label_name": LABEL_NAME,
            "label_latency_us": float(statistics.median(latencies)),
            "latency_unit": "us",
            "aggregation_method": "median",
            "logical_candidate_count": len(group),
            "candidate_ids": [row["candidate_id"] for row in group],
            "latency_min_us": min(latencies),
            "latency_max_us": max(latencies),
            "latency_mean_us": float(statistics.mean(latencies)),
            "feature_join_key": source_hash,
            "feature_join_status": "awaiting_ttir_features",
            "measurement_provenance": "aggregated_historical_evolution_samples",
            "is_current_profile": False,
        })

    summary_rows: List[Dict[str, Any]] = []
    for operator in operators:
        hist = [row for row in historical_rows if row["operator_name"] == operator]
        refs = [row for row in reference_rows if row["operator_name"] == operator]
        base = next((r for r in refs if r["candidate_type"] == "baseline"), None)
        seed = next((r for r in refs if r["candidate_type"] == "seed"), None)
        hist_latencies = [float(row["label_latency_us"]) for row in hist]
        source_groups = [
            row for row in source_rows if row["operator_name"] == operator
        ]
        summary_rows.append({
            "operator_name": operator,
            "historical_candidate_count": len(hist),
            "historical_valid_latency_count": len(hist_latencies),
            "historical_source_available_count": sum(
                1 for row in hist if row["source_available"]
            ),
            "historical_source_missing_count": sum(
                1 for row in hist if not row["source_available"]
            ),
            "historical_compile_eligible_count": sum(
                1 for row in hist if row["compile_eligible"]
            ),
            "historical_unique_source_count": len(source_groups),
            "historical_latency_min_us": min(hist_latencies) if hist_latencies else None,
            "historical_latency_median_us": (
                float(statistics.median(hist_latencies)) if hist_latencies else None
            ),
            "historical_latency_mean_us": (
                float(statistics.mean(hist_latencies)) if hist_latencies else None
            ),
            "historical_latency_max_us": max(hist_latencies) if hist_latencies else None,
            "baseline_latency_us": base["label_latency_us"] if base else None,
            "seed_latency_us": seed["label_latency_us"] if seed else None,
            "seed_speedup_vs_baseline": (
                float(base["label_latency_us"] / seed["label_latency_us"])
                if base and seed and seed["label_latency_us"] > 0 else None
            ),
            "reference_manifest": (
                base.get("manifest_path") if base else None
            ),
        })

    historical_source_count = sum(
        1 for row in historical_rows if row["source_available"]
    )
    historical_missing_source_count = len(historical_rows) - historical_source_count
    manifest_operator_count = len(manifests)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "label_name": LABEL_NAME,
        "label_unit": "us",
        "project_root": str(root),
        "inputs": {
            "samples_jsonl": samples_path.relative_to(root).as_posix()
                if samples_path.is_relative_to(root) else str(samples_path),
            "candidate_pool_jsonl": pool_path.relative_to(root).as_posix()
                if pool_path.is_relative_to(root) else str(pool_path),
            "manifest_dir": manifest_dir.relative_to(root).as_posix()
                if manifest_dir.is_relative_to(root) else str(manifest_dir),
        },
        "counts": {
            "historical_sample_rows_read": len(samples),
            "historical_candidates_with_valid_child_latency": len(historical_rows),
            "historical_candidates_with_source": historical_source_count,
            "historical_candidates_missing_source": historical_missing_source_count,
            "historical_source_groups": len(source_rows),
            "baseline_reference_rows": sum(
                1 for row in reference_rows if row["candidate_type"] == "baseline"
            ),
            "seed_reference_rows": sum(
                1 for row in reference_rows if row["candidate_type"] == "seed"
            ),
            "combined_candidate_latency_rows": len(all_rows),
            "operators_in_samples": len(set(
                str(row.get("kernel")) for row in samples if row.get("kernel")
            )),
            "operators_with_manifest": manifest_operator_count,
            "invalid_or_missing_child_latency_rows": len(invalid_latency_rows),
            "candidate_pool_rows": len(pool_rows),
            "candidate_pool_keys_missing_for_samples": len(missing_pool_keys),
            "duplicate_sample_keys": len(duplicate_sample_keys),
        },
        "warnings": {
            "operators_missing_manifest": missing_manifest_operators,
            "candidate_pool_keys_missing_for_samples": missing_pool_keys[:100],
            "duplicate_sample_keys": duplicate_sample_keys[:100],
            "invalid_latency_rows": invalid_latency_rows[:100],
        },
        "important_notes": [
            "Historical child_latency is a recorded per-candidate runtime, not a fresh re-profile.",
            "Baseline/Seed values come from per-operator experiments/manifest JSON files.",
            "Measurement provenance is preserved; do not assume historical and reference rows share an identical profiling batch.",
            "The historical binary label is retained as historical_improvement_label and is not the regression target.",
            "No historical baseline-relative speedup is computed because exact workload/environment equivalence is not established here.",
            "TTIR features are not created by this script. Join features later using operator_name + source_sha256.",
            "historical_source_latency.jsonl aggregates identical historical source code using the median runtime.",
        ],
        "outputs": {
            "candidate_latency_records": "candidate_latency_records.jsonl",
            "historical_candidate_latency": "historical_candidate_latency.jsonl",
            "operator_reference_latency": "operator_reference_latency.jsonl",
            "historical_source_latency": "historical_source_latency.jsonl",
            "candidate_latency_records_csv": "candidate_latency_records.csv",
            "operator_latency_summary_csv": "operator_latency_summary.csv",
            "summary": "latency_dataset_summary.json",
        },
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidate_latency_records.jsonl", all_rows)
    write_jsonl(output_dir / "historical_candidate_latency.jsonl", historical_rows)
    write_jsonl(output_dir / "operator_reference_latency.jsonl", reference_rows)
    write_jsonl(output_dir / "historical_source_latency.jsonl", source_rows)

    candidate_csv_fields = [
        "operator_name", "candidate_id", "candidate_type", "label_latency_us",
        "latency_unit", "latency_source", "measurement_provenance",
        "is_current_profile", "measurement_count", "source_available",
        "compile_eligible", "source_path", "source_sha256",
        "historical_improvement_label", "parent_latency_us", "seed_time_us",
        "run_id", "generation", "method", "operation", "mutation_type",
        "timestamp", "feature_join_key", "feature_join_status",
    ]
    write_csv(
        output_dir / "candidate_latency_records.csv",
        all_rows,
        candidate_csv_fields,
    )

    summary_csv_fields = [
        "operator_name", "historical_candidate_count",
        "historical_valid_latency_count", "historical_source_available_count",
        "historical_source_missing_count", "historical_compile_eligible_count",
        "historical_unique_source_count", "historical_latency_min_us",
        "historical_latency_median_us", "historical_latency_mean_us",
        "historical_latency_max_us", "baseline_latency_us", "seed_latency_us",
        "seed_speedup_vs_baseline", "reference_manifest",
    ]
    write_csv(
        output_dir / "operator_latency_summary.csv",
        summary_rows,
        summary_csv_fields,
    )
    write_json(output_dir / "latency_dataset_summary.json", summary)

    print("=" * 72)
    print("Historical candidate latency dataset built")
    print("Project root                         :", root)
    print("Historical candidates with latency   :", len(historical_rows))
    print("Historical candidates with source    :", historical_source_count)
    print("Historical candidates missing source:", historical_missing_source_count)
    print("Baseline reference rows              :", sum(
        1 for row in reference_rows if row["candidate_type"] == "baseline"
    ))
    print("Seed reference rows                  :", sum(
        1 for row in reference_rows if row["candidate_type"] == "seed"
    ))
    print("Unique historical source groups      :", len(source_rows))
    print("Combined candidate records           :", len(all_rows))
    print("Operators with manifests             :", manifest_operator_count)
    print("Output directory                     :", output_dir)
    print("=" * 72)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-candidate historical runtime labels for Triton Cost Model."
    )
    parser.add_argument(
        "--root",
        default=str(project_root_from_script()),
        help="Project root (default: inferred from costModel/scripts location).",
    )
    parser.add_argument(
        "--samples",
        default="experiments/dataset/samples.jsonl",
        help="Historical evolution samples JSONL.",
    )
    parser.add_argument(
        "--pool",
        default="costModel/data/manifests/candidate_pool.jsonl",
        help="Candidate pool JSONL with source metadata.",
    )
    parser.add_argument(
        "--manifest-dir",
        default="experiments/manifest",
        help="Directory containing one reference manifest JSON per operator.",
    )
    parser.add_argument(
        "--output-dir",
        default="costModel/data/labels",
        help="Output directory for latency datasets and summaries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without writing output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_dataset(args)
    except Exception as exc:
        print("[latency-dataset-error] {}".format(exc), file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
