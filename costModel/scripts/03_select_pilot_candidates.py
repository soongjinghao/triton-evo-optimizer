#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select a 250-candidate pilot set for the Triton Cost Model.

Selection policy (per operator):
  1. Baseline: one original implementation.
  2. Seed: one initial optimized implementation.
  3. Historical P10: one eligible historical source near the 10th latency percentile.
  4. Historical P50: one eligible historical source near the 50th latency percentile.
  5. Historical P90: one eligible historical source near the 90th latency percentile.

Historical percentiles are computed from historical_source_latency.jsonl, which
already aggregates repeated logical candidates with the same source SHA256 by
the median child_latency. Candidates are restricted to source groups that also
exist in candidate_compile_jobs.jsonl (the authoritative compile-eligible set).

Outputs:
  costModel/data/pilot/selected_candidates.jsonl
  costModel/data/pilot/compile_jobs.jsonl
  costModel/data/pilot/selected_candidates.csv
  costModel/data/pilot/selection_summary.json

This script does NOT compile kernels, run profiling, or extract TTIR.
It only selects candidates and creates a subset of the existing compile jobs.

Run from the Agent project root:
  python costModel/scripts/03_select_pilot_candidates.py --dry-run
  python costModel/scripts/03_select_pilot_candidates.py

Optional:
  --expected-operators 50
  --allow-incomplete
  --skip-file-validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "1.0"
SELECTION_ROLES = (
    ("baseline", "baseline"),
    ("seed", "seed"),
    ("historical_p10", "historical_evolution"),
    ("historical_p50", "historical_evolution"),
    ("historical_p90", "historical_evolution"),
)


def infer_project_root() -> Path:
    script_path = Path(__file__).resolve()
    # Intended location: <Agent>/costModel/scripts/03_select_pilot_candidates.py
    if len(script_path.parents) >= 3:
        return script_path.parents[2]
    return Path.cwd().resolve()


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("Required JSONL file not found: {}".format(path))
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON at {}:{}: {}".format(path, line_no, exc)
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    "Expected JSON object at {}:{}".format(path, line_no)
                )
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def positive_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x <= 0:
        return None
    return x


def clean(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(clean(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "operator_name", "candidate_id", "candidate_type", "selected_role",
        "selection_percentile", "label_latency_us", "latency_unit",
        "latency_source", "measurement_provenance", "source_sha256",
        "source_path", "test_path", "compile_job_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_files(
    root: Path,
    pool_row: Dict[str, Any],
    expected_hash: str,
    skip_file_validation: bool,
) -> Tuple[bool, str]:
    if not pool_row.get("source_available"):
        return False, "source_unavailable"
    if not pool_row.get("compile_eligible"):
        return False, "not_compile_eligible"

    source_rel = pool_row.get("source_path")
    test_rel = pool_row.get("test_path")
    if not source_rel or not test_rel:
        return False, "missing_source_or_test_path"

    if skip_file_validation:
        return True, "validation_skipped"

    source_path = resolve_path(root, str(source_rel))
    test_path = resolve_path(root, str(test_rel))
    if not source_path.is_file():
        return False, "source_file_missing:{}".format(source_rel)
    if not test_path.is_file():
        return False, "test_file_missing:{}".format(test_rel)

    actual_hash = sha256_file(source_path)
    if expected_hash and actual_hash != expected_hash:
        return False, "source_sha256_mismatch"
    return True, "ok"


def percentile_rank_index(count: int, percentile: int) -> int:
    """Nearest-rank index in [0, count-1], deterministic and dependency-free."""
    if count <= 1:
        return 0
    return int(math.floor((percentile / 100.0) * (count - 1) + 0.5))


def build_selection(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    pool_path = resolve_path(root, args.candidate_pool)
    compile_jobs_path = resolve_path(root, args.compile_jobs)
    latency_records_path = resolve_path(root, args.latency_records)
    source_latency_path = resolve_path(root, args.source_latency)
    output_dir = resolve_path(root, args.output_dir)

    pool_rows = read_jsonl(pool_path)
    compile_jobs = read_jsonl(compile_jobs_path)
    latency_rows = read_jsonl(latency_records_path)
    source_latency_rows = read_jsonl(source_latency_path)

    pool_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pool_by_hash: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        op = str(row.get("operator_name") or "").strip()
        cid = str(row.get("candidate_id") or "").strip()
        if not op or not cid:
            continue
        key = (op, cid)
        if key in pool_by_key:
            raise ValueError("Duplicate candidate_pool key: {}".format(key))
        pool_by_key[key] = row
        h = row.get("source_sha256")
        if h:
            pool_by_hash[(op, str(h))].append(row)

    jobs_by_hash: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for job in compile_jobs:
        op = str(job.get("operator_name") or "").strip()
        h = str(job.get("source_sha256") or "").strip()
        if not op or not h:
            continue
        key = (op, h)
        if key in jobs_by_hash:
            raise ValueError("Duplicate compile job for operator/source hash: {}".format(key))
        jobs_by_hash[key] = job

    latency_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in latency_rows:
        op = str(row.get("operator_name") or "").strip()
        cid = str(row.get("candidate_id") or "").strip()
        if op and cid:
            latency_by_key[(op, cid)] = row

    operators = sorted({str(row.get("operator_name") or "").strip()
                        for row in pool_rows if row.get("operator_name")})
    if args.expected_operators and len(operators) != args.expected_operators:
        raise ValueError(
            "Expected {} operators, but candidate_pool contains {}. "
            "Use --expected-operators 0 to disable this check.".format(
                args.expected_operators, len(operators)
            )
        )

    # Index the already-aggregated historical source labels by operator/hash.
    hist_source_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in source_latency_rows:
        op = str(row.get("operator_name") or "").strip()
        h = str(row.get("source_sha256") or "").strip()
        latency = positive_float(row.get("label_latency_us"))
        if op and h and latency is not None:
            key = (op, h)
            if key in hist_source_by_key:
                raise ValueError("Duplicate historical source latency key: {}".format(key))
            hist_source_by_key[key] = row

    selected_rows: List[Dict[str, Any]] = []
    selected_jobs_by_hash: Dict[Tuple[str, str], Dict[str, Any]] = {}
    operator_summaries: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []
    role_counts = defaultdict(int)

    for op in operators:
        op_selected: List[Dict[str, Any]] = []
        used_hashes = set()
        missing_roles: List[str] = []

        # Baseline and Seed must be selected explicitly.
        for role, candidate_type in (("baseline", "baseline"), ("seed", "seed")):
            candidates = [
                row for row in pool_rows
                if row.get("operator_name") == op
                and row.get("candidate_type") == candidate_type
            ]
            valid = []
            for pool_row in candidates:
                cid = str(pool_row.get("candidate_id") or "")
                h = str(pool_row.get("source_sha256") or "")
                latency_row = latency_by_key.get((op, cid))
                latency = positive_float(
                    latency_row.get("label_latency_us") if latency_row else None
                )
                if not h or latency is None or (op, h) not in jobs_by_hash:
                    continue
                ok, reason = validate_candidate_files(
                    root, pool_row, h, args.skip_file_validation
                )
                if not ok:
                    continue
                valid.append((pool_row, latency_row, latency, reason))

            if not valid:
                missing_roles.append(role)
                continue

            # There should normally be one row; deterministic selection if
            # the pool contains more than one baseline/seed record.
            valid.sort(key=lambda item: str(item[0].get("candidate_id")))
            pool_row, latency_row, latency, validation_status = valid[0]
            h = str(pool_row["source_sha256"])
            job = jobs_by_hash[(op, h)]
            selected = clean(pool_row)
            selected.update({
                "schema_version": SCHEMA_VERSION,
                "selected_role": role,
                "selection_percentile": None,
                "label_name": latency_row.get("label_name", "candidate_runtime_us"),
                "label_latency_us": latency,
                "latency_unit": latency_row.get("latency_unit", "us"),
                "latency_source": latency_row.get("latency_source"),
                "measurement_provenance": latency_row.get("measurement_provenance"),
                "is_current_profile": latency_row.get("is_current_profile"),
                "measurement_count": latency_row.get("measurement_count"),
                "feature_join_key": h,
                "feature_join_status": "awaiting_ttir_features",
                "compile_job_id": job.get("compile_job_id"),
                "file_validation": validation_status,
            })
            op_selected.append(selected)
            used_hashes.add(h)
            role_counts[role] += 1

        # Historical source rows are already deduplicated by (operator, SHA256)
        # and their runtime label is the median of repeated logical candidates.
        historical = []
        for (hist_op, source_hash), source_label in hist_source_by_key.items():
            if hist_op != op or source_hash in used_hashes:
                continue
            job = jobs_by_hash.get((op, source_hash))
            if job is None:
                continue

            group_ids = source_label.get("candidate_ids") or []
            matching_pool_rows = []
            for cid in group_ids:
                pool_row = pool_by_key.get((op, str(cid)))
                if pool_row is None:
                    continue
                if str(pool_row.get("source_sha256") or "") != source_hash:
                    continue
                ok, validation_status = validate_candidate_files(
                    root, pool_row, source_hash, args.skip_file_validation
                )
                if ok:
                    matching_pool_rows.append((pool_row, validation_status))

            # Fallback to any compile-eligible pool record with the same hash.
            if not matching_pool_rows:
                for pool_row in pool_by_hash.get((op, source_hash), []):
                    ok, validation_status = validate_candidate_files(
                        root, pool_row, source_hash, args.skip_file_validation
                    )
                    if ok:
                        matching_pool_rows.append((pool_row, validation_status))

            if not matching_pool_rows:
                continue

            matching_pool_rows.sort(key=lambda item: str(item[0].get("candidate_id")))
            pool_row, validation_status = matching_pool_rows[0]
            latency = positive_float(source_label.get("label_latency_us"))
            if latency is None:
                continue
            historical.append({
                "pool_row": pool_row,
                "source_label": source_label,
                "source_hash": source_hash,
                "latency": latency,
                "job": job,
                "validation_status": validation_status,
            })

        historical.sort(key=lambda item: (
            item["latency"],
            item["source_hash"],
            str(item["pool_row"].get("candidate_id")),
        ))

        history_role_specs = (
            ("historical_p10", 10),
            ("historical_p50", 50),
            ("historical_p90", 90),
        )
        if len(historical) < 3:
            missing_roles.extend(role for role, _ in history_role_specs)
        else:
            # Compute all three percentile positions against the same eligible
            # historical population. If a small population causes an index
            # collision, move to the nearest still-unused rank deterministically.
            for role, percentile in history_role_specs:
                target_index = percentile_rank_index(len(historical), percentile)
                available_indices = [
                    i for i, item in enumerate(historical)
                    if item["source_hash"] not in used_hashes
                ]
                if not available_indices:
                    missing_roles.append(role)
                    continue
                index = min(
                    available_indices,
                    key=lambda i: (abs(i - target_index), i)
                )
                chosen = historical[index]
                pool_row = chosen["pool_row"]
                source_label = chosen["source_label"]
                job = chosen["job"]
                selected = clean(pool_row)
                selected.update({
                    "schema_version": SCHEMA_VERSION,
                    "candidate_type": "historical_evolution",
                    "selected_role": role,
                    "selection_percentile": percentile,
                    "selection_target_latency_us": chosen["latency"],
                    "label_name": source_label.get("label_name", "candidate_runtime_us"),
                    "label_latency_us": chosen["latency"],
                    "latency_unit": source_label.get("latency_unit", "us"),
                    "latency_source": "historical_source_latency.jsonl:median(child_latency)",
                    "measurement_provenance": source_label.get(
                        "measurement_provenance",
                        "aggregated_historical_evolution_samples",
                    ),
                    "is_current_profile": False,
                    "measurement_count": source_label.get("logical_candidate_count"),
                    "logical_candidate_count": source_label.get("logical_candidate_count"),
                    "source_candidate_ids": source_label.get("candidate_ids", []),
                    "feature_join_key": chosen["source_hash"],
                    "feature_join_status": "awaiting_ttir_features",
                    "compile_job_id": job.get("compile_job_id"),
                    "file_validation": chosen["validation_status"],
                })
                op_selected.append(selected)
                used_hashes.add(chosen["source_hash"])
                role_counts[role] += 1

        # Add compile jobs for selected source hashes, one per unique hash.
        for selected in op_selected:
            h = str(selected.get("source_sha256") or "")
            job = jobs_by_hash.get((op, h))
            if job is None:
                continue
            key = (op, h)
            if key not in selected_jobs_by_hash:
                selected_jobs_by_hash[key] = clean(job)
                selected_jobs_by_hash[key]["pilot_selected"] = True
                selected_jobs_by_hash[key]["selected_roles"] = []
                selected_jobs_by_hash[key]["selected_candidate_ids"] = []
            selected_jobs_by_hash[key]["selected_roles"].append(selected["selected_role"])
            selected_jobs_by_hash[key]["selected_candidate_ids"].append(
                selected.get("candidate_id")
            )

        op_selected.sort(key=lambda row: (
            {"baseline": 0, "seed": 1, "historical_p10": 2,
             "historical_p50": 3, "historical_p90": 4}.get(row["selected_role"], 9)
        ))
        selected_rows.extend(op_selected)

        operator_summaries.append({
            "operator_name": op,
            "selected_count": len(op_selected),
            "baseline_count": sum(1 for r in op_selected if r["selected_role"] == "baseline"),
            "seed_count": sum(1 for r in op_selected if r["selected_role"] == "seed"),
            "historical_p10_count": sum(1 for r in op_selected if r["selected_role"] == "historical_p10"),
            "historical_p50_count": sum(1 for r in op_selected if r["selected_role"] == "historical_p50"),
            "historical_p90_count": sum(1 for r in op_selected if r["selected_role"] == "historical_p90"),
            "historical_eligible_unique_sources": len(historical),
            "missing_roles": missing_roles,
        })
        if missing_roles or len(op_selected) != 5:
            incomplete.append({
                "operator_name": op,
                "selected_count": len(op_selected),
                "missing_roles": missing_roles,
                "historical_eligible_unique_sources": len(historical),
            })

    selected_rows.sort(key=lambda row: (
        row["operator_name"],
        {"baseline": 0, "seed": 1, "historical_p10": 2,
         "historical_p50": 3, "historical_p90": 4}.get(row["selected_role"], 9),
    ))
    selected_jobs = list(selected_jobs_by_hash.values())
    selected_jobs.sort(key=lambda row: (
        row.get("operator_name", ""), row.get("compile_job_id", "")
    ))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(root),
        "selection_policy": {
            "per_operator": {
                "baseline": 1,
                "seed": 1,
                "historical_p10": 1,
                "historical_p50": 1,
                "historical_p90": 1,
            },
            "historical_label": (
                "median child_latency grouped by operator_name + source_sha256"
            ),
            "historical_candidates_require_compile_job": True,
            "historical_candidates_require_source_and_test_files": (
                not args.skip_file_validation
            ),
            "historical_candidates_deduplicated_by_source_sha256": True,
            "percentile_method": "nearest rank over eligible unique historical source groups",
        },
        "inputs": {
            "candidate_pool": str(pool_path),
            "candidate_compile_jobs": str(compile_jobs_path),
            "candidate_latency_records": str(latency_records_path),
            "historical_source_latency": str(source_latency_path),
        },
        "counts": {
            "operators_found": len(operators),
            "expected_operators": args.expected_operators,
            "selected_logical_candidates": len(selected_rows),
            "selected_candidates_expected": len(operators) * 5,
            "selected_compile_jobs": len(selected_jobs),
            "selected_unique_source_hashes": len(selected_jobs),
            "selected_baseline": role_counts["baseline"],
            "selected_seed": role_counts["seed"],
            "selected_historical_p10": role_counts["historical_p10"],
            "selected_historical_p50": role_counts["historical_p50"],
            "selected_historical_p90": role_counts["historical_p90"],
            "operators_complete": len(operators) - len(incomplete),
            "operators_incomplete": len(incomplete),
        },
        "operators": operator_summaries,
        "incomplete_operators": incomplete,
        "notes": [
            "The 250 logical selections are not necessarily 250 unique compile jobs: exact source duplicates are compiled once.",
            "Baseline/Seed reference latencies and historical child_latency retain their original provenance.",
            "P10/P50/P90 are computed only within each operator's eligible historical source groups, not across different operators.",
            "This is a pilot selection set, not a final model train/test split.",
            "No kernels are compiled or executed by this script.",
        ],
        "outputs": {
            "selected_candidates": "selected_candidates.jsonl",
            "compile_jobs": "compile_jobs.jsonl",
            "selected_candidates_csv": "selected_candidates.csv",
            "summary": "selection_summary.json",
        },
    }

    if incomplete and not args.allow_incomplete:
        details = "; ".join(
            "{}: selected={} missing={}".format(
                item["operator_name"], item["selected_count"], ",".join(item["missing_roles"])
            )
            for item in incomplete[:10]
        )
        raise RuntimeError(
            "Pilot selection is incomplete for {} operator(s): {}. "
            "Inspect the inputs or rerun with --allow-incomplete.".format(
                len(incomplete), details
            )
        )

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "selected_candidates.jsonl", selected_rows)
    write_jsonl(output_dir / "compile_jobs.jsonl", selected_jobs)
    write_csv(output_dir / "selected_candidates.csv", selected_rows)
    write_json(output_dir / "selection_summary.json", summary)

    print("=" * 76)
    print("Cost Model pilot candidate selection completed")
    print("Project root                  :", root)
    print("Operators                     :", len(operators))
    print("Selected logical candidates   :", len(selected_rows))
    print("Unique compile jobs           :", len(selected_jobs))
    print("Baseline / Seed               :", role_counts["baseline"], "/", role_counts["seed"])
    print("Historical P10 / P50 / P90    :",
          role_counts["historical_p10"], "/", role_counts["historical_p50"], "/", role_counts["historical_p90"])
    print("Incomplete operators          :", len(incomplete))
    print("Output directory              :", output_dir)
    print("Summary                       :", output_dir / "selection_summary.json")
    print("=" * 76)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select five representative Cost Model pilot candidates per operator."
    )
    parser.add_argument("--root", default=str(infer_project_root()),
                        help="Agent project root.")
    parser.add_argument("--candidate-pool",
                        default="costModel/data/manifests/candidate_pool.jsonl")
    parser.add_argument("--compile-jobs",
                        default="costModel/data/manifests/candidate_compile_jobs.jsonl")
    parser.add_argument("--latency-records",
                        default="costModel/data/labels/candidate_latency_records.jsonl")
    parser.add_argument("--source-latency",
                        default="costModel/data/labels/historical_source_latency.jsonl")
    parser.add_argument("--output-dir", default="costModel/data/pilot")
    parser.add_argument("--expected-operators", type=int, default=50,
                        help="Expected operator count; set 0 to disable.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Write partial selection even if some operators lack five candidates.")
    parser.add_argument("--skip-file-validation", action="store_true",
                        help="Skip local source/test existence and source SHA256 checks (testing only).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print selection summary without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_selection(args)
    except Exception as exc:
        print("[pilot-selection-error] {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
