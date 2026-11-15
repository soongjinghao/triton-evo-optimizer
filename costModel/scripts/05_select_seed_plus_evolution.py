#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select 5 CostModel candidates/operator: one datasets seed + four evolved versions.

Policy per operator:
  1) dataset_seed: datasets/<op>/<op>.py (the first dataset seed)
  2) historical_best: fastest eligible distinct evolution source
  3) historical_p10
  4) historical_p50
  5) historical_p90

Historical labels are medians of child_latency grouped by (operator, source SHA256).
The second datasets seed is deliberately excluded from the selected set and from
historical selection, so all four evolution candidates are distinct from both
initial dataset seeds.

The dataset seed label is reused from experiments/manifest only if the current
source hash exactly matches datasets2/<op>/<op>.py and the paired test file hash
also matches. Otherwise its label is left pending for script 06 to re-profile.

This script does not compile, warm up, execute, or profile kernels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.0"
ROLES = ("dataset_seed", "historical_best", "historical_p10", "historical_p50", "historical_p90")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n}: expected JSON object")
            rows.append(row)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def positive_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "operator_name", "candidate_id", "candidate_type", "selected_role",
        "label_latency_us", "label_status", "latency_unit", "latency_source",
        "measurement_provenance", "measurement_count", "is_current_profile",
        "source_sha256", "source_path", "test_path", "compile_job_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def manifest_seed_label(root: Path, op: str, filename: str, source_hash: str,
                        test_path: Path, test_hash: str) -> dict[str, Any] | None:
    """Only reuse the old label after exact source AND test identity checks."""
    mp = root / "experiments" / "manifest" / f"{op}.json"
    old_source = root / "datasets2" / op / filename
    old_test = root / "datasets2" / op / test_path.name
    if not mp.is_file() or not old_source.is_file() or not old_test.is_file():
        return None
    manifest = read_json(mp)
    entry = (manifest.get("seeds") or {}).get(filename) or {}
    latency = positive_float(entry.get("latency_us"))
    if not entry.get("success") or latency is None:
        return None
    if sha256_file(old_source) != source_hash:
        return None
    if sha256_file(old_test) != test_hash:
        return None
    # The manifest path should point at this operator's datasets2 seed file.
    manifest_path = str(entry.get("path") or "")
    if manifest_path and not manifest_path.endswith(f"datasets2/{op}/{filename}"):
        return None
    return {
        "label_latency_us": latency,
        "label_status": "reused_exact_manifest_measurement",
        "latency_source": f"experiments/manifest/{op}.json:seeds.{filename}.latency_us",
        "measurement_provenance": "manifest_seed_exact_source_and_test_match",
        "measurement_count": entry.get("measurement_count") or manifest.get("repeats"),
        "measurement_timestamp": manifest.get("created_at"),
        "device_id": manifest.get("device_id"),
        "is_current_profile": False,
    }


def nearest_rank_index(n: int, percentile: int) -> int:
    if n <= 1:
        return 0
    return int(math.floor((percentile / 100.0) * (n - 1) + 0.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--candidate-pool", default="costModel/data/manifests/candidate_pool.jsonl")
    ap.add_argument("--all-compile-jobs", default="costModel/data/manifests/candidate_compile_jobs.jsonl")
    ap.add_argument("--historical-source-latency", default="costModel/data/labels/historical_source_latency.jsonl")
    ap.add_argument("--output-dir", default="costModel/data/pilot_evo")
    ap.add_argument("--expected-operators", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    pool_rows = read_jsonl(resolve(root, args.candidate_pool))
    all_jobs = read_jsonl(resolve(root, args.all_compile_jobs))
    hist_labels = read_jsonl(resolve(root, args.historical_source_latency))
    out_dir = resolve(root, args.output_dir)

    pool_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    pool_by_hash: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    operators = sorted({str(r.get("operator_name") or "").strip() for r in pool_rows if r.get("operator_name")})
    for r in pool_rows:
        op, cid = str(r.get("operator_name") or "").strip(), str(r.get("candidate_id") or "").strip()
        if op and cid:
            if (op, cid) in pool_by_key:
                raise ValueError(f"Duplicate candidate pool key: {op}/{cid}")
            pool_by_key[(op, cid)] = r
        h = str(r.get("source_sha256") or "").strip()
        if op and h:
            pool_by_hash[(op, h)].append(r)
    if args.expected_operators and len(operators) != args.expected_operators:
        raise ValueError(f"Expected {args.expected_operators} operators, found {len(operators)}")

    jobs_by_hash: dict[tuple[str, str], dict[str, Any]] = {}
    for job in all_jobs:
        key = (str(job.get("operator_name") or ""), str(job.get("source_sha256") or ""))
        if key[0] and key[1]:
            if key in jobs_by_hash:
                raise ValueError(f"Duplicate compile job: {key}")
            jobs_by_hash[key] = job

    labels_by_hash: dict[tuple[str, str], dict[str, Any]] = {}
    for row in hist_labels:
        op, h = str(row.get("operator_name") or ""), str(row.get("source_sha256") or "")
        if op and h and positive_float(row.get("label_latency_us")) is not None:
            labels_by_hash[(op, h)] = row

    selected: list[dict[str, Any]] = []
    selected_jobs_by_hash: dict[tuple[str, str], dict[str, Any]] = {}
    per_operator = []
    pending_seed_profiles = []

    for op in operators:
        # The first datasets implementation is kept as the single initial seed.
        seed_pool = next((r for r in pool_rows if r.get("operator_name") == op and r.get("candidate_type") == "baseline"), None)
        second_seed_pool = next((r for r in pool_rows if r.get("operator_name") == op and r.get("candidate_type") == "seed"), None)
        if seed_pool is None:
            raise ValueError(f"{op}: missing datasets/<op>/<op>.py candidate")

        seed_hash = str(seed_pool.get("source_sha256") or "")
        seed_job = jobs_by_hash.get((op, seed_hash))
        if not seed_hash or not seed_job:
            raise ValueError(f"{op}: dataset seed has no compile job")
        seed_path = resolve(root, str(seed_pool.get("source_path") or ""))
        pool_test_path = resolve(root, str(seed_pool.get("test_path") or ""))
        # Historical evolution was evaluated by experiments.common with DATASETS_DIR=datasets2.
        # Keep that exact test harness for every selected candidate so old child_latency labels
        # and newly collected IR/profile data refer to the same workload.
        test_path = root / "datasets2" / op / pool_test_path.name
        if not seed_path.is_file() or not pool_test_path.is_file() or not test_path.is_file():
            raise ValueError(f"{op}: seed source/test missing")
        if sha256_file(seed_path) != seed_hash:
            raise ValueError(f"{op}: dataset seed source SHA256 mismatch")
        test_hash = sha256_file(test_path)

        seed_filename = seed_path.name
        seed_label = manifest_seed_label(root, op, seed_filename, seed_hash, test_path, test_hash)
        seed_row = dict(seed_pool)
        seed_row.update({
            "schema_version": SCHEMA_VERSION,
            "candidate_id": f"dataset_seed_1:{op}",
            "original_inventory_candidate_id": seed_pool.get("candidate_id"),
            "candidate_type": "dataset_seed",
            "selected_role": "dataset_seed",
            "selection_percentile": None,
            "label_name": "candidate_runtime_us",
            "latency_unit": "us",
            "source_sha256": seed_hash,
            "test_path": test_path.relative_to(root).as_posix(),
            "test_sha256": test_hash,
            "test_harness_origin": "datasets2 (same harness used by experiments)",
            "compile_job_id": seed_job.get("compile_job_id"),
            "feature_join_key": seed_hash,
            "feature_join_status": "awaiting_ttir_features",
            "file_validation": "source_and_test_sha256_verified",
        })
        if seed_label:
            seed_row.update(seed_label)
        else:
            seed_row.update({
                "label_latency_us": None,
                "label_status": "pending_profile",
                "latency_source": None,
                "measurement_provenance": None,
                "measurement_count": None,
                "is_current_profile": None,
            })
            pending_seed_profiles.append(op)
        selected.append(seed_row)

        excluded_hashes = {seed_hash}
        if second_seed_pool and second_seed_pool.get("source_sha256"):
            excluded_hashes.add(str(second_seed_pool["source_sha256"]))

        eligible = []
        for (label_op, source_hash), label in labels_by_hash.items():
            if label_op != op or source_hash in excluded_hashes:
                continue
            job = jobs_by_hash.get((op, source_hash))
            if job is None:
                continue
            pool_candidates = pool_by_hash.get((op, source_hash), [])
            valid_pool = []
            for pr in pool_candidates:
                if pr.get("candidate_type") != "historical_evolution":
                    continue
                if not pr.get("source_available") or not pr.get("compile_eligible"):
                    continue
                sp = resolve(root, str(pr.get("source_path") or ""))
                pool_test = resolve(root, str(pr.get("test_path") or ""))
                tp = root / "datasets2" / op / pool_test.name
                if not sp.is_file() or not pool_test.is_file() or not tp.is_file():
                    continue
                if sha256_file(sp) != source_hash:
                    continue
                if not pr.get("jit_functions"):
                    continue
                valid_pool.append(pr)
            if not valid_pool:
                continue
            valid_pool.sort(key=lambda r: str(r.get("candidate_id") or ""))
            pr = valid_pool[0]
            latency = positive_float(label.get("label_latency_us"))
            if latency is None:
                continue
            eligible.append({
                "latency": latency,
                "source_hash": source_hash,
                "pool_row": pr,
                "source_label": label,
                "job": job,
            })

        eligible.sort(key=lambda x: (x["latency"], x["source_hash"], str(x["pool_row"].get("candidate_id"))))
        if len(eligible) < 4:
            raise ValueError(f"{op}: only {len(eligible)} eligible distinct evolution sources; need at least 4")

        chosen_by_role = []
        used_indices: set[int] = set()
        for role, percentile in (("historical_best", 0), ("historical_p10", 10), ("historical_p50", 50), ("historical_p90", 90)):
            target = nearest_rank_index(len(eligible), percentile)
            available = [i for i in range(len(eligible)) if i not in used_indices]
            idx = min(available, key=lambda i: (abs(i - target), i))
            used_indices.add(idx)
            chosen = eligible[idx]
            chosen_by_role.append((role, percentile, chosen))

        for role, percentile, item in chosen_by_role:
            pr = item["pool_row"]
            job = item["job"]
            label = item["source_label"]
            row = dict(pr)
            row.update({
                "schema_version": SCHEMA_VERSION,
                "candidate_type": "historical_evolution",
                "selected_role": role,
                "selection_percentile": percentile,
                "label_name": label.get("label_name", "candidate_runtime_us"),
                "label_latency_us": item["latency"],
                "latency_unit": label.get("latency_unit", "us"),
                "latency_source": "experiments/dataset/samples.jsonl:child_latency (median by source SHA256)",
                "measurement_provenance": "aggregated_historical_evolution_samples",
                "is_current_profile": False,
                "measurement_count": None,
                "source_group_candidate_count": label.get("logical_candidate_count"),
                "historical_latency_min_us": label.get("latency_min_us"),
                "historical_latency_max_us": label.get("latency_max_us"),
                "historical_latency_mean_us": label.get("latency_mean_us"),
                "source_candidate_ids": label.get("candidate_ids", []),
                "selection_target_latency_us": item["latency"],
                "compile_job_id": job.get("compile_job_id"),
                "feature_join_key": item["source_hash"],
                "feature_join_status": "awaiting_ttir_features",
                "test_path": (root / "datasets2" / op / Path(str(pr.get("test_path"))).name).relative_to(root).as_posix(),
                "test_sha256": sha256_file(root / "datasets2" / op / Path(str(pr.get("test_path"))).name),
                "test_harness_origin": "datasets2 (same harness used by experiments)",
                "file_validation": "source_sha256_and_datasets2_test_verified",
            })
            selected.append(row)

        per_operator.append({
            "operator_name": op,
            "selected_count": 5,
            "dataset_seed_label_status": seed_row.get("label_status"),
            "historical_unique_eligible_sources": len(eligible),
            "historical_roles": {role: {"candidate_id": item["pool_row"].get("candidate_id"), "latency_us": item["latency"], "source_sha256": item["source_hash"]}
                                 for role, _, item in chosen_by_role},
        })

    # Build exactly one warmup job per selected unique (operator, source hash).
    selected.sort(key=lambda r: (r["operator_name"], ROLES.index(r["selected_role"])))
    for row in selected:
        op, h = row["operator_name"], row["source_sha256"]
        job = dict(jobs_by_hash[(op, h)])
        job["test_path"] = row.get("test_path")
        job["test_sha256"] = row.get("test_sha256")
        key = (op, h)
        if key not in selected_jobs_by_hash:
            job["pilot_selected"] = True
            job["selected_roles"] = []
            job["selected_candidate_ids"] = []
            selected_jobs_by_hash[key] = job
        selected_jobs_by_hash[key]["selected_roles"].append(row["selected_role"])
        selected_jobs_by_hash[key]["selected_candidate_ids"].append(row["candidate_id"])
        # The selected role/candidate is the display representative for newly compiled IR.
        selected_jobs_by_hash[key]["representative_candidate_id"] = row["candidate_id"]
        if row["selected_role"] == "dataset_seed":
            selected_jobs_by_hash[key]["selected_candidate_type"] = "dataset_seed"

    selected_jobs = sorted(selected_jobs_by_hash.values(), key=lambda j: (j["operator_name"], j["compile_job_id"]))
    role_counts = {role: sum(r["selected_role"] == role for r in selected) for role in ROLES}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": {
            "per_operator": {role: 1 for role in ROLES},
            "historical_label": "median child_latency grouped by operator_name + source_sha256",
            "historical_percentiles": [0, 10, 50, 90],
            "exclude_both_dataset_seed_hashes_from_evolution_pool": True,
            "deduplicate_by_operator_and_source_sha256": True,
            "dataset_seed_label_reuse_requires_exact_source_and_test_hash_match": True,
            "baseline_json_is_not_a_candidate_or_training_label": True,
        },
        "counts": {
            "operators": len(operators),
            "selected_logical_candidates": len(selected),
            "expected_logical_candidates": len(operators) * 5,
            "unique_compile_jobs": len(selected_jobs),
            "role_counts": role_counts,
            "dataset_seed_labels_reused": role_counts["dataset_seed"] - len(pending_seed_profiles),
            "dataset_seed_labels_pending_profile": len(pending_seed_profiles),
            "operators_complete": sum(1 for x in per_operator if x["selected_count"] == 5),
        },
        "pending_seed_profile_operators": pending_seed_profiles,
        "operators": per_operator,
        "notes": [
            "The first datasets implementation is kept as dataset_seed; it is never named baseline.",
            "The second datasets seed is excluded from the five selected candidates.",
            "Four evolution candidates are selected from historical child_latency groups, with distinct source hashes.",
            "Historical labels remain historical measurements; provenance is not rewritten as current profiling.",
            "Warmup is a separate compile-only step and does not produce runtime labels.",
        ],
    }

    if len(selected) != len(operators) * 5 or any(x["selected_count"] != 5 for x in per_operator):
        raise RuntimeError("Selection is incomplete; expected exactly five candidates per operator")

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "selected_candidates.jsonl", selected)
    write_jsonl(out_dir / "compile_jobs.jsonl", selected_jobs)
    write_csv(out_dir / "selected_candidates.csv", selected)
    write_json(out_dir / "selection_summary.json", summary)
    print("=" * 76)
    print("Seed + evolution candidate selection completed")
    print(f"Operators                   : {len(operators)}")
    print(f"Logical candidates          : {len(selected)}")
    print(f"Unique compile jobs         : {len(selected_jobs)}")
    print(f"Dataset seed labels reused  : {role_counts['dataset_seed'] - len(pending_seed_profiles)}")
    print(f"Dataset seed labels pending: {len(pending_seed_profiles)}")
    for role in ROLES:
        print(f"{role:28s}: {role_counts[role]}")
    print(f"Output directory            : {out_dir}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[seed-evolution-selection] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
