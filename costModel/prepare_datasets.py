#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild costModel/datasets from historical Triton evolution logs.

Default inputs:
  experiments/logs/*.jsonl
  experiments/dataset/codes/<candidate_id>.py
  datasets2/<operator>/test__<operator>_1.py (preferred)
  datasets/<operator>/test__<operator>_1.py (fallback)

This script does NOT compile or execute kernels and does NOT create fresh
profiling labels. It only selects historical source variants and stages the
directory layout consumed by costModel/collect.py.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "costmodel_dataset_v1"
SELECTION_METHOD = "pooled_devices_nearest_rank_sha256_dedup"


def now_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def safe_operator_name(name: str) -> bool:
    # Kernel names are expected to be Python identifiers (usually leading "_").
    return bool(name) and all(ch.isalnum() or ch == "_" for ch in name)


def read_history(
    root: Path,
    log_dir: Path,
    code_dir: Path,
    recursive_logs: bool,
    operator_filter: str | None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """
    Returns operator -> source_sha256 -> successful evaluation rows.
    Rows are pooled across all devices and repeated evaluations.
    """
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Historical log directory not found: {log_dir}")
    if not code_dir.is_dir():
        raise FileNotFoundError(f"Historical source directory not found: {code_dir}")

    log_files = sorted(log_dir.rglob("*.jsonl") if recursive_logs else log_dir.glob("*.jsonl"))
    if not log_files:
        raise FileNotFoundError(f"No JSONL logs found under: {log_dir}")

    stats: dict[str, Any] = {
        "log_files": len(log_files),
        "total_log_records": 0,
        "malformed_jsonl_records": 0,
        "eval_records": 0,
        "valid_evaluation_rows": 0,
        "missing_source_candidate_ids": 0,
        "invalid_source_syntax_candidate_ids": 0,
        "unique_candidate_ids": 0,
        "unique_source_hashes": 0,
        "log_files_list": [relpath(p, root) for p in log_files],
        "missing_source_examples": [],
        "invalid_source_examples": [],
        "operators_seen": [],
        "valid_rows_by_operator": {},
        "source_available_rows_by_operator": {},
        "unique_candidate_ids_by_operator": {},
        "missing_source_candidate_ids_by_operator": {},
    }

    operators_seen: set[str] = set()
    valid_rows_by_operator: dict[str, int] = defaultdict(int)
    source_available_rows_by_operator: dict[str, int] = defaultdict(int)
    unique_candidate_ids_by_operator: dict[str, set[str]] = defaultdict(set)
    missing_source_by_operator: dict[str, set[str]] = defaultdict(set)

    # Cache each candidate ID's source validation and hash.
    source_cache: dict[str, tuple[Path | None, str | None, str | None]] = {}
    valid_candidate_ids: set[tuple[str, str]] = set()
    missing_candidate_ids: set[tuple[str, str]] = set()
    invalid_candidate_ids: set[tuple[str, str]] = set()

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for log_path in log_files:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                stats["total_log_records"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_jsonl_records"] += 1
                    continue
                if not isinstance(record, dict):
                    stats["malformed_jsonl_records"] += 1
                    continue

                event = str(record.get("event", "")).lower()
                if event not in {"eval", "evaluation"}:
                    continue
                stats["eval_records"] += 1

                operator = str(record.get("kernel") or record.get("operator_name") or "").strip()
                candidate_id = str(record.get("candidate_id") or "").strip()
                if not operator or not candidate_id:
                    continue
                if operator_filter and operator != operator_filter:
                    continue
                if not safe_operator_name(operator):
                    continue
                operators_seen.add(operator)

                # Exclude explicit failures. A null compile_pass is accepted
                # when correctness and a positive measured latency are present.
                if record.get("ast_pass") is False:
                    continue
                if record.get("compile_pass") is False:
                    continue
                if record.get("correctness_pass") is not True:
                    continue

                latency = positive_float(
                    record.get("latency_us", record.get("child_latency"))
                )
                if latency is None:
                    continue

                stats["valid_evaluation_rows"] += 1
                valid_rows_by_operator[operator] += 1
                valid_candidate_ids.add((operator, candidate_id))
                unique_candidate_ids_by_operator[operator].add(candidate_id)

                if candidate_id not in source_cache:
                    source_path = code_dir / f"{candidate_id}.py"
                    if not source_path.is_file():
                        source_cache[candidate_id] = (None, None, "missing")
                    else:
                        try:
                            code = source_path.read_text(encoding="utf-8")
                            ast.parse(code, filename=str(source_path))
                            source_hash = sha256_file(source_path)
                            source_cache[candidate_id] = (source_path, source_hash, None)
                        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                            source_cache[candidate_id] = (
                                source_path, None, f"invalid_source: {type(exc).__name__}: {exc}"
                            )

                source_path, source_hash, source_error = source_cache[candidate_id]
                if source_error == "missing":
                    missing_candidate_ids.add((operator, candidate_id))
                    missing_source_by_operator[operator].add(candidate_id)
                    continue
                if source_error:
                    invalid_candidate_ids.add((operator, candidate_id))
                    continue
                if source_path is None or source_hash is None:
                    continue

                source_available_rows_by_operator[operator] += 1
                grouped[operator][source_hash].append({
                    "operator_name": operator,
                    "candidate_id": candidate_id,
                    "source_path": source_path,
                    "source_sha256": source_hash,
                    "latency_us": latency,
                    "device_id": record.get("device_id"),
                    "method": record.get("method"),
                    "run_id": record.get("run_id"),
                    "generation": record.get("gen", record.get("generation")),
                    "operation": record.get("operation"),
                    "mutation_type": record.get("mutation_type"),
                    "timestamp": record.get("timestamp"),
                    "log_path": relpath(log_path, root),
                    "line_no": line_no,
                    "logged_source_sha256": record.get("child_code_sha256"),
                })

    stats["unique_candidate_ids"] = len(valid_candidate_ids)
    stats["missing_source_candidate_ids"] = len(missing_candidate_ids)
    stats["invalid_source_syntax_candidate_ids"] = len(invalid_candidate_ids)
    stats["missing_source_examples"] = [
        f"{op}:{cid}" for op, cid in sorted(missing_candidate_ids)[:20]
    ]
    stats["invalid_source_examples"] = [
        f"{op}:{cid}" for op, cid in sorted(invalid_candidate_ids)[:20]
    ]
    stats["unique_source_hashes"] = sum(len(groups) for groups in grouped.values())
    stats["operators_seen"] = sorted(operators_seen)
    stats["valid_rows_by_operator"] = dict(sorted(valid_rows_by_operator.items()))
    stats["source_available_rows_by_operator"] = dict(sorted(source_available_rows_by_operator.items()))
    stats["unique_candidate_ids_by_operator"] = {
        op: len(ids) for op, ids in sorted(unique_candidate_ids_by_operator.items())
    }
    stats["missing_source_candidate_ids_by_operator"] = {
        op: len(ids) for op, ids in sorted(missing_source_by_operator.items())
    }

    return grouped, stats


def summarize_source_groups(
    operator: str,
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_hash, rows in groups.items():
        latencies = [row["latency_us"] for row in rows]
        candidate_ids = sorted({row["candidate_id"] for row in rows})
        # Choose the most frequently observed candidate ID as the representative;
        # lexical order makes ties deterministic.
        id_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            id_counts[row["candidate_id"]] += 1
        representative_id = sorted(
            candidate_ids, key=lambda cid: (-id_counts[cid], cid)
        )[0]
        representative = next(
            row for row in rows if row["candidate_id"] == representative_id
        )
        result.append({
            "operator_name": operator,
            "source_sha256": source_hash,
            "source_path": representative["source_path"],
            "representative_candidate_id": representative_id,
            "candidate_ids": candidate_ids,
            "historical_latency_us": float(statistics.median(latencies)),
            "historical_sample_count": len(latencies),
            "historical_device_ids": sorted({
                str(row["device_id"]) for row in rows if row.get("device_id") is not None
            }),
            "historical_methods": sorted({
                str(row["method"]) for row in rows if row.get("method")
            }),
            "historical_run_ids": sorted({
                str(row["run_id"]) for row in rows if row.get("run_id") is not None
            }),
            "historical_generations": sorted({
                str(row["generation"]) for row in rows if row.get("generation") is not None
            }),
            "historical_operations": sorted({
                str(row["operation"]) for row in rows if row.get("operation")
            }),
            "logged_source_hash_mismatch_count": sum(
                1 for row in rows
                if row.get("logged_source_sha256")
                and row["logged_source_sha256"] != source_hash
            ),
        })
    result.sort(key=lambda item: (
        item["historical_latency_us"], item["source_sha256"]
    ))
    return result


def percentile_specs(population: int) -> list[tuple[str, int]]:
    if population >= 20:
        return [("p0", 0), ("p10", 10), ("p50", 50), ("p90", 90), ("p100", 100)]
    if population >= 5:
        return [("p0", 0), ("p25", 25), ("p50", 50), ("p75", 75), ("p100", 100)]
    if population == 4:
        return [("p0", 0), ("p33", 33), ("p67", 67), ("p100", 100)]
    if population == 3:
        return [("p0", 0), ("p50", 50), ("p100", 100)]
    if population == 2:
        return [("p0", 0), ("p100", 100)]
    if population == 1:
        return [("p50", 50)]
    return []


def nearest_rank_index(population: int, percentile: int) -> int:
    if population <= 0:
        raise ValueError("population must be positive")
    if percentile <= 0:
        return 0
    if percentile >= 100:
        return population - 1
    return max(0, min(population - 1, math.ceil(population * percentile / 100.0) - 1))


def find_test_file(
    root: Path,
    operator: str,
    test_roots: list[str],
) -> tuple[Path | None, str]:
    """
    Find the operator's correctness test. Prefer test__<operator>_1.py.
    If the same preferred filename exists in multiple roots, hashes must agree.
    """
    names = [
        f"test__{operator}_1.py",
        f"test__{operator}.py",
        f"test_{operator}.py",
        "test.py",
    ]
    for name in names:
        matches = [
            (root / test_root / operator / name).resolve()
            for test_root in test_roots
            if (root / test_root / operator / name).is_file()
        ]
        if not matches:
            continue
        hashes = {sha256_file(path) for path in matches}
        if len(hashes) > 1:
            return None, (
                f"ambiguous test file {name}: "
                + ", ".join(relpath(path, root) for path in matches)
            )
        return matches[0], "found"

    # Fallback only when a directory has exactly one test*.py file.
    for test_root in test_roots:
        operator_dir = root / test_root / operator
        if not operator_dir.is_dir():
            continue
        candidates = sorted(
            p.resolve() for p in operator_dir.glob("test*.py") if p.is_file()
        )
        if len(candidates) == 1:
            return candidates[0], "found_fallback"
        if len(candidates) > 1:
            return None, (
                f"ambiguous fallback tests in {relpath(operator_dir, root)}: "
                + ", ".join(p.name for p in candidates)
            )
    return None, "missing test file"


def sample_sufficiency(population: int) -> str:
    if population == 0:
        return "no_candidates"
    if population == 1:
        return "insufficient"
    if population <= 4:
        return "limited"
    if population < 20:
        return "moderate"
    return "sufficient"


def stage_operator(
    stage_root: Path,
    project_root: Path,
    operator: str,
    source_groups: list[dict[str, Any]],
    test_source: Path,
    log_stats: dict[str, Any],
    operator_valid_rows: int,
) -> list[dict[str, Any]]:
    operator_dir = stage_root / operator
    variant_dir = operator_dir / "variant"
    data_dir = operator_dir / "data"
    variant_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    test_dest = operator_dir / "test.py"
    shutil.copy2(test_source, test_dest)

    selected_manifest_rows: list[dict[str, Any]] = []
    population = len(source_groups)
    specs = percentile_specs(population)

    for role, percentile in specs:
        index = nearest_rank_index(population, percentile)
        chosen = source_groups[index]
        source_path = Path(chosen["source_path"])
        variant_dest = variant_dir / f"{role}.py"
        shutil.copy2(source_path, variant_dest)

        role_data_dir = data_dir / role
        role_data_dir.mkdir(parents=True, exist_ok=True)

        candidate_id = (
            f"{operator}__{role}__{chosen['source_sha256'][:12]}"
        )
        test_path_rel = f"costModel/datasets/{operator}/test.py"
        row = {
            "schema_version": SCHEMA_VERSION,
            "operator_name": operator,
            "candidate_id": candidate_id,
            "candidate_role": role,
            "selected_role": role,
            "candidate_type": "historical_evolution",
            "compile_eligible": True,
            "source_path": relpath(variant_dest, project_root),
            "historical_source_path": relpath(source_path, project_root),
            "destination_path": relpath(variant_dest, project_root),
            "test_path": test_path_rel,
            "historical_primary_candidate_id": chosen["representative_candidate_id"],
            "historical_candidate_ids": chosen["candidate_ids"],
            "historical_latency_us": chosen["historical_latency_us"],
            "historical_sample_count": chosen["historical_sample_count"],
            "historical_device_ids": chosen["historical_device_ids"],
            "historical_methods": chosen["historical_methods"],
            "historical_run_ids": chosen["historical_run_ids"],
            "historical_generations": chosen["historical_generations"],
            "historical_operations": chosen["historical_operations"],
            "historical_latency_is_current_label": False,
            "source_sha256": chosen["source_sha256"],
            "selection_percentile": percentile / 100.0,
            "selection_rank": index + 1,
            "selection_population": population,
            "selection_method": SELECTION_METHOD,
            "historical_hash_mismatch_count": chosen["logged_source_hash_mismatch_count"],
        }
        selected_manifest_rows.append(row)

        # Keep historical provenance separate from fresh collect.py metadata.
        write_json(role_data_dir / "selection.json", row)

    roles = [role for role, _ in specs]
    candidate_ids = [row["candidate_id"] for row in selected_manifest_rows]
    test_sha = sha256_file(test_dest)
    operator_config = {
        "schema_version": SCHEMA_VERSION,
        "operator": operator,
        "operator_name": operator,
        "test_file": f"costModel/datasets/{operator}/test.py",
        "test_path": f"costModel/datasets/{operator}/test.py",
        "test_source_path": relpath(test_source, project_root),
        "test_sha256": test_sha,
        "percentiles": roles,
        "candidate_roles": roles,
        "candidate_source_roles": ["historical_evolution"] * len(roles),
        "candidate_count": len(roles),
        "candidates": candidate_ids,
        "selection_method": SELECTION_METHOD,
        "selection_population": population,
        "historical_valid_evaluation_rows": operator_valid_rows,
        "historical_log_files": log_stats["log_files"],
        "variants_ready": bool(roles),
        "correctness_ready": False,
        "profiling_ready": False,
        "ir_ready": False,
        "sample_sufficiency": sample_sufficiency(population),
    }
    write_json(operator_dir / "operator.json", operator_config)
    return selected_manifest_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "=" * 88,
        "COSTMODEL DATASET BUILD SUMMARY",
        "=" * 88,
        f"Historical log files             : {summary['history']['log_files']}",
        f"Total log records                : {summary['history']['total_log_records']}",
        f"Malformed JSONL records          : {summary['history']['malformed_jsonl_records']}",
        f"Valid evaluation rows            : {summary['history']['valid_evaluation_rows']}",
        f"Unique candidate IDs             : {summary['history']['unique_candidate_ids']}",
        f"Missing source candidate IDs      : {summary['history']['missing_source_candidate_ids']}",
        f"Invalid source syntax candidates : {summary['history']['invalid_source_syntax_candidate_ids']}",
        f"Unique source hashes             : {summary['history']['unique_source_hashes']}",
        "",
        f"Operators found                  : {summary['operator_count_found']}",
        f"Operators staged                 : {summary['operator_count_staged']}",
        f"Operators skipped (no test)      : {summary['operator_count_missing_test']}",
        f"Selected variants                : {summary['selected_variant_count']}",
        "",
        "-" * 88,
        f"{'OPERATOR':38} {'UNIQUE':>7} {'PLANNED':>8} {'STAGED':>7} {'TEST':>10} {'SUFFICIENCY':>14}",
        "-" * 88,
    ]
    for item in summary["operators"]:
        lines.append(
            f"{item['operator_name'][:38]:38} "
            f"{item['unique_source_count']:7d} "
            f"{item['planned_selected_count']:8d} "
            f"{item['selected_count']:7d} "
            f"{item['test_status'][:10]:>10} "
            f"{item['sample_sufficiency']:>14}"
        )
        if item.get("skip_reason"):
            lines.append(f"  SKIPPED: {item['skip_reason']}")
    lines += ["-" * 88, ""]
    lines.append("Historical latency is selection provenance only; it is NOT a fresh label.")
    lines.append("Fresh timing/IR/hardware labels must be collected by costModel/collect.py.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select historical Triton candidates and rebuild costModel/datasets."
    )
    parser.add_argument("--root", default=None, help="Project root; default inferred from this script.")
    parser.add_argument("--log-dir", default="experiments/logs")
    parser.add_argument("--codes-dir", default="experiments/dataset/codes")
    parser.add_argument(
        "--test-roots",
        nargs="+",
        default=["datasets2", "datasets"],
        help="Search roots for <root>/<operator>/test*.py; order is priority.",
    )
    parser.add_argument("--recursive-logs", action="store_true")
    parser.add_argument("--operator", default=None, help="Only inspect/build one operator.")
    parser.add_argument("--dry-run", action="store_true", help="Print statistics without writing files.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Move existing costModel/datasets to costModel/audit/backups before rebuilding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else script_path.parent.parent.resolve()
    cost_model_dir = root / "costModel"
    target_dir = cost_model_dir / "datasets"
    log_dir = (root / args.log_dir).resolve()
    code_dir = (root / args.codes_dir).resolve()

    grouped, history_stats = read_history(
        root=root,
        log_dir=log_dir,
        code_dir=code_dir,
        recursive_logs=args.recursive_logs,
        operator_filter=args.operator,
    )

    operators_in_logs = sorted(
        set(grouped) | set(history_stats.get("operators_seen", []))
    )
    # Include an explicitly requested operator even if it has no valid source group.
    if args.operator and args.operator not in operators_in_logs:
        operators_in_logs.append(args.operator)
        operators_in_logs.sort()

    operator_items: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    total_selected = 0
    missing_test_count = 0

    valid_rows_by_operator = history_stats.get("valid_rows_by_operator", {})
    source_available_rows_by_operator = history_stats.get(
        "source_available_rows_by_operator", {}
    )
    unique_candidate_ids_by_operator = history_stats.get(
        "unique_candidate_ids_by_operator", {}
    )

    for operator in operators_in_logs:
        source_groups = summarize_source_groups(operator, grouped.get(operator, {}))
        test_source, test_status = find_test_file(root, operator, args.test_roots)
        planned_selected_count = len(percentile_specs(len(source_groups)))
        skip_reason = None

        if not source_groups:
            skip_reason = "no valid historical candidates with available source"
        elif test_source is None:
            skip_reason = test_status
            missing_test_count += 1

        staged = bool(source_groups and test_source is not None)
        item = {
            "operator_name": operator,
            "valid_evaluation_rows": valid_rows_by_operator.get(operator, 0),
            "source_available_evaluation_rows": source_available_rows_by_operator.get(operator, 0),
            "unique_candidate_ids": unique_candidate_ids_by_operator.get(operator, 0),
            "unique_source_count": len(source_groups),
            "planned_selected_count": planned_selected_count,
            "selected_count": planned_selected_count if staged else 0,
            "test_status": "found" if test_source else test_status,
            "test_source_path": relpath(test_source, root) if test_source else None,
            "sample_sufficiency": sample_sufficiency(len(source_groups)),
            "staged": staged,
            "skip_reason": skip_reason,
        }
        operator_items.append(item)

        if staged:
            # Actual file copying is deferred until after dry-run checks.
            item["_source_groups"] = source_groups
            item["_test_source"] = test_source
            total_selected += planned_selected_count

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(root),
        "inputs": {
            "log_dir": relpath(log_dir, root),
            "codes_dir": relpath(code_dir, root),
            "test_roots": args.test_roots,
            "recursive_logs": args.recursive_logs,
        },
        "history": history_stats,
        "operator_count_found": len(operators_in_logs),
        "operator_count_staged": sum(1 for item in operator_items if item["staged"]),
        "operator_count_missing_test": missing_test_count,
        "selected_variant_count": total_selected,
        "operators": [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in operator_items
        ],
        "notes": [
            "Valid historical rows require event=eval/evaluation, correctness_pass=true, positive latency, and no explicit AST/compile failure.",
            "Measurements from all devices and repeated evaluations are pooled; each unique source hash receives the median latency.",
            "Source variants are deduplicated by SHA256 within each operator.",
            "Percentile selection uses nearest-rank positions over ascending historical median latency.",
            "Historical latency is not a current training label; collect.py must produce fresh labels.",
            "Operators without an unambiguous correctness test are reported but not staged.",
        ],
    }

    print(build_report(summary), end="")
    if args.dry_run:
        print("[DRY RUN] No files or directories were modified.")
        return 0

    if target_dir.exists() and not args.replace:
        raise FileExistsError(
            f"{target_dir} already exists. Re-run with --replace to back it up and rebuild."
        )

    build_id = now_id()
    staging_dir = cost_model_dir / f".datasets_staging_{build_id}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        for item in operator_items:
            if not item["staged"]:
                continue
            operator = item["operator_name"]
            selected_rows = stage_operator(
                stage_root=staging_dir,
                project_root=root,
                operator=operator,
                source_groups=item["_source_groups"],
                test_source=item["_test_source"],
                log_stats=history_stats,
                operator_valid_rows=item["valid_evaluation_rows"],
            )
            selected_rows_all.extend(selected_rows)

        selected_rows_all.sort(
            key=lambda row: (row["operator_name"], int(row["selection_percentile"] * 100))
        )

        # Remove private runtime objects from the persisted summary.
        write_json(staging_dir / "dataset_summary.json", summary)
        write_jsonl(staging_dir / "dataset_manifest.jsonl", selected_rows_all)

        csv_rows = []
        for item in summary["operators"]:
            csv_rows.append({
                "operator_name": item["operator_name"],
                "valid_evaluation_rows": item["valid_evaluation_rows"],
                "source_available_evaluation_rows": item["source_available_evaluation_rows"],
                "unique_candidate_ids": item["unique_candidate_ids"],
                "unique_source_count": item["unique_source_count"],
                "planned_selected_count": item["planned_selected_count"],
                "selected_count": item["selected_count"],
                "test_status": item["test_status"],
                "test_source_path": item["test_source_path"],
                "sample_sufficiency": item["sample_sufficiency"],
                "staged": item["staged"],
                "skip_reason": item["skip_reason"],
            })
        write_csv(
            staging_dir / "dataset_summary.csv",
            csv_rows,
            [
                "operator_name", "valid_evaluation_rows", "source_available_evaluation_rows",
                "unique_candidate_ids", "unique_source_count", "planned_selected_count",
                "selected_count", "test_status", "test_source_path",
                "sample_sufficiency", "staged", "skip_reason",
            ],
        )
        (staging_dir / "dataset_build_report.txt").write_text(
            build_report(summary), encoding="utf-8"
        )

        backup_dir = None
        if target_dir.exists():
            backup_root = cost_model_dir / "audit" / "backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_dir = backup_root / f"datasets_before_rebuild_{build_id}"
            target_dir.rename(backup_dir)

        try:
            staging_dir.rename(target_dir)
        except Exception:
            if backup_dir and backup_dir.exists() and not target_dir.exists():
                backup_dir.rename(target_dir)
            raise

        print(f"[SUCCESS] Dataset rebuilt: {target_dir}")
        if backup_dir:
            print(f"[BACKUP] Previous datasets preserved at: {backup_dir}")
        print(f"[MANIFEST] {target_dir / 'dataset_manifest.jsonl'}")
        print(f"[SUMMARY]  {target_dir / 'dataset_summary.json'}")
        print(f"[REPORT]   {target_dir / 'dataset_build_report.txt'}")
        return 0

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[dataset-build-error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
