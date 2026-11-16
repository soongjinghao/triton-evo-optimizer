#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit CostModel V2 logical candidates against their current source/test files
and the per-candidate IR manifest.

Default inputs:
  costModel_v2/data/selected_candidates.jsonl
  costModel_v2/data/candidate_ir_manifest.jsonl

Default outputs:
  costModel_v2/labels/ir_audit.jsonl
  costModel_v2/labels/ir_audit_summary.json

This is a file/hash audit. It does not compile or execute kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
V2_DIR = SCRIPT.parents[1]
DEFAULT_ROOT = V2_DIR.parent

ROLES = (
    "dataset_seed", "historical_best", "historical_p10",
    "historical_p50", "historical_p90",
)
IR_CATEGORIES = ("ttir", "ttadapter", "npuir", "npubin")


def log(message: str) -> None:
    print(f"[ir-audit] {message}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def resolve_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        parts = list(raw.parts)
        if "Agent" in parts:
            idx = parts.index("Agent")
            candidate = root.joinpath(*parts[idx + 1:])
            if candidate.exists():
                return candidate.resolve()
        return raw
    return (root / raw).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return []


def candidate_key(row: dict[str, Any]) -> str:
    if row.get("candidate_id"):
        return str(row["candidate_id"])
    return "|".join(str(row.get(k) or "") for k in ("operator_name", "selected_role", "source_sha256"))


def audit_one(
    selected: dict[str, Any],
    ir_row: dict[str, Any] | None,
    root: Path,
) -> dict[str, Any]:
    op = str(selected.get("operator_name") or "")
    role = str(selected.get("selected_role") or selected.get("candidate_type") or "")
    cid = candidate_key(selected)
    issues: list[str] = []
    warnings: list[str] = []

    source = resolve_path(root, selected.get("source_path"))
    test = resolve_path(root, selected.get("test_path"))
    source_exists = bool(source and source.is_file())
    test_exists = bool(test and test.is_file())
    source_hash_actual = sha256_file(source) if source_exists else None
    test_hash_actual = sha256_file(test) if test_exists else None
    source_hash_expected = str(selected.get("source_sha256") or "").lower()
    test_hash_expected = str(selected.get("test_sha256") or "").lower()
    source_hash_match = bool(source_exists and source_hash_expected and source_hash_actual == source_hash_expected)
    test_hash_match = bool(test_exists and test_hash_expected and test_hash_actual == test_hash_expected)

    if not source_exists:
        issues.append("source_missing")
    elif not source_hash_expected:
        issues.append("source_sha256_missing")
    elif not source_hash_match:
        issues.append("source_sha256_mismatch")
    if not test_exists:
        issues.append("test_missing")
    elif not test_hash_expected:
        issues.append("test_sha256_missing")
    elif not test_hash_match:
        issues.append("test_sha256_mismatch")

    if ir_row is None:
        issues.append("candidate_ir_manifest_row_missing")
        ir_row = {}
    if str(ir_row.get("operator_name") or "") != op:
        issues.append("ir_manifest_operator_mismatch")
    if str(ir_row.get("source_sha256") or "").lower() != source_hash_expected:
        issues.append("ir_manifest_source_hash_mismatch")
    if str(ir_row.get("test_sha256") or "").lower() != test_hash_expected:
        issues.append("ir_manifest_test_hash_mismatch")

    ir_paths_obj = ir_row.get("ir_paths") or {}
    ir_found_obj = ir_row.get("ir_found") or {}
    category_result: dict[str, dict[str, Any]] = {}

    for category in IR_CATEGORIES:
        raw_paths = as_path_list(ir_paths_obj.get(category) if isinstance(ir_paths_obj, dict) else None)
        existing, missing, empty = [], [], []
        for raw in raw_paths:
            path = resolve_path(root, raw)
            if path is None or not path.is_file():
                missing.append(raw)
            elif path.stat().st_size <= 0:
                empty.append(str(path))
            else:
                existing.append(str(path))
        found_flag = bool(ir_found_obj.get(category)) if isinstance(ir_found_obj, dict) else False
        present = bool(existing)
        category_result[category] = {
            "manifest_flag": found_flag,
            "path_count": len(raw_paths),
            "existing_nonempty_count": len(existing),
            "existing_paths": existing,
            "missing_paths": missing,
            "empty_paths": empty,
            "complete": present,
        }
        if not present:
            issues.append(f"{category}_missing")
        if missing:
            issues.append(f"{category}_path_missing")
        if empty:
            issues.append(f"{category}_empty_file")
        if found_flag and not present:
            warnings.append(f"{category}_flag_true_but_no_valid_file")
        if present and not found_flag:
            warnings.append(f"{category}_file_exists_but_manifest_flag_false")

    compile_pass = ir_row.get("compile_pass")
    compile_status = ir_row.get("compile_status") or ir_row.get("status")
    if compile_pass is False:
        warnings.append("compile_pass_false")
    elif compile_pass is None:
        warnings.append("compile_pass_unknown")
    if compile_status == "no_kernel_compile_observed":
        warnings.append("no_kernel_compile_observed_despite_ir_files")

    full_ir = all(category_result[c]["complete"] for c in IR_CATEGORIES)
    ttir_ok = category_result["ttir"]["complete"]
    source_test_current = source_hash_match and test_hash_match
    # compile_pass=None is allowed to proceed to a real execution test;
    # compile_pass=False is excluded.
    profile_eligible = source_test_current and ttir_ok and compile_pass is not False

    if not full_ir:
        warnings.append("not_full_four_stage_ir")
    if not profile_eligible:
        warnings.append("not_profile_eligible")

    return {
        "candidate_id": cid,
        "operator_name": op,
        "selected_role": role,
        "source_path": selected.get("source_path"),
        "source_sha256_expected": source_hash_expected,
        "source_sha256_actual": source_hash_actual,
        "source_exists": source_exists,
        "source_hash_match": source_hash_match,
        "test_path": selected.get("test_path"),
        "test_sha256_expected": test_hash_expected,
        "test_sha256_actual": test_hash_actual,
        "test_exists": test_exists,
        "test_hash_match": test_hash_match,
        "compile_job_id": selected.get("compile_job_id") or ir_row.get("compile_job_id"),
        "compile_status": compile_status,
        "compile_pass": compile_pass,
        "ir_categories": category_result,
        "ttir_available": ttir_ok,
        "full_ir_complete": full_ir,
        "profile_eligible": profile_eligible,
        "issues": sorted(set(issues)),
        "warnings": sorted(set(warnings)),
        "audit_status": "complete" if full_ir and source_test_current else "incomplete",
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--selected-candidates", default="costModel_v2/data/selected_candidates.jsonl")
    ap.add_argument("--ir-manifest", default="costModel_v2/data/candidate_ir_manifest.jsonl")
    ap.add_argument("--output-dir", default="costModel_v2/labels")
    ap.add_argument("--expected-operators", type=int, default=50)
    ap.add_argument("--expected-per-operator", type=int, default=5)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    selected_path = resolve_path(root, args.selected_candidates)
    ir_manifest_path = resolve_path(root, args.ir_manifest)
    output_dir = resolve_path(root, args.output_dir)
    if selected_path is None or ir_manifest_path is None or output_dir is None:
        raise ValueError("Invalid input/output path")

    selected_rows = read_jsonl(selected_path)
    ir_rows = read_jsonl(ir_manifest_path)
    selected_ids = [candidate_key(row) for row in selected_rows]
    duplicate_selected_ids = sorted({cid for cid in selected_ids if selected_ids.count(cid) > 1})

    ir_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ir_ids = []
    for row in ir_rows:
        key = candidate_key(row)
        if key in ir_by_id:
            duplicate_ir_ids.append(key)
        ir_by_id[key] = row

    audit_rows = [audit_one(row, ir_by_id.get(candidate_key(row)), root) for row in selected_rows]
    op_counts = Counter(row["operator_name"] for row in selected_rows)
    role_counts = Counter(row["selected_role"] for row in selected_rows)
    audit_status_counts = Counter(row["audit_status"] for row in audit_rows)
    compile_pass_counts = Counter(
        "pass" if row["compile_pass"] is True else
        "fail" if row["compile_pass"] is False else "unknown"
        for row in audit_rows
    )
    category_counts = {
        category: sum(bool(row["ir_categories"][category]["complete"]) for row in audit_rows)
        for category in IR_CATEGORIES
    }
    issue_counts = Counter(issue for row in audit_rows for issue in row["issues"])
    warning_counts = Counter(warning for row in audit_rows for warning in row["warnings"])
    expected_total = args.expected_operators * args.expected_per_operator

    summary = {
        "schema_version": "1.0",
        "selected_candidates_path": str(selected_path),
        "ir_manifest_path": str(ir_manifest_path),
        "candidate_count": len(selected_rows),
        "ir_manifest_row_count": len(ir_rows),
        "expected_candidate_count": expected_total,
        "operator_count": len(op_counts),
        "operator_counts": dict(sorted(op_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "candidate_id_duplicates_in_selected_candidates": duplicate_selected_ids,
        "candidate_id_duplicates_in_ir_manifest": duplicate_ir_ids,
        "expected_role_counts_match": all(role_counts.get(role, 0) == args.expected_operators for role in ROLES),
        "audit_status_counts": dict(audit_status_counts),
        "compile_pass_counts": dict(compile_pass_counts),
        "ir_category_complete_counts": category_counts,
        "full_four_stage_ir_count": sum(row["full_ir_complete"] for row in audit_rows),
        "ttir_available_count": sum(row["ttir_available"] for row in audit_rows),
        "profile_eligible_count": sum(row["profile_eligible"] for row in audit_rows),
        "issue_counts": dict(issue_counts),
        "warning_counts": dict(warning_counts),
        "expected_operators_match": len(op_counts) == args.expected_operators,
        "expected_per_operator_match": all(v == args.expected_per_operator for v in op_counts.values()),
        "expected_candidate_count_match": len(selected_rows) == expected_total,
        "outputs": {
            "audit_jsonl": str(output_dir / "ir_audit.jsonl"),
            "summary_json": str(output_dir / "ir_audit_summary.json"),
        },
    }
    write_jsonl(output_dir / "ir_audit.jsonl", audit_rows)
    write_json(output_dir / "ir_audit_summary.json", summary)

    log("=" * 72)
    log("CostModel V2 IR completeness audit")
    log(f"Selected candidates       : {len(selected_rows)}")
    log(f"IR manifest rows          : {len(ir_rows)}")
    log(f"Operators                 : {len(op_counts)}")
    log(f"Full four-stage IR        : {summary['full_four_stage_ir_count']}")
    log(f"TTIR available            : {summary['ttir_available_count']}")
    log(f"Profiling eligible        : {summary['profile_eligible_count']}")
    log(f"IR category counts        : {category_counts}")
    log(f"Compile pass/fail/unknown : {dict(compile_pass_counts)}")
    log(f"Issues                    : {dict(issue_counts)}")
    log(f"Audit JSONL               : {output_dir / 'ir_audit.jsonl'}")
    log(f"Summary                   : {output_dir / 'ir_audit_summary.json'}")
    log("=" * 72)
    for row in [r for r in audit_rows if not r["profile_eligible"]][:20]:
        log(f"NOT PROFILE-ELIGIBLE: {row['operator_name']} / {row['selected_role']} / {row['candidate_id']} :: {row['issues']} {row['warnings']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import sys
        print(f"[ir-audit] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
