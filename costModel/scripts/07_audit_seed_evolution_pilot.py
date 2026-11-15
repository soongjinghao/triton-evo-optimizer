#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the V2 seed+evolution candidate labels, source/test identity, and TTIR coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROLES = {"dataset_seed", "historical_best", "historical_p10", "historical_p50", "historical_p90"}
SUCCESS_WITH_TTIR = {"success", "compiled_with_test_error"}


def resolve(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--selected-candidates", default="costModel/data/pilot_evo/selected_candidates.jsonl")
    ap.add_argument("--ir-manifest", default="costModel/data/pilot_evo/candidate_ir_manifest.jsonl")
    ap.add_argument("--output", default="costModel/data/pilot_evo/data_collection_audit.json")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    selected_path = resolve(root, args.selected_candidates)
    manifest_path = resolve(root, args.ir_manifest)
    output_path = resolve(root, args.output)
    selected = read_jsonl(selected_path)
    ir_rows = read_jsonl(manifest_path) if manifest_path.is_file() else []

    errors: list[str] = []
    roles = Counter(r.get("selected_role") for r in selected)
    by_op = defaultdict(list)
    for row in selected:
        by_op[str(row.get("operator_name"))].append(row)
        source = resolve(root, str(row.get("source_path") or ""))
        test = resolve(root, str(row.get("test_path") or ""))
        if not source.is_file():
            errors.append(f"{row.get('operator_name')}/{row.get('candidate_id')}: source missing")
        elif sha256_file(source) != row.get("source_sha256"):
            errors.append(f"{row.get('operator_name')}/{row.get('candidate_id')}: source SHA256 mismatch")
        if not test.is_file():
            errors.append(f"{row.get('operator_name')}/{row.get('candidate_id')}: test missing")
        elif row.get("test_sha256") and sha256_file(test) != row.get("test_sha256"):
            errors.append(f"{row.get('operator_name')}/{row.get('candidate_id')}: test SHA256 mismatch")
        try:
            label = float(row.get("label_latency_us"))
            if not (label > 0):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{row.get('operator_name')}/{row.get('candidate_id')}: missing/non-positive runtime label")

    for op, rows in sorted(by_op.items()):
        op_roles = Counter(r.get("selected_role") for r in rows)
        if len(rows) != 5:
            errors.append(f"{op}: expected 5 selected rows, found {len(rows)}")
        if set(op_roles) != EXPECTED_ROLES or any(op_roles[r] != 1 for r in EXPECTED_ROLES):
            errors.append(f"{op}: role composition invalid: {dict(op_roles)}")
        hashes = [r.get("source_sha256") for r in rows]
        if len(set(hashes)) != 5:
            errors.append(f"{op}: selected source hashes are not all distinct")

    ir_by_key = {(str(r.get("operator_name")), str(r.get("source_sha256"))): r for r in ir_rows}
    missing_ir = []
    status_counts = Counter()
    for row in selected:
        key = (str(row.get("operator_name")), str(row.get("source_sha256")))
        ir = ir_by_key.get(key)
        if not ir:
            missing_ir.append(key)
            errors.append(f"{key[0]}/{row.get('candidate_id')}: IR manifest row missing")
            continue
        status = str(ir.get("compile_status") or "unknown")
        status_counts[status] += 1
        if status not in SUCCESS_WITH_TTIR or not (ir.get("ir_found") or {}).get("ttir"):
            errors.append(f"{key[0]}/{row.get('candidate_id')}: TTIR unavailable ({status})")

    invalid_label_count = 0
    for row in selected:
        try:
            value = float(row.get("label_latency_us"))
            if not (value > 0):
                invalid_label_count += 1
        except (TypeError, ValueError):
            invalid_label_count += 1

    summary = {
        "operators": len(by_op),
        "selected_logical_candidates": len(selected),
        "selected_role_counts": dict(roles),
        "unique_ir_manifest_rows": len(ir_rows),
        "ir_status_counts": dict(status_counts),
        "missing_ir_rows": len(missing_ir),
        "label_missing_or_invalid_count": invalid_label_count,
        "error_count": len(errors),
        "errors": errors[:100],
        "complete": len(selected) == 250 and len(by_op) == 50 and len(ir_rows) == 250 and not missing_ir and not errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[pilot-audit] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
