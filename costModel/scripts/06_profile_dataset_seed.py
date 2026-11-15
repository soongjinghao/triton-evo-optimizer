#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile selected datasets seed candidates whose exact manifest labels are unavailable.

Uses the project's existing TritonExecutor + msprof-op measurement path. By
 default it only measures rows marked pending_profile (expected: the four seed
sources whose current datasets source hash differs from datasets2). Use
--reprofile-all to remeasure all 50 dataset_seed candidates in one fresh batch.

Successful median labels are written back into selected_candidates.jsonl/csv.
This script measures runtime; it is NOT a warmup-only IR collector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EAConfig  # noqa: E402
from executor import TritonExecutor  # noqa: E402


class DatasetSeedExecutor(TritonExecutor):
    """Reuse executor staging but also copy non-kernel helper modules beside the test."""
    def _prepare_test_dir(self, code: str, device_id: int = 0):
        temp_dir, test_file = super()._prepare_test_dir(code, device_id)
        support_root = self.test_code_path.parent
        for src in support_root.rglob("*.py"):
            if "__pycache__" in src.parts or src.name.lower().startswith("test_"):
                continue
            if src.name == f"{self.kernel_name}.py":
                continue
            if src.stem.startswith(self.kernel_name + "_") and src.stem[len(self.kernel_name) + 1:].isdigit():
                continue
            rel = src.relative_to(support_root)
            dst = temp_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return temp_dir, test_file


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["operator_name", "candidate_id", "candidate_type", "selected_role", "label_latency_us",
              "label_status", "latency_unit", "latency_source", "measurement_provenance",
              "measurement_count", "is_current_profile", "source_sha256", "source_path",
              "test_path", "compile_job_id"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--selected-candidates", default="costModel/data/pilot_evo/selected_candidates.jsonl")
    ap.add_argument("--output-dir", default="costModel/artifacts/profiling/dataset_seed")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="0 means all pending candidates")
    ap.add_argument("--reprofile-all", action="store_true", help="Remeasure all 50 dataset_seed candidates")
    ap.add_argument("--force", action="store_true", help="Ignore prior successful profile records")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    selected_path = Path(args.selected_candidates).expanduser()
    selected_path = selected_path.resolve() if selected_path.is_absolute() else (root / selected_path).resolve()
    out_dir = Path(args.output_dir).expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (root / out_dir).resolve()
    if args.repeats < 1 or args.limit < 0 or args.timeout < 1:
        raise ValueError("repeats/timeout must be positive and limit must be >= 0")

    rows = read_jsonl(selected_path)
    seed_rows = [r for r in rows if r.get("selected_role") == "dataset_seed"]
    if len(seed_rows) != 50:
        raise ValueError(f"Expected 50 dataset_seed rows, found {len(seed_rows)}")

    records_path = out_dir / "dataset_seed_profile_records.jsonl"
    prior = read_jsonl(records_path) if records_path.is_file() else []
    prior_by_key = {(r.get("operator_name"), r.get("source_sha256")): r for r in prior}

    pending = []
    for row in seed_rows:
        key = (row.get("operator_name"), row.get("source_sha256"))
        old = prior_by_key.get(key)
        if not args.force and old and old.get("success") and old.get("label_latency_us"):
            continue
        if args.reprofile_all or row.get("label_status") in {"pending_profile", "profile_failed"}:
            pending.append(row)
    if args.limit:
        pending = pending[:args.limit]

    print(f"Dataset seed rows       : {len(seed_rows)}")
    print(f"Need profiling this run : {len(pending)}")
    print(f"Repeats per candidate  : {args.repeats}")
    print(f"Device                 : npu:{args.device_id}")
    print(f"Output                 : {records_path}")
    if args.dry_run:
        for r in pending:
            print(f"  {r['operator_name']} | {r['source_path']} | {r['source_sha256'][:16]}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    profile_rows = list(prior)
    for i, row in enumerate(pending, 1):
        op = str(row["operator_name"])
        source = Path(str(row["source_path"]))
        test = Path(str(row["test_path"]))
        if not source.is_absolute():
            source = root / source
        if not test.is_absolute():
            test = root / test
        if not source.is_file() or not test.is_file():
            raise FileNotFoundError(f"{op}: source/test missing: {source} / {test}")
        actual_source_hash = sha256_file(source)
        actual_test_hash = sha256_file(test)
        if actual_source_hash != row.get("source_sha256"):
            raise ValueError(f"{op}: source SHA256 mismatch; selection is stale")
        if row.get("test_sha256") and actual_test_hash != row.get("test_sha256"):
            raise ValueError(f"{op}: test SHA256 mismatch; selection is stale")

        print(f"\n[{i}/{len(pending)}] Profiling {op}", flush=True)
        config = EAConfig()
        config.msprof_backend = "op"
        config.pre_compile_check = False
        work_dir = out_dir / "work" / op
        executor = DatasetSeedExecutor(
            baseline_time=1.0,  # speedup is unused; only execution_time is the label
            test_code_path=str(test),
            config=config,
            kernel_name=op,
            work_dir=work_dir,
        )
        code = source.read_text(encoding="utf-8")
        values: list[float] = []
        errors: list[str] = []
        consecutive_failures = 0
        for repeat_idx in range(args.repeats):
            result = executor.evaluate(code, timeout=args.timeout, device_id=args.device_id)
            if result.success and math.isfinite(float(result.execution_time)) and result.execution_time > 0:
                values.append(float(result.execution_time))
                consecutive_failures = 0
                print(f"  repeat {repeat_idx + 1}/{args.repeats}: {result.execution_time:.6f} us", flush=True)
            else:
                errors.append(str(result.error or "profile failed"))
                consecutive_failures += 1
                print(f"  repeat {repeat_idx + 1}/{args.repeats}: FAILED {result.error}", flush=True)
                if consecutive_failures >= 2:
                    break

        # A partial set is useful diagnostic evidence, but it is not a complete
        # five-repeat label. Keep it retryable instead of silently accepting a
        # median based on only one or two successful measurements.
        success = len(values) == args.repeats
        median = float(statistics.median(values)) if success else None
        if not success and values:
            errors.append(
                f"incomplete_profile: {len(values)}/{args.repeats} successful measurements; label not committed"
            )
        rec = {
            "schema_version": "1.0",
            "operator_name": op,
            "candidate_id": row.get("candidate_id"),
            "selected_role": "dataset_seed",
            "source_path": row.get("source_path"),
            "source_sha256": actual_source_hash,
            "test_path": row.get("test_path"),
            "test_sha256": actual_test_hash,
            "success": success,
            "label_latency_us": median,
            "latency_unit": "us",
            "measurement_values_us": values,
            "measurement_count": len(values),
            "requested_repeats": args.repeats,
            "device_id": args.device_id,
            "measurement_provenance": "current_dataset_seed_msprof_op_reprofile",
            "errors": errors,
        }
        key = (op, actual_source_hash)
        profile_rows = [r for r in profile_rows if (r.get("operator_name"), r.get("source_sha256")) != key]
        profile_rows.append(rec)
        profile_rows.sort(key=lambda r: (str(r.get("operator_name")), str(r.get("source_sha256"))))
        write_jsonl(records_path, profile_rows)

        # Update the selected row immediately so an interrupted batch is resumable.
        for selected in rows:
            if selected.get("selected_role") == "dataset_seed" and selected.get("operator_name") == op:
                if success:
                    selected.update({
                        "label_latency_us": median,
                        "label_status": "profiled_current_source",
                        "latency_source": str(records_path.relative_to(root)) if records_path.is_relative_to(root) else str(records_path),
                        "measurement_provenance": rec["measurement_provenance"],
                        "measurement_count": len(values),
                        "measurement_values_us": values,
                        "measurement_timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                        "device_id": args.device_id,
                        "is_current_profile": True,
                    })
                else:
                    selected.update({"label_status": "profile_failed", "profile_errors": errors})
        write_jsonl(selected_path, rows)
        write_csv(selected_path.with_suffix(".csv"), rows)
        print(f"  => {'OK' if success else 'FAILED'} median={median}", flush=True)

    current_seed_rows = [r for r in rows if r.get("selected_role") == "dataset_seed"]
    pending_seed_count = sum(1 for r in current_seed_rows if not (isinstance(r.get("label_latency_us"), (int, float)) and r.get("label_latency_us", 0) > 0))
    current_profiled_count = sum(1 for r in current_seed_rows if r.get("measurement_provenance") == "current_dataset_seed_msprof_op_reprofile" and isinstance(r.get("label_latency_us"), (int, float)) and r.get("label_latency_us", 0) > 0)
    summary = {
        "dataset_seed_count": len(seed_rows),
        "profiled_this_run": len(pending),
        "successful_records": sum(1 for r in profile_rows if r.get("success")),
        "failed_records": sum(1 for r in profile_rows if not r.get("success")),
        "pending_seed_labels": pending_seed_count,
        "current_reprofiled_seed_labels": current_profiled_count,
        "repeats": args.repeats,
        "device_id": args.device_id,
        "records_path": str(records_path),
        "selected_candidates_path": str(selected_path),
    }
    selection_summary_path = selected_path.parent / "selection_summary.json"
    if selection_summary_path.is_file():
        selection_summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))
        selection_counts = selection_summary.setdefault("counts", {})
        selection_counts["dataset_seed_labels_pending_profile"] = pending_seed_count
        selection_counts["dataset_seed_labels_profiled_current"] = current_profiled_count
        selection_counts["dataset_seed_labels_reused"] = len(seed_rows) - pending_seed_count - current_profiled_count
        selection_summary_path.write_text(json.dumps(selection_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "dataset_seed_profile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[dataset-seed-profile] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
