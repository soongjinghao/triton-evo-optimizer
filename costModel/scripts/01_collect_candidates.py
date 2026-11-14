#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a unified Candidate Pool for the Triton Cost Model V1.

Data sources (datasets/ only for the current operator definitions):
  1. Baseline source: datasets/<operator>/<operator>.py
  2. Seed source:     datasets/<operator>/<operator>_1.py
  3. Historical evolution candidates:
       experiments/dataset/samples.jsonl
       experiments/dataset/codes/<candidate_id>.py

Outputs:
  costModel/data/manifests/candidate_pool.jsonl
      One row per logical candidate. Historical records are preserved even
      when their source file is missing; such rows are marked unavailable.
  costModel/data/manifests/candidate_compile_jobs.jsonl
      One row per unique (operator, source SHA256) that has source, JIT code,
      and an associated test file. Exact duplicate source versions are
      represented by one compile job while all logical candidates remain in
      candidate_pool.jsonl.
  costModel/data/manifests/candidate_pool_summary.json
      Counts, coverage, deduplication and warnings.

This script only reads files and parses Python AST. It does not import Triton,
run tests, compile kernels, or execute NPU code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
COST_MODEL_DIR = SCRIPT_DIR.parent
DEFAULT_PROJECT_ROOT = COST_MODEL_DIR.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_jit_functions(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Find common Triton JIT decorator forms without importing the module."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return [], f"{type(exc).__name__}: {exc}"

    triton_aliases: set[str] = set()
    jit_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton":
                    triton_aliases.add(alias.asname or "triton")
        elif isinstance(node, ast.ImportFrom) and node.module == "triton":
            for alias in node.names:
                if alias.name == "jit":
                    jit_aliases.add(alias.asname or alias.name)

    found: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            is_jit = (
                isinstance(target, ast.Name) and target.id in jit_aliases
            ) or (
                isinstance(target, ast.Attribute)
                and target.attr == "jit"
                and isinstance(target.value, ast.Name)
                and target.value.id in triton_aliases
            )
            if is_jit:
                found.append({"name": node.name, "line": getattr(node, "lineno", None)})
                break
    found.sort(key=lambda item: (item["line"] or 0, item["name"]))
    return found, None


def inspect_test_imports(test_path: Path | None) -> tuple[list[str], str | None]:
    if test_path is None or not test_path.is_file():
        return [], None
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return sorted(modules), None


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    if not path.is_file():
        return rows, malformed
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                value["_source_line"] = line_no
                rows.append(value)
            else:
                malformed += 1
    return rows, malformed


def load_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Inventory not found: {path}. Run 00_scan_operators.py first."
        )
    rows, malformed = load_jsonl(path)
    if malformed:
        raise ValueError(f"Inventory contains {malformed} malformed JSONL rows: {path}")
    return rows


def safe_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def candidate_base(
    *,
    candidate_id: str,
    operator_name: str,
    candidate_type: str,
    source_path: Path | None,
    test_path: Path | None,
    project_root: Path,
    source_origin: str,
) -> dict[str, Any]:
    source_exists = bool(source_path and source_path.is_file())
    test_exists = bool(test_path and test_path.is_file())

    if source_exists:
        assert source_path is not None
        jit_functions, parse_error = parse_jit_functions(source_path)
        source_hash = sha256_file(source_path)
        source_size = source_path.stat().st_size
    else:
        jit_functions, parse_error = [], None
        source_hash, source_size = None, None

    if test_exists:
        assert test_path is not None
        test_hash = sha256_file(test_path)
        test_modules, test_parse_error = inspect_test_imports(test_path)
    else:
        test_hash, test_modules, test_parse_error = None, [], None

    target_jit_names = {item["name"] for item in jit_functions}
    exact_target_match = operator_name in target_jit_names
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "operator_name": operator_name,
        "candidate_type": candidate_type,
        "source_origin": source_origin,
        "source_path": relative_path(source_path, project_root),
        "source_available": source_exists,
        "source_sha256": source_hash,
        "source_size_bytes": source_size,
        "jit_functions": jit_functions,
        "jit_function_count": len(jit_functions),
        "target_operator_jit_exact_match": exact_target_match,
        "ast_parse_error": parse_error,
        "test_path": relative_path(test_path, project_root),
        "test_available": test_exists,
        "test_sha256": test_hash,
        "test_import_modules": test_modules,
        "test_imports_expected_module": operator_name in test_modules,
        "test_ast_parse_error": test_parse_error,
        "expected_test_module_name": operator_name,
        "requires_candidate_staging": candidate_type == "historical_evolution",
        "compile_eligible": bool(source_exists and jit_functions and test_exists and not parse_error),
        "compile_status": "not_attempted",
        "compile_group_id": None,
        "is_compile_representative": False,
        "duplicate_of_candidate_id": None,
    }


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def build_candidate_pool(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    project_root = Path(args.project_root).expanduser().resolve()
    inventory_path = resolve_path(project_root, args.inventory_jsonl)
    samples_path = resolve_path(project_root, args.samples_jsonl)
    code_root = resolve_path(project_root, args.candidate_code_dir)
    output_dir = resolve_path(project_root, args.output_dir)

    operators = load_inventory(inventory_path)
    operator_lookup = {str(row["operator_name"]): row for row in operators}
    if not operator_lookup:
        raise ValueError(f"No operators found in inventory: {inventory_path}")

    candidate_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    source_type_counts: Counter[str] = Counter()

    # Add baseline and Seed records from datasets/ inventory.
    for operator_name in sorted(operator_lookup):
        op = operator_lookup[operator_name]
        test_info = op.get("preferred_test_file") or {}
        test_path = resolve_path(project_root, test_info["path"]) if test_info.get("path") else None
        for candidate_type, source_key in (("baseline", "baseline_source"), ("seed", "seed_source")):
            source_info = op.get(source_key) or {}
            if not source_info.get("path"):
                warnings.append({"type": f"missing_{candidate_type}_source", "operator_name": operator_name})
                continue
            source_path = resolve_path(project_root, source_info["path"])
            candidate_id = f"{candidate_type}:{operator_name}"
            record = candidate_base(
                candidate_id=candidate_id,
                operator_name=operator_name,
                candidate_type=candidate_type,
                source_path=source_path,
                test_path=test_path,
                project_root=project_root,
                source_origin="datasets",
            )
            record["inventory_compile_ready"] = bool(op.get("compile_ready"))
            record["test_file_name"] = test_info.get("file_name")
            source_type_counts[candidate_type] += 1
            candidate_rows.append(record)

    # Add every historical evolution row; missing source is retained as metadata.
    historical_rows, malformed_rows = load_jsonl(samples_path)
    if not samples_path.is_file():
        warnings.append({"type": "historical_samples_missing", "path": str(samples_path)})
    if malformed_rows:
        warnings.append({"type": "malformed_historical_rows", "count": malformed_rows})

    seen_historical_ids: set[tuple[str, str]] = set()
    duplicate_historical_rows = 0
    for index, sample in enumerate(historical_rows):
        operator_name = str(sample.get("kernel") or "__missing_operator__")
        raw_candidate_id = sample.get("candidate_id")
        historical_id = str(raw_candidate_id) if raw_candidate_id is not None else f"missing_id_line_{sample.get('_source_line', index + 1)}"
        dedup_key = (operator_name, historical_id)
        if dedup_key in seen_historical_ids:
            duplicate_historical_rows += 1
            continue
        seen_historical_ids.add(dedup_key)

        if args.limit_history is not None and source_type_counts["historical_evolution"] >= args.limit_history:
            continue

        source_path = code_root / f"{historical_id}.py" if raw_candidate_id is not None else None
        op = operator_lookup.get(operator_name, {})
        test_info = op.get("preferred_test_file") or {}
        test_path = resolve_path(project_root, test_info["path"]) if test_info.get("path") else None

        record = candidate_base(
            candidate_id=historical_id,
            operator_name=operator_name,
            candidate_type="historical_evolution",
            source_path=source_path,
            test_path=test_path,
            project_root=project_root,
            source_origin="experiments/dataset/codes",
        )
        record.update({
            "historical_sample": {
                "source_file": args.samples_jsonl,
                "source_line": sample.get("_source_line"),
                "method": sample.get("method"),
                "run_id": sample.get("run_id"),
                "tau": sample.get("tau"),
                "generation": sample.get("gen"),
                "operation": sample.get("operation"),
                "mutation_type": sample.get("mutation_type"),
                "parent_ids": sample.get("parent_ids"),
                "parent_latency": sample.get("parent_latency"),
                "child_latency": sample.get("child_latency"),
                "seed_time": sample.get("seed_time"),
                "g_mut": sample.get("g_mut"),
                "label": sample.get("label"),
                "applied_strategy": sample.get("applied_strategy"),
                "child_profile": sample.get("child_profile"),
                "model": sample.get("model"),
                "timestamp": sample.get("timestamp"),
                "latency_provenance": "historical_samples_jsonl",
                "is_current_profile": False,
                "parent_latency_valid_positive": safe_positive_float(sample.get("parent_latency")) is not None,
                "child_latency_valid_positive": safe_positive_float(sample.get("child_latency")) is not None,
            },
            "test_file_name": test_info.get("file_name"),
        })
        source_type_counts["historical_evolution"] += 1
        candidate_rows.append(record)

    # Exact-source deduplication is scoped to an operator: identical code for
    # different operators is not assumed to have the same test/config context.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        source_hash = row.get("source_sha256")
        if source_hash:
            grouped[(row["operator_name"], source_hash)].append(row)

    compile_jobs: list[dict[str, Any]] = []
    duplicate_source_records = 0
    for (operator_name, source_hash), group in grouped.items():
        # Candidate rows are ordered baseline -> seed -> history, so this picks
        # the most stable canonical representative when source is identical.
        representative = group[0]
        group_id = f"{operator_name}:{source_hash[:16]}"
        representative["compile_group_id"] = group_id
        representative["is_compile_representative"] = True
        representative["duplicate_of_candidate_id"] = None
        for duplicate in group[1:]:
            duplicate["compile_group_id"] = group_id
            duplicate["is_compile_representative"] = False
            duplicate["duplicate_of_candidate_id"] = representative["candidate_id"]
            duplicate_source_records += 1

        if representative.get("compile_eligible"):
            compile_jobs.append({
                "schema_version": "1.0",
                "compile_job_id": f"compile:{group_id}",
                "operator_name": operator_name,
                "representative_candidate_id": representative["candidate_id"],
                "candidate_type": representative["candidate_type"],
                "source_path": representative["source_path"],
                "source_sha256": source_hash,
                "test_path": representative["test_path"],
                "test_sha256": representative["test_sha256"],
                "expected_test_module_name": representative["expected_test_module_name"],
                "test_import_modules": representative["test_import_modules"],
                "test_imports_expected_module": representative["test_imports_expected_module"],
                "jit_functions": representative["jit_functions"],
                "requires_candidate_staging": representative["requires_candidate_staging"],
                "logical_candidate_ids": [item["candidate_id"] for item in group],
                "logical_candidate_count": len(group),
                "compile_status": "pending",
            })

    compile_jobs.sort(key=lambda row: (row["operator_name"], row["compile_job_id"]))
    candidate_rows.sort(key=lambda row: (
        row["operator_name"],
        {"baseline": 0, "seed": 1, "historical_evolution": 2}.get(row["candidate_type"], 9),
        row["candidate_id"],
    ))

    total_by_type = Counter(row["candidate_type"] for row in candidate_rows)
    source_by_type = Counter(row["candidate_type"] for row in candidate_rows if row["source_available"])
    compile_eligible_by_type = Counter(row["candidate_type"] for row in candidate_rows if row["compile_eligible"])
    missing_source_by_type = Counter(row["candidate_type"] for row in candidate_rows if not row["source_available"])
    jit_missing_by_type = Counter(row["candidate_type"] for row in candidate_rows if row["source_available"] and row["jit_function_count"] == 0)
    operator_sample_counts = Counter(row["operator_name"] for row in candidate_rows if row["candidate_type"] == "historical_evolution")
    historical_code_found = sum(1 for row in candidate_rows if row["candidate_type"] == "historical_evolution" and row["source_available"])
    historical_target_match = sum(1 for row in candidate_rows if row["candidate_type"] == "historical_evolution" and row["target_operator_jit_exact_match"])
    historical_jit_without_exact_target = sum(1 for row in candidate_rows if row["candidate_type"] == "historical_evolution" and row["source_available"] and row["jit_function_count"] > 0 and not row["target_operator_jit_exact_match"])
    historical_missing_sources = missing_source_by_type.get("historical_evolution", 0)
    if historical_missing_sources:
        warnings.append({
            "type": "historical_candidate_source_missing",
            "count": historical_missing_sources,
            "message": "历史样本存在记录，但对应 experiments/dataset/codes/<candidate_id>.py 文件缺失；这些记录保留在候选池中，但不会生成编译任务。",
        })
    if historical_jit_without_exact_target:
        warnings.append({
            "type": "historical_jit_name_not_exact",
            "count": historical_jit_without_exact_target,
            "message": "部分历史候选包含 Triton JIT 函数，但函数名与算子目录名不完全相同；已保留，后续需要结合测试入口确认。",
        })

    summary = {
        "schema_version": "1.0",
        "project_root": str(project_root),
        "primary_dataset": "datasets",
        "policy": {
            "include_baseline": True,
            "include_seed": True,
            "include_historical_evolution": True,
            "historical_latency_is_current_label": False,
            "deduplicate_compile_jobs_by_operator_and_source_sha256": True,
            "missing_historical_sources_are_preserved_in_pool": True,
        },
        "input_paths": {
            "inventory_jsonl": relative_path(inventory_path, project_root),
            "samples_jsonl": relative_path(samples_path, project_root),
            "candidate_code_dir": relative_path(code_root, project_root),
        },
        "operator_count_in_inventory": len(operator_lookup),
        "candidate_record_count": len(candidate_rows),
        "candidate_counts_by_type": dict(total_by_type),
        "source_available_counts_by_type": dict(source_by_type),
        "missing_source_counts_by_type": dict(missing_source_by_type),
        "compile_eligible_counts_by_type": dict(compile_eligible_by_type),
        "source_without_jit_counts_by_type": dict(jit_missing_by_type),
        "historical_sample_rows_read": len(historical_rows),
        "historical_malformed_rows": malformed_rows,
        "duplicate_historical_rows_skipped": duplicate_historical_rows,
        "historical_candidate_sources_found": historical_code_found,
        "historical_candidate_sources_missing": historical_missing_sources,
        "historical_candidates_with_exact_operator_jit_name": historical_target_match,
        "historical_candidates_with_jit_but_no_exact_operator_name": historical_jit_without_exact_target,
        "unique_source_compile_group_count": len(grouped),
        "duplicate_source_records": duplicate_source_records,
        "compile_job_count": len(compile_jobs),
        "compile_jobs_by_type": dict(Counter(row["candidate_type"] for row in compile_jobs)),
        "operators_with_historical_candidates": len(operator_sample_counts),
        "historical_candidates_per_operator": dict(sorted(operator_sample_counts.items())),
        "warnings": warnings,
        "outputs": {
            "candidate_pool": relative_path(output_dir / "candidate_pool.jsonl", project_root),
            "compile_jobs": relative_path(output_dir / "candidate_compile_jobs.jsonl", project_root),
            "summary": relative_path(output_dir / "candidate_pool_summary.json", project_root),
        },
        "notes": [
            "candidate_pool.jsonl preserves logical historical candidate records, including records whose source file is missing.",
            "candidate_compile_jobs.jsonl contains one representative per exact source hash within an operator, and only if source, JIT and test are present.",
            "Historical parent_latency/child_latency/label are retained as provenance and must not be treated as fresh profiling labels.",
            "Historical candidate source files need to be staged under the expected module name before running the operator test; this script does not compile them.",
            "One source file can contain multiple Triton JIT functions and can produce multiple compiled kernel instances during a test.",
        ],
    }
    return candidate_rows, compile_jobs, summary


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Baseline, Seed and historical evolution candidates for Cost Model V1."
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="Agent project root")
    parser.add_argument("--inventory-jsonl", default="costModel/data/manifests/operator_inventory.jsonl")
    parser.add_argument("--samples-jsonl", default="experiments/dataset/samples.jsonl")
    parser.add_argument("--candidate-code-dir", default="experiments/dataset/codes")
    parser.add_argument("--output-dir", default="costModel/data/manifests")
    parser.add_argument("--limit-history", type=int, default=None, help="Optional limit for smoke testing")
    parser.add_argument("--dry-run", action="store_true", help="Build statistics but do not write output files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit_history is not None and args.limit_history < 0:
        print("[collect-candidates] ERROR: --limit-history must be >= 0", file=sys.stderr)
        return 2
    try:
        candidate_rows, compile_jobs, summary = build_candidate_pool(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[collect-candidates] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = resolve_path(project_root, args.output_dir)
    if not args.dry_run:
        pool_path = output_dir / "candidate_pool.jsonl"
        jobs_path = output_dir / "candidate_compile_jobs.jsonl"
        summary_path = output_dir / "candidate_pool_summary.json"
        atomic_write_jsonl(pool_path, candidate_rows)
        atomic_write_jsonl(jobs_path, compile_jobs)
        atomic_write_json(summary_path, summary)
    else:
        pool_path = output_dir / "candidate_pool.jsonl"
        jobs_path = output_dir / "candidate_compile_jobs.jsonl"
        summary_path = output_dir / "candidate_pool_summary.json"

    print("=" * 76)
    print("Cost Model candidate pool collection completed")
    print(f"Project root                       : {summary['project_root']}")
    print(f"Primary dataset                    : {summary['primary_dataset']}")
    print(f"Operators                          : {summary['operator_count_in_inventory']}")
    print(f"Baseline candidates                : {summary['candidate_counts_by_type'].get('baseline', 0)}")
    print(f"Seed candidates                    : {summary['candidate_counts_by_type'].get('seed', 0)}")
    print(f"Historical evolution candidates    : {summary['candidate_counts_by_type'].get('historical_evolution', 0)}")
    print(f"Historical candidate sources found: {summary['historical_candidate_sources_found']}")
    print(f"Historical candidate sources missing: {summary['historical_candidate_sources_missing']}")
    print(f"Unique source compile groups       : {summary['unique_source_compile_group_count']}")
    print(f"Deduplicated source records        : {summary['duplicate_source_records']}")
    print(f"Compile-eligible unique jobs       : {summary['compile_job_count']}")
    print(f"Compile jobs by type               : {summary['compile_jobs_by_type']}")
    if args.dry_run:
        print("Mode                               : DRY RUN (no files written)")
    else:
        print(f"Candidate pool                     : {pool_path}")
        print(f"Compile jobs                       : {jobs_path}")
        print(f"Summary                            : {summary_path}")
    print(f"Warnings                           : {len(summary['warnings'])}")
    for warning in summary["warnings"][:20]:
        print(f"  - {warning}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
