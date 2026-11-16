#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare CostModel V2 compile jobs and safely reuse previously collected IR.

Expected input:
  costModel_v2/data/candidate_manifest.jsonl
  costModel_v2/data/<operator>/*.py                 (or paths recorded in manifest)
  costModel/data/manifests/operator_inventory.jsonl (preferred test-file map)

Outputs:
  costModel_v2/data/selected_candidates.jsonl
  costModel_v2/data/compile_jobs.jsonl
  costModel_v2/data/compile_jobs_summary.json
  costModel_v2/ir/{ttir,ttadapter,npuir,npubin,metadata}/...

Reuse is deliberately strict: an old result is reusable only when operator,
source SHA256, test SHA256 and an existing TTIR artifact all match. Historical
latency/labels are not read or modified by this script.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT = Path(__file__).resolve()
V2_DIR = SCRIPT.parents[1]
ROOT = V2_DIR.parent
DEFAULT_MANIFEST = V2_DIR / "data" / "candidate_manifest.jsonl"
DEFAULT_INVENTORY = ROOT / "costModel" / "data" / "manifests" / "operator_inventory.jsonl"
DEFAULT_IR_DIR = V2_DIR / "ir"
ROLE_NAMES = (
    "dataset_seed", "historical_best", "historical_p10",
    "historical_p50", "historical_p90",
)
IR_CATEGORIES = ("ttir", "ttadapter", "npuir", "npubin")
SUCCESS_STATUSES = {"success", "compiled_with_test_error"}


def log(msg: str) -> None:
    print(f"[prepare-jobs] {msg}", flush=True)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip(".") or "unnamed"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        if p.exists():
            return p.resolve()
        # Recover paths recorded on the same HiDevLab mount but with a different
        # project-root prefix, e.g. /workspace/Agent/...
        parts = list(p.parts)
        if "Agent" in parts:
            idx = parts.index("Agent")
            candidate = root.joinpath(*parts[idx + 1:])
            if candidate.exists():
                return candidate.resolve()
        return p
    return (root / p).resolve()


def walk_values(obj: Any, wanted: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in wanted and value not in (None, ""):
                found.append(value)
            found.extend(walk_values(value, wanted))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(walk_values(value, wanted))
    return found


def first_value(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    values = walk_values(row, {k.lower() for k in keys})
    return values[0] if values else default


def role_for(row: dict[str, Any]) -> str:
    keys = {
        "selected_role", "candidate_role", "role", "selection_role",
        "candidate_type", "source_role", "variant", "candidate_variant",
    }
    vals = walk_values(row, keys)
    text = " ".join(str(v).lower() for v in vals)
    for role in ROLE_NAMES:
        if role in text:
            return role
    # Candidate IDs and filenames often carry the role even if the manifest
    # uses a generic field name.
    text += " " + " ".join(str(v).lower() for v in walk_values(
        row, {"candidate_id", "candidate_file", "source_file", "file_name", "filename", "name"}
    ))
    for role in ROLE_NAMES:
        if role in text:
            return role
    return "unknown"


def candidate_id_for(row: dict[str, Any], operator: str, role: str, index: int) -> str:
    value = first_value(row, {"candidate_id", "candidate_key", "selection_id", "record_id", "id"})
    if value:
        return str(value)
    return f"{operator}:{role}:{index:03d}"


def operator_for(row: dict[str, Any]) -> str:
    value = first_value(row, {"operator_name", "operator", "kernel_name", "kernel"})
    return str(value).strip() if value else ""


def candidate_path_values(row: dict[str, Any]) -> list[str]:
    keys = {
        "source_path", "candidate_source_path", "candidate_path", "source_file_path",
        "code_path", "file_path", "source_file", "candidate_file", "file_name", "filename",
    }
    values = walk_values(row, keys)
    source_obj = first_value(row, {"source", "candidate_source", "source_info"})
    if isinstance(source_obj, dict):
        for key in ("path", "source_path", "file_path", "file_name", "filename", "name"):
            if source_obj.get(key):
                values.insert(0, source_obj[key])
    return [str(v) for v in values if isinstance(v, (str, Path)) and str(v).strip()]


def resolve_source(
    row: dict[str, Any], operator: str, role: str, root: Path, data_dir: Path,
    hash_cache: dict[Path, str],
) -> tuple[Path, str]:
    expected_hash = first_value(row, {"source_sha256", "sha256", "code_sha256", "content_sha256"})
    expected_hash = str(expected_hash).lower() if expected_hash else ""
    candidates: list[Path] = []

    for value in candidate_path_values(row):
        p = resolve_path(root, value)
        if p.is_file():
            candidates.append(p)
        else:
            # A basename-only source_file is common in compact manifests.
            if Path(value).name == value:
                candidates.extend(data_dir.rglob(value))
            else:
                p2 = (data_dir / value).resolve()
                if p2.is_file():
                    candidates.append(p2)
                p3 = (data_dir / operator / value).resolve()
                if p3.is_file():
                    candidates.append(p3)

    # If the manifest's path is absent or stale, exact source hash is the
    # authoritative locator within this selected data snapshot.
    op_dir = data_dir / operator
    search_root = op_dir if op_dir.is_dir() else data_dir
    if expected_hash:
        matched = []
        for p in search_root.rglob("*.py"):
            try:
                actual = hash_cache.setdefault(p, sha256_file(p))
            except OSError:
                continue
            if actual.lower() == expected_hash:
                matched.append(p)
        if matched:
            candidates = matched + candidates

    # If no path/hash locator worked, try role and candidate-id file stems.
    if not candidates:
        cid = str(first_value(row, {"candidate_id", "candidate_key", "selection_id", "record_id", "id"}, ""))
        py_files = list(search_root.rglob("*.py")) if search_root.exists() else []
        role_matches = [p for p in py_files if role != "unknown" and role.lower() in p.stem.lower()]
        id_matches = [p for p in py_files if cid and cid.lower() in p.stem.lower()]
        candidates = id_matches or role_matches

    # Deduplicate paths, then enforce the hash if one was supplied.
    unique = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if expected_hash:
        unique = [p for p in unique if hash_cache.setdefault(p, sha256_file(p)).lower() == expected_hash]
    if not unique:
        detail = f"operator={operator}, role={role}, source_sha256={expected_hash or '<missing>'}"
        raise FileNotFoundError(f"Cannot resolve candidate source ({detail}); check candidate_manifest.jsonl and data/<operator>/")
    if len(unique) > 1:
        # Exact-hash duplicates within one operator are equivalent source files;
        # use the shortest deterministic path and record the canonical source.
        unique.sort(key=lambda p: (len(str(p)), str(p)))
    source = unique[0]
    actual_hash = hash_cache.setdefault(source, sha256_file(source))
    return source, actual_hash


def load_inventory(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    for row in read_jsonl(path):
        op = operator_for(row)
        if op:
            result[op] = row
    return result


def resolve_test(
    row: dict[str, Any], operator: str, root: Path, inventory: dict[str, dict[str, Any]],
    allow_hash_mismatch: bool = False,
) -> tuple[Path, str, str]:
    values = walk_values(row, {"test_path", "test_file_path", "preferred_test_path", "test_file"})
    test_obj = first_value(row, {"test_file", "preferred_test_file", "test_info"})
    if isinstance(test_obj, dict):
        for key in ("path", "test_path", "file_path", "file_name", "filename", "name"):
            if test_obj.get(key):
                values.insert(0, test_obj[key])
    candidates: list[Path] = []
    for value in values:
        if not isinstance(value, str):
            continue
        p = resolve_path(root, value)
        if p.is_file():
            candidates.append(p)
        elif Path(value).name == value:
            candidates.extend(root.glob(f"datasets/{operator}/{value}"))
            candidates.extend(root.glob(f"**/{value}"))

    inv = inventory.get(operator, {})
    test_info = inv.get("preferred_test_file") or {}
    if isinstance(test_info, dict) and test_info.get("path"):
        p = resolve_path(root, test_info["path"])
        if p.is_file():
            candidates.append(p)
    for item in inv.get("test_files") or []:
        if isinstance(item, dict) and item.get("path"):
            p = resolve_path(root, item["path"])
            if p.is_file():
                candidates.append(p)

    if not candidates:
        op_dir = root / "datasets" / operator
        if op_dir.is_dir():
            found = sorted(op_dir.glob("test*.py"))
            if not found:
                found = sorted(op_dir.rglob("test*.py"))
            candidates.extend(found[:1])
    unique = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if not unique:
        raise FileNotFoundError(f"No test file found for operator {operator}; checked candidate manifest, operator_inventory.jsonl and datasets/{operator}/")

    preferred_path = test_info.get("path") if isinstance(test_info, dict) else None
    if preferred_path:
        preferred = resolve_path(root, preferred_path)
        if preferred in unique:
            test_path = preferred
        else:
            test_path = unique[0]
    else:
        unique.sort(key=lambda p: (0 if p.name.startswith("test_") else 1, len(str(p)), str(p)))
        test_path = unique[0]

    declared_hash = first_value(row, {"test_sha256"})
    if not declared_hash and isinstance(test_info, dict):
        declared_hash = test_info.get("sha256")
    actual_hash = sha256_file(test_path)
    if declared_hash and str(declared_hash).lower() != actual_hash.lower():
        message = (
            f"Test SHA256 mismatch for {operator}: manifest/inventory={declared_hash}, "
            f"actual={actual_hash}, file={test_path}"
        )
        if not allow_hash_mismatch:
            raise ValueError(message + ". Review the test file before compiling, or pass --allow-test-hash-mismatch deliberately.")
        log("WARNING: " + message + "; using the current test file hash.")
    module_name = first_value(row, {"expected_test_module_name", "test_module_name"}) or operator
    if isinstance(test_info, dict) and not first_value(row, {"expected_test_module_name", "test_module_name"}):
        module_name = operator
    return test_path, actual_hash, str(module_name)


def parse_jit_functions(source_path: Path) -> list[dict[str, Any]]:
    """Return Triton JIT functions, including call-style decorators.

    Both ``@triton.jit`` and ``@triton.jit(...)`` are valid. In the latter
    form the decorator AST node is an ``ast.Call`` and its target is stored
    in ``node.func``; inspecting the Call node directly misses the JIT.
    """
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    result: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        decorated = False
        for dec in node.decorator_list:
            # Handle @triton.jit(...), @jit(...), as well as bare decorators.
            target = dec.func if isinstance(dec, ast.Call) else dec
            names: list[str] = []
            cur = target
            while isinstance(cur, ast.Attribute):
                names.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                names.append(cur.id)

            # ``triton.jit`` and ``from triton import jit`` both end in jit.
            if names and names[0] == "jit":
                decorated = True
                break

        if decorated:
            result.append({"name": node.name, "line": getattr(node, "lineno", None)})

    result.sort(key=lambda x: (x["line"] or 0, x["name"]))
    return result


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def collect_reuse_records(reuse_roots: list[Path], reuse_manifests: list[Path]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Collect only metadata that can be tied to exact source + test hashes."""
    records: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    def add(row: dict[str, Any], origin: Path) -> None:
        op = str(row.get("operator_name") or "").strip()
        sh = str(row.get("source_sha256") or "").lower()
        th = str(row.get("test_sha256") or "").lower()
        if not (op and sh and th):
            return
        ir_paths = row.get("ir_paths") or {}
        if not isinstance(ir_paths, dict):
            return
        valid_paths: dict[str, list[str]] = {}
        for category in IR_CATEGORIES:
            values = ir_paths.get(category) or []
            if isinstance(values, str):
                values = [values]
            existing = [str(Path(v).expanduser()) for v in values if Path(str(v)).is_file()]
            if existing:
                valid_paths[category] = existing
        if not valid_paths.get("ttir"):
            return
        if row.get("compile_pass") is False:
            return
        if row.get("status") and row.get("status") not in SUCCESS_STATUSES:
            return
        records[(op, sh, th)].append({"row": row, "origin": str(origin), "ir_paths": valid_paths})

    for root in reuse_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            if "metadata" not in p.parts and p.name not in {"compile_summary.json"}:
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    add(obj, p)
            except Exception:
                continue
        for p in root.rglob("compile_results.jsonl"):
            try:
                for row in read_jsonl(p):
                    add(row, p)
            except Exception:
                pass

    for p in reuse_manifests:
        if not p.is_file():
            continue
        try:
            for row in read_jsonl(p):
                meta_path = row.get("compile_metadata_path") or row.get("metadata_path")
                if meta_path:
                    mp = resolve_path(ROOT, str(meta_path))
                    if mp.is_file():
                        try:
                            meta = json.loads(mp.read_text(encoding="utf-8"))
                            if isinstance(meta, dict):
                                add(meta, mp)
                        except Exception:
                            pass
                add(row, p)
        except Exception:
            continue

    for key in records:
        records[key].sort(key=lambda item: sum(len(v) for v in item["ir_paths"].values()), reverse=True)
    return records


def metadata_path_for(job: dict[str, Any], ir_dir: Path) -> Path:
    return ir_dir / "metadata" / safe_name(job["operator_name"]) / (safe_name(job["compile_job_id"]) + ".json")


def copy_reused_ir(job: dict[str, Any], reuse_record: dict[str, Any], ir_dir: Path) -> dict[str, Any]:
    op_dir = safe_name(job["operator_name"])
    job_key = safe_name(job["compile_job_id"])
    ir_paths: dict[str, list[str]] = {category: [] for category in IR_CATEGORIES}
    ir_found = {category: False for category in IR_CATEGORIES}
    for category, source_values in reuse_record["ir_paths"].items():
        target_dir = ir_dir / category / op_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, source_value in enumerate(source_values):
            source = Path(source_value)
            if not source.is_file():
                continue
            if category == "ttir":
                suffix = ".ttir.mlir"
            elif category == "ttadapter":
                suffix = ".ttadapter.mlir"
            elif category == "npuir":
                suffix = ".npuir.mlir"
            else:
                suffix = source.suffix or ".npubin"
            target = target_dir / f"{job_key}__reuse_k{index:02d}{suffix}"
            shutil.copy2(source, target)
            ir_paths[category].append(str(target.resolve()))
        ir_found[category] = bool(ir_paths[category])

    metadata = {
        **job,
        "status": "success",
        "compile_pass": True,
        "kernel_executed": False,
        "compile_only_patch": True,
        "reused": True,
        "reuse_source_metadata": reuse_record["origin"],
        "reuse_source_compile_job_id": reuse_record["row"].get("compile_job_id"),
        "ir_paths": ir_paths,
        "ir_found": ir_found,
        "ir_counts": {k: len(v) for k, v in ir_paths.items()},
    }
    meta_path = metadata_path_for(job, ir_dir)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(meta_path.resolve())
    return metadata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT), help="Agent project root")
    ap.add_argument("--manifest", default="costModel_v2/data/candidate_manifest.jsonl")
    ap.add_argument("--inventory", default="costModel/data/manifests/operator_inventory.jsonl")
    ap.add_argument("--output-dir", default="costModel_v2/data")
    ap.add_argument("--ir-dir", default="costModel_v2/ir")
    ap.add_argument("--reuse-root", action="append", default=None,
                    help="Old IR root; repeatable. Default: costModel/artifacts/ir")
    ap.add_argument("--reuse-manifest", action="append", default=None,
                    help="Old candidate IR manifest; repeatable. Default: costModel/data/pilot/candidate_ir_manifest.jsonl")
    ap.add_argument("--expected-operators", type=int, default=50)
    ap.add_argument("--expected-per-operator", type=int, default=5)
    ap.add_argument("--allow-test-hash-mismatch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    manifest_path = resolve_path(root, args.manifest)
    inventory_path = resolve_path(root, args.inventory)
    output_dir = resolve_path(root, args.output_dir)
    ir_dir = resolve_path(root, args.ir_dir)
    reuse_roots = [resolve_path(root, x) for x in (args.reuse_root or ["costModel/artifacts/ir"])]
    reuse_manifests = [resolve_path(root, x) for x in (args.reuse_manifest or ["costModel/data/pilot/candidate_ir_manifest.jsonl"])]

    rows = read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"Candidate manifest is empty: {manifest_path}")
    inventory = load_inventory(inventory_path)
    hash_cache: dict[Path, str] = {}
    normalized: list[dict[str, Any]] = []
    problems: list[str] = []

    for index, row in enumerate(rows, 1):
        op = operator_for(row)
        if not op:
            problems.append(f"row {index}: missing operator_name")
            continue
        role = role_for(row)
        cid = candidate_id_for(row, op, role, index)
        try:
            source, source_hash = resolve_source(row, op, role, root, resolve_path(root, args.output_dir), hash_cache)
            test, test_hash, module_name = resolve_test(
                row, op, root, inventory, allow_hash_mismatch=args.allow_test_hash_mismatch
            )
            if not parse_jit_functions(source):
                raise ValueError(f"No @triton.jit function found in {source}")
            normalized.append({
                **row,
                "candidate_id": cid,
                "operator_name": op,
                "selected_role": role,
                "source_path": relative_or_absolute(source, root),
                "source_sha256": source_hash,
                "test_path": relative_or_absolute(test, root),
                "test_sha256": test_hash,
                "expected_test_module_name": module_name,
                "jit_functions": parse_jit_functions(source),
                "compile_eligible": True,
            })
        except Exception as exc:
            problems.append(f"row {index} [{op}/{role}/{cid}]: {type(exc).__name__}: {exc}")

    if problems:
        preview = "\n".join("  - " + p for p in problems[:30])
        more = f"\n  ... and {len(problems) - 30} more" if len(problems) > 30 else ""
        raise RuntimeError(f"Could not prepare {len(problems)} candidate row(s):\n{preview}{more}")

    operator_counts = Counter(row["operator_name"] for row in normalized)
    expected_total = args.expected_operators * args.expected_per_operator
    if len(normalized) != expected_total:
        log(f"WARNING: expected {expected_total} candidates ({args.expected_operators}×{args.expected_per_operator}), found {len(normalized)}")
    bad_counts = {op: count for op, count in operator_counts.items() if count != args.expected_per_operator}
    if bad_counts:
        log(f"WARNING: operators not having exactly {args.expected_per_operator} candidates: {bad_counts}")

    # Exact-source deduplication is scoped to an operator and test signature.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[(row["operator_name"], row["source_sha256"], row["test_sha256"])].append(row)

    jobs: list[dict[str, Any]] = []
    for (op, source_hash, test_hash), group in sorted(grouped.items()):
        group.sort(key=lambda r: (ROLE_NAMES.index(r["selected_role"]) if r["selected_role"] in ROLE_NAMES else 99, r["candidate_id"]))
        representative = group[0]
        job_id = f"compile:{safe_name(op)}:{source_hash[:16]}:{test_hash[:8]}"
        jobs.append({
            "schema_version": "2.0",
            "compile_job_id": job_id,
            "operator_name": op,
            "representative_candidate_id": representative["candidate_id"],
            "candidate_type": representative["selected_role"],
            "selected_roles": sorted({r["selected_role"] for r in group}),
            "source_path": representative["source_path"],
            "source_sha256": source_hash,
            "test_path": representative["test_path"],
            "test_sha256": test_hash,
            "expected_test_module_name": representative["expected_test_module_name"],
            "jit_functions": representative["jit_functions"],
            "logical_candidate_ids": [r["candidate_id"] for r in group],
            "logical_candidate_count": len(group),
            "compile_status": "pending",
            "reuse_status": "not_checked",
        })
        for row in group:
            row["compile_job_id"] = job_id
            row["duplicate_source"] = len(group) > 1
            row["logical_candidate_count"] = len(group)

    reuse_index = collect_reuse_records(reuse_roots, reuse_manifests)
    reused_jobs = 0
    reuse_stats = Counter()
    for job in jobs:
        key = (job["operator_name"], job["source_sha256"].lower(), job["test_sha256"].lower())
        matches = reuse_index.get(key, [])
        if matches:
            try:
                if args.dry_run:
                    job["reuse_status"] = "would_reuse"
                    job["reuse_source_metadata"] = matches[0]["origin"]
                    reused_jobs += 1
                    reuse_stats["would_reuse"] += 1
                else:
                    meta = copy_reused_ir(job, matches[0], ir_dir)
                    job["reuse_status"] = "reused"
                    job["reuse_metadata_path"] = meta["metadata_path"]
                    job["reused_ir_paths"] = meta["ir_paths"]
                    reused_jobs += 1
                    reuse_stats["reused"] += 1
            except Exception as exc:
                job["reuse_status"] = "reuse_copy_failed"
                job["reuse_error"] = f"{type(exc).__name__}: {exc}"
                reuse_stats["reuse_copy_failed"] += 1
        else:
            job["reuse_status"] = "needs_compile"
            reuse_stats["needs_compile"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_candidates.jsonl"
    jobs_path = output_dir / "compile_jobs.jsonl"
    summary_path = output_dir / "compile_jobs_summary.json"
    if not args.dry_run:
        write_jsonl(selected_path, normalized)
        write_jsonl(jobs_path, jobs)
        summary = {
            "schema_version": "2.0",
            "candidate_manifest": str(manifest_path),
            "candidate_count": len(normalized),
            "operator_count": len(operator_counts),
            "candidate_counts_by_operator": dict(sorted(operator_counts.items())),
            "unique_compile_job_count": len(jobs),
            "deduplicated_logical_candidates": len(normalized) - len(jobs),
            "reuse_roots": [str(p) for p in reuse_roots],
            "reuse_manifests": [str(p) for p in reuse_manifests],
            "reuse_status_counts": dict(reuse_stats),
            "reused_jobs": reused_jobs,
            "jobs_needing_compile": len(jobs) - reused_jobs,
            "outputs": {
                "selected_candidates": str(selected_path),
                "compile_jobs": str(jobs_path),
                "ir_dir": str(ir_dir),
            },
            "note": "Reuse requires exact operator + source SHA256 + test SHA256 match and an existing TTIR file.",
        }
        write_json(summary_path, summary)
    else:
        summary = {
            "candidate_count": len(normalized), "operator_count": len(operator_counts),
            "unique_compile_job_count": len(jobs), "reused_jobs": reused_jobs,
            "jobs_needing_compile": len(jobs) - reused_jobs,
            "reuse_status_counts": dict(reuse_stats),
        }

    log("=" * 72)
    log("CostModel V2 compile-job preparation")
    log(f"Candidate rows       : {len(normalized)}")
    log(f"Operators            : {len(operator_counts)}")
    log(f"Unique compile jobs  : {len(jobs)}")
    log(f"Deduplicated rows    : {len(normalized) - len(jobs)}")
    log(f"Reused IR jobs       : {reused_jobs}")
    log(f"Need warmup compile  : {len(jobs) - reused_jobs}")
    log(f"Selected candidates  : {selected_path}")
    log(f"Compile jobs         : {jobs_path}")
    if not args.dry_run:
        log(f"Summary              : {summary_path}")
    log("=" * 72)
    if args.dry_run:
        log("DRY RUN: no output files or reused IR files were written.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[prepare-jobs] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
