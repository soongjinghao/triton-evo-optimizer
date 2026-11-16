
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare 50 operators x 5 candidates for CostModel V2.

Per operator:
    1 dataset_seed
    1 historical_best
    1 historical_p10
    1 historical_p50
    1 historical_p90

Sources:
    datasets/<operator>/<operator>_1.py

    experience/**/*.jsonl
    experience/**/*.py

    or:
    experiments/dataset/samples.jsonl
    experiments/dataset/codes/<candidate_id>.py

Historical child_latency is used ONLY for candidate selection.
It is NOT used as the new CostModel profiling label.

Default behavior: dry-run.
Use --apply to actually copy files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import statistics
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any


ROLES = (
    "dataset_seed",
    "historical_best",
    "historical_p10",
    "historical_p50",
    "historical_p90",
)

PERCENTILES = {
    "historical_p10": 10,
    "historical_p50": 50,
    "historical_p90": 90,
}

SOURCE_PATH_KEYS = (
    "source_path",
    "code_path",
    "candidate_path",
    "file_path",
    "source_file",
)


# ============================================================
# Basic utilities
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def positive_float(value: Any):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value) or value <= 0:
        return None

    return value


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def atomic_write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")

    temp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        ) + "\n",
        encoding="utf-8"
    )

    temp.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True
                ) + "\n"
            )

    temp.replace(path)


# ============================================================
# Triton source validation
# ============================================================

def inspect_triton_source(path: Path):
    """
    Check Python syntax and detect Triton JIT functions.
    """

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

    except Exception as exc:
        return False, [], str(exc)

    triton_aliases = set()
    jit_aliases = set()

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton":
                    triton_aliases.add(alias.asname or "triton")

        elif isinstance(node, ast.ImportFrom):
            if node.module == "triton":
                for alias in node.names:
                    if alias.name == "jit":
                        jit_aliases.add(alias.asname or "jit")

    functions = []

    for node in ast.walk(tree):

        if not isinstance(node, ast.FunctionDef):
            continue

        for decorator in node.decorator_list:

            target = (
                decorator.func
                if isinstance(decorator, ast.Call)
                else decorator
            )

            is_jit = (
                isinstance(target, ast.Name)
                and target.id in jit_aliases
            ) or (
                isinstance(target, ast.Attribute)
                and target.attr == "jit"
                and isinstance(target.value, ast.Name)
                and target.value.id in triton_aliases
            )

            if is_jit:
                functions.append(node.name)
                break

    if not functions:
        return False, [], "No Triton JIT function found"

    return True, functions, None


# ============================================================
# Dataset seeds
# ============================================================

def discover_operators(datasets_dir: Path, expected: int):

    if not datasets_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {datasets_dir}"
        )

    operators = []

    for operator_dir in sorted(datasets_dir.iterdir()):

        if not operator_dir.is_dir():
            continue

        operator = operator_dir.name

        # Only select the second source file.
        # Do NOT copy <operator>.py (reference baseline).
        seed_path = operator_dir / f"{operator}_1.py"

        if seed_path.is_file():
            operators.append({
                "operator_name": operator,
                "operator_dir": operator_dir,
                "seed_path": seed_path,
            })

    if len(operators) != expected:
        raise ValueError(
            f"Expected {expected} operators, "
            f"but found {len(operators)} valid seed directories."
        )

    return operators


# ============================================================
# Historical evolution records
# ============================================================

def load_history_rows(history_dir: Path):

    rows = []
    malformed = 0
    history_files = []

    jsonl_files = sorted(history_dir.rglob("*.jsonl"))

    for jsonl_path in jsonl_files:

        history_files.append(jsonl_path)

        with jsonl_path.open("r", encoding="utf-8") as f:

            for line_no, line in enumerate(f, 1):

                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue

                if not isinstance(row, dict):
                    malformed += 1
                    continue

                operator = str(
                    row.get("kernel")
                    or row.get("operator_name")
                    or ""
                ).strip()

                candidate_id = str(
                    row.get("candidate_id") or ""
                ).strip()

                latency = positive_float(
                    row.get("child_latency")
                )

                if not operator or not candidate_id or latency is None:
                    continue

                row["_history_file"] = str(jsonl_path)
                row["_history_line"] = line_no
                row["_operator"] = operator
                row["_candidate_id"] = candidate_id
                row["_child_latency_us"] = latency

                rows.append(row)

    return rows, malformed, history_files


def build_code_index(history_dir: Path):

    index = defaultdict(list)

    for path in history_dir.rglob("*.py"):

        if "__pycache__" in path.parts:
            continue

        index[path.stem].append(path.resolve())

    return index


def resolve_history_source(
    row: dict,
    project_root: Path,
    history_dir: Path,
    code_index: dict
):

    # First use an explicit source path if available.
    for key in SOURCE_PATH_KEYS:

        value = row.get(key)

        if not value:
            continue

        p = Path(str(value)).expanduser()

        candidates = []

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([
                project_root / p,
                history_dir / p,
            ])

        for candidate in candidates:

            if candidate.is_file():
                return candidate.resolve()

    # Existing historical dataset convention:
    # codes/<candidate_id>.py
    candidate_id = row["_candidate_id"]
    stem = Path(candidate_id).stem

    matches = code_index.get(stem, [])

    if not matches:
        return None

    # Deduplicate identical files.
    by_hash = {}

    for path in matches:
        h = sha256_file(path)
        by_hash.setdefault(h, []).append(path)

    if len(by_hash) > 1:
        raise ValueError(
            f"Ambiguous historical source for candidate {candidate_id}: "
            f"{matches}"
        )

    return sorted(matches)[0]


def build_history_groups(
    history_rows,
    history_dir: Path,
    project_root: Path
):

    code_index = build_code_index(history_dir)

    groups = defaultdict(lambda: {
        "source_path": None,
        "source_sha256": None,
        "candidate_ids": set(),
        "latencies": [],
        "run_ids": set(),
        "generations": set(),
        "methods": set(),
        "operations": set(),
        "sample_rows": 0,
    })

    missing_sources = 0
    invalid_sources = 0

    for row in history_rows:

        source_path = resolve_history_source(
            row,
            project_root,
            history_dir,
            code_index
        )

        if source_path is None:
            missing_sources += 1
            continue

        valid, jit_functions, error = inspect_triton_source(
            source_path
        )

        if not valid:
            invalid_sources += 1
            continue

        source_hash = sha256_file(source_path)

        operator = row["_operator"]

        key = (operator, source_hash)

        group = groups[key]

        group["source_path"] = source_path
        group["source_sha256"] = source_hash

        group["candidate_ids"].add(row["_candidate_id"])
        group["latencies"].append(row["_child_latency_us"])

        group["sample_rows"] += 1

        if row.get("run_id") is not None:
            group["run_ids"].add(str(row["run_id"]))

        if row.get("gen") is not None:
            group["generations"].add(str(row["gen"]))

        if row.get("method") is not None:
            group["methods"].add(str(row["method"]))

        if row.get("operation") is not None:
            group["operations"].add(str(row["operation"]))

    result = defaultdict(list)

    for (operator, source_hash), group in groups.items():

        group["historical_median_latency_us"] = statistics.median(
            group["latencies"]
        )

        group["candidate_ids"] = sorted(group["candidate_ids"])
        group["run_ids"] = sorted(group["run_ids"])
        group["generations"] = sorted(group["generations"])
        group["methods"] = sorted(group["methods"])
        group["operations"] = sorted(group["operations"])

        del group["latencies"]

        result[operator].append(group)

    return result, {
        "missing_source_rows": missing_sources,
        "invalid_source_rows": invalid_sources,
        "unique_historical_sources": sum(
            len(v) for v in result.values()
        ),
    }


# ============================================================
# Select four distinct evolution candidates
# ============================================================

def percentile_index(n: int, percentile: int):

    # Nearest-rank percentile.
    rank = math.ceil(percentile / 100 * n)

    return max(0, min(n - 1, rank - 1))


def select_four_historical_candidates(groups):

    # Each item represents a unique source hash.
    ordered = sorted(
        groups,
        key=lambda x: (
            x["historical_median_latency_us"],
            x["source_sha256"]
        )
    )

    if len(ordered) < 4:
        raise ValueError(
            f"Only {len(ordered)} unique valid historical sources; "
            "at least 4 are required."
        )

    selected = {}

    # 1. Historical best.
    best = ordered[0]

    selected["historical_best"] = best

    used_hashes = {best["source_sha256"]}

    # 2. P10 / P50 / P90.
    for role, percentile in PERCENTILES.items():

        target_index = percentile_index(
            len(ordered),
            percentile
        )

        available = [
            i for i, item in enumerate(ordered)
            if item["source_sha256"] not in used_hashes
        ]

        if not available:
            raise ValueError(
                f"No unique candidate available for {role}"
            )

        selected_index = min(
            available,
            key=lambda i: (
                abs(i - target_index),
                i
            )
        )

        chosen = ordered[selected_index]

        selected[role] = chosen
        used_hashes.add(chosen["source_sha256"])

    if len({
        item["source_sha256"]
        for item in selected.values()
    }) != 4:
        raise RuntimeError("Historical candidates are not unique.")

    return selected


# ============================================================
# Build candidate records
# ============================================================

def build_candidate_record(
    operator: str,
    role: str,
    source_path: Path,
    project_root: Path,
    destination: Path,
    history_group: dict | None = None
):

    valid, jit_functions, error = inspect_triton_source(
        source_path
    )

    if not valid:
        raise ValueError(
            f"Invalid candidate source: {source_path}: {error}"
        )

    source_hash = sha256_file(source_path)

    candidate_id = (
        f"{operator}__{role}__{source_hash[:12]}"
    )

    record = {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "operator_name": operator,
        "candidate_role": role,
        "candidate_type": (
            "dataset_seed"
            if role == "dataset_seed"
            else "historical_evolution"
        ),
        "source_path": relative_path(
            source_path,
            project_root
        ),
        "destination_path": relative_path(
            destination,
            project_root
        ),
        "source_sha256": source_hash,
        "source_size_bytes": source_path.stat().st_size,
        "jit_functions": jit_functions,
        "selection_status": "selected",
    }

    if history_group is not None:

        record.update({
            "selection_method": role,
            "historical_median_latency_us":
                history_group["historical_median_latency_us"],
            "historical_sample_count":
                history_group["sample_rows"],
            "historical_candidate_ids":
                history_group["candidate_ids"],
            "historical_run_ids":
                history_group["run_ids"],
            "historical_generations":
                history_group["generations"],
            "historical_methods":
                history_group["methods"],
            "historical_operations":
                history_group["operations"],
            "historical_latency_is_current_label": False,
        })

    else:

        record.update({
            "selection_method": "select_dataset_seed_1",
            "historical_latency_is_current_label": False,
        })

    return record


# ============================================================
# Main preparation
# ============================================================

def choose_history_directory(root: Path, specified: str | None):

    if specified:
        path = (root / specified).resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"History directory not found: {path}"
            )

        return path

    candidates = [
        root / "experience",
        root / "experiments" / "dataset",
    ]

    available = [
        path.resolve()
        for path in candidates
        if path.is_dir()
    ]

    if not available:
        raise FileNotFoundError(
            "Neither experience/ nor experiments/dataset/ exists. "
            "Specify --history-dir explicitly."
        )

    # Prefer experience when it contains valid history records.
    # Otherwise use the existing experiments/dataset convention.
    scored = []

    for path in available:

        rows, _, _ = load_history_rows(path)
        index = build_code_index(path)

        matched = sum(
            1 for row in rows
            if Path(row["_candidate_id"]).stem in index
        )

        scored.append((matched, path))

    scored.sort(
        key=lambda x: (
            x[0],
            x[1].name == "experience"
        ),
        reverse=True
    )

    if scored[0][0] == 0:
        raise RuntimeError(
            "No historical candidate source files matched "
            "the JSONL records."
        )

    return scored[0][1]


def prepare(args):

    root = Path(args.root).expanduser().resolve()

    datasets_dir = (root / args.datasets_dir).resolve()
    output_dir = (root / args.output_dir).resolve()

    history_dir = choose_history_directory(
        root,
        args.history_dir
    )

    print("=" * 76)
    print("CostModel V2 - Candidate Preparation")
    print("=" * 76)

    print("Project root :", root)
    print("Dataset dir  :", datasets_dir)
    print("History dir  :", history_dir)
    print("Output dir   :", output_dir)
    print("Mode         :", "APPLY" if args.apply else "DRY-RUN")

    # 1. Discover exactly 50 operators.
    operators = discover_operators(
        datasets_dir,
        args.expected_operators
    )

    print("\nOperators found:", len(operators))

    # 2. Load historical evolution records.
    history_rows, malformed, history_files = load_history_rows(
        history_dir
    )

    print("Historical valid records:", len(history_rows))
    print("Malformed JSONL rows:", malformed)

    if not history_rows:
        raise RuntimeError("No valid historical records found.")

    history_groups, history_stats = build_history_groups(
        history_rows,
        history_dir,
        root
    )

    # 3. Build all selections before copying anything.
    candidate_manifest = []
    operator_manifest = []
    incomplete = []

    for op_info in operators:

        operator = op_info["operator_name"]
        seed_path = op_info["seed_path"]

        print(f"\n[{operator}]")

        # Only copy <operator>_1.py.
        # The reference baseline <operator>.py is excluded.
        valid, _, error = inspect_triton_source(seed_path)

        if not valid:
            incomplete.append({
                "operator": operator,
                "reason": f"Invalid dataset seed: {error}"
            })
            continue

        selected_candidates = []

        # Dataset seed.
        seed_destination = (
            output_dir / "operators" / operator /
            "candidates" / "dataset_seed" / "kernel.py"
        )

        seed_record = build_candidate_record(
            operator,
            "dataset_seed",
            seed_path,
            root,
            seed_destination
        )

        selected_candidates.append(
            (seed_record, seed_path, None)
        )

        # Four historical candidates.
        groups = history_groups.get(operator, [])

        try:
            historical_selected = (
                select_four_historical_candidates(groups)
            )

        except Exception as exc:
            incomplete.append({
                "operator": operator,
                "reason": str(exc)
            })
            continue

        for role in (
            "historical_best",
            "historical_p10",
            "historical_p50",
            "historical_p90",
        ):

            group = historical_selected[role]

            destination = (
                output_dir / "operators" / operator /
                "candidates" / role / "kernel.py"
            )

            record = build_candidate_record(
                operator,
                role,
                group["source_path"],
                root,
                destination,
                history_group=group
            )

            selected_candidates.append(
                (record, group["source_path"], group)
            )

        # Per-operator uniqueness check.
        hashes = [
            item[0]["source_sha256"]
            for item in selected_candidates
        ]

        if len(hashes) != 5 or len(set(hashes)) != 5:
            incomplete.append({
                "operator": operator,
                "reason": "Five candidates are not source-distinct."
            })
            continue

        candidate_manifest.extend(
            item[0] for item in selected_candidates
        )

        operator_manifest.append({
            "operator_name": operator,
            "candidate_count": 5,
            "candidates": [
                item[0]["candidate_id"]
                for item in selected_candidates
            ],
        })

        print("  dataset_seed    : OK")
        print("  historical_best : OK")
        print("  historical_p10  : OK")
        print("  historical_p50  : OK")
        print("  historical_p90  : OK")

    # 4. Strict completeness validation.
    if incomplete:

        print("\nINCOMPLETE OPERATORS:")

        for item in incomplete:
            print(
                "  - {}: {}".format(
                    item["operator"],
                    item["reason"]
                )
            )

        raise RuntimeError(
            f"{len(incomplete)} operators are incomplete. "
            "No files were copied."
        )

    if len(operator_manifest) != args.expected_operators:
        raise RuntimeError(
            f"Expected {args.expected_operators} operators, "
            f"got {len(operator_manifest)}."
        )

    if len(candidate_manifest) != args.expected_operators * 5:
        raise RuntimeError(
            "Candidate count validation failed."
        )

    # 5. Preflight output files.
    if args.apply:

        for record in candidate_manifest:

            source = root / record["source_path"]
            destination = root / record["destination_path"]

            if destination.exists():

                existing_hash = sha256_file(destination)

                if existing_hash != record["source_sha256"]:
                    if not args.overwrite:
                        raise FileExistsError(
                            f"Destination already exists with "
                            f"different content: {destination}. "
                            "Use --overwrite if intentional."
                        )

    # 6. Print summary.
    summary = {
        "schema_version": "1.0",
        "expected_operators": args.expected_operators,
        "operator_count": len(operator_manifest),
        "candidates_per_operator": 5,
        "candidate_count": len(candidate_manifest),
        "candidate_role_counts": {
            role: len([
                row for row in candidate_manifest
                if row["candidate_role"] == role
            ])
            for role in ROLES
        },
        "dataset_seed_source": (
            "datasets/<operator>/<operator>_1.py"
        ),
        "excluded_source": (
            "datasets/<operator>/<operator>.py"
        ),
        "history_directory": relative_path(
            history_dir,
            root
        ),
        "history_jsonl_files": [
            relative_path(path, root)
            for path in history_files
        ],
        "history_statistics": history_stats,
        "malformed_history_rows": malformed,
        "selection_policy": {
            "historical_best": "Minimum median child_latency",
            "historical_p10": "Nearest-rank P10",
            "historical_p50": "Nearest-rank P50",
            "historical_p90": "Nearest-rank P90",
            "deduplication": "operator + source SHA256",
            "historical_latency_used_as_current_label": False,
        },
        "outputs": {
            "candidate_manifest": "candidate_manifest.jsonl",
            "operator_manifest": "operator_manifest.json",
            "summary": "selection_summary.json",
        },
    }

    print("\n" + "=" * 76)
    print("SELECTION SUMMARY")
    print("=" * 76)

    print("Operators :", len(operator_manifest))
    print("Candidates:", len(candidate_manifest))

    for role, count in summary["candidate_role_counts"].items():
        print(f"{role:22s}: {count}")

    print("Historical source stats:", history_stats)

    if not args.apply:
        print("\nDRY-RUN completed. No files were copied.")
        print("Run again with --apply to copy the candidates.")
        print("=" * 76)
        return

    # 7. Copy candidate source files.
    for record in candidate_manifest:

        source = root / record["source_path"]
        destination = root / record["destination_path"]

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if (
            destination.exists()
            and sha256_file(destination) == record["source_sha256"]
        ):
            continue

        shutil.copy2(source, destination)

        candidate_json = destination.parent / "candidate.json"

        atomic_write_json(candidate_json, record)

    # 8. Write operator metadata.
    for op_record in operator_manifest:

        operator_dir = (
            output_dir / "operators" /
            op_record["operator_name"]
        )

        atomic_write_json(
            operator_dir / "operator.json",
            op_record
        )

    # 9. Write global manifests.
    atomic_write_jsonl(
        output_dir / "candidate_manifest.jsonl",
        candidate_manifest
    )

    atomic_write_json(
        output_dir / "operator_manifest.json",
        operator_manifest
    )

    atomic_write_json(
        output_dir / "selection_summary.json",
        summary
    )

    print("\n" + "=" * 76)
    print("CANDIDATE PREPARATION COMPLETED")
    print("=" * 76)
    print("Operators :", len(operator_manifest))
    print("Candidates:", len(candidate_manifest))
    print("Output    :", output_dir)
    print("Manifest  :", output_dir / "candidate_manifest.jsonl")
    print("=" * 76)


# ============================================================
# CLI
# ============================================================

def parse_args():

    script_path = Path(__file__).resolve()

    default_root = script_path.parents[2]

    parser = argparse.ArgumentParser(
        description="Prepare CostModel V2 candidate dataset."
    )

    parser.add_argument(
        "--root",
        default=str(default_root)
    )

    parser.add_argument(
        "--datasets-dir",
        default="datasets"
    )

    parser.add_argument(
        "--history-dir",
        default=None,
        help="Optional: experience or experiments/dataset"
    )

    parser.add_argument(
        "--output-dir",
        default="costModel_v2/data"
    )

    parser.add_argument(
        "--expected-operators",
        type=int,
        default=50
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy files. Without this flag, run dry-run."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing candidate files with different hashes."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    try:
        prepare(args)

    except Exception as exc:

        print(
            f"\n[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr
        )

        raise SystemExit(1)
