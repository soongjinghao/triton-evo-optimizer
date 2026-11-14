#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Cost Model inventory from Agent/datasets only.

The current V1 policy is intentionally narrow:
- datasets/ is the only source dataset;
- <operator>/<operator>_1.py is the Seed candidate to compile;
- <operator>/<operator>.py is retained as the baseline/reference source;
- historical manifests are read as provenance only. They are NOT used to
  decide whether a datasets/ Seed is eligible, because the existing manifests
  may point to datasets2/.

This module never imports or executes user Triton kernels. Python AST is used
only for static source inspection.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class InventoryPaths:
    """Input and output paths for a single inventory scan."""

    project_root: Path = DEFAULT_PROJECT_ROOT
    dataset_dir: str = "datasets"
    manifest_dir: str = "experiments/manifest"
    baseline_json: str = "baseline/baseline.json"
    samples_jsonl: str = "experiments/dataset/samples.jsonl"
    candidate_code_dir: str = "experiments/dataset/codes"
    output_dir: str = "costModel/data/manifests"

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


class InventoryScanner:
    """Scan operators, Seed sources, tests and historical metadata."""

    def __init__(self, paths: InventoryPaths):
        self.paths = paths
        self.root = paths.project_root.expanduser().resolve()
        self.dataset_root = paths.resolve(paths.dataset_dir)
        self.manifest_root = paths.resolve(paths.manifest_dir)
        self.baseline_path = paths.resolve(paths.baseline_json)
        self.samples_path = paths.resolve(paths.samples_jsonl)
        self.code_root = paths.resolve(paths.candidate_code_dir)

    def scan(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._validate_paths()
        manifests = self._load_manifests()
        baseline = self._load_baseline()
        sample_stats, sample_summary = self._load_sample_statistics()

        operator_dirs = sorted(
            (p for p in self.dataset_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
        records = [
            self._scan_operator(
                operator_dir=operator_dir,
                legacy_manifest=manifests.get(operator_dir.name),
                baseline_records=baseline.get(operator_dir.name, []),
                sample_stats=sample_stats.get(operator_dir.name, self._empty_sample_stats()),
            )
            for operator_dir in operator_dirs
        ]
        return records, self._build_summary(records, manifests, baseline, sample_summary)

    def write(self) -> tuple[Path, Path, dict[str, Any]]:
        records, summary = self.scan()
        output_dir = self.paths.resolve(self.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = output_dir / "operator_inventory.jsonl"
        summary_path = output_dir / "inventory_summary.json"
        self._atomic_write_jsonl(inventory_path, records)
        self._atomic_write_json(summary_path, summary)
        return inventory_path, summary_path, summary

    def _validate_paths(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Project root does not exist: {self.root}")
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Primary dataset directory does not exist: {self.dataset_root}")

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _decorator_target(node: ast.expr) -> ast.expr:
        return node.func if isinstance(node, ast.Call) else node

    @classmethod
    def _find_triton_jit_functions(cls, path: Path) -> tuple[list[dict[str, Any]], str | None]:
        """Find @triton.jit and from triton import jit aliases without importing."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            return [], f"{type(exc).__name__}: {exc}"

        triton_aliases = {"triton"}
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
                target = cls._decorator_target(decorator)
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

    def _load_manifests(self) -> dict[str, dict[str, Any]]:
        """Load old manifests as historical provenance, never as compile authority."""
        result: dict[str, dict[str, Any]] = {}
        if not self.manifest_root.is_dir():
            return result
        for path in sorted(self.manifest_root.glob("*.json")):
            try:
                data = self._read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                result[path.stem] = {"_load_error": f"{type(exc).__name__}: {exc}"}
                continue
            if isinstance(data, dict):
                result[str(data.get("kernel") or path.stem)] = data
        return result

    @staticmethod
    def _dataset_name_from_recorded_path(value: Any) -> str | None:
        if not value:
            return None
        parts = PurePosixPath(str(value).replace("\\", "/")).parts
        for part in parts:
            if part in {"datasets", "datasets2"}:
                return part
        return None

    def _load_baseline(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if not self.baseline_path.is_file():
            return grouped
        try:
            data = self._read_json(self.baseline_path)
        except (OSError, json.JSONDecodeError):
            return grouped
        for item in data.get("results", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict) or not item.get("kernel_name"):
                continue
            kernel = str(item["kernel_name"])
            test_name = item.get("test_file")
            test_exists = bool(
                test_name and (self.dataset_root / kernel / str(test_name)).is_file()
            )
            grouped[kernel].append({
                "test_file": test_name,
                "task_duration_us": item.get("task_duration_us"),
                "success": item.get("success"),
                "timestamp": item.get("timestamp"),
                "test_file_exists_in_datasets": test_exists,
                "source": "baseline/baseline.json",
            })
        return grouped

    @staticmethod
    def _empty_sample_stats() -> dict[str, Any]:
        return {
            "sample_count": 0,
            "label_0_count": 0,
            "label_1_count": 0,
            "valid_parent_latency_count": 0,
            "valid_child_latency_count": 0,
            "candidate_code_found_count": 0,
            "candidate_code_missing_count": 0,
            "unique_candidate_ids": 0,
            "min_generation": None,
            "max_generation": None,
        }

    def _load_sample_statistics(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        per_kernel: dict[str, dict[str, Any]] = defaultdict(self._empty_sample_stats)
        label_counts: Counter[str] = Counter()
        total = 0
        unique_ids: set[str] = set()
        unique_by_kernel: dict[str, set[str]] = defaultdict(set)
        code_found = code_missing = 0
        malformed = 0

        if self.samples_path.is_file():
            with self.samples_path.open("r", encoding="utf-8") as handle:
                for line in handle:
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
                    kernel = str(row.get("kernel") or "__missing_kernel__")
                    stats = per_kernel[kernel]
                    stats["sample_count"] += 1
                    total += 1
                    label = row.get("label")
                    if label in (0, "0", False):
                        stats["label_0_count"] += 1
                        label_counts["0"] += 1
                    elif label in (1, "1", True):
                        stats["label_1_count"] += 1
                        label_counts["1"] += 1
                    else:
                        label_counts["other"] += 1

                    for field, output_key in (
                        ("parent_latency", "valid_parent_latency_count"),
                        ("child_latency", "valid_child_latency_count"),
                    ):
                        try:
                            value = float(row.get(field))
                            if math.isfinite(value) and value > 0:
                                stats[output_key] += 1
                        except (TypeError, ValueError):
                            pass

                    candidate_id = row.get("candidate_id")
                    if candidate_id is not None:
                        candidate_id = str(candidate_id)
                        unique_ids.add(candidate_id)
                        unique_by_kernel[kernel].add(candidate_id)
                        if (self.code_root / f"{candidate_id}.py").is_file():
                            stats["candidate_code_found_count"] += 1
                            code_found += 1
                        else:
                            stats["candidate_code_missing_count"] += 1
                            code_missing += 1

                    try:
                        generation = int(row.get("gen"))
                    except (TypeError, ValueError):
                        generation = None
                    if generation is not None:
                        old_min, old_max = stats["min_generation"], stats["max_generation"]
                        stats["min_generation"] = generation if old_min is None else min(old_min, generation)
                        stats["max_generation"] = generation if old_max is None else max(old_max, generation)

        for kernel, stats in per_kernel.items():
            stats["unique_candidate_ids"] = len(unique_by_kernel[kernel])

        summary = {
            "path": self._relative(self.samples_path),
            "file_exists": self.samples_path.is_file(),
            "sample_count": total,
            "operator_count": len(per_kernel),
            "label_counts": dict(label_counts),
            "unique_candidate_ids": len(unique_ids),
            "candidate_code_found_count": code_found,
            "candidate_code_missing_count": code_missing,
            "malformed_rows": malformed,
        }
        return dict(per_kernel), summary

    def _scan_operator(
        self,
        operator_dir: Path,
        legacy_manifest: dict[str, Any] | None,
        baseline_records: list[dict[str, Any]],
        sample_stats: dict[str, Any],
    ) -> dict[str, Any]:
        operator_name = operator_dir.name
        py_files = sorted(p for p in operator_dir.glob("*.py") if p.is_file())
        test_paths = [p for p in py_files if p.name.lower().startswith("test_")]
        source_paths = [p for p in py_files if p not in test_paths]

        manifest_seeds = (legacy_manifest or {}).get("seeds", {})
        manifest_by_name = {
            Path(str(name)).name: value
            for name, value in manifest_seeds.items()
            if isinstance(value, dict)
        } if isinstance(manifest_seeds, dict) else {}

        source_records: list[dict[str, Any]] = []
        for path in source_paths:
            if path.name == f"{operator_name}.py":
                role = "baseline"
            elif path.name == f"{operator_name}_1.py":
                role = "seed"
            else:
                role = "other_source"

            jit_functions, parse_error = self._find_triton_jit_functions(path)
            old = manifest_by_name.get(path.name)
            old_path = old.get("path") if old else None
            old_dataset = self._dataset_name_from_recorded_path(old_path)
            source_records.append({
                "file_name": path.name,
                "path": self._relative(path),
                "role": role,
                "is_seed_candidate": role == "seed",
                "sha256": self._sha256(path),
                "size_bytes": path.stat().st_size,
                "jit_functions": jit_functions,
                "jit_function_count": len(jit_functions),
                "ast_parse_error": parse_error,
                # Old manifest entries are retained for traceability only.
                # They frequently point at datasets2 and must not gate V1.
                "legacy_manifest": ({
                    "success": old.get("success"),
                    "latency_us": old.get("latency_us"),
                    "recorded_path": old_path,
                    "recorded_dataset": old_dataset,
                    "points_to_primary_dataset": old_dataset == "datasets",
                } if old else None),
            })

        test_records = [{
            "file_name": path.name,
            "path": self._relative(path),
            "sha256": self._sha256(path),
            "size_bytes": path.stat().st_size,
        } for path in test_paths]

        preferred_test = next(
            (item for item in test_records if item["file_name"] == f"test_{operator_name}_1.py"),
            test_records[0] if len(test_records) == 1 else None,
        )
        seed_source = next((item for item in source_records if item["role"] == "seed"), None)
        baseline_source = next((item for item in source_records if item["role"] == "baseline"), None)
        compile_ready = bool(
            seed_source and seed_source["jit_function_count"] > 0 and preferred_test
        )

        successful_legacy_seeds = sum(
            1 for item in manifest_by_name.values() if item.get("success") is True
        )
        legacy_seed_files = len(manifest_by_name)
        legacy_points_to_datasets2 = sum(
            1 for item in manifest_by_name.values()
            if self._dataset_name_from_recorded_path(item.get("path")) == "datasets2"
        )

        baseline_success_count = sum(1 for item in baseline_records if item.get("success") is True)
        return {
            "operator_name": operator_name,
            "primary_dataset": self._relative(operator_dir),
            "source_file_count": len(source_records),
            "test_file_count": len(test_records),
            "triton_jit_function_count": sum(item["jit_function_count"] for item in source_records),
            "has_triton_jit": any(item["jit_function_count"] > 0 for item in source_records),
            "source_files": source_records,
            "seed_source": seed_source,
            "baseline_source": baseline_source,
            "test_files": test_records,
            "preferred_test_file": preferred_test,
            "compile_ready": compile_ready,
            "legacy_manifest": {
                "path": self._relative(self.manifest_root / f"{operator_name}.json")
                if (self.manifest_root / f"{operator_name}.json").is_file() else None,
                "created_at": (legacy_manifest or {}).get("created_at"),
                "seed_file_count": legacy_seed_files,
                "successful_seed_count": successful_legacy_seeds,
                "seed_entries_pointing_to_datasets2": legacy_points_to_datasets2,
                "used_for_compile_selection": False,
                "load_error": (legacy_manifest or {}).get("_load_error"),
            },
            "baseline_history": {
                "record_count": len(baseline_records),
                "successful_record_count": baseline_success_count,
                "records": baseline_records,
                "used_as_current_seed_label": False,
            },
            "historical_evolution_samples": sample_stats,
            "scan_flags": {
                "has_seed_source": seed_source is not None,
                "seed_has_triton_jit": bool(seed_source and seed_source["jit_function_count"] > 0),
                "has_preferred_test": preferred_test is not None,
                "compile_ready": compile_ready,
            },
        }

    def _build_summary(
        self,
        records: list[dict[str, Any]],
        manifests: dict[str, dict[str, Any]],
        baseline: dict[str, list[dict[str, Any]]],
        sample_summary: dict[str, Any],
    ) -> dict[str, Any]:
        source_count = sum(row["source_file_count"] for row in records)
        test_count = sum(row["test_file_count"] for row in records)
        source_files_with_jit = sum(
            1 for row in records for source in row["source_files"]
            if source["jit_function_count"] > 0
        )
        seed_count = sum(1 for row in records if row.get("seed_source"))
        seed_with_jit = sum(1 for row in records if row.get("seed_source") and row["seed_source"]["jit_function_count"] > 0)
        compile_ready_count = sum(1 for row in records if row.get("compile_ready"))
        no_seed = [row["operator_name"] for row in records if not row.get("seed_source")]
        no_test = [row["operator_name"] for row in records if not row.get("preferred_test_file")]
        seed_without_jit = [row["operator_name"] for row in records if row.get("seed_source") and row["seed_source"]["jit_function_count"] == 0]

        all_legacy_entries = [
            item
            for manifest in manifests.values()
            for item in (manifest.get("seeds", {}) or {}).values()
            if isinstance(item, dict)
        ]
        legacy_dataset2_count = sum(
            1 for item in all_legacy_entries
            if self._dataset_name_from_recorded_path(item.get("path")) == "datasets2"
        )
        baseline_rows = [item for values in baseline.values() for item in values]
        warnings: list[dict[str, Any]] = []
        if no_seed:
            warnings.append({"type": "missing_seed_source", "operators": no_seed})
        if no_test:
            warnings.append({"type": "missing_test_file", "operators": no_test})
        if seed_without_jit:
            warnings.append({"type": "seed_without_triton_jit", "operators": seed_without_jit})
        if legacy_dataset2_count:
            warnings.append({
                "type": "legacy_manifest_dataset_mismatch",
                "count": legacy_dataset2_count,
                "message": "历史 manifest 路径指向 datasets2；仅作历史元数据，不参与当前 datasets Seed 的编译筛选或标签绑定。",
            })

        return {
            "schema_version": "2.0",
            "project_root": str(self.root),
            "primary_dataset": self._relative(self.dataset_root),
            "dataset_policy": "datasets_only",
            "operator_count": len(records),
            "source_file_count": source_count,
            "source_files_with_triton_jit": source_files_with_jit,
            "seed_source_count": seed_count,
            "seed_sources_with_triton_jit": seed_with_jit,
            "test_file_count": test_count,
            "compile_ready_operator_count": compile_ready_count,
            "operators_without_seed": no_seed,
            "operators_without_preferred_test": no_test,
            "seed_sources_without_triton_jit": seed_without_jit,
            "legacy_manifest": {
                "operator_count": len(manifests),
                "seed_entry_count": len(all_legacy_entries),
                "seed_entries_pointing_to_datasets2": legacy_dataset2_count,
                "used_for_compile_selection": False,
            },
            "baseline_history": {
                "path": self._relative(self.baseline_path),
                "record_count": len(baseline_rows),
                "successful_record_count": sum(1 for row in baseline_rows if row.get("success") is True),
                "records_with_test_file_in_datasets": sum(1 for row in baseline_rows if row.get("test_file_exists_in_datasets")),
                "used_as_current_seed_label": False,
            },
            "historical_evolution_samples": sample_summary,
            "warnings": warnings,
        }

    @staticmethod
    def _atomic_write_json(path: Path, data: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(path)

    @staticmethod
    def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp.replace(path)
