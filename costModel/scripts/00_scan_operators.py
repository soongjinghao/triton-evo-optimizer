#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan Agent/datasets and build the V1 Cost Model operator inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COST_MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = COST_MODEL_DIR.parent
sys.path.insert(0, str(COST_MODEL_DIR / "src"))

from cost_model.data.inventory import InventoryPaths, InventoryScanner  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan datasets/ only; identify one *_1.py Seed per operator."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Agent 仓库根目录")
    parser.add_argument("--dataset-dir", default="datasets", help="V1 唯一数据源，默认 datasets")
    parser.add_argument("--manifest-dir", default="experiments/manifest", help="历史 manifest（仅作参考）")
    parser.add_argument("--baseline-json", default="baseline/baseline.json", help="历史 baseline（仅作参考）")
    parser.add_argument("--samples-jsonl", default="experiments/dataset/samples.jsonl", help="历史进化样本统计")
    parser.add_argument("--candidate-code-dir", default="experiments/dataset/codes", help="历史候选源码目录")
    parser.add_argument("--output-dir", default="costModel/data/manifests", help="清单输出目录")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    paths = InventoryPaths(
        project_root=project_root,
        dataset_dir=args.dataset_dir,
        manifest_dir=args.manifest_dir,
        baseline_json=args.baseline_json,
        samples_jsonl=args.samples_jsonl,
        candidate_code_dir=args.candidate_code_dir,
        output_dir=args.output_dir,
    )

    try:
        inventory_path, summary_path, summary = InventoryScanner(paths).write()
    except Exception as exc:  # noqa: BLE001
        print(f"[scan-operators] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    hist = summary.get("historical_evolution_samples", {})
    legacy = summary.get("legacy_manifest", {})
    print("=" * 72)
    print("Cost Model V1 operator inventory completed")
    print(f"Project root                 : {summary['project_root']}")
    print(f"Primary dataset              : {summary['primary_dataset']}")
    print(f"Dataset policy               : {summary['dataset_policy']}")
    print(f"Operators                    : {summary['operator_count']}")
    print(f"Source .py files             : {summary['source_file_count']}")
    print(f"Source files with Triton JIT : {summary['source_files_with_triton_jit']}")
    print(f"Seed source files            : {summary['seed_source_count']}")
    print(f"Seed files with Triton JIT   : {summary['seed_sources_with_triton_jit']}")
    print(f"Test .py files               : {summary['test_file_count']}")
    print(f"Compile-ready operators      : {summary['compile_ready_operator_count']}")
    print(f"Historical evolution rows    : {hist.get('sample_count', 0)}")
    print(f"Historical candidate codes  : {hist.get('candidate_code_found_count', 0)}")
    print(f"Legacy manifest seed entries : {legacy.get('seed_entry_count', 0)}")
    print(f"Legacy entries from datasets2: {legacy.get('seed_entries_pointing_to_datasets2', 0)}")
    print(f"Inventory JSONL              : {inventory_path}")
    print(f"Summary JSON                 : {summary_path}")
    warnings = summary.get("warnings", [])
    print(f"Warnings                     : {len(warnings)}")
    for warning in warnings:
        print(f"  - {warning.get('type')}: {warning.get('message', warning)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
