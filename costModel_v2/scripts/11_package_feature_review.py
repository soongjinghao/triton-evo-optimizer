#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package the first-round CostModel V2 feature-analysis artifacts for review.

Run from the Agent project:
    python costModel_v2/scripts/11_package_feature_review.py

Optional:
    python costModel_v2/scripts/11_package_feature_review.py --output /tmp/my_review.zip
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COST = ROOT / "costModel_v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Package CostModel V2 feature-review files")
    parser.add_argument(
        "--output",
        default="/tmp/costmodel_v2_feature_review.zip",
        help="Output ZIP path (default: /tmp/costmodel_v2_feature_review.zip)",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    def add(path: Path) -> None:
        if path.is_file():
            try:
                rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                rel = path.name
            files[rel] = path

    # All generated first-round feature artifacts, including datasets/reports.
    feature_dir = COST / "features"
    if feature_dir.exists():
        for p in feature_dir.rglob("*"):
            if p.is_file():
                add(p)

    # Candidate selection and compile provenance.
    for name in (
        "candidate_manifest.jsonl",
        "selected_candidates.jsonl",
        "compile_jobs.jsonl",
    ):
        add(COST / "data" / name)

    # Compile/IR metadata only. Do NOT package large raw IR binaries or MLIR trees.
    ir_dir = COST / "ir"
    if ir_dir.exists():
        for p in ir_dir.iterdir():
            if p.is_file() and (
                p.suffix.lower() in {".json", ".jsonl", ".csv"}
                or "manifest" in p.name.lower()
            ):
                add(p)

    # Main profiling labels and IR audit records.
    labels_dir = COST / "labels"
    for name in (
        "candidate_labels.csv",
        "candidate_profile_records.jsonl",
        "profile_summary.json",
        "ir_audit.jsonl",
        "ir_audit_summary.json",
    ):
        add(labels_dir / name)

    # Include retry results (e.g. RMSNorm retries) so they can be compared
    # with the original profiling records.
    if COST.exists():
        for p in COST.glob("labels_retry*"):
            if p.is_dir():
                for q in p.rglob("*"):
                    if q.is_file() and q.name in {
                        "candidate_labels.csv",
                        "candidate_profile_records.jsonl",
                        "profile_summary.json",
                    }:
                        add(q)

    # Include the exact scripts used to prepare labels, extract/build features,
    # analyze features, and this packaging script.
    scripts_dir = COST / "scripts"
    for name in (
        "07_prepare_feature_labels.py",
        "08_extract_ttir_features.py",
        "09_build_feature_dataset.py",
        "10_analyze_features.py",
        "11_package_feature_review.py",
        "README_first_round_feature_pipeline.md",
    ):
        add(scripts_dir / name)

    if not files:
        print(f"[bundle] ERROR: no review files found under {COST}")
        return 2

    # Runtime metadata, without importing torch/CANN.
    packages = {}
    for dist_name in (
        "torch", "torch-npu", "triton", "triton-ascend",
        "numpy", "pandas", "scipy", "scikit-learn",
    ):
        try:
            packages[dist_name] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            packages[dist_name] = None

    cann_version = None
    for candidate in (
        Path("/usr/local/Ascend/cann-9.1.0/compiler/version.info"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"),
    ):
        if candidate.exists():
            cann_version = candidate.read_text(errors="replace")[:4000]
            break

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cann_version_info": cann_version,
        "file_count": len(files),
        "note": (
            "Raw TTIR/NPUIR/NPUBIN files are intentionally excluded. "
            "The bundle contains extracted feature tables, labels, analysis reports, "
            "candidate/compile metadata, and the scripts used to produce them."
        ),
        "files": [],
    }

    output_resolved = output.resolve()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in sorted(files.items()):
            # Never include the output archive in itself.
            if path.resolve() == output_resolved:
                continue
            zf.write(path, arcname=arcname)
            metadata["files"].append({
                "path": arcname,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

        zf.writestr(
            "costModel_v2/feature_review_bundle_meta.json",
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )

    with zipfile.ZipFile(output, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"[bundle] ERROR: ZIP integrity check failed at {bad}")
            return 3

    print("[bundle] CostModel V2 feature-review package created")
    print(f"[bundle] Project root : {ROOT}")
    print(f"[bundle] Files        : {len(metadata['files'])}")
    print(f"[bundle] Output       : {output}")
    print(f"[bundle] Size         : {output.stat().st_size / 1024:.1f} KB")
    print("[bundle] ZIP integrity: OK")
    print("[bundle] Raw IR binaries were not included.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
