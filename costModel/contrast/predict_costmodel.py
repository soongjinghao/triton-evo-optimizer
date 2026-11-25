#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concise Triton-Ascend official cost model prediction entry point."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Optional


PID_RE = re.compile(r"\btt\.get_program_id\s+([xyz])\b")
FUNC_RE = re.compile(r'\btt\.func(?:\s+public)?\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))')


def fmt(value: Optional[float]) -> str:
    return f"{value:.6f}" if value is not None else "-"


def positive_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def profile(candidate_dir: Path, metadata: dict):
    paths = [candidate_dir / "profiling" / "profile_summary.json"]
    rel = (metadata.get("profiling") or {}).get("summary_path")
    if rel:
        paths.append(candidate_dir / str(rel))
    for path in paths:
        if path.is_file():
            value = read_json(path)
            if value:
                return value, path
    return None, None


def kernel_name(profile_data: Optional[dict], metadata: dict) -> str:
    if profile_data:
        kernel = profile_data.get("kernel") or {}
        for key in ("Op Name", "OP Type", "Kernel Name", "kernel_name"):
            if kernel.get(key):
                return str(kernel[key]).strip().strip('"')
        if profile_data.get("kernel_name"):
            return str(profile_data["kernel_name"]).strip().strip('"')
    return str(metadata.get("operator_name") or "")


def real_latency(metadata: dict, profile_data: Optional[dict]):
    timing = (metadata.get("profiling") or {}).get("timing") or {}
    candidates = [
        (timing.get("label_execution_time_us"), "metadata.profiling.timing.label_execution_time_us"),
        (timing.get("execution_time_us"), "metadata.profiling.timing.execution_time_us"),
        ((metadata.get("evaluation") or {}).get("execution_time_us"), "metadata.evaluation.execution_time_us"),
    ]
    if profile_data:
        timing = profile_data.get("timing") or {}
        candidates.extend([
            (timing.get("label_execution_time_us"), "profile_summary.timing.label_execution_time_us"),
            (timing.get("execution_time_us"), "profile_summary.timing.execution_time_us"),
        ])
    for value, source in candidates:
        value = positive_float(value)
        if value is not None:
            return value, source
    return None, "-"


def choose_ttir(candidate_dir: Path, kernel: str):
    directory = candidate_dir / "ir" / "ttir"
    files = sorted(directory.glob("*.ttir.mlir")) if directory.is_dir() else []
    if not files:
        files = sorted(path for path in directory.glob("*ttir*") if path.is_file()) if directory.is_dir() else []
    if not files:
        return None

    matches = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        symbols = [a or b for a, b in FUNC_RE.findall(text)]
        if any(symbol == kernel for symbol in symbols):
            matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if len(files) == 1:
        return files[0]
    return None


def load_bindings(path: Path):
    config = read_json(path)
    if not config:
        return {"default": {}, "operators": {}, "candidates": {}}
    return config


def parse_bindings(value):
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items() if val is not None}
    if isinstance(value, str):
        result = {}
        for item in value.split(","):
            if "=" in item:
                key, val = item.split("=", 1)
                result[key.strip()] = val.strip()
        return result
    return {}


def nested_binding(metadata: dict):
    for key in ("arg_bindings", "costmodel_arg_bindings", "runtime_bindings"):
        if key in metadata and metadata[key] not in (None, ""):
            return metadata[key]
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in {"arg_bindings", "costmodel_arg_bindings", "runtime_bindings"} and child not in (None, ""):
                    return child
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found is not None:
                    return found
        return None
    return walk(metadata)


def call_costmodel(costmodel_bench, version: str, item: dict):
    key = item["config"]
    if str(version).startswith("3.2."):
        result = costmodel_bench([item])
    else:
        try:
            result = costmodel_bench(item)
        except (TypeError, AttributeError):
            result = costmodel_bench([item])
    if isinstance(result, dict):
        value = result.get(key)
    elif isinstance(result, tuple) and len(result) == 2:
        returned_key, value = result
        if returned_key != key:
            return None, f"unexpected_key:{returned_key}"
    else:
        return None, f"unexpected_type:{type(result).__name__}"
    value = positive_float(value)
    return (value, "ok") if value is not None else (None, "invalid_latency")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print Triton-Ascend Cost Model predictions")
    parser.add_argument("--root", type=Path, default=Path("/workspace/Agent/costModel"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bindings-json", type=Path)
    parser.add_argument("--no-auto-pid", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    root = args.root.resolve()
    dataset = (args.dataset_dir or root / "datasets").resolve()
    bindings_path = (args.bindings_json or root / "contrast" / "official_costmodel_bindings.json").resolve()
    config = load_bindings(bindings_path)

    records = []
    if not dataset.is_dir():
        print(f"dataset directory not found: {dataset}", file=sys.stderr)
        return 2

    for operator_dir in sorted(path for path in dataset.iterdir() if path.is_dir()):
        operator = operator_dir.name
        if args.operator and operator not in args.operator:
            continue
        data_dir = operator_dir / "data"
        if not data_dir.is_dir():
            continue
        for candidate_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            role = candidate_dir.name
            if args.role and role not in args.role:
                continue
            metadata = read_json(candidate_dir / "metadata.json")
            if not metadata:
                continue
            if metadata.get("status") not in (None, "success"):
                continue
            profile_data, _ = profile(candidate_dir, metadata)
            candidate_id = str(metadata.get("candidate_id") or f"{operator}__{role}")
            kernel = kernel_name(profile_data, metadata)
            ttir_path = choose_ttir(candidate_dir, kernel)
            if ttir_path is None:
                continue
            real, real_source = real_latency(metadata, profile_data)
            records.append({
                "operator": operator,
                "role": role,
                "candidate": candidate_id,
                "metadata": metadata,
                "ttir": ttir_path.read_text(encoding="utf-8", errors="replace"),
                "real": real,
                "real_source": real_source,
            })

    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        print("no eligible candidates", file=sys.stderr)
        return 2

    try:
        import triton
        from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench
    except Exception as exc:
        print(f"cannot import official costmodel_bench: {exc}", file=sys.stderr)
        return 3

    version = getattr(triton, "__version__", "unknown")
    if not args.json_output:
        print("operator\trole\tcandidate\tpredicted_us\treal_us\treal_source\tstatus")

    output = []
    for record in records:
        bindings = {}
        bindings.update(parse_bindings(config.get("default", {})))
        bindings.update(parse_bindings((config.get("operators") or {}).get(record["operator"], {})))
        bindings.update(parse_bindings((config.get("candidates") or {}).get(record["candidate"], {})))
        bindings.update(parse_bindings(nested_binding(record["metadata"])))

        for axis in sorted(set(PID_RE.findall(record["ttir"]))):
            key = f"pid_{axis}"
            if args.no_auto_pid:
                bindings.pop(key, None)
            elif key not in bindings:
                bindings[key] = "0"

        binding_text = ",".join(f"{key}={bindings[key]}" for key in sorted(bindings))
        key = f"{record['operator']}::{record['candidate']}"
        item = {"config": key, "ttir": record["ttir"]}
        if binding_text:
            item["arg_bindings"] = binding_text

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted, status = call_costmodel(costmodel_bench, version, item)

        row = {
            "operator": record["operator"],
            "role": record["role"],
            "candidate": record["candidate"],
            "predicted_us": predicted,
            "real_us": record["real"],
            "real_source": record["real_source"],
            "status": status,
        }
        output.append(row)

        if not args.json_output:
            print("\t".join([
                str(row["operator"]), str(row["role"]), str(row["candidate"]),
                fmt(predicted), fmt(record["real"]), str(record["real_source"]), status,
            ]))

    if args.json_output:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
