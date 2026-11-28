#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print Triton-Ascend official Cost Model predictions for one operator."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Optional


FUNC_RE = re.compile(r'\btt\.func(?:\s+public)?\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))')
PID_RE = re.compile(r"\btt\.get_program_id\s+([xyz])\b")


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


def get_real_latency(metadata: dict, profile: Optional[dict]):
    timing = (metadata.get("profiling") or {}).get("timing") or {}
    candidates = [
        timing.get("label_execution_time_us"),
        timing.get("execution_time_us"),
        (metadata.get("evaluation") or {}).get("execution_time_us"),
    ]
    if profile:
        candidates.extend([
            (profile.get("timing") or {}).get("label_execution_time_us"),
            (profile.get("timing") or {}).get("execution_time_us"),
        ])
    for value in candidates:
        value = positive_float(value)
        if value is not None:
            return value
    return None


def get_kernel_name(profile: Optional[dict], metadata: dict) -> str:
    if profile:
        kernel = profile.get("kernel") or {}
        for key in ("Op Name", "OP Type", "Kernel Name", "kernel_name"):
            if kernel.get(key):
                return str(kernel[key]).strip().strip('"')
    return str(metadata.get("operator_name") or "")


def choose_ttir(candidate_dir: Path, kernel_name: str):
    ttir_dir = candidate_dir / "ir" / "ttir"
    files = sorted(ttir_dir.glob("*.ttir.mlir")) if ttir_dir.is_dir() else []
    if not files:
        return None

    matches = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        symbols = [a or b for a, b in FUNC_RE.findall(text)]
        if any(symbol == kernel_name for symbol in symbols):
            matches.append((path, text))

    if len(matches) == 1:
        return matches[0]
    if len(files) == 1:
        path = files[0]
        return path, path.read_text(encoding="utf-8", errors="replace")
    return None


def load_bindings(path: Path):
    value = read_json(path)
    return value if value else {"default": {}, "operators": {}, "candidates": {}}


def parse_bindings(value):
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v is not None}
    if isinstance(value, str):
        result = {}
        for item in value.split(","):
            if "=" in item:
                key, val = item.split("=", 1)
                result[key.strip()] = val.strip()
        return result
    return {}


def nested_bindings(metadata: dict):
    result = {}
    for key, value in metadata.items():
        if key in ("arg_bindings", "costmodel_arg_bindings", "runtime_bindings"):
            result.update(parse_bindings(value))
    return result


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
        value = result[1]
    else:
        return None
    return positive_float(value)


def main():
    parser = argparse.ArgumentParser(description="Predict Triton-Ascend Cost Model latency for one operator")
    parser.add_argument("--operator", required=True, help="Operator directory name, e.g. _rms_norm_kernel")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    operator_dir = root / "datasets" / args.operator
    data_dir = operator_dir / "data"
    if not data_dir.is_dir():
        print(f"operator not found: {args.operator}", file=sys.stderr)
        return 2

    try:
        import triton
        from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench
    except Exception as exc:
        print(f"cannot import Triton-Ascend costmodel_bench: {exc}", file=sys.stderr)
        return 3

    binding_config = load_bindings(Path(__file__).resolve().parent / "official_costmodel_bindings.json")
    version = getattr(triton, "__version__", "unknown")

    print("role\tcandidate\tpredicted_us\treal_us")
    count = 0
    for candidate_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        metadata = read_json(candidate_dir / "metadata.json")
        if not metadata or metadata.get("status") not in (None, "success"):
            continue

        profile = None
        profile_path = candidate_dir / "profiling" / "profile_summary.json"
        if profile_path.is_file():
            profile = read_json(profile_path)

        candidate_id = str(metadata.get("candidate_id") or f"{args.operator}__{candidate_dir.name}")
        kernel_name = get_kernel_name(profile, metadata)
        chosen = choose_ttir(candidate_dir, kernel_name)
        if chosen is None:
            continue

        ttir_path, ttir_text = chosen
        bindings = {}
        bindings.update(parse_bindings(binding_config.get("default", {})))
        bindings.update(parse_bindings((binding_config.get("operators") or {}).get(args.operator, {})))
        bindings.update(parse_bindings((binding_config.get("candidates") or {}).get(candidate_id, {})))
        bindings.update(nested_bindings(metadata))

        for axis in sorted(set(PID_RE.findall(ttir_text))):
            bindings.setdefault(f"pid_{axis}", "0")

        binding_text = ",".join(f"{key}={bindings[key]}" for key in sorted(bindings))
        item = {
            "config": f"{args.operator}::{candidate_id}",
            "ttir": ttir_text,
        }
        if binding_text:
            item["arg_bindings"] = binding_text

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = call_costmodel(costmodel_bench, version, item)

        real = get_real_latency(metadata, profile)
        print("\t".join([
            candidate_dir.name,
            candidate_id,
            f"{predicted:.6f}" if predicted is not None else "-",
            f"{real:.6f}" if real is not None else "-",
        ]))
        count += 1

    if count == 0:
        print("no eligible candidates", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
