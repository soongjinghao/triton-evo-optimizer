#!/usr/bin/env python3
"""Extract per-specialization Triton TTIR semantic features from feature-analysis output."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "feature_analysis_20261029"

TTIR_OPS = (
    "tt.load",
    "tt.store",
    "tt.addptr",
    "tt.get_program_id",
    "tt.make_range",
    "tt.make_tensor_ptr",
    "tt.gather",
    "tt.reduce",
    "tt.scan",
    "tt.dot",
    "tt.dot_scaled",
    "tt.trans",
    "tt.reshape",
    "tt.expand_dims",
    "tt.broadcast",
    "tt.splat",
    "tt.join",
    "tt.split",
    "tt.select",
    "tt.program_id",
)
ARITH_OPS = (
    "arith.addf",
    "arith.addi",
    "arith.subf",
    "arith.subi",
    "arith.mulf",
    "arith.muli",
    "arith.divf",
    "arith.divsi",
    "arith.divui",
    "arith.remsi",
    "arith.remui",
    "arith.cmpf",
    "arith.cmpi",
    "arith.select",
    "arith.constant",
    "arith.extf",
    "arith.extsi",
    "arith.extui",
    "arith.truncf",
    "arith.trunci",
    "arith.sitofp",
    "arith.uitofp",
    "arith.fptosi",
    "arith.maxf",
    "arith.minf",
    "arith.negf",
    "arith.fma",
    "arith.andi",
    "arith.ori",
    "arith.xori",
)
SCF_OPS = (
    "scf.for",
    "scf.if",
    "scf.while",
    "scf.yield",
    "scf.condition",
)
MATH_OPS = (
    "math.exp",
    "math.exp2",
    "math.log",
    "math.log2",
    "math.sqrt",
    "math.rsqrt",
    "math.absf",
    "math.tanh",
)
DTYPES = ("i1", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64",
          "fp8e4nv", "fp8e5")
LAYOUTS = ("blocked", "slice", "linear", "dot_operand", "shared", "mma")
TENSOR_TYPE_RE = re.compile(r"tensor<([^>]+)>")
OP_RE = re.compile(
    r'^\s*(?:(?:%[\w.$-]+(?:\s*,\s*%[\w.$-]+)*)\s*=\s*)?'
    r'(?:"(?P<quoted>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)"|'
    r'(?P<plain>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))\b'
)

FEATURE_NAMES = [
    "operation_count",
    "tt_operation_count",
    "arith_operation_count",
    "scf_operation_count",
    "math_operation_count",
    "other_dialect_operation_count",
    *[f"op_{op.replace('.', '_')}" for op in TTIR_OPS],
    *[f"op_{op.replace('.', '_')}" for op in ARITH_OPS],
    *[f"op_{op.replace('.', '_')}" for op in SCF_OPS],
    *[f"op_{op.replace('.', '_')}" for op in MATH_OPS],
    "program_id_axis_x_count",
    "program_id_axis_y_count",
    "program_id_axis_z_count",
    "load_mask_operand_count",
    "store_mask_operand_count",
    "tensor_type_occurrence_count",
    "unique_tensor_type_count",
    "tensor_static_elements_unique_sum",
    "tensor_static_elements_max",
    "tensor_dynamic_dim_count",
    "tensor_rank_max",
    *[f"tensor_rank_{rank}_unique_type_count" for rank in range(6)],
    *[f"dtype_{dtype}_ir_type_occurrence_count" for dtype in DTYPES],
    "make_range_count",
    "make_range_max_extent",
    "boundary_check_attribute_count",
    "padding_attribute_count",
    "divisibility_attribute_count",
    "divisibility_max",
    *[f"layout_{layout}_attribute_count" for layout in LAYOUTS],
    "tensor_layout_attribute_count",
    "masked_load_ratio",
    "masked_store_ratio",
]


def _parse_tensor_type(type_body):
    parts = type_body.split("x")
    if not parts:
        return None
    dtype = parts[-1]
    dims = parts[:-1]
    if dtype.startswith("!"):
        return None
    if not re.fullmatch(r"(?:i\d+|f\d+|bf16|fp8[a-zA-Z0-9_]*)", dtype):
        return None
    shape = []
    dynamic_dims = 0
    for dim in dims:
        if dim == "?":
            shape.append(None)
            dynamic_dims += 1
        elif dim.isdigit():
            shape.append(int(dim))
        else:
            return None
    return dtype, tuple(shape), dynamic_dims


def extract_ttir_features(text):
    operations = []
    load_mask_count = 0
    store_mask_count = 0
    range_extents = []
    for line in text.splitlines():
        match = OP_RE.match(line)
        if not match:
            continue
        op = match.group("quoted") or match.group("plain")
        if not op or "." not in op:
            continue
        operations.append(op)
        if op in ("tt.load", "tt.store"):
            operands = line[match.end():].split(":", 1)[0].split("{", 1)[0].strip()
            operand_count = len([part for part in operands.split(",") if part.strip()])
            if op == "tt.load" and operand_count >= 2:
                load_mask_count += 1
            elif op == "tt.store" and operand_count >= 3:
                store_mask_count += 1
        if op == "tt.make_range":
            end = re.search(r"\bend\s*=\s*(\d+)", line)
            start = re.search(r"\bstart\s*=\s*(\d+)", line)
            if end:
                range_extents.append(max(0, int(end.group(1)) - int(start.group(1) if start else 0)))

    counts = Counter(operations)
    tt_count = sum(count for op, count in counts.items() if op.startswith("tt."))
    arith_count = sum(count for op, count in counts.items() if op.startswith("arith."))
    scf_count = sum(count for op, count in counts.items() if op.startswith("scf."))
    math_count = sum(count for op, count in counts.items() if op.startswith("math."))
    all_known_ops = set(TTIR_OPS) | set(ARITH_OPS) | set(SCF_OPS) | set(MATH_OPS)
    other_count = len(operations) - sum(counts[op] for op in all_known_ops)

    tensor_types = []
    for match in TENSOR_TYPE_RE.finditer(text):
        parsed = _parse_tensor_type(match.group(1))
        if parsed:
            tensor_types.append((match.group(1), *parsed))
    unique_tensor_types = {entry[0]: entry[1:] for entry in tensor_types}
    rank_counts = Counter(len(shape) for _, (_, shape, _) in unique_tensor_types.items())
    static_sizes = [
        _product(shape) for _, (_, shape, dynamic_dims) in unique_tensor_types.items()
        if dynamic_dims == 0
    ]
    dtype_counts = {
        dtype: len(re.findall(rf"(?<![\w]){re.escape(dtype)}(?![\w])", text))
        for dtype in DTYPES
    }

    values = {
        "operation_count": len(operations),
        "tt_operation_count": tt_count,
        "arith_operation_count": arith_count,
        "scf_operation_count": scf_count,
        "math_operation_count": math_count,
        "other_dialect_operation_count": other_count,
    }
    for op in (*TTIR_OPS, *ARITH_OPS, *SCF_OPS, *MATH_OPS):
        values[f"op_{op.replace('.', '_')}"] = counts[op]

    values.update({
        "program_id_axis_x_count": len(re.findall(r"\btt\.get_program_id\s+x\b", text)),
        "program_id_axis_y_count": len(re.findall(r"\btt\.get_program_id\s+y\b", text)),
        "program_id_axis_z_count": len(re.findall(r"\btt\.get_program_id\s+z\b", text)),
        "load_mask_operand_count": load_mask_count,
        "store_mask_operand_count": store_mask_count,
        "tensor_type_occurrence_count": len(tensor_types),
        "unique_tensor_type_count": len(unique_tensor_types),
        "tensor_static_elements_unique_sum": sum(static_sizes),
        "tensor_static_elements_max": max(static_sizes, default=0),
        "tensor_dynamic_dim_count": sum(dynamic_dims for _, _, _, dynamic_dims in tensor_types),
        "tensor_rank_max": max(rank_counts, default=0),
    })
    for rank in range(6):
        values[f"tensor_rank_{rank}_unique_type_count"] = rank_counts[rank]
    for dtype in DTYPES:
        values[f"dtype_{dtype}_ir_type_occurrence_count"] = dtype_counts[dtype]

    values.update({
        "make_range_count": counts["tt.make_range"],
        "make_range_max_extent": max(range_extents, default=0),
        "boundary_check_attribute_count": len(re.findall(r"\bboundaryCheck\s*=", text)),
        "padding_attribute_count": len(re.findall(r"\bpadding\s*=", text)),
        "divisibility_attribute_count": len(re.findall(r"\btt\.divisibility\s*=", text)),
        "divisibility_max": max(
            (int(value) for value in re.findall(r"\btt\.divisibility\s*=\s*(\d+)", text)),
            default=0,
        ),
    })
    for layout in LAYOUTS:
        values[f"layout_{layout}_attribute_count"] = len(
            re.findall(rf"#(?:ttg|tt)\.{re.escape(layout)}\b", text)
        )
    values["tensor_layout_attribute_count"] = sum(
        values[f"layout_{layout}_attribute_count"] for layout in LAYOUTS
    )
    load_count = counts["tt.load"]
    store_count = counts["tt.store"]
    values["masked_load_ratio"] = load_mask_count / load_count if load_count else 0.0
    values["masked_store_ratio"] = store_mask_count / store_count if store_count else 0.0

    if set(values) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(values))
        extra = sorted(set(values) - set(FEATURE_NAMES))
        raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
    if not all(isinstance(value, (int, float)) for value in values.values()):
        raise TypeError("All semantic features must be numeric")
    return values


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _load_metadata(cache_instance_dir, kernel):
    paths = sorted(
        path for path in cache_instance_dir.glob("*.json")
        if not path.name.startswith("__grp__")
    )
    if not paths:
        return {}, None
    preferred = [path for path in paths if path.stem == kernel]
    path = preferred[0] if preferred else paths[0]
    return json.loads(path.read_text(encoding="utf-8")), path


def _write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = (args.output_dir or analysis_dir / "semantic_v2_ttir").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_rows = []
    instance_rows = []
    parse_errors = []
    for kernel_record in manifest["kernels"]:
        kernel = kernel_record["name"]
        operator_type = kernel_record["operator_type"]
        source_path = analysis_dir / "artifacts" / kernel / "source.py"
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else ""
        ir_root = analysis_dir / "ir" / kernel
        cache_root = analysis_dir / "cache" / kernel
        for ir_path in sorted(ir_root.glob("*/kernel.ttir.mlir")):
            instance_id = ir_path.parent.name
            cache_instance_dir = cache_root / instance_id
            metadata, metadata_path = _load_metadata(cache_instance_dir, kernel)
            text = ir_path.read_text(encoding="utf-8", errors="replace")
            extraction_started = time.perf_counter_ns()
            try:
                features = extract_ttir_features(text)
            except Exception as error:
                parse_errors.append({
                    "kernel": kernel,
                    "instance_id": instance_id,
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            extraction_time_ms = (time.perf_counter_ns() - extraction_started) / 1_000_000
            ir_hash = hashlib.sha256(ir_path.read_bytes()).hexdigest()
            cache_ir_paths = sorted(
                path for path in cache_instance_dir.glob("*.ttir")
                if path.is_file()
            )
            matching_cache_irs = [
                path for path in cache_ir_paths
                if hashlib.sha256(path.read_bytes()).hexdigest() == ir_hash
            ]
            exact_cache_match = bool(matching_cache_irs)
            if matching_cache_irs:
                matching_metadata_path = matching_cache_irs[0].with_suffix(".json")
                if matching_metadata_path.is_file():
                    metadata_path = matching_metadata_path
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            input_signature_status = "not_captured_in_round_1"
            row = {
                "kernel": kernel,
                "operator_type": operator_type,
                "instance_id": instance_id,
                "source_sha256": source_hash,
                "ir_sha256": ir_hash,
                "cache_metadata_found": metadata_path is not None,
                "cache_ir_exact_match": exact_cache_match,
                "compile_ok": exact_cache_match,
                "correctness_status": "not_measured_compile_only",
                "feature_extraction_ok": True,
                "performance_label_valid": False,
                "kernel_name": metadata.get("kernel_name", ""),
                "target_backend": metadata.get("target", {}).get("backend", ""),
                "target_arch": metadata.get("target", {}).get("arch", metadata.get("arch", "")),
                "num_warps": metadata.get("num_warps", ""),
                "num_stages": metadata.get("num_stages", ""),
                "num_ctas": metadata.get("num_ctas", ""),
                "warp_size": metadata.get("warp_size", ""),
                "mix_mode": metadata.get("mix_mode", ""),
                "parallel_mode": metadata.get("parallel_mode", ""),
                "compile_metadata_sha256": (
                    hashlib.sha256(metadata_path.read_bytes()).hexdigest()
                    if metadata_path is not None else ""
                ),
                "runtime_input_signature_status": input_signature_status,
                "semantic_extraction_time_ms": round(extraction_time_ms, 6),
            }
            row.update(features)
            feature_rows.append(row)
            instance_rows.append({
                key: row[key]
                for key in (
                    "kernel", "operator_type", "instance_id", "source_sha256", "ir_sha256",
                    "cache_metadata_found", "cache_ir_exact_match", "compile_ok",
                    "correctness_status", "feature_extraction_ok", "performance_label_valid",
                    "kernel_name",
                    "target_backend", "target_arch", "num_warps", "num_stages", "num_ctas",
                    "warp_size", "mix_mode", "parallel_mode", "compile_metadata_sha256",
                    "runtime_input_signature_status", "semantic_extraction_time_ms",
                )
            })

    metadata_columns = [
        "kernel", "operator_type", "instance_id", "source_sha256", "ir_sha256",
        "cache_metadata_found", "cache_ir_exact_match", "compile_ok", "correctness_status",
        "feature_extraction_ok", "performance_label_valid", "kernel_name", "target_backend",
        "target_arch", "num_warps", "num_stages", "num_ctas", "warp_size", "mix_mode",
        "parallel_mode", "compile_metadata_sha256", "runtime_input_signature_status",
        "semantic_extraction_time_ms",
    ]
    _write_csv(output_dir / "ttir_semantic_features.csv", feature_rows, metadata_columns + FEATURE_NAMES)
    _write_csv(output_dir / "ttir_instances.csv", instance_rows, metadata_columns)

    per_kernel = []
    for kernel in manifest["kernels"]:
        subset = [row for row in feature_rows if row["kernel"] == kernel["name"]]
        per_kernel.append({
            "kernel": kernel["name"],
            "operator_type": kernel["operator_type"],
            "ttir_instances": len(subset),
            "cache_metadata_match_count": sum(row["cache_metadata_found"] for row in subset),
            "exact_cache_ir_match_count": sum(row["cache_ir_exact_match"] for row in subset),
            "distinct_ir_hashes": len({row["ir_sha256"] for row in subset}),
            "num_warps_values": sorted({row["num_warps"] for row in subset if row["num_warps"] != ""}),
            "num_stages_values": sorted({row["num_stages"] for row in subset if row["num_stages"] != ""}),
        })
    times_path = analysis_dir / "summary.json"
    baseline_summary = json.loads(times_path.read_text(encoding="utf-8"))
    summary = {
        "feature_version": "ttir_semantics_v2",
        "analysis_dir": str(analysis_dir),
        "kernel_count": len(manifest["kernels"]),
        "ttir_instance_count": len(feature_rows),
        "semantic_feature_width": len(FEATURE_NAMES),
        "semantic_feature_schema": FEATURE_NAMES,
        "all_values_numeric": all(
            all(isinstance(row[name], (int, float)) for name in FEATURE_NAMES)
            for row in feature_rows
        ),
        "instance_policy": "One row per raw TTIR dump hash; specializations are never averaged.",
        "instance_linkage": "Dump directory hash is joined to the same-named Triton cache directory and exact cached TTIR bytes are verified where available.",
        "runtime_input_signature_captured": False,
        "runtime_input_signature_note": "Round 1 did not persist actual runtime tensor shapes/strides/dtypes per JIT warmup; these are left explicitly uncaptured rather than inferred from incomplete IR types.",
        "cache_config_fields": ["target_arch", "num_warps", "num_stages", "num_ctas", "warp_size", "mix_mode", "parallel_mode"],
        "failed_parse_count": len(parse_errors),
        "failed_parses": parse_errors,
        "correctness_status": "not_measured_compile_only",
        "performance_label_valid": False,
        "semantic_extraction_time_ms": {
            "total": round(sum(row["semantic_extraction_time_ms"] for row in feature_rows), 6),
            "mean": round(
                statistics.fmean(row["semantic_extraction_time_ms"] for row in feature_rows), 6
            ) if feature_rows else None,
            "median": round(
                statistics.median(row["semantic_extraction_time_ms"] for row in feature_rows), 6
            ) if feature_rows else None,
        },
        "aggregate_counts": {
            "instances_per_kernel": per_kernel,
            "distinct_specialization_config_count": len({
                (
                    row["kernel"], row["ir_sha256"], row["num_warps"], row["num_stages"],
                    row["num_ctas"], row["mix_mode"], row["parallel_mode"],
                )
                for row in feature_rows
            }),
        },
        "baseline_v1_preserved": True,
        "baseline_summary_path": str(times_path),
        "baseline_ttir_extraction_time_ms": baseline_summary["stages"]["ttir"]["feature_extraction_time_ms"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_version": "ttir_semantics_v2",
                "feature_names": FEATURE_NAMES,
                "definitions": {
                    "op_*": "Exact count of the corresponding TTIR MLIR operation.",
                    "tensor_rank_*_unique_type_count": "Number of distinct static/dynamic tensor type strings at this rank.",
                    "tensor_static_elements_unique_sum": "Sum of element counts over distinct fully static tensor type strings; a structural-size proxy, not total runtime traffic.",
                    "load_mask_operand_count": "tt.load operations with an explicit mask operand.",
                    "store_mask_operand_count": "tt.store operations with an explicit mask operand.",
                    "loads_per_program_id": "tt.load count divided by get_program_id axis count, or zero when no axis op exists.",
                    "layout_*": "Count of explicit TTG/TT layout attributes; zero means no explicit layout was present in the captured TTIR.",
                },
                "metadata_columns": metadata_columns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "kernel_count": len(manifest["kernels"]),
        "ttir_instances": len(feature_rows),
        "feature_width": len(FEATURE_NAMES),
        "metadata_matches": sum(row["cache_metadata_found"] for row in feature_rows),
        "exact_cache_ir_matches": sum(row["cache_ir_exact_match"] for row in feature_rows),
        "parse_errors": len(parse_errors),
        "unique_instances": len({
            (row["kernel"], row["instance_id"]) for row in feature_rows
        }),
        "distinct_ir_hashes": len({row["ir_sha256"] for row in feature_rows}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
