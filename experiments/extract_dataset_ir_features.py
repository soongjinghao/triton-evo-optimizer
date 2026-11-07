#!/usr/bin/env python3
"""Extract fixed-width text features and compiler IR for representative dataset kernels."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
KERNELS = [
    ("_act_quant_kernel", "quantization"),
    ("_chunk_cumsum_fwd_kernel", "prefix_scan"),
    ("_copy_page_indices_kernel", "memory_copy"),
    ("_log_softmax_kernel", "softmax_reduction"),
    ("_pack_seq_kernel", "sequence_packing"),
    ("_per_group_transpose", "layout_transform"),
    ("_rms_norm_kernel", "normalization"),
    ("_selective_scan_update_kernel", "state_space_scan"),
    ("_fwd_kernel_ep_gather", "gather"),
    ("fused_qkvzba_split_reshape_cat_kernel", "fused_reshape"),
    ("matmul_kernel_simplified", "matrix_multiplication"),
    ("moe_align_block_size_stage", "moe_routing"),
    ("reshape_and_cache_flash_kernel", "kv_cache"),
    ("softplus", "elementwise_activation"),
    ("solve_tril_16x16_kernel", "triangular_solve"),
]

FEATURE_PATTERNS = {
    "jit_decorator_count": r"@(?:triton\.)?jit\b",
    "function_def_count": r"\bdef\s+[A-Za-z_]\w*|\bfunc\.func\b",
    "program_id_count": r"\b(?:program_id|get_program_id)\b",
    "range_count": r"\b(?:arange|make_range|range)\b",
    "load_count": r"\b(?:tl\.)?load\b",
    "store_count": r"\b(?:tl\.)?store\b",
    "add_count": r"\badd[fhi]?\b|(?<![+\w])\+(?![+=])",
    "sub_count": r"\bsub[fhi]?\b|(?<![-\w])-(?![-=>])",
    "mul_count": r"\bmul[fhi]?\b|(?<![*\w])\*(?![=*])",
    "div_count": r"\bdiv[fui]?\b|(?<![/\w])/(?![/=*])",
    "compare_count": r"\b(?:cmpi|cmpf|cmp|compare|eq|ne|slt|sle|sgt|sge)\b|==|!=|<=|>=",
    "where_select_count": r"\b(?:where|select|select_sc|cmp_select)\b",
    "reduce_count": r"\b(?:reduce|reduce_sum|reduce_max|reduce_min)\b",
    "sum_count": r"\b(?:sum|reduce_sum)\b",
    "max_count": r"\b(?:max|max_contiguous|reduce_max)\b",
    "min_count": r"\b(?:min|reduce_min)\b",
    "dot_count": r"\b(?:dot|dot_scaled|matmul|mm)\b",
    "exp_count": r"\b(?:exp|exp2|expm1)\b",
    "log_count": r"\b(?:log|log2|log1p)\b",
    "sqrt_count": r"\b(?:sqrt|rsqrt)\b",
    "cast_count": r"\b(?:cast|bitcast|to|convert|index_cast|trunci|extsi|extui)\b",
    "loop_count": r"\b(?:for|while|scf\.for|scf\.while)\b",
    "conditional_count": r"\b(?:if|scf\.if|cf\.cond_br)\b",
    "gather_count": r"\b(?:gather|gather_nd|index_select)\b",
    "transpose_count": r"\b(?:transpose|trans|permute|swapdims)\b",
    "scan_count": r"\b(?:scan|associative_scan|cum_sum|cumsum)\b",
    "softmax_count": r"\bsoftmax\b",
    "barrier_sync_count": r"\b(?:barrier|sync|fence|wait_flag|set_flag)\b",
    "allocation_count": r"\b(?:alloc|allocate|empty|zeros|full)\b",
    "pointer_op_count": r"\b(?:addptr|ptr|pointer|reinterpret_cast)\b",
    "tensor_memref_count": r"\b(?:tensor|memref)\b",
    "linalg_count": r"\blinalg\.[A-Za-z_]\w*",
    "hivm_count": r"\bhivm\.[A-Za-z_]\w*",
    "vector_count": r"\b(?:vector|vadd|vmul|vdiv|vmax|vmin)\b",
    "mask_count": r"\b(?:mask|masked|mask_fallback)\b",
    "python_import_count": r"^\s*(?:from\s+\S+\s+)?import\s+",
    "return_count": r"\breturn\b",
    "call_count": r"\b[A-Za-z_]\w*\s*\(",
}

BASE_FEATURES = [
    "char_count",
    "byte_count",
    "line_count",
    "nonempty_line_count",
    "comment_line_count",
    "token_count",
    "identifier_count",
    "unique_identifier_count",
    "numeric_literal_count",
    "string_literal_count",
    "delimiter_count",
]
FEATURE_NAMES = BASE_FEATURES + list(FEATURE_PATTERNS)
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[^\s]")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
NUMBER_RE = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
STRING_RE = re.compile(r"""(?s)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")
DELIMITER_RE = re.compile(r"[{}()\[\],;]")
COMMENT_RE = re.compile(r"^\s*(?:#|//)")
COMPILED_PATTERNS = {
    name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for name, pattern in FEATURE_PATTERNS.items()
}


def extract_features(text):
    lines = text.splitlines()
    identifiers = IDENTIFIER_RE.findall(text)
    values = [
        len(text),
        len(text.encode("utf-8")),
        len(lines),
        sum(bool(line.strip()) for line in lines),
        sum(bool(COMMENT_RE.match(line)) for line in lines),
        len(TOKEN_RE.findall(text)),
        len(identifiers),
        len(set(identifiers)),
        len(NUMBER_RE.findall(text)),
        len(STRING_RE.findall(text)),
        len(DELIMITER_RE.findall(text)),
    ]
    values.extend(len(COMPILED_PATTERNS[name].findall(text)) for name in FEATURE_PATTERNS)
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise ValueError("Feature extraction did not produce a finite fixed-width vector")
    return values


def rewrite_test_imports(text, kernel_name):
    escaped = re.escape(kernel_name)
    text = re.sub(
        rf"from\s+({escaped}|kernel)\s*(?:_\d+)?\s+import",
        f"from {kernel_name} import",
        text,
        flags=re.MULTILINE,
    )
    return re.sub(
        rf"import\s+({escaped}|kernel)\b((?:\s+as\s+\w+)?)",
        lambda match: f"import {kernel_name}{match.group(2)}",
        text,
        flags=re.MULTILINE,
    )


def find_source_file(kernel_dir, kernel_name, test_text):
    imported_modules = re.findall(r"^\s*from\s+([A-Za-z_]\w*)\s+import\b", test_text, re.MULTILINE)
    for module_name in imported_modules:
        if module_name == kernel_name or module_name.startswith(f"{kernel_name}_"):
            candidate = kernel_dir / f"{module_name}.py"
            if candidate.is_file():
                return candidate
    fallback = kernel_dir / f"{kernel_name}.py"
    return fallback if fallback.is_file() else None


WORKER = r'''
import json, os, runpy, sys, time, traceback
from pathlib import Path

kernel_name, source_file, test_file, output_file = sys.argv[1:]
import torch_npu
import torch
import triton
from triton.runtime.jit import JITFunction

attempts = 0
successes = 0
original_run = JITFunction.run
def compile_only(self, *args, **kwargs):
    global attempts, successes
    kwargs["warmup"] = True
    attempts += 1
    result = original_run(self, *args, **kwargs)
    successes += 1
    return result
JITFunction.run = compile_only
torch.testing.assert_close = lambda *args, **kwargs: None
started = time.perf_counter()
error = None
target = None
try:
    target = str(triton.runtime.driver.active.get_current_target())
    sys.path.insert(0, str(Path(source_file).parent))
    runpy.run_path(test_file, run_name="__main__")
except BaseException as exc:
    error = {
        "type": type(exc).__name__,
        "message": str(exc)[:1000],
        "traceback": traceback.format_exc()[-3000:],
    }
elapsed_ms = (time.perf_counter() - started) * 1000
status = "unknown" if attempts == 0 else ("fail" if successes < attempts else "pass")
Path(output_file).write_text(json.dumps({
    "kernel": kernel_name,
    "compile_status": status,
    "jit_warmup_calls": attempts,
    "successful_warmup_calls": successes,
    "compile_wall_ms": round(elapsed_ms, 3),
    "target": target,
    "error": error,
}, ensure_ascii=False), encoding="utf-8")
'''


def summarize_times(values):
    if not values:
        return {"total_ms": 0.0, "mean_ms": None, "median_ms": None}
    return {
        "total_ms": round(sum(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
    }


def find_ir_files(dump_dir, cache_dir, extension):
    files = sorted(dump_dir.rglob(f"kernel.{extension}.mlir")) if dump_dir.exists() else []
    if files:
        return files
    cache_suffix = {"ttir": ".ttir", "ttadapter": ".ttadapter"}.get(extension)
    if cache_suffix and cache_dir.exists():
        return sorted(
            path for path in cache_dir.rglob(f"*{cache_suffix}")
            if path.is_file() and not path.name.startswith("__grp__")
        )
    return []


def compile_kernel(kernel_name, test_path, source_path, output_dir, device_id, timeout):
    work_dir = Path(tempfile.mkdtemp(prefix=f".{kernel_name}-work-", dir=output_dir))
    cache_dir = output_dir / "cache" / kernel_name
    dump_dir = output_dir / "ir" / kernel_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / f"{kernel_name}.py").write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    test_text = rewrite_test_imports(test_path.read_text(encoding="utf-8"), kernel_name)
    test_copy = work_dir / f"test_{kernel_name}.py"
    test_copy.write_text(test_text, encoding="utf-8")
    result_path = work_dir / "compile_result.json"
    env = os.environ.copy()
    env.update({
        "ASCEND_DEVICE_ID": str(device_id),
        "ASCEND_RT_VISIBLE_DEVICES": str(device_id),
        "TRITON_CACHE_DIR": str(cache_dir),
        "TRITON_DUMP_DIR": str(dump_dir),
        "TRITON_DEBUG": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(work_dir), str(test_path.parent), str(ROOT)]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        ),
    })
    started = time.perf_counter()
    log_path = output_dir / "worker_logs" / f"{kernel_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [sys.executable, "-c", WORKER, kernel_name, str(work_dir / f"{kernel_name}.py"),
                 str(test_copy), str(result_path)],
                cwd=test_path.parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        wall_ms = (time.perf_counter() - started) * 1000
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "kernel": kernel_name,
                "compile_status": "unknown",
                "jit_warmup_calls": 0,
                "successful_warmup_calls": 0,
                "compile_wall_ms": round(wall_ms, 3),
                "target": None,
                "error": {"type": "WorkerError", "message": f"worker exited {process.returncode}"},
            }
    except subprocess.TimeoutExpired:
        wall_ms = (time.perf_counter() - started) * 1000
        result = {
            "kernel": kernel_name,
            "compile_status": "unknown",
            "jit_warmup_calls": 0,
            "successful_warmup_calls": 0,
            "compile_wall_ms": round(wall_ms, 3),
            "target": None,
            "error": {"type": "Timeout", "message": f"compile check exceeded {timeout}s"},
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    result["compile_wall_ms"] = round(wall_ms, 3)
    result["log_path"] = str(log_path.relative_to(output_dir))
    return result


def aggregate_stage(stage, paths, output_dir, kernel_name):
    vectors = []
    times = []
    artifact_paths = []
    for index, path in enumerate(paths):
        started = time.perf_counter_ns()
        text = path.read_text(encoding="utf-8", errors="replace")
        vector = extract_features(text)
        times.append((time.perf_counter_ns() - started) / 1_000_000)
        vectors.append(vector)
        artifact_dir = output_dir / "artifacts" / kernel_name / stage
        artifact_dir.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix or ".txt"
        copied = artifact_dir / f"{index:03d}_{path.stem}{suffix}"
        shutil.copyfile(path, copied)
        artifact_paths.append(str(copied.relative_to(output_dir)))
    if not vectors:
        return {
            "vector": None,
            "artifact_count": 0,
            "extraction_time_ms": 0.0,
            "artifact_paths": [],
        }
    width = len(FEATURE_NAMES)
    mean_vector = [
        math.fsum(vector[i] for vector in vectors) / len(vectors)
        for i in range(width)
    ]
    return {
        "vector": mean_vector,
        "artifact_count": len(vectors),
        "extraction_time_ms": round(sum(times), 3),
        "artifact_paths": artifact_paths,
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=240, help="per-kernel compile timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--kernels", nargs="+", choices=[name for name, _ in KERNELS])
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    compile_rows = []
    selected_kernels = [
        item for item in KERNELS
        if args.kernels is None or item[0] in args.kernels
    ]
    for index, (kernel_name, operator_type) in enumerate(selected_kernels, start=1):
        kernel_dir = DATASETS / kernel_name
        test_candidates = sorted(kernel_dir.glob(f"test*{kernel_name}*.py"))
        if not test_candidates:
            compile_rows.append({
                "kernel": kernel_name,
                "operator_type": operator_type,
                "compile_status": "unknown",
                "jit_warmup_calls": 0,
                "successful_warmup_calls": 0,
                "compile_wall_ms": "",
                "target": "",
                "error_type": "MissingSourceOrTest",
                "error_message": "",
                "log_path": "",
            })
            print(f"[{index}/{len(selected_kernels)}] {kernel_name}: missing source/test")
            continue
        test_path = test_candidates[0]
        source_path = find_source_file(
            kernel_dir, kernel_name, test_path.read_text(encoding="utf-8")
        )
        if source_path is None:
            compile_rows.append({
                "kernel": kernel_name,
                "operator_type": operator_type,
                "compile_status": "unknown",
                "jit_warmup_calls": 0,
                "successful_warmup_calls": 0,
                "compile_wall_ms": "",
                "target": "",
                "error_type": "MissingSource",
                "error_message": "",
                "log_path": "",
            })
            print(f"[{index}/{len(selected_kernels)}] {kernel_name}: missing source")
            continue
        source_stage = aggregate_stage("source", [source_path], output_dir, kernel_name)
        shutil.copyfile(source_path, output_dir / "artifacts" / kernel_name / "source.py")
        print(f"[{index}/{len(selected_kernels)}] compiling {kernel_name} ({operator_type})...", flush=True)
        compile_result = compile_kernel(
            kernel_name, test_path, source_path, output_dir, args.device_id, args.timeout
        )
        compile_result.update({
            "operator_type": operator_type,
            "source_file": str(source_path.relative_to(ROOT)),
            "test_file": str(test_path.relative_to(ROOT)),
        })
        compile_rows.append(compile_result)

        stage_data = {"source": source_stage}
        dump_dir = output_dir / "ir" / kernel_name
        cache_dir = output_dir / "cache" / kernel_name
        for stage, extension in (("ttir", "ttir"), ("ttadapter", "ttadapter"), ("npuir", "npuir")):
            paths = find_ir_files(dump_dir, cache_dir, extension)
            stage_data[stage] = aggregate_stage(stage, paths, output_dir, kernel_name)
        for stage, data in stage_data.items():
            records.append({
                "kernel": kernel_name,
                "operator_type": operator_type,
                "stage": stage,
                "vector_width": len(FEATURE_NAMES),
                "missing": data["vector"] is None,
                "artifact_count": data["artifact_count"],
                "extraction_time_ms": data["extraction_time_ms"],
                "artifact_paths": data["artifact_paths"],
                "vector": data["vector"],
            })
        print(
            f"  compile={compile_result['compile_status']} "
            f"jit-warmups={compile_result['jit_warmup_calls']} "
            f"IR(ttir/adapter/npuir)="
            f"{stage_data['ttir']['artifact_count']}/"
            f"{stage_data['ttadapter']['artifact_count']}/"
            f"{stage_data['npuir']['artifact_count']}",
            flush=True,
        )

    feature_rows = []
    for record in records:
        row = {
            "kernel": record["kernel"],
            "operator_type": record["operator_type"],
            "stage": record["stage"],
            "vector_width": record["vector_width"],
            "missing": record["missing"],
            "artifact_count": record["artifact_count"],
            "extraction_time_ms": record["extraction_time_ms"],
            "artifact_paths": ";".join(record["artifact_paths"]),
        }
        row.update(
            {name: ("" if record["vector"] is None else record["vector"][i])
             for i, name in enumerate(FEATURE_NAMES)}
        )
        feature_rows.append(row)
    write_csv(
        output_dir / "feature_vectors.csv",
        feature_rows,
        ["kernel", "operator_type", "stage", "vector_width", "missing",
         "artifact_count", "extraction_time_ms", "artifact_paths"] + FEATURE_NAMES,
    )

    normalized_compile_rows = []
    for item in compile_rows:
        error = item.get("error") or {}
        normalized_compile_rows.append({
            "kernel": item["kernel"],
            "operator_type": item["operator_type"],
            "compile_status": item["compile_status"],
            "jit_warmup_calls": item["jit_warmup_calls"],
            "successful_warmup_calls": item["successful_warmup_calls"],
            "compile_wall_ms": item.get("compile_wall_ms", ""),
            "target": item.get("target") or "",
            "error_type": error.get("type") or item.get("error_type") or "",
            "error_stage": (
                "post_compile_test" if item["compile_status"] == "pass" and error
                else "compile" if item["compile_status"] == "fail"
                else "test_or_setup" if error
                else ""
            ),
            "error_message": error.get("message") or item.get("error_message") or "",
            "source_file": item.get("source_file", ""),
            "test_file": item.get("test_file", ""),
            "log_path": item.get("log_path", ""),
        })
    write_csv(
        output_dir / "compile_results.csv",
        normalized_compile_rows,
        ["kernel", "operator_type", "compile_status", "jit_warmup_calls",
         "successful_warmup_calls", "compile_wall_ms", "target", "error_type",
         "error_stage",
         "error_message", "source_file", "test_file", "log_path"],
    )

    stages_summary = {}
    for stage in ("source", "ttir", "ttadapter", "npuir"):
        stage_records = [row for row in records if row["stage"] == stage]
        missing_count = sum(row["missing"] for row in stage_records)
        widths = sorted({row["vector_width"] for row in stage_records if not row["missing"]})
        times = [row["extraction_time_ms"] for row in stage_records]
        stages_summary[stage] = {
            "rows": len(stage_records),
            "artifacts": sum(row["artifact_count"] for row in stage_records),
            "missing_count": missing_count,
            "missing_rate": round(missing_count / len(selected_kernels), 6) if selected_kernels else None,
            "observed_vector_widths": widths,
            "fixed_width": len(widths) == 1 and widths[0] == len(FEATURE_NAMES),
            "feature_extraction_time_ms": summarize_times(times),
        }
    statuses = [row["compile_status"] for row in compile_rows]
    denominator = len(compile_rows)
    confirmed_failures = statuses.count("fail")
    summary = {
        "sample_count": len(selected_kernels),
        "feature_width": len(FEATURE_NAMES),
        "feature_vector_schema": FEATURE_NAMES,
        "fixed_width_verified": all(item["fixed_width"] for item in stages_summary.values()),
        "compile": {
            "confirmed_failure_count": confirmed_failures,
            "confirmed_failure_rate": round(confirmed_failures / denominator, 6) if denominator else None,
            "pass_count": statuses.count("pass"),
            "unknown_count": statuses.count("unknown"),
            "status_counts": {status: statuses.count(status) for status in ("pass", "fail", "unknown")},
            "compile_wall_time_ms": summarize_times([
                float(row["compile_wall_ms"])
                for row in compile_rows
                if row.get("compile_wall_ms") not in ("", None)
            ]),
            "post_compile_test_exception_count": sum(
                row["compile_status"] == "pass" and bool(row.get("error"))
                for row in compile_rows
            ),
            "wall_time_note": "compile_wall_ms includes worker startup, imports, and test warmup; no kernel execution or performance profiling.",
            "note": "Unknown means no JIT warmup was observed or the worker timed out; it is not counted as a compile failure. Post-compile test assertions can fail because warmup does not execute the kernel and are not counted as compile failures.",
        },
        "stages": stages_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(DATASETS),
        "output_dir": str(output_dir),
        "device_id": args.device_id,
        "compile_timeout_seconds": args.timeout,
        "python": sys.version.split()[0],
        "packages": {},
        "kernels": [{"name": name, "operator_type": kind} for name, kind in selected_kernels],
        "feature_extraction": "Shared lexical/statistical numeric schema applied to source and each IR text.",
    }
    for package in ("triton", "triton-ascend", "torch", "torch-npu"):
        try:
            manifest["packages"][package] = version(package)
        except PackageNotFoundError:
            manifest["packages"][package] = None
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults: {output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
