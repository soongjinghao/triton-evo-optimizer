#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile CostModel V2 candidates with Triton-Ascend's do_bench_npu.

For each candidate, the original dataset test is used as the correctness harness.
A generated sitecustomize patch intercepts target JITFunction.run calls:
  1. snapshot the first arguments for each unique launch signature;
  2. execute all real launches normally so the original assertions run;
  3. only after the full test passes, benchmark each unique signature once
     with do_bench_npu (not once per repeated launch in the test loop).

A candidate label is committed only when all requested outer repeats pass
correctness, observe target JIT launches, and return positive finite timings.
For moe_align_block_size_stage, which is a sequential three-kernel pipeline,
the per-stage medians are summed. Other operators use the median of unique
launch-signature timings; component timings are retained in each repeat record.

In the user's verified Triton-Ascend environment do_bench_npu returns ms.
The script converts to us before writing CostModel labels.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
V2_DIR = SCRIPT.parents[1]
DEFAULT_ROOT = V2_DIR.parent
ROLES = (
    "dataset_seed", "historical_best", "historical_p10",
    "historical_p50", "historical_p90",
)
MEASUREMENT_PROVENANCE = "correctness_first_all_jit_launches_do_bench_npu_v3"


def log(message: str) -> None:
    print(f"[profile] {message}", flush=True)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip(".") or "unnamed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "operator_name", "candidate_id", "selected_role", "source_sha256",
        "test_sha256", "label_latency_us", "latency_unit", "label_status",
        "measurement_count", "requested_repeats", "device_id",
        "latency_source", "compile_pass", "compile_status", "compile_job_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def resolve_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        parts = list(raw.parts)
        if "Agent" in parts:
            idx = parts.index("Agent")
            candidate = root.joinpath(*parts[idx + 1:])
            if candidate.exists():
                return candidate.resolve()
        return raw
    return (root / raw).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_key(row: dict[str, Any]) -> str:
    if row.get("candidate_id"):
        return str(row["candidate_id"])
    return "|".join(str(row.get(k) or "") for k in ("operator_name", "selected_role", "source_sha256"))


def as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return []


def parse_jit_names(source_path: Path) -> list[str]:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_jit = False
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            chain = []
            cur = target
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
            if chain and chain[0] == "jit":
                is_jit = True
                break
        if is_jit:
            names.append(node.name)
    return list(dict.fromkeys(names))


def rewrite_test_imports(test_source: str, module_name: str) -> str:
    base = re.escape(module_name)

    def replace_import(match: re.Match[str]) -> str:
        prefix, module, suffix = match.groups()
        if module == module_name or re.fullmatch(base + r"_\d+", module):
            return prefix + module_name + suffix
        if module == "kernel" or re.fullmatch(r"kernel_\d+", module):
            return prefix + module_name + suffix
        return match.group(0)

    test_source = re.sub(
        r"(?m)^(\s*from\s+)([A-Za-z_]\w*)(\s+import\b)",
        replace_import,
        test_source,
    )
    test_source = re.sub(
        r"(?m)^(\s*import\s+)([A-Za-z_]\w*)(\s*(?:as\s+\w+)?\s*(?:,|$))",
        replace_import,
        test_source,
    )
    if not re.search(r"(?m)^\s*import\s+torch_npu\b", test_source):
        test_source = "import torch_npu  # register Ascend backend\n" + test_source
    return test_source


def make_profile_sitecustomize_source(target_jit_names: list[str]) -> str:
    target_json = json.dumps(target_jit_names, ensure_ascii=False)
    template = r"""# Generated by costModel V2. Bench unique launch signatures after correctness.
import hashlib
import json
import math
import traceback

_STATE = {
    "patch_error": None,
    "target_jit_names": TARGET_NAMES_PLACEHOLDER,
    "launch_events": [],
    "event_by_signature": {},
    "pending_benchmarks": [],
    "observed_launch_count": 0,
}

try:
    import torch
    import torch_npu  # noqa: F401
    from triton.runtime.jit import JITFunction
    from triton.backends.ascend.testing import do_bench_npu

    _ORIGINAL_RUN = JITFunction.run
    _TARGETS = set(_STATE["target_jit_names"])

    def _clone_tree(value, memo=None):
        if memo is None:
            memo = {}
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        if isinstance(value, torch.Tensor):
            cloned = value.detach().clone(memory_format=torch.preserve_format)
            memo[obj_id] = cloned
            return cloned
        if isinstance(value, tuple):
            cloned = tuple(_clone_tree(v, memo) for v in value)
            memo[obj_id] = cloned
            return cloned
        if isinstance(value, list):
            cloned = [_clone_tree(v, memo) for v in value]
            memo[obj_id] = cloned
            return cloned
        if isinstance(value, dict):
            cloned = {k: _clone_tree(v, memo) for k, v in value.items()}
            memo[obj_id] = cloned
            return cloned
        return value

    def _kernel_name(jit_fn):
        return (
            getattr(jit_fn, "__name__", None)
            or getattr(jit_fn, "name", None)
            or getattr(getattr(jit_fn, "fn", None), "__name__", None)
            or "<unknown>"
        )

    def _describe(value, depth=0, include_scalar_values=True):
        if depth > 6:
            return {"type": type(value).__name__}
        if isinstance(value, torch.Tensor):
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "stride": list(value.stride()) if value.layout == torch.strided else None,
            }
        if isinstance(value, (tuple, list)):
            return [ _describe(v, depth + 1, include_scalar_values) for v in value ]
        if isinstance(value, dict):
            return {
                str(k): _describe(v, depth + 1, include_scalar_values)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                if str(k) not in {"stream", "warmup"}
            }
        if value is None or isinstance(value, (bool, int, float, str)):
            if not include_scalar_values:
                return {"type": type(value).__name__}
            if isinstance(value, float) and not math.isfinite(value):
                return {"type": "float", "value": repr(value)}
            return {"type": type(value).__name__, "value": value}
        # Do not put object addresses into the signature (e.g. stream objects).
        return {"type": type(value).__name__}

    def _launch_signature(jit_fn, args, kwargs):
        name = _kernel_name(jit_fn)
        payload = {
            "kernel": name,
            # Positional scalar arguments are commonly runtime indices that
            # change inside test loops. Ignore their values to deduplicate them;
            # retain tensor metadata and keyword launch/constexpr settings.
            "args": _describe(args, include_scalar_values=False),
            "kwargs": _describe(kwargs, include_scalar_values=True),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), payload

    def _profiled_run(self, *args, **kwargs):
        name = _kernel_name(self)
        if name not in _TARGETS or kwargs.get("warmup", False):
            return _ORIGINAL_RUN(self, *args, **kwargs)

        _STATE["observed_launch_count"] += 1
        signature, signature_desc = _launch_signature(self, args, kwargs)
        event = _STATE["event_by_signature"].get(signature)

        if event is None:
            event = {
                "kernel_name": name,
                "launch_signature": signature,
                "signature_description": signature_desc,
                "observed_launch_count": 0,
                "latency_ms": None,
                "latency_us": None,
                "benchmark_error": None,
                "launch_error": None,
                "benchmark_phase": "after_correctness_test",
            }
            _STATE["event_by_signature"][signature] = event
            _STATE["launch_events"].append(event)

            # Snapshot tensors at the first real launch, before the test can
            # mutate/reuse them. The benchmark itself runs only after the test
            # has completed successfully.
            _STATE["pending_benchmarks"].append((self, _clone_tree(args), _clone_tree(kwargs), event))

        event["observed_launch_count"] += 1

        try:
            return _ORIGINAL_RUN(self, *args, **kwargs)
        except BaseException as exc:
            event["launch_error"] = "%s: %s" % (type(exc).__name__, exc)
            event["launch_traceback"] = traceback.format_exc()
            raise

    def _run_pending_benchmarks():
        # Called only after the original test and all its assertions pass.
        for jit_fn, bench_args, bench_kwargs, event in _STATE["pending_benchmarks"]:
            try:
                torch.npu.synchronize()
                def _bench_once():
                    return _ORIGINAL_RUN(jit_fn, *bench_args, **bench_kwargs)
                elapsed_ms = float(do_bench_npu(_bench_once))
                torch.npu.synchronize()
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise RuntimeError("do_bench_npu returned invalid latency: %r" % elapsed_ms)
                event["latency_ms"] = elapsed_ms
                event["latency_us"] = elapsed_ms * 1000.0
                print(
                    "[costmodel-bench] %s signature=%s latency=%.6f ms observed=%d"
                    % (event["kernel_name"], event["launch_signature"][:12], elapsed_ms,
                       event["observed_launch_count"]),
                    flush=True,
                )
            except BaseException as exc:
                event["benchmark_error"] = "%s: %s" % (type(exc).__name__, exc)
                event["benchmark_traceback"] = traceback.format_exc()
                print(
                    "[costmodel-bench] FAILED %s: %s" % (event["kernel_name"], event["benchmark_error"]),
                    flush=True,
                )
        return _STATE["launch_events"]

    JITFunction.run = _profiled_run

except BaseException as exc:
    _STATE["patch_error"] = "%s: %s" % (type(exc).__name__, exc)
    _STATE["patch_traceback"] = traceback.format_exc()
"""
    return template.replace("TARGET_NAMES_PLACEHOLDER", target_json)


def stage_candidate(
    row: dict[str, Any],
    root: Path,
    test_dir: Path,
    patch_dir: Path,
    target_jit_names: list[str],
) -> tuple[Path, str]:
    source_path = resolve_path(root, row.get("source_path"))
    test_path = resolve_path(root, row.get("test_path"))
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError(f"candidate source missing: {source_path}")
    if test_path is None or not test_path.is_file():
        raise FileNotFoundError(f"candidate test missing: {test_path}")

    module_name = str(row.get("expected_test_module_name") or row.get("operator_name"))
    if not re.fullmatch(r"[A-Za-z_]\w*", module_name):
        raise ValueError(f"Invalid expected_test_module_name: {module_name!r}")

    test_dir.mkdir(parents=True, exist_ok=True)
    support_root = test_path.parent
    for support_path in support_root.rglob("*.py"):
        if "__pycache__" in support_path.parts or support_path.name.lower().startswith("test_"):
            continue
        relative = support_path.relative_to(support_root)
        target = test_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(support_path, target)

    candidate_module = test_dir / (module_name + ".py")
    candidate_module.parent.mkdir(parents=True, exist_ok=True)
    candidate_module.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    test_code = test_path.read_text(encoding="utf-8")
    worker_test = test_dir / ("test_" + safe_name(module_name) + ".py")
    worker_test.write_text(rewrite_test_imports(test_code, module_name), encoding="utf-8")

    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "sitecustomize.py").write_text(
        make_profile_sitecustomize_source(target_jit_names),
        encoding="utf-8",
    )
    return worker_test, module_name


def worker_main() -> int:
    result_path = Path(os.environ["COSTMODEL_WORKER_RESULT"]).resolve()
    test_file = Path(os.environ["COSTMODEL_TEST_FILE"]).resolve()
    test_dir = Path(os.environ["COSTMODEL_TEST_DIR"]).resolve()
    device_id = int(os.environ.get("COSTMODEL_DEVICE_ID", "0"))

    result: dict[str, Any] = {
        "status": "failed",
        "test_error": None,
        "test_traceback": None,
        "device_id": device_id,
        "torch_version": None,
        "triton_version": None,
        "launch_events": [],
        "benchmark_phase": "after_correctness_test",
    }
    patch_module = None
    patch_state: dict[str, Any] = {}
    try:
        import torch
        import torch_npu  # noqa: F401
        import triton
        result["torch_version"] = str(torch.__version__)
        result["triton_version"] = str(triton.__version__)
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is False")
        torch.npu.set_device(device_id)

        patch_module = sys.modules.get("sitecustomize")
        patch_state = getattr(patch_module, "_STATE", {}) if patch_module else {}
        if not patch_module:
            raise RuntimeError("Generated sitecustomize.py was not imported")
        if patch_state.get("patch_error"):
            raise RuntimeError("Profiling patch initialization failed: " + str(patch_state["patch_error"]))

        import runpy
        os.chdir(test_dir)
        sys.path.insert(0, str(test_dir))

        # Phase 1: run the unmodified test. JITFunction.run is wrapped only to
        # execute the original launch and snapshot unique launch arguments.
        runpy.run_path(str(test_file), run_name="__main__")
        torch.npu.synchronize()
        result["status"] = "test_passed"

        # Phase 2: correctness has passed. Now benchmark each unique signature
        # once, regardless of how many times the test loop launched it.
        benchmark_fn = getattr(patch_module, "_run_pending_benchmarks", None)
        if not callable(benchmark_fn):
            raise RuntimeError("Profiling benchmark callback is unavailable")
        benchmark_events = benchmark_fn()
        torch.npu.synchronize()

        # An empty event list is NOT a successful benchmark. It means the
        # selected JIT names did not match any kernel actually launched by
        # the correctness test (or the test path bypassed the patched entry).
        if not benchmark_events:
            result["benchmark_status"] = "failed"
            result["benchmark_error"] = (
                "no_target_jit_launch_observed: target_jit_names="
                + repr(patch_state.get("target_jit_names", []))
            )
        else:
            invalid_events = []
            for event in benchmark_events:
                latency = event.get("latency_us")
                valid_latency = (
                    isinstance(latency, (int, float))
                    and math.isfinite(float(latency))
                    and float(latency) > 0
                )
                if (
                    event.get("benchmark_error")
                    or event.get("launch_error")
                    or not valid_latency
                ):
                    invalid_events.append(event)

            if invalid_events:
                result["benchmark_status"] = "failed"
                result["benchmark_error"] = (
                    "invalid_or_failed_benchmark_events: "
                    + repr([
                        {
                            "kernel_name": e.get("kernel_name"),
                            "benchmark_error": e.get("benchmark_error"),
                            "launch_error": e.get("launch_error"),
                            "latency_us": e.get("latency_us"),
                        }
                        for e in invalid_events
                    ])
                )
            else:
                result["benchmark_status"] = "success"
                result["benchmark_error"] = None

    except BaseException as exc:
        if result.get("status") != "test_passed":
            result["status"] = "test_failed"
        else:
            result["benchmark_status"] = "failed"
        result["test_error"] = f"{type(exc).__name__}: {exc}"
        result["test_traceback"] = traceback.format_exc()

    patch_module = sys.modules.get("sitecustomize")
    patch_state = getattr(patch_module, "_STATE", {}) if patch_module else {}
    if result.get("status") != "test_passed":
        for event in patch_state.get("launch_events", []):
            if event.get("latency_us") is None and not event.get("benchmark_error"):
                event["benchmark_skipped_reason"] = "correctness_test_failed"
    result["patch_error"] = patch_state.get("patch_error")
    result["target_jit_names"] = patch_state.get("target_jit_names", [])
    result["launch_events"] = patch_state.get("launch_events", [])
    result["observed_launch_count"] = patch_state.get("observed_launch_count", 0)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def run_one_repeat(
    row: dict[str, Any],
    root: Path,
    output_dir: Path,
    repeat_index: int,
    device_id: int,
    timeout_seconds: int,
    keep_work: bool,
) -> dict[str, Any]:
    op = str(row["operator_name"])
    cid = candidate_key(row)
    source_path = resolve_path(root, row.get("source_path"))
    test_path = resolve_path(root, row.get("test_path"))
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError(f"{cid}: source missing: {source_path}")
    if test_path is None or not test_path.is_file():
        raise FileNotFoundError(f"{cid}: test missing: {test_path}")

    actual_source_hash = sha256_file(source_path)
    actual_test_hash = sha256_file(test_path)
    if actual_source_hash != str(row.get("source_sha256") or "").lower():
        raise ValueError(f"{cid}: source SHA256 mismatch; selected_candidates.jsonl is stale")
    if actual_test_hash != str(row.get("test_sha256") or "").lower():
        raise ValueError(f"{cid}: test SHA256 mismatch; selected_candidates.jsonl is stale")

    jit_names = parse_jit_names(source_path)
    if not jit_names:
        raise RuntimeError(f"{cid}: no @triton.jit function found in source")

    # The operator name is not necessarily the JIT kernel that the test uses.
    # A candidate may contain several JIT implementations selected by runtime
    # shape/configuration branches (e.g. RMSNorm's full_row vs generic kernel).
    # Monitor every JIT function declared by this candidate; only functions
    # actually launched by the correctness test will create benchmark events.
    target_names = list(dict.fromkeys(jit_names))

    work_root = output_dir / "work"
    logs_root = output_dir / "logs"
    work_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    safe_id = safe_name(cid)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{safe_id}_r{repeat_index:02d}_", dir=str(work_root)))
    test_dir = work_dir / "test_env"
    patch_dir = work_dir / "patch"
    home_dir = work_dir / "home"
    cache_dir = work_dir / "triton_cache"
    result_path = work_dir / "worker_result.json"
    for directory in (test_dir, patch_dir, home_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    worker_test, module_name = stage_candidate(row, root, test_dir, patch_dir, target_names)
    env = os.environ.copy()
    env["COSTMODEL_TEST_FILE"] = str(worker_test.resolve())
    env["COSTMODEL_TEST_DIR"] = str(test_dir.resolve())
    env["COSTMODEL_WORKER_RESULT"] = str(result_path.resolve())
    env["COSTMODEL_DEVICE_ID"] = str(device_id)
    env["HOME"] = str(home_dir.resolve())
    env["TRITON_CACHE_DIR"] = str(cache_dir.resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        [str(patch_dir.resolve()), str(test_dir.resolve()), str(root.resolve())]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    log_path = logs_root / f"{safe_id}_repeat_{repeat_index:02d}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--_worker"],
                cwd=str(test_dir),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if result_path.is_file():
            worker_result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            worker_result = {
                "status": "worker_no_result",
                "test_error": f"worker exited {completed.returncode} without worker_result.json",
            }
        worker_result["worker_returncode"] = completed.returncode
    except subprocess.TimeoutExpired:
        worker_result = {"status": "timeout", "test_error": f"worker timed out after {timeout_seconds}s"}
    except Exception as exc:
        worker_result = {
            "status": "worker_error",
            "test_error": f"{type(exc).__name__}: {exc}",
            "test_traceback": traceback.format_exc(),
        }

    events = worker_result.get("launch_events") or []
    valid_latencies = [
        float(event["latency_us"])
        for event in events
        if isinstance(event, dict)
        and event.get("latency_us") is not None
        and math.isfinite(float(event["latency_us"]))
        and float(event["latency_us"]) > 0
        and not event.get("benchmark_error")
        and not event.get("launch_error")
    ]
    event_errors = [
        event for event in events
        if event.get("benchmark_error") or event.get("launch_error")
    ]
    repeat_success = (
        worker_result.get("status") == "test_passed"
        and worker_result.get("benchmark_status") == "success"
        and bool(events)
        and bool(valid_latencies)
        and not event_errors
    )

    benchmark_error = worker_result.get("benchmark_error")
    if not benchmark_error and worker_result.get("status") == "test_passed" and not events:
        benchmark_error = (
            "no_target_jit_launch_observed: target_jit_names="
            + repr(worker_result.get("target_jit_names", target_names))
        )
    if not benchmark_error and worker_result.get("benchmark_status") != "success":
        benchmark_error = "benchmark_status_not_success"

    kernel_values: dict[str, list[float]] = {}
    for event in events:
        if (
            isinstance(event, dict)
            and event.get("latency_us") is not None
            and not event.get("benchmark_error")
            and not event.get("launch_error")
        ):
            kernel_values.setdefault(str(event.get("kernel_name") or "<unknown>"), []).append(
                float(event["latency_us"])
            )
    kernel_medians = {
        name: float(statistics.median(values))
        for name, values in kernel_values.items() if values
    }

    aggregation_error = None
    if repeat_success and op == "moe_align_block_size_stage":
        # This operator is a known sequential pipeline. The token kernel is
        # launched twice with stage=0 and stage=1, followed by stage2/stage3.
        # The launch signature includes the stage keyword, so sum all unique
        # stage/signature measurements to approximate the full pipeline time.
        required_stages = {
            "moe_align_block_size_token",
            "moe_align_block_size_stage2",
            "moe_align_block_size_stage3",
        }
        missing_stages = sorted(required_stages - set(kernel_values))
        token_signatures = sum(
            1 for event in events
            if event.get("kernel_name") == "moe_align_block_size_token"
            and event.get("latency_us") is not None
        )
        if missing_stages or token_signatures < 2:
            repeat_success = False
            repeat_latency = None
            aggregation_error = (
                f"incomplete_moe_pipeline: missing_stages={missing_stages}, "
                f"token_stage_signatures={token_signatures}/2"
            )
            aggregation_policy = "sum_of_unique_stage_launch_signature_latencies"
        else:
            repeat_latency = float(sum(valid_latencies))
            aggregation_policy = "sum_of_unique_stage_launch_signature_latencies"
    elif repeat_success:
        # A test may exercise multiple shapes or alternative JIT variants.
        # Use the median of unique launch signatures, not repeated test-loop
        # launches (which are deduplicated before benchmarking).
        repeat_latency = float(statistics.median(valid_latencies))
        aggregation_policy = "median_of_unique_launch_signatures"
    else:
        repeat_latency = None
        aggregation_policy = "not_available"

    result = {
        "repeat_index": repeat_index,
        "success": repeat_success,
        "latency_us": repeat_latency,
        "launch_latencies_us": valid_latencies,
        "kernel_latency_medians_us": kernel_medians,
        "aggregation_policy": aggregation_policy,
        "aggregation_error": aggregation_error,
        "unique_launch_signature_count": len(events),
        "observed_launch_count": worker_result.get("observed_launch_count"),
        "launch_events": events,
        "target_jit_names": worker_result.get("target_jit_names", target_names),
        "test_status": worker_result.get("status"),
        "benchmark_status": worker_result.get("benchmark_status"),
        "benchmark_error": benchmark_error,
        "test_error": worker_result.get("test_error"),
        "test_traceback": worker_result.get("test_traceback"),
        "patch_error": worker_result.get("patch_error"),
        "worker_returncode": worker_result.get("worker_returncode"),
        "log_path": str(log_path.resolve()),
        "module_name": module_name,
    }
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        result["work_dir"] = str(work_dir.resolve())
    return result


def load_ir_index(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        key = candidate_key(row)
        if key in result:
            raise ValueError(f"Duplicate candidate_id in IR manifest: {key}")
        result[key] = row
    return result


def eligible_reason(row: dict[str, Any], ir_row: dict[str, Any] | None, root: Path) -> tuple[bool, str]:
    if ir_row is None:
        return False, "missing_ir_manifest_row"
    if str(ir_row.get("operator_name") or "") != str(row.get("operator_name") or ""):
        return False, "ir_manifest_operator_mismatch"
    if str(ir_row.get("source_sha256") or "").lower() != str(row.get("source_sha256") or "").lower():
        return False, "ir_manifest_source_hash_mismatch"
    if str(ir_row.get("test_sha256") or "").lower() != str(row.get("test_sha256") or "").lower():
        return False, "ir_manifest_test_hash_mismatch"
    selected_role = str(row.get("selected_role") or row.get("candidate_type") or "")
    ir_role = str(ir_row.get("selected_role") or ir_row.get("candidate_type") or "")
    if ir_role and selected_role and ir_role != selected_role:
        return False, "ir_manifest_role_mismatch"
    if ir_row.get("compile_pass") is False:
        return False, "compile_pass_false"

    ir_paths = ir_row.get("ir_paths") or {}
    ttir_paths = as_path_list(ir_paths.get("ttir") if isinstance(ir_paths, dict) else None)
    ttir_ok = any(
        (path := resolve_path(root, value)) is not None and path.is_file() and path.stat().st_size > 0
        for value in ttir_paths
    )
    if not ttir_ok:
        return False, "ttir_missing"

    source = resolve_path(root, row.get("source_path"))
    test = resolve_path(root, row.get("test_path"))
    if source is None or not source.is_file():
        return False, "source_missing"
    if test is None or not test.is_file():
        return False, "test_missing"
    if sha256_file(source) != str(row.get("source_sha256") or "").lower():
        return False, "source_sha256_mismatch"
    if sha256_file(test) != str(row.get("test_sha256") or "").lower():
        return False, "test_sha256_mismatch"
    return True, "eligible"


def record_key(row: dict[str, Any], device_id: int, repeats: int) -> tuple[str, str, str, int, int]:
    return (
        candidate_key(row),
        str(row.get("source_sha256") or ""),
        str(row.get("test_sha256") or ""),
        device_id,
        repeats,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--selected-candidates", default="costModel_v2/data/selected_candidates.jsonl")
    ap.add_argument("--ir-manifest", default="costModel_v2/data/candidate_ir_manifest.jsonl")
    ap.add_argument("--output-dir", default="costModel_v2/labels")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5, help="Successful outer test+benchmark repeats per candidate")
    ap.add_argument("--timeout", type=int, default=900, help="Timeout seconds for one repeat subprocess")
    ap.add_argument("--limit", type=int, default=0, help="0 means all pending candidates")
    ap.add_argument("--operator", action="append", default=[], help="Filter by operator; repeatable")
    ap.add_argument("--role", action="append", default=[], choices=list(ROLES), help="Filter by role; repeatable")
    ap.add_argument("--force", action="store_true", help="Rerun even if a successful label exists")
    ap.add_argument("--keep-work", action="store_true", help="Keep per-repeat temporary work directories")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args._worker:
        return worker_main()
    if args.repeats < 1 or args.timeout < 1 or args.limit < 0:
        raise ValueError("--repeats/--timeout must be positive and --limit must be >= 0")

    root = Path(args.root).expanduser().resolve()
    selected_path = resolve_path(root, args.selected_candidates)
    ir_manifest_path = resolve_path(root, args.ir_manifest)
    output_dir = resolve_path(root, args.output_dir)
    if selected_path is None or ir_manifest_path is None or output_dir is None:
        raise ValueError("Invalid input/output path")

    selected_rows = read_jsonl(selected_path)
    ir_index = load_ir_index(ir_manifest_path)
    operators = set(args.operator or [])
    roles = set(args.role or [])
    records_path = output_dir / "candidate_profile_records.jsonl"
    summary_path = output_dir / "profile_summary.json"
    csv_path = output_dir / "candidate_labels.csv"
    prior = read_jsonl(records_path) if records_path.is_file() else []

    prior_by_key = {
        (
            str(rec.get("candidate_id") or ""),
            str(rec.get("source_sha256") or ""),
            str(rec.get("test_sha256") or ""),
            int(rec.get("device_id", -1)),
            int(rec.get("requested_repeats", -1)),
        ): rec
        for rec in prior
    }

    candidates = []
    skipped_reasons = Counter()
    for row in selected_rows:
        op = str(row.get("operator_name") or "")
        role = str(row.get("selected_role") or row.get("candidate_type") or "")
        if operators and op not in operators:
            continue
        if roles and role not in roles:
            continue
        ok, reason = eligible_reason(row, ir_index.get(candidate_key(row)), root)
        if not ok:
            skipped_reasons[reason] += 1
            continue
        candidates.append(row)

    candidates.sort(key=lambda row: (
        str(row.get("operator_name", "")),
        ROLES.index(str(row.get("selected_role") or row.get("candidate_type")))
        if str(row.get("selected_role") or row.get("candidate_type")) in ROLES else 99,
        candidate_key(row),
    ))

    pending = []
    for row in candidates:
        key = record_key(row, args.device_id, args.repeats)
        old = prior_by_key.get(key)
        if not args.force and old and old.get("label_status") == "success":
            old_values_ok = len(old.get("measurement_values_us") or []) == args.repeats
            aggregation_changed = (
                str(row.get("operator_name") or "") == "moe_align_block_size_stage"
                and old.get("measurement_provenance") != MEASUREMENT_PROVENANCE
            )
            if old_values_ok and not aggregation_changed:
                continue
        pending.append(row)
    if args.limit:
        pending = pending[:args.limit]

    log("=" * 72)
    log("CostModel V2 do_bench_npu candidate profiling")
    log(f"Selected candidate rows : {len(selected_rows)}")
    log(f"IR/profile eligible     : {len(candidates)}")
    log(f"Need profiling this run : {len(pending)}")
    log(f"Repeats per candidate   : {args.repeats}")
    log(f"Device                  : npu:{args.device_id}")
    log(f"Skipped reasons         : {dict(skipped_reasons)}")
    log(f"Records                 : {records_path}")
    if args.dry_run:
        for row in pending:
            log(f"  {row.get('operator_name')} | {row.get('selected_role')} | {candidate_key(row)}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_key = {
        (
            str(rec.get("candidate_id") or ""),
            str(rec.get("source_sha256") or ""),
            str(rec.get("test_sha256") or ""),
            int(rec.get("device_id", -1)),
            int(rec.get("requested_repeats", -1)),
        ): rec
        for rec in prior
    }

    for candidate_index, row in enumerate(pending, 1):
        op = str(row["operator_name"])
        cid = candidate_key(row)
        key = record_key(row, args.device_id, args.repeats)
        old = records_by_key.get(key)
        resume_values: list[float] = []
        run_details: list[dict[str, Any]] = []
        if (
            not args.force and old
            and old.get("measurement_provenance") == MEASUREMENT_PROVENANCE
            and old.get("label_status") == "profiling_in_progress"
            and int(old.get("requested_repeats", 0)) == args.repeats
        ):
            resume_values = [
                float(v) for v in (old.get("measurement_values_us") or [])
                if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0
            ]
            run_details = list(old.get("run_details") or [])

        ir_row = ir_index.get(cid) or {}
        previous_attempts = list(old.get("previous_attempts") or []) if old else []
        if old and old.get("measurement_provenance") != MEASUREMENT_PROVENANCE:
            previous_attempts.append({
                "measurement_provenance": old.get("measurement_provenance"),
                "label_status": old.get("label_status"),
                "profile_error": old.get("profile_error"),
                "measurement_values_us": old.get("measurement_values_us", []),
                "run_details": old.get("run_details", []),
                "updated_at": old.get("updated_at"),
            })
        record = {
            "schema_version": "2.0",
            "operator_name": op,
            "candidate_id": cid,
            "selected_role": row.get("selected_role") or row.get("candidate_type"),
            "candidate_type": row.get("selected_role") or row.get("candidate_type"),
            "source_path": row.get("source_path"),
            "source_sha256": row.get("source_sha256"),
            "test_path": row.get("test_path"),
            "test_sha256": row.get("test_sha256"),
            "compile_job_id": row.get("compile_job_id"),
            "compile_status": ir_row.get("compile_status") or ir_row.get("status"),
            "compile_pass": ir_row.get("compile_pass"),
            "ir_paths": ir_row.get("ir_paths") or {},
            "ir_found": ir_row.get("ir_found") or {},
            "device_id": args.device_id,
            "requested_repeats": args.repeats,
            "latency_unit": "us",
            "latency_source": "triton.backends.ascend.testing.do_bench_npu",
            "measurement_provenance": MEASUREMENT_PROVENANCE,
            "aggregation_policy": (
                "sum_of_unique_stage_kernel_medians"
                if op == "moe_align_block_size_stage"
                else "median_of_unique_launch_signatures"
            ),
            "previous_attempts": previous_attempts,
            "measurement_values_us": resume_values,
            "measurement_count": len(resume_values),
            "run_details": run_details,
            "label_latency_us": None,
            "label_status": "profiling_in_progress",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        records_by_key[key] = record
        write_jsonl(records_path, list(records_by_key.values()))

        log(f"\n[{candidate_index}/{len(pending)}] {op} | {record['selected_role']} | {cid}")
        for repeat_index in range(len(resume_values) + 1, args.repeats + 1):
            log(f"  repeat {repeat_index}/{args.repeats}: correctness test + do_bench_npu")
            try:
                detail = run_one_repeat(
                    row=row,
                    root=root,
                    output_dir=output_dir,
                    repeat_index=repeat_index,
                    device_id=args.device_id,
                    timeout_seconds=args.timeout,
                    keep_work=args.keep_work,
                )
            except Exception as exc:
                detail = {
                    "repeat_index": repeat_index,
                    "success": False,
                    "test_status": "parent_error",
                    "test_error": f"{type(exc).__name__}: {exc}",
                    "test_traceback": traceback.format_exc(),
                }
            record["run_details"].append(detail)

            if detail.get("success") and isinstance(detail.get("latency_us"), (int, float)):
                record["measurement_values_us"].append(float(detail["latency_us"]))
                record["measurement_count"] = len(record["measurement_values_us"])
                log(f"    OK: {float(detail['latency_us']):.3f} us")
            else:
                record["label_status"] = "profile_failed"
                record["profile_error"] = (
                    detail.get("test_error")
                    or detail.get("patch_error")
                    or detail.get("benchmark_error")
                    or detail.get("aggregation_error")
                    or "benchmark/test failed"
                )
                log(f"    FAILED: {record['profile_error']}")
                break

            record["updated_at"] = datetime.now().isoformat(timespec="seconds")
            records_by_key[key] = record
            write_jsonl(records_path, list(records_by_key.values()))

        if record["label_status"] != "profile_failed":
            if len(record["measurement_values_us"]) == args.repeats:
                record["label_latency_us"] = float(statistics.median(record["measurement_values_us"]))
                record["label_status"] = "success"
            else:
                record["label_status"] = "profile_failed"
                record["profile_error"] = f"incomplete_measurements:{len(record['measurement_values_us'])}/{args.repeats}"
        record["updated_at"] = datetime.now().isoformat(timespec="seconds")
        records_by_key[key] = record
        write_jsonl(records_path, list(records_by_key.values()))
        if record["label_status"] == "success":
            log(f"  => LABEL OK: median={record['label_latency_us']:.3f} us")
        else:
            log(f"  => LABEL NOT COMMITTED: {record.get('profile_error')}")

    all_records = list(records_by_key.values())
    all_records.sort(key=lambda rec: (
        str(rec.get("operator_name", "")),
        ROLES.index(str(rec.get("selected_role"))) if rec.get("selected_role") in ROLES else 99,
        str(rec.get("candidate_id", "")),
    ))
    write_jsonl(records_path, all_records)
    write_csv(csv_path, all_records)

    status_counts = Counter(str(rec.get("label_status") or "unknown") for rec in all_records)
    summary = {
        "schema_version": "1.0",
        "selected_candidate_count": len(selected_rows),
        "profile_eligible_after_filters": len(candidates),
        "profiled_or_resumed_this_run": len(pending),
        "profile_record_count": len(all_records),
        "label_status_counts": dict(status_counts),
        "successful_labels": sum(rec.get("label_status") == "success" for rec in all_records),
        "failed_labels": sum(rec.get("label_status") == "profile_failed" for rec in all_records),
        "pending_labels": sum(rec.get("label_status") == "profiling_in_progress" for rec in all_records),
        "repeats": args.repeats,
        "device_id": args.device_id,
        "latency_unit": "us",
        "latency_source": "triton.backends.ascend.testing.do_bench_npu",
        "measurement_provenance": MEASUREMENT_PROVENANCE,
        "benchmark_policy": "correctness_first_then_once_per_unique_launch_signature",
        "skipped_reasons_for_current_filter": dict(skipped_reasons),
        "outputs": {
            "records_jsonl": str(records_path),
            "labels_csv": str(csv_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    log("\n" + "=" * 72)
    log("CostModel V2 profiling finished")
    log(f"Profile records : {len(all_records)}")
    log(f"Label statuses  : {dict(status_counts)}")
    log(f"Records JSONL   : {records_path}")
    log(f"Labels CSV      : {csv_path}")
    log(f"Summary         : {summary_path}")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[profile] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
