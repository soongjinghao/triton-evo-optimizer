#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect Triton IR and MSProf data into costModel/datasets/<operator>/data/<role>."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

VERSION = "2.1.0"


# ---------------------------------------------------------------------------
# Utilities and project discovery
# ---------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def default_root() -> Path:
    # Expected location: <project_root>/costModel/collect.py
    return Path(__file__).resolve().parent.parent


def percentile_key(role: str) -> Tuple[int, float, str]:
    match = re.fullmatch(r"p(\d+(?:\.\d+)?)", role.lower())
    if match:
        return (0, float(match.group(1)), role)
    return (1, 0.0, role)


def resolve_test_path(root: Path, operator_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    for candidate in ((root / path).resolve(), (operator_dir / path).resolve()):
        if candidate.is_file():
            return candidate
    return (root / path).resolve()


# ---------------------------------------------------------------------------
# Operator config and variant discovery
# ---------------------------------------------------------------------------

def load_operator(root: Path, operator: str) -> Dict[str, Any]:
    operator_dir = root / "costModel" / "datasets" / operator
    config_path = operator_dir / "operator.json"
    variant_dir = operator_dir / "variant"

    if not operator_dir.is_dir():
        raise FileNotFoundError(f"Operator directory not found: {operator_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"operator.json not found: {config_path}")
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"variant directory not found: {variant_dir}")

    config = read_json(config_path)
    operator_name = str(config.get("operator_name") or config.get("operator") or operator)

    variants = sorted(
        [path for path in variant_dir.glob("*.py") if path.is_file()],
        key=lambda path: percentile_key(path.stem),
    )
    if not variants:
        raise FileNotFoundError(f"No .py variants found in {variant_dir}")

    old_roles = config.get("candidate_roles") or config.get("percentiles") or []
    old_ids = config.get("candidates") or []
    candidate_ids = {
        str(role): str(old_ids[i])
        for i, role in enumerate(old_roles)
        if i < len(old_ids)
    }

    raw_test = config.get("test_path") or config.get("test_file") or "test.py"
    test_path = resolve_test_path(root, operator_dir, str(raw_test))
    if not test_path.is_file():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    return {
        "operator": operator_name,
        "operator_dir": operator_dir.resolve(),
        "config_path": config_path.resolve(),
        "config": config,
        "variants": variants,
        "candidate_ids": candidate_ids,
        "test_path": test_path.resolve(),
        "data_dir": (operator_dir / "data").resolve(),
    }


def update_operator_config(info: Dict[str, Any], roles: List[str]) -> None:
    """Make variant/*.py the source of truth for role/percentile metadata."""
    config = dict(info["config"])
    config["operator"] = info["operator"]
    config["operator_name"] = info["operator"]
    config["percentiles"] = list(roles)
    config["candidate_roles"] = list(roles)
    config["candidate_count"] = len(roles)
    config["test_path"] = relpath(info["test_path"], info["operator_dir"].parent.parent.parent)
    config["variants_ready"] = bool(roles)

    metadata_items = []
    for role in roles:
        metadata_path = info["data_dir"] / role / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata_items.append(read_json(metadata_path))
            except Exception:
                pass

    complete_set = len(metadata_items) == len(roles) and bool(roles)
    def profile_record_ready(item: Dict[str, Any]) -> bool:
        evaluation = item.get("evaluation", {})
        profiling = item.get("profiling", {})
        correctness = item.get("correctness", {})
        mode = evaluation.get("profile_mode", "full")
        timing = profiling.get("timing", {})
        timing_ready = (
            timing.get("label_valid") is True
            if mode == "full"
            else (
                timing.get("execution_time_us") is not None
                and timing.get("sample_count", 0) >= 1
            )
        )
        return bool(
            evaluation.get("success")
            and correctness.get("success") is True
            and profiling.get("summary_available")
            and timing_ready
            and (mode != "full" or evaluation.get("hardware_profile_success") is True)
        )

    config["profiling_ready"] = bool(
        complete_set and all(profile_record_ready(item) for item in metadata_items)
    )
    config["correctness_ready"] = bool(
        complete_set
        and all(item.get("correctness", {}).get("success") is True for item in metadata_items)
    )
    config["ir_ready"] = bool(
        complete_set
        and all(
            item.get("correctness", {}).get("success") is True
            and item.get("ir_found", {}).get("ttir")
            and item.get("ir_found", {}).get("ttadapter")
            for item in metadata_items
        )
    )
    config["m3_ir_ready"] = bool(
        complete_set
        and all(
            item.get("correctness", {}).get("success") is True
            and item.get("ir_found", {}).get("ttir")
            and item.get("ir_found", {}).get("ttadapter")
            and item.get("ir_found", {}).get("hivm_opt")
            for item in metadata_items
        )
    )
    config["complete_data_ready"] = bool(
        config["profiling_ready"] and config["m3_ir_ready"]
    )
    config["last_collector_version"] = VERSION
    write_json(info["config_path"], config)
    info["config"] = config


# ---------------------------------------------------------------------------
# Capture hook: save compiler artifacts from Triton JIT compilation
# ---------------------------------------------------------------------------

def write_capture_hook(patch_dir: Path) -> Path:
    patch_dir.mkdir(parents=True, exist_ok=True)
    hook = patch_dir / "costmodel_capture_hook.py"

    # Keep hivm and hivm_opt separate: M3 must require a real optimized HIVM file.
    source = r"""
import json
import os
import re
import threading
from pathlib import Path

_LOCK = threading.Lock()
_COUNTER = 0

def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "kernel"

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (bytes, bytearray, memoryview)):
        path.write_bytes(bytes(value))
    else:
        path.write_text(str(value), encoding="utf-8")

def _capture(jit_function, compiled):
    global _COUNTER
    output_dir = os.environ.get("COSTMODEL_CAPTURE_DIR")
    target = os.environ.get("COSTMODEL_CAPTURE_KERNEL", "")
    if not output_dir:
        return

    function_name = getattr(jit_function, "__name__", None)
    if not function_name:
        function = getattr(jit_function, "fn", None)
        function_name = getattr(function, "__name__", "")
    function_text = str(function_name)
    target_alt = target.lstrip("_")
    if target and target not in function_text and target_alt not in function_text:
        return

    asm = getattr(compiled, "asm", None)
    if not isinstance(asm, dict):
        return

    extensions = {
        "ttir": "ttir.mlir",
        "ttadapter": "ttadapter.mlir",
        "npuir": "npuir.mlir",
        "hivm": "module.hivm.mlir",
        "hivm_opt": "module.hivm.opt.mlir",
        "npubin": "npubin",
    }

    with _LOCK:
        _COUNTER += 1
        call_index = _COUNTER

    root = Path(output_dir)
    saved = []
    for key, value in asm.items():
        normalized = str(key).lower().replace("-", "_")
        extension = extensions.get(normalized)
        if extension is None:
            continue
        filename = "p%06d_k%03d_%s.%s" % (
            os.getpid(), call_index, _safe(function_name), extension
        )
        path = root / filename
        try:
            _write(path, value)
            saved.append(str(path))
        except Exception:
            pass

    if saved:
        event = {
            "pid": os.getpid(),
            "call_index": call_index,
            "jit_function": str(function_name),
            "asm_keys": [str(key) for key in asm.keys()],
            "saved_files": saved,
        }
        event_path = root.parent / "capture_events.jsonl"
        try:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK:
                with event_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

try:
    from triton.runtime.jit import JITFunction
    _ORIGINAL_RUN = JITFunction.run

    def _costmodel_run(self, *args, **kwargs):
        compiled = _ORIGINAL_RUN(self, *args, **kwargs)
        try:
            _capture(self, compiled)
        except Exception:
            pass
        return compiled

    JITFunction.run = _costmodel_run
except Exception as exc:
    try:
        error_file = Path(os.environ.get("COSTMODEL_CAPTURE_DIR", ".")).parent / "capture_hook_error.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text(repr(exc), encoding="utf-8")
    except Exception:
        pass
"""
    hook.write_text(source, encoding="utf-8")
    return hook


def make_instrumented_test(source: Path, target: Path) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines()
    insertion = 1 if lines and lines[0].startswith("#!") else 0
    coding = re.compile(r"coding[:=]\s*[-\w.]+")
    for index in range(min(2, len(lines))):
        if coding.search(lines[index]):
            insertion = max(insertion, index + 1)
    lines.insert(insertion, "import costmodel_capture_hook")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# IR collection and organization
# ---------------------------------------------------------------------------

def ir_stage(path: Path) -> Optional[str]:
    name = path.name.lower()
    if ".hivm.opt.mlir" in name:
        return "hivm_opt"
    if name.endswith(".hivm.mlir"):
        return "hivm"
    if name.endswith(".ttadapter.mlir"):
        return "ttadapter"
    if name.endswith(".ttir.mlir"):
        return "ttir"
    if name.endswith(".npuir.mlir"):
        return "npuir"
    if name.endswith(".npubin") or name.endswith(".bin"):
        return "npubin"
    if name.endswith(".o"):
        return "object"
    if name.endswith(".so"):
        return "shared"
    return None


def copy_ir(source: Path, ir_root: Path, stage: str, prefix: str) -> Path:
    target_dir = ir_root / stage
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    target = target_dir / f"{prefix}{source.name}"

    if target.exists():
        if sha256(target) == digest:
            return target
        target = target_dir / f"{prefix}{source.stem}_{digest[:10]}{source.suffix}"

    if not target.exists():
        shutil.copy2(source, target)
    return target


def collect_ir(source_root: Path, ir_root: Path, prefix: str) -> Dict[str, List[Path]]:
    found: Dict[str, List[Path]] = {}
    if not source_root.exists():
        return found

    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        stage = ir_stage(source)
        if stage is None:
            continue
        saved = copy_ir(source, ir_root, stage, prefix)
        found.setdefault(stage, []).append(saved)
    return found


def merge_ir(*groups: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
    merged: Dict[str, List[Path]] = {}
    for group in groups:
        for stage, paths in group.items():
            bucket = merged.setdefault(stage, [])
            for path in paths:
                if path not in bucket:
                    bucket.append(path)
    return {key: sorted(value, key=str) for key, value in merged.items()}


def rewrite_moved_paths(value: Any, old_root: Path, new_root: Path) -> Any:
    """Rewrite absolute paths embedded in profile_summary after moving performance/."""
    old_prefix = str(old_root.resolve())
    new_prefix = str(new_root.resolve())
    if isinstance(value, str):
        if value == old_prefix:
            return new_prefix
        if value.startswith(old_prefix + os.sep):
            return new_prefix + value[len(old_prefix):]
        return value
    if isinstance(value, list):
        return [rewrite_moved_paths(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {
            key: rewrite_moved_paths(item, old_root, new_root)
            for key, item in value.items()
        }
    return value


# ---------------------------------------------------------------------------
# Existing executor integration
# ---------------------------------------------------------------------------

def get_executor(root: Path):
    executor_path = root / "executor.py"
    config_path = root / "config.py"
    if not executor_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Expected project-root executor.py and config.py; checked "
            f"{executor_path} and {config_path}"
        )

    root_string = str(root.resolve())
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)

    from executor import TritonExecutor  # type: ignore
    return TritonExecutor


def candidate_id(info: Dict[str, Any], role: str) -> str:
    return info["candidate_ids"].get(role) or f"{info['operator']}__{role}"


def run_one(
    root: Path,
    info: Dict[str, Any],
    role: str,
    variant: Path,
    profile_mode: str,
    timeout: int,
    baseline_time: float,
) -> Dict[str, Any]:
    operator = info["operator"]
    role_dir = info["data_dir"] / role
    ir_root = role_dir / "ir"
    capture_dir = ir_root / "capture" / "asm"
    profiling_dir = role_dir / "profiling"
    raw_profile_dir = profiling_dir / "raw"
    work_dir = role_dir / ".collector_work"

    role_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    profiling_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    kernel_snapshot = role_dir / "kernel.py"
    shutil.copy2(variant, kernel_snapshot)

    record = {
        "schema_version": "costmodel_candidate_v1",
        "operator_name": operator,
        "candidate_id": candidate_id(info, role),
        "candidate_role": role,
        "variant_path": relpath(variant, root),
        "source_code_sha256": sha256(variant),
        "kernel_snapshot": relpath(kernel_snapshot, root),
        "test_path": relpath(info["test_path"], root),
        "created_at": timestamp(),
    }
    write_json(role_dir / "candidate.json", record)

    patch_dir = work_dir / "patch"
    write_capture_hook(patch_dir)
    instrumented_test = make_instrumented_test(
        info["test_path"], work_dir / "capture_test.py"
    )

    isolated_home = work_dir / "home"
    triton_home = isolated_home / ".triton"
    cache_dir = triton_home / "cache"
    dump_dir = triton_home / "dump"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    env_keys = [
        "HOME", "TRITON_CACHE_DIR", "PYTHONPATH",
        "COSTMODEL_CAPTURE_DIR", "COSTMODEL_CAPTURE_KERNEL",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    start_clock = time.monotonic()
    start_time = timestamp()

    result_ok = False
    execution_time_us = None
    execution_error = None
    profile_summary: Dict[str, Any] = {}
    profile_summary_path = None
    stored_profile_dir = None
    ir_found: Dict[str, List[Path]] = {}
    hook_error = None

    try:
        os.environ["HOME"] = str(isolated_home)
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

        old_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = [str(patch_dir), str(root)]
        if old_pythonpath:
            pythonpath.append(old_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)

        os.environ["COSTMODEL_CAPTURE_DIR"] = str(capture_dir)
        os.environ["COSTMODEL_CAPTURE_KERNEL"] = operator

        TritonExecutor = get_executor(root)
        executor = TritonExecutor(
            baseline_time=baseline_time,
            test_code_path=str(instrumented_test),
            config=SimpleNamespace(),
            kernel_name=operator,
            work_dir=work_dir,
            profile_mode=profile_mode,
            correctness_preflight=True,
        )
        result = executor.evaluate(variant.read_text(encoding="utf-8"), timeout=timeout)

        result_ok = bool(result.success)
        execution_time_us = float(result.execution_time) if result.success else None
        execution_error = getattr(result, "error", None)
        correctness_ok = getattr(result, "correctness_success", None)
        preflight_log_source = getattr(executor, "last_preflight_log", None)
        preflight_log_path = None
        if preflight_log_source is not None and Path(preflight_log_source).is_file():
            preflight_log_path = profiling_dir / "preflight.log"
            shutil.copy2(preflight_log_source, preflight_log_path)

        last_profile = getattr(executor, "last_profile_dir", None)
        last_profile = Path(last_profile) if last_profile is not None else None

        # Move raw MSProf files first, then rewrite paths in the copied summary.
        performance_dir = work_dir / "performance"
        if performance_dir.is_dir():
            if raw_profile_dir.exists():
                shutil.rmtree(raw_profile_dir)
            shutil.move(str(performance_dir), str(raw_profile_dir))
            if last_profile is not None:
                try:
                    profile_relative = last_profile.relative_to(performance_dir)
                    stored_profile_dir = raw_profile_dir / profile_relative
                except ValueError:
                    stored_profile_dir = raw_profile_dir / last_profile.name

        summary_source = (
            stored_profile_dir / "profile_summary.json"
            if stored_profile_dir is not None else None
        )
        if summary_source is not None and summary_source.is_file():
            profile_summary = read_json(summary_source)
            profile_summary = rewrite_moved_paths(
                profile_summary, performance_dir, raw_profile_dir
            )
            profile_summary_path = profiling_dir / "profile_summary.json"
            write_json(profile_summary_path, profile_summary)

        captured = collect_ir(capture_dir, ir_root, "asm_")
        dumped = collect_ir(dump_dir, ir_root, "dump_")
        ir_found = merge_ir(captured, dumped)

        hook_error_path = ir_root / "capture" / "capture_hook_error.txt"
        if hook_error_path.is_file():
            hook_error = hook_error_path.read_text(encoding="utf-8")

    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    has_ttir = bool(ir_found.get("ttir"))
    has_ttadapter = bool(ir_found.get("ttadapter"))
    has_hivm_opt = bool(ir_found.get("hivm_opt"))
    hardware_ok = profile_summary.get("collection", {}).get("hardware_profile_success")
    summary_available = bool(profile_summary_path and profile_summary_path.is_file())

    timing_label_valid = profile_summary.get("timing", {}).get("label_valid")
    if correctness_ok is False:
        status = "correctness_failed"
    elif not result_ok:
        status = "execution_failed"
    elif profile_mode == "full" and hardware_ok is not True:
        status = "hardware_profile_incomplete"
    elif profile_mode == "full" and timing_label_valid is not True:
        status = "measurement_inconsistent"
    elif not (has_ttir and has_ttadapter):
        status = "ir_incomplete"
    elif not has_hivm_opt:
        status = "partial_success"
    else:
        status = "success"

    elapsed = round(time.monotonic() - start_clock, 3)
    metadata = {
        "schema_version": "costmodel_collection_v2",
        "script_version": VERSION,
        "operator_name": operator,
        "candidate_id": record["candidate_id"],
        "candidate_role": role,
        "status": status,
        "started_at": start_time,
        "finished_at": timestamp(),
        "duration_seconds": elapsed,
        "source": {
            "variant_path": relpath(variant, root),
            "source_code_sha256": sha256(variant),
            "kernel_snapshot": relpath(kernel_snapshot, root),
            "test_path": relpath(info["test_path"], root),
            "test_sha256": sha256(info["test_path"]),
        },
        "evaluation": {
            "success": result_ok,
            "execution_time_us": execution_time_us,
            "error": execution_error,
            "profile_mode": profile_mode,
            "profile_dir": relpath(stored_profile_dir, role_dir),
            "hardware_profile_success": hardware_ok,
        },
        "correctness": {
            "preflight_enabled": True,
            "success": correctness_ok,
            "log_path": relpath(preflight_log_path, role_dir),
            "error": execution_error if correctness_ok is False else None,
        },
        "profiling": {
            "summary_available": summary_available,
            "summary_path": relpath(profile_summary_path, role_dir),
            "raw_profile_dir": relpath(raw_profile_dir, role_dir)
            if raw_profile_dir.exists() else None,
            "timing": profile_summary.get("timing", {}),
            "kernel": profile_summary.get("kernel", {}),
            "core_utilization": profile_summary.get("core_utilization", {}),
            "memory": profile_summary.get("memory", {}),
            "collection": profile_summary.get("collection", {}),
        },
        "ir_found": {stage: bool(paths) for stage, paths in ir_found.items()},
        "ir_counts": {stage: len(paths) for stage, paths in ir_found.items()},
        "ir_paths": {
            stage: [relpath(path, role_dir) for path in paths]
            for stage, paths in ir_found.items()
        },
        "ir_stage_status": {
            "M1_ttir": "available" if has_ttir else "missing",
            "M2_ttir_ttadapter": "available" if has_ttir and has_ttadapter else "incomplete",
            "M3_ttir_ttadapter_hivm_opt": (
                "available" if has_ttir and has_ttadapter and has_hivm_opt else "incomplete"
            ),
            "hivm_unoptimized": "available" if ir_found.get("hivm") else "missing_optional",
            "npuir_debug_dump": "available" if ir_found.get("npuir") else "missing_optional",
        },
        "capture_hook_error": hook_error,
        "notes": [
            "Read candidate directly from variant/<role>.py; no manifest is used.",
            "No historical IR or profiling labels were reused.",
            "Triton cache was isolated for this candidate.",
            "A direct correctness preflight must pass before any MSProf pass is run.",
            "HIVM and optimized HIVM are stored as separate IR stages.",
            "M3 requires an actual module.hivm.opt.mlir artifact.",
        ],
    }
    write_json(role_dir / "metadata.json", metadata)

    # Keep only stable outputs: IR, profile data, candidate snapshot and metadata.
    shutil.rmtree(work_dir, ignore_errors=True)
    return metadata


def write_failed_metadata(
    info: Dict[str, Any], role: str, variant: Path, exc: Exception
) -> Dict[str, Any]:
    role_dir = info["data_dir"] / role
    role_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "costmodel_collection_v2",
        "script_version": VERSION,
        "operator_name": info["operator"],
        "candidate_role": role,
        "status": "collector_exception",
        "finished_at": timestamp(),
        "source": {
            "variant_path": relpath(variant, info["operator_dir"].parent.parent.parent),
            "source_code_sha256": sha256(variant) if variant.is_file() else None,
        },
        "evaluation": {
            "success": False,
            "execution_time_us": None,
            "error": repr(exc),
        },
        "traceback": traceback.format_exc(),
    }
    write_json(role_dir / "metadata.json", metadata)
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect IR and MSProf data from costModel/datasets/<operator>/variant."
    )
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--operator", required=True, help="e.g. _act_quant_kernel")
    parser.add_argument("--role", default=None, help="e.g. p50; default collects all variants")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile-mode", choices=["full", "timing_only"], default="full")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--baseline-time", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing data/<role> output for selected variants.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be >= 1")

    info = load_operator(root, args.operator)
    all_roles = [path.stem for path in info["variants"]]
    selected = list(info["variants"])

    if args.role:
        selected = [path for path in selected if path.stem == args.role]
        if not selected:
            raise ValueError(f"Unknown role {args.role!r}; available: {all_roles}")
    if args.limit is not None:
        selected = selected[:args.limit]

    executor_path = root / "executor.py"
    config_path = root / "config.py"
    if not executor_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Expected project-root files: {executor_path} and {config_path}"
        )

    print("=" * 76)
    print("CostModel local collection")
    print(f"Root         : {root}")
    print(f"Operator     : {info['operator']}")
    print(f"Test         : {info['test_path']}")
    print(f"Variants     : {[path.stem for path in selected]}")
    print(f"Profile mode : {args.profile_mode}")
    print(f"Output       : {info['data_dir']}")
    print("=" * 76)

    # Preflight all selected output paths before starting any NPU job.
    for variant in selected:
        role_dir = info["data_dir"] / variant.stem
        if role_dir.exists() and any(role_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Existing output: {role_dir}\n"
                "Use --overwrite to replace this selected role."
            )

    if args.dry_run:
        for variant in selected:
            print(f"[DRY-RUN] {variant} -> {info['data_dir'] / variant.stem}")
        print("DRY RUN PASSED. No NPU/MSProf command was executed.")
        return 0

    info["data_dir"].mkdir(parents=True, exist_ok=True)
    # Repairs stale percentile entries by reading the actual variant filenames.
    update_operator_config(info, all_roles)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = info["data_dir"] / "collection_history.jsonl"
    results: List[Dict[str, Any]] = []

    for index, variant in enumerate(selected, 1):
        role = variant.stem
        role_dir = info["data_dir"] / role

        print("\n" + "-" * 76)
        print(f"[{index}/{len(selected)}] {info['operator']} / {role}")
        print(f"Variant: {variant}")
        print(f"Output : {role_dir}")

        if args.overwrite and role_dir.exists():
            shutil.rmtree(role_dir)

        try:
            metadata = run_one(
                root=root,
                info=info,
                role=role,
                variant=variant,
                profile_mode=args.profile_mode,
                timeout=args.timeout,
                baseline_time=args.baseline_time,
            )
        except Exception as exc:
            print(f"[ERROR] {role}: {exc}")
            traceback.print_exc()
            metadata = write_failed_metadata(info, role, variant, exc)

        metadata["run_id"] = run_id
        append_jsonl(history_path, metadata)
        results.append(metadata)

        print(f"Status       : {metadata.get('status')}")
        print(f"Correctness  : {metadata.get('correctness', {}).get('success')}")
        print(
            "Execution us : "
            f"{metadata.get('evaluation', {}).get('execution_time_us')}"
        )
        print(f"IR status    : {metadata.get('ir_stage_status', {})}")

    # Readiness flags reflect all variants, not only this invocation's subset.
    update_operator_config(info, all_roles)

    success_count = sum(item.get("status") == "success" for item in results)
    correctness_failure_count = sum(item.get("status") == "correctness_failed" for item in results)
    partial_count = sum(item.get("status") == "partial_success" for item in results)
    measurement_inconsistent_count = sum(
        item.get("status") == "measurement_inconsistent" for item in results
    )
    summary = {
        "schema_version": "costmodel_collection_run_v2",
        "script_version": VERSION,
        "run_id": run_id,
        "operator_name": info["operator"],
        "selected_roles": [path.stem for path in selected],
        "selected_count": len(selected),
        "success_count": success_count,
        "partial_success_count": partial_count,
        "correctness_failure_count": correctness_failure_count,
        "measurement_inconsistent_count": measurement_inconsistent_count,
        "incomplete_or_failed_count": len(results) - success_count,
        "profile_mode": args.profile_mode,
        "history_path": relpath(history_path, root),
        "finished_at": timestamp(),
    }
    write_json(info["data_dir"] / "run_summary.json", summary)

    print("\n" + "=" * 76)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 76)
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
