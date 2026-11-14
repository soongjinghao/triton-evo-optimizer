#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Warmup-compile Seed Triton kernels and collect Ascend IR artifacts.

This is the second-stage entry point of Agent/costModel.

Important guarantees:
- Only datasets/<operator>/<operator>_1.py is selected: one Seed per operator.
- The matching test script provides real arguments and launch configurations.
- JITFunction.run is patched in an isolated subprocess to force warmup=True.
- The target Triton JIT kernels are not launched; PyTorch setup/reference ops in
  the test script may still execute on the NPU. Correctness is not validated here.
- TTIR / TTAdapterIR are collected from CompiledKernel.asm when available;
  NPUIR is collected from ~/.triton/dump when the Ascend backend emits it.
- The default run is a small 5-candidate pilot. Use --limit 0 for all unique
  successful Seed sources after the pilot has been inspected.

Run from the Agent repository root:
    python costModel/scripts/01_compile_candidates.py --limit 5
    python costModel/scripts/01_compile_candidates.py --limit 0 --workers 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
COST_MODEL_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = COST_MODEL_DIR.parent
DEFAULT_INVENTORY = COST_MODEL_DIR / "data" / "manifests" / "operator_inventory.jsonl"
DEFAULT_OUTPUT_DIR = COST_MODEL_DIR / "artifacts" / "ir"

IR_CATEGORIES = ("ttir", "ttadapter", "npuir", "npubin")
SUCCESS_STATUSES = {"success", "compiled_with_test_error"}


def log(message: str) -> None:
    print(f"[compile-candidates] {message}", flush=True)


def safe_name(value: str) -> str:
    # Keep leading underscores: they are part of many original operator names.
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip(".")
    return value or "unnamed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Inventory file not found: {path}\nRun 00_scan_operators.py first.")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Inventory row {line_no} is not a JSON object")
            rows.append(item)
    return rows


def build_candidates(
    rows: list[dict[str, Any]],
    project_root: Path,
    dataset_dir: str = "datasets",
    operator_filters: list[str] | None = None,
    allow_stale_inventory: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build exactly one V1 compile task per operator from datasets/<op>/<op>_1.py.

    Historical manifest success flags are deliberately ignored: the repository\'s
    existing manifests may refer to datasets2, while V1 compiles datasets only.
    """
    allowed = set(operator_filters or [])
    dataset_root = (project_root / dataset_dir).resolve()
    candidates: list[dict[str, Any]] = []
    skipped: dict[str, int] = {
        "operator_filter": 0,
        "missing_seed_file": 0,
        "missing_test_file": 0,
        "seed_without_triton_jit": 0,
        "missing_source": 0,
        "stale_source_hash": 0,
        "source_outside_primary_dataset": 0,
    }

    for operator in sorted(rows, key=lambda item: str(item.get("operator_name", ""))):
        operator_name = str(operator.get("operator_name") or "").strip()
        if not operator_name:
            continue
        if allowed and operator_name not in allowed:
            skipped["operator_filter"] += 1
            continue

        # Prefer the explicit field emitted by inventory schema v2.0.
        seed = operator.get("seed_source")
        if not isinstance(seed, dict):
            seed = next((
                item for item in operator.get("source_files", [])
                if item.get("file_name") == f"{operator_name}_1.py"
            ), None)
        if not isinstance(seed, dict):
            skipped["missing_seed_file"] += 1
            continue

        relative_source = str(seed.get("path") or "")
        source_path = (project_root / relative_source).resolve()
        try:
            source_path.relative_to(dataset_root)
        except ValueError:
            skipped["source_outside_primary_dataset"] += 1
            continue
        if not source_path.is_file():
            skipped["missing_source"] += 1
            continue

        jit_functions = seed.get("jit_functions") or []
        if not jit_functions:
            skipped["seed_without_triton_jit"] += 1
            continue

        preferred_test = operator.get("preferred_test_file")
        if not isinstance(preferred_test, dict):
            tests = operator.get("test_files") or []
            preferred_test = next((
                item for item in tests
                if item.get("file_name") == f"test_{operator_name}_1.py"
            ), tests[0] if len(tests) == 1 else None)
        if not isinstance(preferred_test, dict):
            skipped["missing_test_file"] += 1
            continue

        test_path = (project_root / str(preferred_test.get("path") or "")).resolve()
        if not test_path.is_file():
            skipped["missing_test_file"] += 1
            continue

        current_hash = sha256_file(source_path)
        expected_hash = str(seed.get("sha256") or "")
        if expected_hash and current_hash != expected_hash and not allow_stale_inventory:
            skipped["stale_source_hash"] += 1
            continue

        source_records = operator.get("source_files") or []
        support_paths = []
        for item in source_records:
            item_path = (project_root / str(item.get("path") or "")).resolve()
            if item_path.is_file():
                try:
                    item_path.relative_to(dataset_root)
                except ValueError:
                    continue
                support_paths.append(str(item_path))

        candidate_id = safe_name(f"{operator_name}__seed1")
        candidates.append({
            "candidate_id": candidate_id,
            "operator_name": operator_name,
            "dataset": dataset_dir,
            "source_file": source_path.name,
            "source_path": str(source_path),
            "source_relative_path": relative_source,
            "source_sha256": current_hash,
            "inventory_sha256": expected_hash or None,
            "source_role": "seed",
            "jit_functions": jit_functions,
            "support_source_paths": support_paths,
            "test_file": test_path.name,
            "test_path": str(test_path),
            "test_sha256": preferred_test.get("sha256"),
            # Historical manifest is preserved for audit only, not used to
            # decide whether this datasets/ Seed can be compiled.
            "legacy_manifest": seed.get("legacy_manifest"),
            "aliases": [],
        })

    diagnostics = {
        "inventory_operator_count": len(rows),
        "primary_dataset": dataset_dir,
        "selection_policy": "one_seed_file_per_operator: <operator>_1.py from datasets/",
        "selected_candidate_count": len(candidates),
        "skipped": skipped,
    }
    return candidates, diagnostics

def rewrite_test_imports(test_source: str, operator_name: str) -> str:
    """Redirect imports of the operator's numbered Seed module to one module.

    The original executor uses the same basic strategy: every candidate is
    written to <operator_name>.py, while tests may import <operator_name>_1.
    """
    base = re.escape(operator_name)

    def replace_from(match: re.Match[str]) -> str:
        prefix, module, suffix = match.groups()
        if module == operator_name or re.fullmatch(base + r"_\d+", module):
            return prefix + operator_name + suffix
        if module == "kernel" or re.fullmatch(r"kernel_\d+", module):
            return prefix + operator_name + suffix
        return match.group(0)

    def replace_import(match: re.Match[str]) -> str:
        prefix, module, suffix = match.groups()
        if module == operator_name or re.fullmatch(base + r"_\d+", module):
            return prefix + operator_name + suffix
        if module == "kernel" or re.fullmatch(r"kernel_\d+", module):
            return prefix + operator_name + suffix
        return match.group(0)

    test_source = re.sub(
        r"(?m)^(\s*from\s+)([A-Za-z_]\w*)(\s+import\b)",
        replace_from,
        test_source,
    )
    test_source = re.sub(
        r"(?m)^(\s*import\s+)([A-Za-z_]\w*)(\s*(?:as\s+\w+)?\s*(?:,|$))",
        replace_import,
        test_source,
    )

    # Register the Ascend backend in test subprocesses, matching executor.py.
    if not re.search(r"(?m)^\s*import\s+torch_npu\b", test_source):
        test_source = "import torch_npu  # register Ascend device backend\n" + test_source
    return test_source


def make_sitecustomize_source() -> str:
    """Return the isolated compile-only monkey patch loaded by Python startup."""
    return r'''# Auto-generated by costModel/scripts/01_compile_candidates.py
import atexit
import json
import os
import re
import time
from pathlib import Path

_STATE = {
    "n_attempt": 0,
    "n_ok": 0,
    "warmup_time_ms": 0.0,
    "kernel_calls": [],
    "compile_errors": [],
    "patch_error": None,
}
_CAPTURE_DIR = Path(os.environ.get("COSTMODEL_CAPTURE_DIR", "."))
_ASM_DIR = _CAPTURE_DIR / "asm"
_ASM_DIR.mkdir(parents=True, exist_ok=True)


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unnamed"


def _save_state():
    path = os.environ.get("COSTMODEL_STATE_PATH")
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(_STATE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _capture_compiled(compiled, call_index):
    call = {
        "call_index": call_index,
        "compiled_type": type(compiled).__name__,
        "asm_keys": [],
        "saved_files": [],
        "capture_error": None,
    }
    try:
        asm = getattr(compiled, "asm", None)
        if not isinstance(asm, dict):
            call["capture_error"] = "CompiledKernel.asm is not a dict"
            return call
        call["asm_keys"] = [str(key) for key in asm.keys()]
        for key, value in asm.items():
            key_name = _safe(key).lower()
            if isinstance(value, str):
                if "ttadapter" in key_name:
                    suffix = ".ttadapter.mlir"
                elif "ttir" in key_name:
                    suffix = ".ttir.mlir"
                elif "npuir" in key_name:
                    suffix = ".npuir.mlir"
                else:
                    suffix = ".txt"
                target = _ASM_DIR / ("k%03d_%s%s" % (call_index, key_name, suffix))
                target.write_text(value, encoding="utf-8")
                call["saved_files"].append(str(target))
            elif isinstance(value, (bytes, bytearray)):
                target = _ASM_DIR / ("k%03d_%s.bin" % (call_index, key_name))
                target.write_bytes(bytes(value))
                call["saved_files"].append(str(target))
    except Exception as exc:
        call["capture_error"] = "%s: %s" % (type(exc).__name__, exc)
    return call


try:
    from triton.runtime.jit import JITFunction
    _ORIGINAL_RUN = JITFunction.run

    def _costmodel_run(self, *args, **kwargs):
        # Force the JIT compile path while preventing the actual NPU kernel launch.
        kwargs["warmup"] = True
        _STATE["n_attempt"] += 1
        call_index = _STATE["n_attempt"]
        start = time.perf_counter()
        try:
            compiled = _ORIGINAL_RUN(self, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000.0
            _STATE["n_ok"] += 1
            _STATE["warmup_time_ms"] += elapsed
            call = _capture_compiled(compiled, call_index)
            call["kernel_name"] = getattr(self, "__name__", None) or getattr(self, "name", None)
            call["warmup_time_ms"] = round(elapsed, 3)
            _STATE["kernel_calls"].append(call)
            _save_state()
            return compiled
        except BaseException as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            _STATE["compile_errors"].append({
                "call_index": call_index,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "warmup_time_ms": round(elapsed, 3),
            })
            _save_state()
            raise

    JITFunction.run = _costmodel_run
except Exception as exc:
    _STATE["patch_error"] = "%s: %s" % (type(exc).__name__, exc)

atexit.register(_save_state)
'''


def classify_ir_file(path: Path) -> str | None:
    name = path.name.lower()
    if "ttadapter" in name:
        return "ttadapter"
    if "npuir" in name:
        return "npuir"
    if "ttir" in name:
        return "ttir"
    if "npubin" in name or path.suffix.lower() == ".bin":
        return "npubin"
    return None


def copy_dump_files(home_dir: Path, capture_dir: Path) -> list[str]:
    dump_root = home_dir / ".triton" / "dump"
    copied: list[str] = []
    if not dump_root.is_dir():
        return copied
    target_root = capture_dir / "dump"
    for source in dump_root.rglob("*"):
        if not source.is_file():
            continue
        if source.suffix.lower() not in {".mlir", ".npubin", ".bin"} and not source.name.lower().endswith((".ttir", ".ttadapter", ".npuir", ".npubin")):
            continue
        relative = source.relative_to(dump_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
            copied.append(str(target))
        except OSError:
            continue
    return copied


def patch_test_assertions() -> None:
    """Disable output assertions in this compile-only subprocess only.

    Warmup returns CompiledKernel objects rather than computed output tensors.
    The assertions therefore cannot validate correctness and may stop a test
    before it reaches subsequent independent JIT calls.
    """
    try:
        import torch
        if hasattr(torch, "testing"):
            torch.testing.assert_close = lambda *args, **kwargs: None
            if hasattr(torch.testing, "assert_allclose"):
                torch.testing.assert_allclose = lambda *args, **kwargs: None
        torch.allclose = lambda *args, **kwargs: True
        torch.equal = lambda *args, **kwargs: True
    except Exception:
        pass
    try:
        import numpy as np
        np.allclose = lambda *args, **kwargs: True
        np.testing.assert_allclose = lambda *args, **kwargs: None
        np.testing.assert_array_equal = lambda *args, **kwargs: None
    except Exception:
        pass


def worker_main() -> int:
    """Run one operator test under the sitecustomize compile-only patch."""
    test_file = Path(os.environ["COSTMODEL_TEST_FILE"]).resolve()
    test_dir = Path(os.environ["COSTMODEL_TEST_DIR"]).resolve()
    capture_dir = Path(os.environ["COSTMODEL_CAPTURE_DIR"]).resolve()
    result_path = Path(os.environ["COSTMODEL_WORKER_RESULT"]).resolve()
    device_index = int(os.environ.get("COSTMODEL_DEVICE_INDEX", "0"))
    result: dict[str, Any] = {
        "status": "failed",
        "triton_kernel_launched": False,
        "torch_reference_ops_may_execute": True,
        "test_error": None,
        "test_traceback": None,
        "torch_version": None,
        "triton_version": None,
        "device": None,
        "dump_files": [],
    }

    try:
        import torch
        import torch_npu  # noqa: F401
        import triton

        result["torch_version"] = str(torch.__version__)
        result["triton_version"] = str(triton.__version__)
        patch_module = sys.modules.get("sitecustomize")
        patch_state = getattr(patch_module, "_STATE", {}) if patch_module else {}
        if not patch_module or patch_state.get("patch_error"):
            raise RuntimeError(
                "Compile-only JIT patch is unavailable; aborting before test execution. "
                f"patch_error={patch_state.get('patch_error') if patch_state else 'sitecustomize not loaded'}"
            )
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is False")
        torch.npu.set_device(device_index)
        result["device"] = f"npu:{device_index}"
        patch_test_assertions()

        os.chdir(test_dir)
        sys.path.insert(0, str(test_dir))
        runpy.run_path(str(test_file), run_name="__main__")
        result["status"] = "test_completed"
    except BaseException as exc:
        result["test_error"] = f"{type(exc).__name__}: {exc}"
        result["test_traceback"] = traceback.format_exc()

    try:
        module = sys.modules.get("sitecustomize")
        state = getattr(module, "_STATE", {}) if module else {}
        result["compile_state"] = state
    except Exception as exc:
        result["compile_state_error"] = f"{type(exc).__name__}: {exc}"
        result["compile_state"] = {}

    result["dump_files"] = copy_dump_files(Path.home(), capture_dir)
    result["triton_kernel_launched"] = False
    result["torch_reference_ops_may_execute"] = True
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def collect_and_save_ir(
    candidate: dict[str, Any],
    work_dir: Path,
    output_dir: Path,
    worker_result: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, bool]]:
    capture_dir = work_dir / "capture"
    sources: list[Path] = []
    for subdir in (capture_dir / "asm", capture_dir / "dump"):
        if subdir.is_dir():
            sources.extend(path for path in subdir.rglob("*") if path.is_file())

    ir_paths: dict[str, list[str]] = {category: [] for category in IR_CATEGORIES}
    seen_hashes: dict[str, set[str]] = {category: set() for category in IR_CATEGORIES}
    operator_name = safe_name(candidate["operator_name"])
    candidate_id = safe_name(candidate["candidate_id"])

    for source in sorted(sources, key=lambda item: str(item)):
        category = classify_ir_file(source)
        if category is None:
            continue
        digest = sha256_file(source)
        if digest in seen_hashes[category]:
            continue
        seen_hashes[category].add(digest)
        ordinal = len(ir_paths[category])
        if category == "ttir":
            suffix = ".ttir.mlir"
        elif category == "ttadapter":
            suffix = ".ttadapter.mlir"
        elif category == "npuir":
            suffix = ".npuir.mlir"
        else:
            suffix = ".bin"
        target_dir = output_dir / category / operator_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{candidate_id}__k{ordinal:02d}{suffix}"
        shutil.copy2(source, target)
        ir_paths[category].append(str(target.resolve()))

    ir_found = {category: bool(ir_paths[category]) for category in IR_CATEGORIES}
    return ir_paths, ir_found


def remove_previous_candidate_ir(candidate: dict[str, Any], output_dir: Path) -> None:
    """Remove stale artifacts for this candidate before a fresh compile."""
    operator_name = safe_name(candidate["operator_name"])
    candidate_id = safe_name(candidate["candidate_id"])
    for category in IR_CATEGORIES:
        category_dir = output_dir / category / operator_name
        if not category_dir.is_dir():
            continue
        for path in category_dir.glob(f"{candidate_id}__k*"):
            if path.is_file():
                path.unlink()


def run_one_candidate(
    candidate: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    device_index: int,
    timeout: int,
) -> dict[str, Any]:
    candidate_id = safe_name(candidate["candidate_id"])
    remove_previous_candidate_ir(candidate, output_dir)
    work_root = output_dir / "_work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{candidate_id}_", dir=str(work_root)))
    test_dir = work_dir / "test_env"
    patch_dir = work_dir / "patch"
    home_dir = work_dir / "home"
    cache_dir = work_dir / "cache"
    capture_dir = work_dir / "capture"
    for directory in (test_dir, patch_dir, home_dir, cache_dir, capture_dir):
        directory.mkdir(parents=True, exist_ok=True)

    operator_name = candidate["operator_name"]
    source_path = Path(candidate["source_path"])
    test_path = Path(candidate["test_path"])
    result: dict[str, Any] = {
        **candidate,
        "status": "failed",
        "compile_pass": None,
        "triton_kernel_launched": False,
        "torch_reference_ops_may_execute": True,
        "device_index": device_index,
        "ir_paths": {},
        "ir_found": {category: False for category in IR_CATEGORIES},
    }

    try:
        source_code = source_path.read_text(encoding="utf-8")
        test_code = test_path.read_text(encoding="utf-8")
        # Copy sibling source modules as possible helper imports, then replace
        # the canonical operator module with the selected Seed implementation.
        for support_path_text in candidate.get("support_source_paths", []):
            support_path = Path(support_path_text)
            if support_path.is_file():
                shutil.copy2(support_path, test_dir / support_path.name)
        (test_dir / f"{operator_name}.py").write_text(source_code, encoding="utf-8")
        rewritten_test = rewrite_test_imports(test_code, operator_name)
        worker_test = test_dir / f"test_{operator_name}.py"
        worker_test.write_text(rewritten_test, encoding="utf-8")
        (patch_dir / "sitecustomize.py").write_text(make_sitecustomize_source(), encoding="utf-8")

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_parts = [str(patch_dir), str(test_dir), str(project_root)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env.update({
            "HOME": str(home_dir),
            "TRITON_CACHE_DIR": str(cache_dir),
            "TRITON_DEBUG": "1",
            "TRITON_ALWAYS_COMPILE": "1",
            "TRITON_DISABLE_CACHE": "1",
            "ASCEND_DEVICE_ID": str(device_index),
            "ASCEND_RT_VISIBLE_DEVICES": str(device_index),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "COSTMODEL_TEST_FILE": str(worker_test),
            "COSTMODEL_TEST_DIR": str(test_dir),
            "COSTMODEL_CAPTURE_DIR": str(capture_dir),
            "COSTMODEL_STATE_PATH": str(capture_dir / "compile_state.json"),
            "COSTMODEL_WORKER_RESULT": str(work_dir / "worker_result.json"),
            "COSTMODEL_DEVICE_INDEX": str(device_index),
        })
        env.pop("MLIR_ENABLE_DUMP", None)

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--_worker"],
                cwd=str(test_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            process_wall_seconds = time.perf_counter() - start
            result["process_return_code"] = proc.returncode
            result["process_wall_seconds"] = round(process_wall_seconds, 3)
            result["worker_log"] = proc.stdout[-20000:] if proc.stdout else ""
        except subprocess.TimeoutExpired as exc:
            result.update({
                "status": "timeout",
                "error": f"Candidate exceeded timeout of {timeout}s",
                "process_wall_seconds": round(time.perf_counter() - start, 3),
            })
            return result

        worker_result_path = work_dir / "worker_result.json"
        if worker_result_path.is_file():
            worker_result = json.loads(worker_result_path.read_text(encoding="utf-8"))
        else:
            worker_result = {
                "status": "worker_failed",
                "test_error": "Worker did not produce worker_result.json",
                "compile_state": {},
            }
        state = worker_result.get("compile_state") or {}
        n_attempt = int(state.get("n_attempt", 0) or 0)
        n_ok = int(state.get("n_ok", 0) or 0)
        compile_pass: bool | None
        if n_attempt == 0:
            compile_pass = None
        else:
            compile_pass = n_ok == n_attempt

        result.update({
            "torch_version": worker_result.get("torch_version"),
            "triton_version": worker_result.get("triton_version"),
            "device": worker_result.get("device"),
            "n_attempt": n_attempt,
            "n_ok": n_ok,
            "warmup_time_ms": round(float(state.get("warmup_time_ms", 0.0) or 0.0), 3),
            "kernel_calls": state.get("kernel_calls", []),
            "compile_errors": state.get("compile_errors", []),
            "test_error": worker_result.get("test_error"),
            "test_traceback": worker_result.get("test_traceback"),
            "sitecustomize_patch_error": state.get("patch_error"),
            "compile_pass": compile_pass,
            "triton_kernel_launched": False,
            "torch_reference_ops_may_execute": True,
        })

        ir_paths, ir_found = collect_and_save_ir(candidate, work_dir, output_dir, worker_result)
        result["ir_paths"] = ir_paths
        result["ir_found"] = ir_found

        if compile_pass is True:
            result["status"] = "success" if not worker_result.get("test_error") else "compiled_with_test_error"
        elif compile_pass is False:
            result["status"] = "compile_failed"
        else:
            result["status"] = "no_kernel_compile_observed"

        result["dump_file_count"] = len(worker_result.get("dump_files", []))
        result["ir_counts"] = {category: len(paths) for category, paths in ir_paths.items()}

    except Exception as exc:
        result.update({
            "status": "collector_error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    finally:
        # Keep a compact log in the durable output tree; temporary caches and
        # test workspaces are removed to avoid filling the remote disk.
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if result.get("worker_log"):
            (logs_dir / f"{candidate_id}.log").write_text(result["worker_log"], encoding="utf-8")
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warmup-compile the single *_1.py Seed from each operator in datasets/."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Agent 仓库根目录")
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY), help="00_scan_operators.py 生成的 JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="IR 产物根目录")
    parser.add_argument("--dataset-dir", default="datasets", help="V1 唯一数据源（默认 datasets）")
    parser.add_argument("--limit", type=int, default=5, help="最多编译多少个 Seed；0 表示全部 50 个（默认先试编译 5 个）")
    parser.add_argument("--operator", action="append", default=[], help="只编译指定算子，可重复传入")
    parser.add_argument("--device-index", type=int, default=0, help="Ascend NPU device index")
    parser.add_argument("--workers", type=int, default=1, help="并发编译进程数；首轮建议保持 1")
    parser.add_argument("--timeout", type=int, default=300, help="每个候选的超时时间（秒）")
    parser.add_argument("--allow-stale-inventory", action="store_true", help="允许源码哈希与 inventory 不一致")
    parser.add_argument("--force", action="store_true", help="覆盖已有成功结果并重新编译")
    parser.add_argument("--dry-run", action="store_true", help="只显示候选清单，不启动 NPU 编译")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args._worker:
        return worker_main()

    project_root = Path(args.project_root).expanduser().resolve()
    inventory_path = Path(args.inventory).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")

    rows = read_jsonl(inventory_path)
    candidates, diagnostics = build_candidates(
        rows=rows,
        project_root=project_root,
        dataset_dir=args.dataset_dir,
        operator_filters=args.operator,
        allow_stale_inventory=args.allow_stale_inventory,
    )
    all_candidates_count = len(candidates)
    if args.limit > 0:
        candidates = candidates[:args.limit]

    log(f"Project root       : {project_root}")
    log(f"Inventory          : {inventory_path}")
    log(f"Output directory   : {output_dir}")
    log(f"Inventory operators: {diagnostics['inventory_operator_count']}")
    log(f"Primary dataset    : {diagnostics['primary_dataset']}")
    log(f"Selection policy   : {diagnostics['selection_policy']}")
    log(f"Seed compile jobs  : {all_candidates_count}")
    log(f"Selected this run  : {len(candidates)}")
    log(f"Device             : npu:{args.device_index}")
    log(f"Workers            : {args.workers}")
    log(f"Skipped            : {diagnostics['skipped']}")

    if args.dry_run:
        for index, candidate in enumerate(candidates, start=1):
            log(f"[{index}/{len(candidates)}] {candidate['candidate_id']} | {candidate['source_relative_path']}")
        return 0

    if args.workers > 1:
        log("WARNING: multiple workers share one NPU. Use workers=1 for the first real run.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    start_all = time.perf_counter()

    def should_skip(candidate: dict[str, Any]) -> bool:
        meta_path = output_dir / "metadata" / safe_name(candidate["operator_name"]) / f"{safe_name(candidate['candidate_id'])}.json"
        if not meta_path.is_file() or args.force:
            return False
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            return old.get("status") in SUCCESS_STATUSES and old.get("ir_found", {}).get("ttir", False)
        except Exception:
            return False

    pending = []
    for candidate in candidates:
        if should_skip(candidate):
            log(f"SKIP existing: {candidate['candidate_id']}")
            continue
        pending.append(candidate)

    if args.workers == 1:
        for index, candidate in enumerate(pending, start=1):
            log(f"[{index}/{len(pending)}] compiling {candidate['candidate_id']}")
            result = run_one_candidate(candidate, project_root, output_dir, args.device_index, args.timeout)
            results.append(result)
            log(
                f"  status={result.get('status')} compile_pass={result.get('compile_pass')} "
                f"warmup={result.get('warmup_time_ms', 0)}ms IR={result.get('ir_found')}"
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(run_one_candidate, candidate, project_root, output_dir, args.device_index, args.timeout): candidate
                for candidate in pending
            }
            for future in concurrent.futures.as_completed(future_map):
                candidate = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **candidate,
                        "status": "collector_error",
                        "compile_pass": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "triton_kernel_launched": False,
                        "torch_reference_ops_may_execute": True,
                    }
                results.append(result)
                log(f"{candidate['candidate_id']}: {result.get('status')} IR={result.get('ir_found')}")

    elapsed = time.perf_counter() - start_all
    results.sort(key=lambda item: item.get("candidate_id", ""))

    # Store each candidate's complete metadata independently.
    metadata_root = output_dir / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for result in results:
        op_dir = metadata_root / safe_name(result.get("operator_name", "unknown"))
        write_json_atomic(op_dir / f"{safe_name(result.get('candidate_id', 'candidate'))}.json", result)

    jsonl_path = output_dir / "compile_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema_version": "2.0",
        "project_root": str(project_root),
        "primary_dataset": args.dataset_dir,
        "selection_policy": diagnostics["selection_policy"],
        "inventory_path": str(inventory_path),
        "output_dir": str(output_dir),
        "device_index": args.device_index,
        "workers": args.workers,
        "timeout_seconds": args.timeout,
        "requested_limit": args.limit,
        "inventory_diagnostics": diagnostics,
        "candidate_count_before_limit": all_candidates_count,
        "candidate_count_selected": len(candidates),
        "candidate_count_run": len(results),
        "elapsed_seconds": round(elapsed, 3),
        "execution_note": "Target Triton JIT kernels are warmup-compiled but not launched; PyTorch setup/reference operations in test scripts may still execute.",
        "triton_kernel_launched": False,
        "torch_reference_ops_may_execute": True,
        "status_counts": {},
        "compile_pass_count": sum(item.get("compile_pass") is True for item in results),
        "compile_fail_count": sum(item.get("compile_pass") is False for item in results),
        "compile_unknown_count": sum(item.get("compile_pass") is None for item in results),
        "ir_file_counts": {
            category: sum(len(item.get("ir_paths", {}).get(category, [])) for item in results)
            for category in IR_CATEGORIES
        },
        "results": [
            {
                "candidate_id": item.get("candidate_id"),
                "operator_name": item.get("operator_name"),
                "status": item.get("status"),
                "compile_pass": item.get("compile_pass"),
                "warmup_time_ms": item.get("warmup_time_ms"),
                "ir_found": item.get("ir_found"),
            }
            for item in results
        ],
    }
    for item in results:
        status = str(item.get("status", "unknown"))
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    summary_path = output_dir / "compile_summary.json"
    write_json_atomic(summary_path, summary)

    log("=" * 72)
    log("Warmup IR collection finished")
    log(f"Candidates run      : {len(results)}")
    log(f"Compile pass        : {summary['compile_pass_count']}")
    log(f"Compile fail        : {summary['compile_fail_count']}")
    log(f"Compile unknown     : {summary['compile_unknown_count']}")
    log(f"IR file counts      : {summary['ir_file_counts']}")
    log(f"Summary             : {summary_path}")
    log(f"Results JSONL       : {jsonl_path}")
    log("Kernel executed     : False (warmup only)")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
