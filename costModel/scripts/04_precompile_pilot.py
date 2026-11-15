#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warmup-precompile the selected Cost Model pilot candidates and collect Ascend IR.

This adapts the previously validated benchmark_warmup.py isolation model to
the real 50-operator pilot:

- Reads costModel/data/pilot/compile_jobs.jsonl (248 unique source jobs).
- Stages each selected candidate under the module name expected by its test.
- Reuses the real operator test file to provide actual input shapes, dtypes,
  launch grids, and constexpr arguments.
- Patches Triton JITFunction.run so every JIT call uses warmup=True.
- Runs every candidate in an isolated Python subprocess with its own HOME,
  Triton cache, and dump directory.
- Saves TTIR, TTAdapterIR, NPUIR, npubin, other artifacts, logs, and metadata.
- Maps the 248 unique compile results back to the 250 logical candidate rows.

Recommended:
  # Validate all 248 jobs without starting NPU compilation.
  python costModel/scripts/04_precompile_pilot.py --dry-run --limit 0

  # Smoke test: 10 jobs across 5 operators, including reference and historical.
  python costModel/scripts/04_precompile_pilot.py --limit 10 --workers 1

  # After checking the smoke test, compile the remaining unique jobs.
  python costModel/scripts/04_precompile_pilot.py --limit 0 --workers 2

  # Force recompilation of jobs that already succeeded.
  python costModel/scripts/04_precompile_pilot.py --limit 0 --workers 2 --force

Important:
- Triton JIT launches are intercepted and changed to warmup-only.
- Test scripts still run to supply real call arguments; PyTorch setup/reference
  operations inside a test may still execute.
- Warmup success confirms compilation, not correctness or runtime performance.
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
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
COST_MODEL_DIR = SCRIPT_PATH.parents[1]
DEFAULT_PROJECT_ROOT = COST_MODEL_DIR.parent
DEFAULT_COMPILE_JOBS = COST_MODEL_DIR / "data" / "pilot" / "compile_jobs.jsonl"
DEFAULT_SELECTED_CANDIDATES = COST_MODEL_DIR / "data" / "pilot" / "selected_candidates.jsonl"
DEFAULT_OUTPUT_DIR = COST_MODEL_DIR / "artifacts" / "ir"
DEFAULT_MANIFEST_OUTPUT = COST_MODEL_DIR / "data" / "pilot" / "candidate_ir_manifest.jsonl"

IR_CATEGORIES = ("ttir", "ttadapter", "npuir", "npubin")
ARTIFACT_SUFFIXES = {".mlir", ".ll", ".o", ".npubin", ".bin", ".so"}
SUCCESS_STATUSES = {"success", "compiled_with_test_error"}
DUMP_LINE = re.compile(r"Dumping intermediate results to\s+(.+?)\s*$")


def log(message: str) -> None:
    print("[pilot-precompile] {}".format(message), flush=True)


def safe_name(value: Any) -> str:
    # Keep leading underscores: they are meaningful in many operator names.
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip(".")
    return value or "unnamed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("JSONL file not found: {}".format(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{}: {}".format(path, line_no, exc)) from exc
            if not isinstance(row, dict):
                raise ValueError("{}:{} is not a JSON object".format(path, line_no))
            rows.append(row)
    return rows


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(path)


def load_compile_jobs(path: Path) -> list[dict[str, Any]]:
    jobs = read_jsonl(path)
    seen: set[str] = set()
    for index, job in enumerate(jobs, 1):
        job_id = str(job.get("compile_job_id") or "").strip()
        if not job_id:
            raise ValueError("compile_jobs row {} has no compile_job_id".format(index))
        if job_id in seen:
            raise ValueError("Duplicate compile_job_id: {}".format(job_id))
        seen.add(job_id)
        for field in ("operator_name", "source_path", "source_sha256", "test_path"):
            if not job.get(field):
                raise ValueError("{} missing required field {!r}".format(job_id, field))
    return jobs


def validate_job_files(jobs: list[dict[str, Any]], project_root: Path) -> None:
    """Fail before NPU work if a source/test file is missing or has changed."""
    problems: list[str] = []
    for job in jobs:
        job_id = str(job["compile_job_id"])
        source_path = resolve_path(project_root, str(job["source_path"]))
        test_path = resolve_path(project_root, str(job["test_path"]))

        if not source_path.is_file():
            problems.append("{}: source missing: {}".format(job_id, source_path))
            continue
        if not test_path.is_file():
            problems.append("{}: test missing: {}".format(job_id, test_path))
            continue

        expected_source_hash = str(job.get("source_sha256") or "").strip()
        if expected_source_hash:
            actual_source_hash = sha256_file(source_path)
            if actual_source_hash != expected_source_hash:
                problems.append(
                    "{}: source SHA256 mismatch (expected {}, got {})".format(
                        job_id, expected_source_hash, actual_source_hash
                    )
                )

        expected_test_hash = str(job.get("test_sha256") or "").strip()
        if expected_test_hash:
            actual_test_hash = sha256_file(test_path)
            if actual_test_hash != expected_test_hash:
                problems.append(
                    "{}: test SHA256 mismatch (expected {}, got {})".format(
                        job_id, expected_test_hash, actual_test_hash
                    )
                )

    if problems:
        preview = "\n".join("  - " + item for item in problems[:20])
        more = "\n  ... and {} more".format(len(problems) - 20) if len(problems) > 20 else ""
        raise RuntimeError(
            "Input source/test validation failed for {} job(s):\n{}{}".format(
                len(problems), preview, more
            )
        )


def role_priority(job: dict[str, Any]) -> tuple[int, str]:
    roles = set(job.get("selected_roles") or [])
    if "baseline" in roles or job.get("candidate_type") == "baseline":
        rank = 0
    elif "seed" in roles or job.get("candidate_type") == "seed":
        rank = 1
    elif "historical_p10" in roles:
        rank = 2
    elif "historical_p50" in roles:
        rank = 3
    elif "historical_p90" in roles:
        rank = 4
    else:
        rank = 5
    return rank, str(job.get("compile_job_id", ""))


def select_jobs(
    jobs: list[dict[str, Any]],
    limit: int,
    operators: list[str],
    sample_strategy: str,
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    allowed = set(operators or [])
    filtered = [
        job for job in jobs
        if not allowed or str(job.get("operator_name")) in allowed
    ]
    filtered.sort(key=lambda item: (
        str(item.get("operator_name", "")),
        role_priority(item),
    ))

    if limit == 0 or limit >= len(filtered):
        return filtered
    if sample_strategy == "file_order":
        return filtered[:limit]

    # Balanced smoke test: for each operator, take one reference job
    # (Baseline/Seed) and one historical job before moving to the next operator.
    # Thus --limit 10 samples five operators and exercises both source paths.
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in filtered:
        by_operator[str(job.get("operator_name", ""))].append(job)

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for op in sorted(by_operator):
        op_jobs = by_operator[op]
        reference = [
            item for item in op_jobs
            if item.get("candidate_type") in {"baseline", "seed"}
            or bool(set(item.get("selected_roles") or []) & {"baseline", "seed"})
        ]
        historical = [
            item for item in op_jobs
            if item.get("candidate_type") == "historical_evolution"
        ]
        reference.sort(key=role_priority)
        historical.sort(key=role_priority)

        for group in (reference, historical):
            chosen = next(
                (item for item in group
                 if str(item["compile_job_id"]) not in used),
                None,
            )
            if chosen is not None:
                selected.append(chosen)
                used.add(str(chosen["compile_job_id"]))
                if len(selected) >= limit:
                    return selected[:limit]

    # Fill any remaining slots if an operator lacks one of the two role groups.
    for job in filtered:
        job_id = str(job["compile_job_id"])
        if job_id not in used:
            selected.append(job)
            used.add(job_id)
            if len(selected) >= limit:
                break
    return selected[:limit]


def rewrite_test_imports(test_source: str, module_name: str) -> str:
    """Redirect numbered/kernel module imports to the staged candidate module."""
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

    # Register the Ascend backend before test code imports torch/creates NPU tensors.
    if not re.search(r"(?m)^\s*import\s+torch_npu\b", test_source):
        test_source = "import torch_npu  # register Ascend device backend\n" + test_source
    return test_source


def make_sitecustomize_source() -> str:
    """Compile-only patch: force every Triton JITFunction.run through warmup."""
    return r"""# Generated by costModel/scripts/04_precompile_pilot.py
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
        Path(path).write_text(
            json.dumps(_STATE, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
                elif "npuir" in key_name:
                    suffix = ".npuir.mlir"
                elif "ttir" in key_name:
                    suffix = ".ttir.mlir"
                elif "ir" in key_name or "mlir" in key_name:
                    suffix = ".mlir"
                else:
                    suffix = ".txt"
                target = _ASM_DIR / ("k%03d_%s%s" % (call_index, key_name, suffix))
                target.write_text(value, encoding="utf-8")
                call["saved_files"].append(str(target))
            elif isinstance(value, (bytes, bytearray)):
                suffix = ".npubin" if "npubin" in key_name else ".bin"
                target = _ASM_DIR / ("k%03d_%s%s" % (call_index, key_name, suffix))
                target.write_bytes(bytes(value))
                call["saved_files"].append(str(target))
    except Exception as exc:
        call["capture_error"] = "%s: %s" % (type(exc).__name__, exc)
    return call


try:
    from triton.runtime.jit import JITFunction
    _ORIGINAL_RUN = JITFunction.run

    def _costmodel_run(self, *args, **kwargs):
        # The key switch: compile and return CompiledKernel, never launch Triton.
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
            call["kernel_name"] = (
                getattr(self, "__name__", None) or getattr(self, "name", None)
            )
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
"""


def classify_artifact(path: Path) -> str | None:
    # Inspect the full path because some backend dumps use generic filenames.
    name = "/".join(part.lower() for part in path.parts)
    suffix = path.suffix.lower()
    if "ttadapter" in name:
        return "ttadapter"
    if "npuir" in name:
        return "npuir"
    if "ttir" in name:
        return "ttir"
    if "npubin" in name or suffix == ".npubin":
        return "npubin"
    if suffix in ARTIFACT_SUFFIXES:
        return "other"
    return None


def copy_artifact_tree(source_root: Path, destination_root: Path) -> list[str]:
    """Copy compiler outputs while preserving relative paths."""
    copied: list[str] = []
    if not source_root.exists():
        return copied
    files = [source_root] if source_root.is_file() else source_root.rglob("*")
    for source in files:
        try:
            if not source.is_file() or source.suffix.lower() not in ARTIFACT_SUFFIXES:
                continue
            if source.name == "compile_state.json":
                continue
            relative = Path(source.name) if source_root.is_file() else source.relative_to(source_root)
            target = destination_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(target.resolve()))
        except OSError:
            continue
    return copied


def copy_worker_dump_files(home_dir: Path, cache_dir: Path, capture_dir: Path) -> list[str]:
    copied: list[str] = []
    copied.extend(copy_artifact_tree(home_dir / ".triton" / "dump", capture_dir / "dump_home"))
    copied.extend(copy_artifact_tree(cache_dir, capture_dir / "cache"))
    return copied


def parse_dump_paths(log_text: str) -> list[Path]:
    paths: list[Path] = []
    for line in log_text.splitlines():
        match = DUMP_LINE.search(line.strip())
        if not match:
            continue
        raw = match.group(1).strip().strip("'\"")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (DEFAULT_PROJECT_ROOT / path).resolve()
        if path not in paths:
            paths.append(path)
    return paths


def copy_reported_dump_paths(log_text: str, capture_dir: Path) -> list[str]:
    copied: list[str] = []
    for index, dump_path in enumerate(parse_dump_paths(log_text)):
        copied.extend(
            copy_artifact_tree(
                dump_path,
                capture_dir / "dump_reported" / "{:03d}".format(index),
            )
        )
    return copied


def patch_test_assertions() -> None:
    """Output assertions are not meaningful after warmup-only Triton calls."""
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
    """Execute one real operator test with the compile-only sitecustomize patch."""
    test_file = Path(os.environ["COSTMODEL_TEST_FILE"]).resolve()
    test_dir = Path(os.environ["COSTMODEL_TEST_DIR"]).resolve()
    capture_dir = Path(os.environ["COSTMODEL_CAPTURE_DIR"]).resolve()
    result_path = Path(os.environ["COSTMODEL_WORKER_RESULT"]).resolve()
    device_index = int(os.environ.get("COSTMODEL_DEVICE_INDEX", "0"))
    home_dir = Path.home()
    cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", test_dir / "cache"))

    result: dict[str, Any] = {
        "status": "failed",
        "kernel_executed": False,
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
                "Compile-only JIT patch unavailable: {}".format(
                    patch_state.get("patch_error") if patch_state else "sitecustomize not loaded"
                )
            )

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is False")

        torch.npu.set_device(device_index)
        result["device"] = "npu:{}".format(device_index)
        patch_test_assertions()

        os.chdir(test_dir)
        sys.path.insert(0, str(test_dir))
        runpy.run_path(str(test_file), run_name="__main__")
        result["status"] = "test_completed"
    except BaseException as exc:
        result["test_error"] = "{}: {}".format(type(exc).__name__, exc)
        result["test_traceback"] = traceback.format_exc()

    try:
        patch_module = sys.modules.get("sitecustomize")
        state = getattr(patch_module, "_STATE", {}) if patch_module else {}
        result["compile_state"] = state
    except Exception as exc:
        result["compile_state_error"] = "{}: {}".format(type(exc).__name__, exc)
        result["compile_state"] = {}

    result["dump_files"] = copy_worker_dump_files(home_dir, cache_dir, capture_dir)
    result["kernel_executed"] = False
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def rewrite_and_stage_job(
    job: dict[str, Any],
    project_root: Path,
    test_dir: Path,
    patch_dir: Path,
) -> tuple[Path, str]:
    operator_name = str(job["operator_name"])
    module_name = str(job.get("expected_test_module_name") or operator_name)
    source_path = resolve_path(project_root, str(job["source_path"]))
    test_path = resolve_path(project_root, str(job["test_path"]))

    source_code = source_path.read_text(encoding="utf-8")
    test_code = test_path.read_text(encoding="utf-8")

    # Copy sibling implementation/helper modules from the operator's dataset
    # folder. The chosen candidate is installed last under the expected module.
    support_root = test_path.parent
    for support_path in support_root.rglob("*.py"):
        if "__pycache__" in support_path.parts:
            continue
        if support_path.name.lower().startswith("test_"):
            continue
        relative = support_path.relative_to(support_root)
        target = test_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(support_path, target)

    candidate_module = test_dir / (module_name + ".py")
    candidate_module.parent.mkdir(parents=True, exist_ok=True)
    candidate_module.write_text(source_code, encoding="utf-8")

    rewritten_test = rewrite_test_imports(test_code, module_name)
    worker_test = test_dir / ("test_" + safe_name(module_name) + ".py")
    worker_test.write_text(rewritten_test, encoding="utf-8")

    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "sitecustomize.py").write_text(
        make_sitecustomize_source(),
        encoding="utf-8",
    )
    return worker_test, module_name


def classify_and_save_ir(
    job: dict[str, Any],
    capture_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, bool], list[str]]:
    sources = [
        path for path in capture_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES
    ]

    ir_paths: dict[str, list[str]] = {category: [] for category in IR_CATEGORIES}
    other_paths: list[str] = []
    seen_hashes: dict[str, set[str]] = {
        category: set() for category in (*IR_CATEGORIES, "other")
    }
    operator_dir = safe_name(job["operator_name"])
    candidate_key = safe_name(
        job.get("representative_candidate_id") or job["compile_job_id"]
    )

    for source in sorted(sources, key=lambda item: str(item)):
        category = classify_artifact(source)
        if category is None:
            continue
        digest = sha256_file(source)
        if digest in seen_hashes[category]:
            continue
        seen_hashes[category].add(digest)

        if category == "other":
            target_dir = output_dir / "other" / operator_dir
            suffix = source.suffix.lower() or ".bin"
            ordinal = len(other_paths)
        else:
            target_dir = output_dir / category / operator_dir
            if category == "ttir":
                suffix = ".ttir.mlir"
            elif category == "ttadapter":
                suffix = ".ttadapter.mlir"
            elif category == "npuir":
                suffix = ".npuir.mlir"
            else:
                suffix = source.suffix.lower() or ".npubin"
            ordinal = len(ir_paths[category])

        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "{}__k{:02d}{}".format(candidate_key, ordinal, suffix)
        shutil.copy2(source, target)
        if category == "other":
            other_paths.append(str(target.resolve()))
        else:
            ir_paths[category].append(str(target.resolve()))

    ir_found = {category: bool(ir_paths[category]) for category in IR_CATEGORIES}
    return ir_paths, ir_found, other_paths


def remove_previous_job_artifacts(job: dict[str, Any], output_dir: Path) -> None:
    operator_dir = safe_name(job["operator_name"])
    candidate_key = safe_name(
        job.get("representative_candidate_id") or job["compile_job_id"]
    )
    for category in (*IR_CATEGORIES, "other"):
        category_dir = output_dir / category / operator_dir
        if category_dir.is_dir():
            for path in category_dir.glob(candidate_key + "__k*"):
                if path.is_file():
                    path.unlink()


def metadata_path_for(job: dict[str, Any], output_dir: Path) -> Path:
    operator_dir = safe_name(job["operator_name"])
    job_key = safe_name(job["compile_job_id"])
    return output_dir / "metadata" / operator_dir / (job_key + ".json")


def run_one_job(
    job: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    device_index: int,
    timeout_seconds: int,
    force: bool,
) -> dict[str, Any]:
    work_root = output_dir / "_work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(
        prefix=safe_name(job["compile_job_id"]) + "_",
        dir=str(work_root),
    ))
    test_dir = work_dir / "test_env"
    patch_dir = work_dir / "patch"
    home_dir = work_dir / "home"
    cache_dir = work_dir / "cache"
    capture_dir = work_dir / "capture"
    for directory in (test_dir, patch_dir, home_dir, cache_dir, capture_dir):
        directory.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        **job,
        "status": "failed",
        "compile_pass": None,
        "kernel_executed": False,
        "device_index": device_index,
        "source_sha256_verified": True,
        "test_sha256_verified": True,
        "ir_paths": {},
        "ir_found": {category: False for category in IR_CATEGORIES},
    }

    try:
        if force:
            remove_previous_job_artifacts(job, output_dir)

        worker_test, module_name = rewrite_and_stage_job(
            job, project_root, test_dir, patch_dir
        )

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
                timeout=timeout_seconds,
            )
            process_wall_seconds = time.perf_counter() - start
            process_log = proc.stdout or ""
            result["process_return_code"] = proc.returncode
            result["process_wall_seconds"] = round(process_wall_seconds, 6)
            result["worker_log_tail"] = process_log[-30000:]
        except subprocess.TimeoutExpired as exc:
            result.update({
                "status": "timeout",
                "error": "Candidate exceeded timeout of {} seconds".format(timeout_seconds),
                "process_wall_seconds": round(time.perf_counter() - start, 6),
                "timeout_output_tail": (
                    exc.stdout[-10000:] if isinstance(exc.stdout, str) else None
                ),
            })
            return result

        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / (
            safe_name(job["operator_name"]) + "__" + safe_name(job["compile_job_id"]) + ".log"
        )
        log_path.write_text(process_log, encoding="utf-8")
        result["process_log"] = str(log_path.resolve())

        # Keep the earlier benchmark behavior: also collect the dump directory
        # printed by Triton, not only the isolated HOME and cache directories.
        copy_reported_dump_paths(process_log, capture_dir)

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
        compile_pass = None if n_attempt == 0 else (n_ok == n_attempt)

        ir_paths, ir_found, other_paths = classify_and_save_ir(
            job, capture_dir, output_dir
        )

        if compile_pass is True:
            status = (
                "success" if not worker_result.get("test_error")
                else "compiled_with_test_error"
            )
        elif compile_pass is False:
            status = "compile_failed"
        else:
            status = "no_kernel_compile_observed"

        result.update({
            "status": status,
            "compile_pass": compile_pass,
            "torch_version": worker_result.get("torch_version"),
            "triton_version": worker_result.get("triton_version"),
            "device": worker_result.get("device"),
            "n_attempt": n_attempt,
            "n_ok": n_ok,
            "warmup_time_ms": round(
                float(state.get("warmup_time_ms", 0.0) or 0.0), 3
            ),
            "kernel_calls": state.get("kernel_calls", []),
            "compile_errors": state.get("compile_errors", []),
            "test_error": worker_result.get("test_error"),
            "test_traceback": worker_result.get("test_traceback"),
            "sitecustomize_patch_error": state.get("patch_error"),
            "dump_file_count": len(worker_result.get("dump_files", [])),
            "ir_paths": ir_paths,
            "ir_found": ir_found,
            "other_artifact_paths": other_paths,
            "ir_counts": {
                **{category: len(paths) for category, paths in ir_paths.items()},
                "other": len(other_paths),
            },
            "kernel_executed": False,
            "compile_only_patch": True,
            "module_staged_as": module_name,
            "worker_status": worker_result.get("status"),
        })
    except Exception as exc:
        result.update({
            "status": "collector_error",
            "error": "{}: {}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        job_id = str(row.get("compile_job_id") or "")
        if job_id:
            results[job_id] = row
    return results


def can_skip_existing(job: dict[str, Any], output_dir: Path, force: bool) -> bool:
    if force:
        return False
    path = metadata_path_for(job, output_dir)
    if not path.is_file():
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        same_inputs = (
            old.get("source_sha256") == job.get("source_sha256")
            and old.get("test_sha256") == job.get("test_sha256")
        )
        return (
            same_inputs
            and old.get("status") in SUCCESS_STATUSES
            and bool((old.get("ir_found") or {}).get("ttir"))
        )
    except Exception:
        return False


def expand_candidate_manifest(
    selected_path: Path,
    manifest_path: Path,
    results_by_job: dict[str, dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> int:
    if not selected_path.is_file():
        return 0

    selected_rows = read_jsonl(selected_path)
    job_by_key = {
        (str(job.get("operator_name")), str(job.get("source_sha256"))): job
        for job in jobs
    }
    result_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results_by_job.values():
        result_by_key[
            (str(result.get("operator_name")), str(result.get("source_sha256")))
        ] = result

    manifest_rows: list[dict[str, Any]] = []
    for selected in selected_rows:
        row = dict(selected)
        key = (
            str(selected.get("operator_name")),
            str(selected.get("source_sha256")),
        )
        job = job_by_key.get(key)
        result = result_by_key.get(key)
        if job:
            row["compile_job_id"] = job.get("compile_job_id")
        if result:
            row["compile_status"] = result.get("status")
            row["compile_pass"] = result.get("compile_pass")
            row["warmup_time_ms"] = result.get("warmup_time_ms")
            row["ir_found"] = result.get("ir_found")
            row["ir_paths"] = result.get("ir_paths")
            row["compile_metadata_path"] = result.get("metadata_path")
            row["feature_join_status"] = (
                "awaiting_ttir_features"
                if (result.get("ir_found") or {}).get("ttir")
                else "ttir_not_available"
            )
        else:
            row["compile_status"] = "not_compiled_yet"
            row["compile_pass"] = None
            row["feature_join_status"] = "awaiting_precompile"
        manifest_rows.append(row)

    write_jsonl_atomic(manifest_path, manifest_rows)
    return len(manifest_rows)


def summarize_results(
    project_root: Path,
    compile_jobs_path: Path,
    output_dir: Path,
    device_index: int,
    workers: int,
    timeout_seconds: int,
    requested_limit: int,
    selected_jobs: list[dict[str, Any]],
    results_by_job: dict[str, dict[str, Any]],
    skipped_existing: int,
) -> dict[str, Any]:
    rows = list(results_by_job.values())
    rows.sort(key=lambda row: (
        str(row.get("operator_name", "")),
        str(row.get("compile_job_id", "")),
    ))
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "schema_version": "1.0",
        "project_root": str(project_root),
        "compile_jobs_path": str(compile_jobs_path),
        "output_dir": str(output_dir),
        "device_index": device_index,
        "workers": workers,
        "timeout_seconds": timeout_seconds,
        "requested_limit": requested_limit,
        "selected_job_count_this_run": len(selected_jobs),
        "skipped_existing_count_this_run": skipped_existing,
        "unique_compile_jobs_total": len(read_jsonl(compile_jobs_path)),
        "results_recorded_total": len(rows),
        "success_count": sum(row.get("status") in SUCCESS_STATUSES for row in rows),
        "compile_pass_count": sum(row.get("compile_pass") is True for row in rows),
        "compile_fail_count": sum(row.get("compile_pass") is False for row in rows),
        "compile_unknown_count": sum(row.get("compile_pass") is None for row in rows),
        "ir_file_counts": {
            category: sum(
                len((row.get("ir_paths") or {}).get(category, []))
                for row in rows
            )
            for category in IR_CATEGORIES
        },
        "status_counts": status_counts,
        "kernel_executed": False,
        "compile_only_patch": True,
        "results": [
            {
                "compile_job_id": row.get("compile_job_id"),
                "operator_name": row.get("operator_name"),
                "representative_candidate_id": row.get("representative_candidate_id"),
                "candidate_type": row.get("candidate_type"),
                "selected_roles": row.get("selected_roles"),
                "status": row.get("status"),
                "compile_pass": row.get("compile_pass"),
                "n_attempt": row.get("n_attempt"),
                "warmup_time_ms": row.get("warmup_time_ms"),
                "ir_found": row.get("ir_found"),
                "ir_counts": row.get("ir_counts"),
            }
            for row in rows
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warmup-precompile the selected 250-candidate Cost Model pilot."
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--compile-jobs", default="costModel/data/pilot/compile_jobs.jsonl")
    parser.add_argument("--selected-candidates", default="costModel/data/pilot/selected_candidates.jsonl")
    parser.add_argument("--output-dir", default="costModel/artifacts/ir")
    parser.add_argument("--manifest-output", default="costModel/data/pilot/candidate_ir_manifest.jsonl")
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Jobs to run; default 10 for smoke test; use 0 for all 248 unique jobs.",
    )
    parser.add_argument(
        "--sample-strategy", choices=["balanced", "file_order"], default="balanced",
        help="For a limited run, balanced samples reference+historical jobs across operators.",
    )
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args._worker:
        return worker_main()

    project_root = Path(args.project_root).expanduser().resolve()
    compile_jobs_path = resolve_path(project_root, args.compile_jobs)
    selected_path = resolve_path(project_root, args.selected_candidates)
    output_dir = resolve_path(project_root, args.output_dir)
    manifest_output = resolve_path(project_root, args.manifest_output)

    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")

    jobs = load_compile_jobs(compile_jobs_path)
    validate_job_files(jobs, project_root)
    selected_jobs = select_jobs(
        jobs, args.limit, args.operator, args.sample_strategy
    )

    log("Project root       : {}".format(project_root))
    log("Compile jobs       : {}".format(compile_jobs_path))
    log("Output directory   : {}".format(output_dir))
    log("Unique jobs in file: {}".format(len(jobs)))
    log("Selected this run  : {}".format(len(selected_jobs)))
    log("Device             : npu:{}".format(args.device_index))
    log("Workers            : {}".format(args.workers))
    log("Triton JIT launch  : disabled (JIT warmup only)")

    if args.dry_run:
        for index, job in enumerate(selected_jobs, 1):
            log(
                "[{}/{}] {} | {} | roles={} | {}".format(
                    index,
                    len(selected_jobs),
                    job.get("operator_name"),
                    job.get("representative_candidate_id"),
                    job.get("selected_roles"),
                    job.get("source_path"),
                )
            )
        log("Dry-run completed; no NPU compilation was started.")
        return 0

    if args.workers > 1:
        log(
            "WARNING: workers share one NPU. Parallel warmup was validated "
            "previously on this environment; start with workers=1 for smoke "
            "test and use workers=2 for the full pilot."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compile_jobs.used.jsonl").write_text(
        compile_jobs_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    results_path = output_dir / "compile_results.jsonl"
    results_by_job = load_existing_results(results_path)

    skipped_existing = 0
    pending: list[dict[str, Any]] = []
    for job in selected_jobs:
        if can_skip_existing(job, output_dir, args.force):
            metadata_path = metadata_path_for(job, output_dir)
            old = json.loads(metadata_path.read_text(encoding="utf-8"))
            old["metadata_path"] = str(metadata_path.resolve())
            results_by_job[str(job["compile_job_id"])] = old
            skipped_existing += 1
            log("SKIP existing success: {}".format(job["compile_job_id"]))
        else:
            pending.append(job)

    def persist_state() -> None:
        rows = list(results_by_job.values())
        rows.sort(key=lambda row: (
            str(row.get("operator_name", "")),
            str(row.get("compile_job_id", "")),
        ))
        write_jsonl_atomic(results_path, rows)
        expand_candidate_manifest(
            selected_path,
            manifest_output,
            results_by_job,
            jobs,
        )

    start_all = time.perf_counter()
    if args.workers == 1:
        for index, job in enumerate(pending, 1):
            log("[{}/{}] compiling {}".format(index, len(pending), job["compile_job_id"]))
            result = run_one_job(
                job, project_root, output_dir,
                args.device_index, args.timeout, args.force,
            )
            metadata_path = metadata_path_for(job, output_dir)
            result["metadata_path"] = str(metadata_path.resolve())
            write_json_atomic(metadata_path, result)
            results_by_job[str(job["compile_job_id"])] = result
            persist_state()
            log(
                "  status={} compile_pass={} warmup={}ms IR={}".format(
                    result.get("status"),
                    result.get("compile_pass"),
                    result.get("warmup_time_ms", 0),
                    result.get("ir_found"),
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(
                    run_one_job,
                    job,
                    project_root,
                    output_dir,
                    args.device_index,
                    args.timeout,
                    args.force,
                ): job
                for job in pending
            }
            for future in concurrent.futures.as_completed(future_map):
                job = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **job,
                        "status": "collector_error",
                        "compile_pass": None,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                        "kernel_executed": False,
                    }
                metadata_path = metadata_path_for(job, output_dir)
                result["metadata_path"] = str(metadata_path.resolve())
                write_json_atomic(metadata_path, result)
                results_by_job[str(job["compile_job_id"])] = result
                persist_state()
                log(
                    "{}: status={} compile_pass={} IR={}".format(
                        job["compile_job_id"],
                        result.get("status"),
                        result.get("compile_pass"),
                        result.get("ir_found"),
                    )
                )

    elapsed = time.perf_counter() - start_all
    summary = summarize_results(
        project_root,
        compile_jobs_path,
        output_dir,
        args.device_index,
        args.workers,
        args.timeout,
        args.limit,
        selected_jobs,
        results_by_job,
        skipped_existing,
    )
    summary["elapsed_seconds_this_run"] = round(elapsed, 3)
    summary["manifest_output"] = str(manifest_output)
    summary_path = output_dir / "compile_summary.json"
    write_json_atomic(summary_path, summary)

    manifest_count = expand_candidate_manifest(
        selected_path,
        manifest_output,
        results_by_job,
        jobs,
    )

    log("=" * 76)
    log("Pilot warmup precompilation finished")
    log("Jobs selected this run : {}".format(len(selected_jobs)))
    log("Jobs newly run         : {}".format(len(pending)))
    log("Existing successes     : {}".format(skipped_existing))
    log("Recorded unique jobs   : {}".format(len(results_by_job)))
    log("Compile pass           : {}".format(summary["compile_pass_count"]))
    log("Compile fail           : {}".format(summary["compile_fail_count"]))
    log("Compile unknown        : {}".format(summary["compile_unknown_count"]))
    log("IR file counts         : {}".format(summary["ir_file_counts"]))
    log("Logical candidate map  : {} rows".format(manifest_count))
    log("Summary                : {}".format(summary_path))
    log("Results JSONL          : {}".format(results_path))
    log("Triton kernel executed : False (JIT warmup only)")
    log("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
