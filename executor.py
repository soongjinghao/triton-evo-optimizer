#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Triton executor with unified MSProf timing + hardware profiling.

Compatibility:
- Keeps the original TritonExecutor constructor/evaluate interface.
- Keeps EvaluationResult fields and speedup/fitness calculation.
- Uses the regular `msprof` application command instead of `msprof op`.

Profile modes (environment variable COSTMODEL_PROFILE_MODE):
- full (default): separate low-overhead timing pass and hardware-metrics pass.
- timing_only: only collect the timing pass; useful for fast evolutionary search.

The raw MSProf PROF directories are preserved. A compact profile_summary.json is
written beside them and contains only useful, normalized fields.
"""

import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import EAConfig


ICONS = {
    "rocket": "🚀",
    "timer": "⏱️",
    "check": "✅",
    "cross": "❌",
    "gear": "⚙️",
    "chart": "📊",
    "warning": "⚠️",
    "target": "🎯",
    "zap": "⚡",
    "trophy": "🏆",
    "microscope": "🔬",
}


@dataclass
class EvaluationResult:
    """Evaluation result. execution_time is measured in microseconds."""

    success: bool
    execution_time: float
    speedup: float
    fitness: float
    error: Optional[str] = None


class TritonExecutor:
    """Evaluate Triton candidates and collect a consistent MSProf profile."""

    TIMING_FLAGS = [
        "--task-time=l0",
        "--ai-core=off",
        "--ascendcl=off",
        "--runtime-api=off",
    ]

    # This is the confirmed V1 hardware profile: PipeUtilization + HBM/LLC
    # + NPU task memory. Other AIC metric groups remain separate experiments.
    HARDWARE_FLAGS = [
        "--task-time=l1",
        "--task-memory=on",
        "--ai-core=on",
        "--aic-mode=task-based",
        "--aic-metrics=PipeUtilization",
        "--sys-hardware-mem=on",
        "--sys-hardware-mem-freq=50",
        "--llc-profiling=read",
        "--ascendcl=off",
        "--runtime-api=off",
        "--ge-api=off",
    ]

    # Useful columns from op_summary. The parser only emits columns actually
    # present in a given CANN/MSProf version/profile.
    OP_SUMMARY_KEEP = [
        "Device_id",
        "Task ID",
        "Stream ID",
        "Op Name",
        "OP Type",
        "OP State",
        "Task Type",
        "Task Start Time(us)",
        "Task Duration(us)",
        "Block Num",
        "Mix Block Num",
        "HF32 Eligible",
        "Input Shapes",
        "Input Data Types",
        "Output Shapes",
        "Output Data Types",
        "aicore_time(us)",
        "aic_total_cycles",
        "aic_mac_time(us)",
        "aic_mac_ratio",
        "aic_scalar_time(us)",
        "aic_scalar_ratio",
        "aic_mte1_time(us)",
        "aic_mte1_ratio",
        "aic_mte2_time(us)",
        "aic_mte2_ratio",
        "aic_fixpipe_time(us)",
        "aic_fixpipe_ratio",
        "aic_icache_miss_rate",
        "aiv_time(us)",
        "aiv_total_cycles",
        "aiv_vec_time(us)",
        "aiv_vec_ratio",
        "aiv_scalar_time(us)",
        "aiv_scalar_ratio",
        "aiv_mte2_time(us)",
        "aiv_mte2_ratio",
        "aiv_mte3_time(us)",
        "aiv_mte3_ratio",
        "aiv_icache_miss_rate",
        "cube_utilization(%)",
        "aiv_vec_bankgroup_cflt_ratio",
        "aiv_vec_bank_cflt_ratio",
        "aiv_vec_resc_cflt_ratio",
        "aic_l0a_read_bw(GB/s)",
        "aic_l0a_write_bw(GB/s)",
        "aic_l0b_read_bw(GB/s)",
        "aic_l0b_write_bw(GB/s)",
        "aic_l0c_read_bw_cube(GB/s)",
        "aic_l0c_write_bw_cube(GB/s)",
        "aic_l1_read_bw(GB/s)",
        "aic_l1_write_bw(GB/s)",
        "aic_main_mem_read_bw(GB/s)",
        "aic_main_mem_write_bw(GB/s)",
        "aic_l2_read_bw(GB/s)",
        "aic_l2_write_bw(GB/s)",
        "aiv_l0c_read_bw(GB/s)",
        "aiv_l0c_write_bw(GB/s)",
        "aiv_ub_read_bw(GB/s)",
        "aiv_ub_write_bw(GB/s)",
        "aiv_ub_read_bw_vector(GB/s)",
        "aiv_ub_write_bw_vector(GB/s)",
        "aiv_ub_read_bw_scalar(GB/s)",
        "aiv_ub_write_bw_scalar(GB/s)",
        "aiv_main_mem_read_bw(GB/s)",
        "aiv_main_mem_write_bw(GB/s)",
        "aiv_l2_read_bw(GB/s)",
        "aiv_l2_write_bw(GB/s)",
        "aic_read_main_memory_datas(KB)",
        "aic_write_main_memory_datas(KB)",
        "aic_GM_to_L1_datas(KB)",
        "aic_L0C_to_L1_datas(KB)",
        "aic_L0C_to_GM_datas(KB)",
        "aic_GM_to_UB_datas(KB)",
        "aic_UB_to_GM_datas(KB)",
        "aiv_read_main_memory_datas(KB)",
        "aiv_write_main_memory_datas(KB)",
        "aiv_GM_to_L1_datas(KB)",
        "aiv_L0C_to_L1_datas(KB)",
        "aiv_L0C_to_GM_datas(KB)",
        "aiv_GM_to_UB_datas(KB)",
        "aiv_UB_to_GM_datas(KB)",
    ]

    def __init__(
        self,
        baseline_time: float,
        test_code_path: str,
        config: EAConfig,
        kernel_name: str = "kernel",
        work_dir: Optional[Path] = None,
        profile_mode: Optional[str] = None,
    ):
        self.baseline_time = float(baseline_time)
        self.test_code_path = Path(test_code_path)
        self.config = config
        self.kernel_name = kernel_name
        self.work_dir = Path(work_dir) if work_dir is not None else Path(".")
        self.performance_dir = self.work_dir / "performance"
        self.performance_dir.mkdir(parents=True, exist_ok=True)

        self.profile_mode = (
            profile_mode or os.environ.get("COSTMODEL_PROFILE_MODE", "full")
        ).strip().lower()
        if self.profile_mode not in {"full", "timing_only"}:
            raise ValueError(
                "COSTMODEL_PROFILE_MODE must be 'full' or 'timing_only', "
                f"got {self.profile_mode!r}"
            )

        self.last_profile_dir: Optional[Path] = None
        self.last_profile_error: Optional[str] = None

        print(f"\n[{ICONS['rocket']}] [Executor] Initializing Triton executor...")
        print(f"       └─ Operator name: {kernel_name}")
        print(f"       └─ Working directory: {self.work_dir}")
        print(f"       └─ Baseline time: {self.baseline_time:.2f}us")
        print(f"       └─ Test file: {self.test_code_path}")
        print(f"       └─ Profiling mode: {self.profile_mode}")

    @staticmethod
    def _read_csv(path: Path) -> List[Dict[str, str]]:
        if not path or not path.exists() or path.stat().st_size == 0:
            return []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "-"}:
            return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _median(values: List[float]) -> Optional[float]:
        clean = [float(v) for v in values if v is not None]
        return float(statistics.median(clean)) if clean else None

    def _new_eval_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        eval_dir = self.performance_dir / self.kernel_name / f"eval_{stamp}_{os.getpid()}"
        suffix = 1
        base = eval_dir
        while eval_dir.exists():
            eval_dir = Path(f"{base}_{suffix}")
            suffix += 1
        eval_dir.mkdir(parents=True, exist_ok=True)
        self.last_profile_dir = eval_dir
        return eval_dir

    @staticmethod
    def _find_latest_prof_dir(output_dir: Path) -> Optional[Path]:
        if not output_dir.exists():
            return None
        prof_dirs = [
            p for p in output_dir.glob("PROF_*") if p.is_dir()
        ]
        if not prof_dirs:
            return None
        return max(prof_dirs, key=lambda p: p.stat().st_mtime)

    @classmethod
    def _find_csv(cls, output_dir: Path, stem: str) -> Optional[Path]:
        prof_dirs = sorted(
            [p for p in output_dir.glob("PROF_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if output_dir.exists() else []
        for prof_dir in prof_dirs:
            out = prof_dir / "mindstudio_profiler_output"
            matches = sorted(out.glob(f"{stem}_*.csv")) if out.exists() else []
            for path in reversed(matches):
                if path.exists() and path.stat().st_size > 0:
                    return path
        return None

    def _run_msprof_pass(
        self,
        test_file: Path,
        output_dir: Path,
        flags: List[str],
        timeout: int,
        pass_name: str,
    ) -> Tuple[bool, List[str]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "msprof",
            f"--output={output_dir}",
            *flags,
            "python3",
            str(test_file.resolve()),
        ]
        log_file = output_dir / "msprof.log"
        print(f"[Executor] {pass_name} command: {' '.join(cmd)}")
        try:
            with log_file.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    cwd=str(test_file.parent),
                )
            if result.returncode != 0:
                self.last_profile_error = (
                    f"{pass_name} MSProf exited with code {result.returncode}; "
                    f"see {log_file}"
                )
                print(f"[Executor] WARNING: {self.last_profile_error}")
                return False, cmd
            if self._find_latest_prof_dir(output_dir) is None:
                self.last_profile_error = (
                    f"{pass_name} MSProf completed but no PROF_* directory was found; "
                    f"see {log_file}"
                )
                print(f"[Executor] WARNING: {self.last_profile_error}")
                return False, cmd
            return True, cmd
        except subprocess.TimeoutExpired:
            self.last_profile_error = f"{pass_name} MSProf timed out; see {log_file}"
            print(f"[Executor] WARNING: {self.last_profile_error}")
            return False, cmd
        except FileNotFoundError:
            self.last_profile_error = "msprof executable was not found in PATH"
            print(f"[Executor] ERROR: {self.last_profile_error}")
            return False, cmd
        except Exception as exc:
            self.last_profile_error = f"{pass_name} MSProf failed: {exc}"
            print(f"[Executor] ERROR: {self.last_profile_error}")
            return False, cmd

    def _kernel_matches(self, row: Dict[str, str]) -> bool:
        target = self.kernel_name.lower().strip()
        target_alt = target.lstrip("_")
        fields = [
            row.get("Op Name", ""),
            row.get("OP Type", ""),
            row.get("kernel_name", ""),
            row.get("kernel_type", ""),
        ]
        values = [str(v).lower().strip() for v in fields if v]
        return any(
            target in value or (target_alt and target_alt in value)
            for value in values
        )

    def _parse_timing(self, output_dir: Path) -> Tuple[Optional[float], str, int, List[Dict[str, str]]]:
        """Return median target-kernel duration, source, sample count, matched rows."""
        op_summary = self._find_csv(output_dir, "op_summary")
        rows = self._read_csv(op_summary) if op_summary else []
        matched = [r for r in rows if self._kernel_matches(r)]
        durations = [
            n for n in (self._number(r.get("Task Duration(us)")) for r in matched)
            if n is not None and n > 0
        ]
        if durations:
            return self._median(durations), "op_summary.Task Duration(us)", len(durations), matched

        task_time = self._find_csv(output_dir, "task_time")
        task_rows = self._read_csv(task_time) if task_time else []
        task_matched = [r for r in task_rows if self._kernel_matches(r)]
        task_durations = [
            n for n in (self._number(r.get("task_time(us)")) for r in task_matched)
            if n is not None and n > 0
        ]
        if task_durations:
            return self._median(task_durations), "task_time.task_time(us)", len(task_durations), task_matched

        # Last-resort fallback: op_statistic contains Avg Time(us) by OP Type.
        op_stat = self._find_csv(output_dir, "op_statistic")
        stat_rows = self._read_csv(op_stat) if op_stat else []
        stat_matched = [r for r in stat_rows if self._kernel_matches(r)]
        stat_durations = [
            n for n in (self._number(r.get("Avg Time(us)")) for r in stat_matched)
            if n is not None and n > 0
        ]
        if stat_durations:
            return self._median(stat_durations), "op_statistic.Avg Time(us)", len(stat_durations), stat_matched

        return None, "unavailable", 0, []

    def _aggregate_core_csv(self, path: Optional[Path], ratio_fields: List[str]) -> Dict[str, Any]:
        rows = self._read_csv(path) if path else []
        result: Dict[str, Any] = {"core_count": len(rows)}
        active = 0
        for field in ratio_fields:
            vals = [
                n for n in (self._number(r.get(field)) for r in rows)
                if n is not None
            ]
            result[f"{field}_mean"] = statistics.fmean(vals) if vals else None
            result[f"{field}_max"] = max(vals) if vals else None
        for row in rows:
            vals = [self._number(row.get(f)) for f in ratio_fields]
            if any(v is not None and v > 0 for v in vals):
                active += 1
        result["active_core_count"] = active
        return result

    def _extract_compact_profile(
        self,
        timing_dir: Path,
        metrics_dir: Optional[Path],
        eval_dir: Path,
        timing: Optional[float],
        timing_source: str,
        timing_samples: int,
        commands: Dict[str, List[str]],
        metrics_success: bool,
    ) -> None:
        """Write a compact, useful profile summary; raw MSProf files stay untouched."""
        source_dir = metrics_dir if metrics_success and metrics_dir else timing_dir
        op_path = self._find_csv(source_dir, "op_summary")
        if op_path is None and source_dir != timing_dir:
            op_path = self._find_csv(timing_dir, "op_summary")
        op_rows = self._read_csv(op_path) if op_path else []
        kernel_rows = [r for r in op_rows if self._kernel_matches(r)]

        kernel_record: Dict[str, Any] = {}
        if kernel_rows:
            # Keep identity/shape fields from the first matching task, and use
            # median for numeric performance counters across repeated launches.
            first = kernel_rows[0]
            for field in self.OP_SUMMARY_KEEP:
                if field not in first:
                    continue
                values = [r.get(field) for r in kernel_rows if r.get(field) not in (None, "", "N/A")]
                numeric = [self._number(v) for v in values]
                numeric = [v for v in numeric if v is not None]
                if field in {
                    "Task ID", "Stream ID", "Device_id", "Op Name", "OP Type", "OP State",
                    "Task Type", "Task Start Time(us)", "HF32 Eligible", "Input Shapes",
                    "Input Data Types", "Output Shapes", "Output Data Types", "Block Num", "Mix Block Num",
                }:
                    kernel_record[field] = first.get(field)
                elif numeric:
                    kernel_record[field] = self._median(numeric)
                else:
                    kernel_record[field] = first.get(field)

        ai_core_path = self._find_csv(source_dir, "ai_core_utilization")
        ai_vector_path = self._find_csv(source_dir, "ai_vector_core_utilization")
        ai_core = self._aggregate_core_csv(
            ai_core_path,
            ["mac_ratio", "scalar_ratio", "mte1_ratio", "mte2_ratio", "fixpipe_ratio", "icache_miss_rate"],
        ) if ai_core_path else {}
        ai_vector = self._aggregate_core_csv(
            ai_vector_path,
            ["vec_ratio", "mac_ratio", "scalar_ratio", "mte1_ratio", "mte2_ratio", "mte3_ratio", "icache_miss_rate", "memory_bound"],
        ) if ai_vector_path else {}

        hbm_path = self._find_csv(source_dir, "hbm")
        hbm_rows = self._read_csv(hbm_path) if hbm_path else []
        hbm_avg = next((r for r in hbm_rows if str(r.get("Metric", "")).lower() == "average"), None)
        if hbm_avg is None and hbm_rows:
            hbm_avg = hbm_rows[0]
        hbm = {
            "read_mb_s": self._number(hbm_avg.get("Read(MB/s)")) if hbm_avg else None,
            "write_mb_s": self._number(hbm_avg.get("Write(MB/s)")) if hbm_avg else None,
        }

        llc_path = self._find_csv(source_dir, "llc_read_write")
        llc_rows = self._read_csv(llc_path) if llc_path else []
        llc: Dict[str, Any] = {}
        for mode in ("read", "write"):
            candidates = [r for r in llc_rows if str(r.get("Mode", "")).lower() == mode]
            avg = next((r for r in candidates if str(r.get("Task", "")).lower() == "average"), None)
            if avg is None and candidates:
                avg = candidates[0]
            llc[f"{mode}_hit_rate_pct"] = self._number(avg.get("Hit Rate(%)")) if avg else None
            llc[f"{mode}_throughput_mb_s"] = self._number(avg.get("Throughput(MB/s)")) if avg else None

        task_time_path = self._find_csv(source_dir, "task_time")
        if task_time_path is None and source_dir != timing_dir:
            task_time_path = self._find_csv(timing_dir, "task_time")
        task_rows = self._read_csv(task_time_path) if task_time_path else []
        target_task_rows = [r for r in task_rows if self._kernel_matches(r)]
        task_time_values = [
            n for n in (self._number(r.get("task_time(us)")) for r in target_task_rows)
            if n is not None and n > 0
        ]
        task_time_validation = {
            "median_task_time_us": self._median(task_time_values),
            "sample_count": len(task_time_values),
            "task_id": target_task_rows[0].get("task_id") if target_task_rows else None,
            "stream_id": target_task_rows[0].get("stream_id") if target_task_rows else None,
        }

        npu_mem_path = self._find_csv(source_dir, "npu_mem")
        npu_mem_rows = self._read_csv(npu_mem_path) if npu_mem_path else []
        app_rows = [r for r in npu_mem_rows if str(r.get("event", "")).upper() == "APP"]
        if not app_rows:
            app_rows = npu_mem_rows
        npu_memory: Dict[str, Any] = {}
        for source_field, output_field in (
            ("ddr(KB)", "ddr_kb"),
            ("hbm(KB)", "hbm_kb"),
            ("memory(KB)", "total_memory_kb"),
        ):
            values = [
                n for n in (self._number(r.get(source_field)) for r in app_rows)
                if n is not None
            ]
            npu_memory[f"{output_field}_peak"] = max(values) if values else None
            npu_memory[f"{output_field}_mean"] = self._median(values)

        summary = {
            "schema_version": "costmodel_msprof_v1",
            "kernel_name": self.kernel_name,
            "profile_mode": self.profile_mode,
            "timing": {
                "execution_time_us": timing,
                "source": timing_source,
                "sample_count": timing_samples,
                "task_time_cross_check": task_time_validation,
            },
            "kernel": kernel_record,
            "core_utilization": {
                "ai_core": ai_core,
                "ai_vector_core": ai_vector,
            },
            "memory": {
                "hbm": hbm,
                "llc": llc,
                "npu_memory": npu_memory,
            },
            "collection": {
                "hardware_profile_success": metrics_success,
                "op_summary_rows_for_kernel": len(kernel_rows),
                "source_csv": {
                    "op_summary": str(op_path) if op_path else None,
                    "ai_core_utilization": str(ai_core_path) if ai_core_path else None,
                    "ai_vector_core_utilization": str(ai_vector_path) if ai_vector_path else None,
                    "hbm": str(hbm_path) if hbm_path else None,
                    "llc_read_write": str(llc_path) if llc_path else None,
                    "npu_mem": str(npu_mem_path) if npu_mem_path else None,
                    "task_time": str(task_time_path) if task_time_path else None,
                },
                "commands": commands,
                "raw_profile_directories": {
                    "timing": str(timing_dir),
                    "hardware": str(metrics_dir) if metrics_dir else None,
                },
            },
        }
        summary_path = eval_dir / "profile_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _run_msprof(self, test_file: Path, timeout: int = 300) -> Optional[float]:
        """Run timing and optional hardware passes; return the timing label in us."""
        eval_dir = self._new_eval_dir()
        timing_dir = eval_dir / "timing"
        timing_ok, timing_cmd = self._run_msprof_pass(
            test_file=test_file,
            output_dir=timing_dir,
            flags=self.TIMING_FLAGS,
            timeout=timeout,
            pass_name="timing",
        )
        if not timing_ok:
            return None

        execution_time, source, sample_count, _ = self._parse_timing(timing_dir)
        if execution_time is None or execution_time <= 0:
            self.last_profile_error = (
                "MSProf completed, but no positive duration was found for target kernel "
                f"{self.kernel_name!r}; inspect {timing_dir}"
            )
            print(f"[Executor] ERROR: {self.last_profile_error}")
            return None

        metrics_dir: Optional[Path] = None
        metrics_ok = False
        metrics_cmd: List[str] = []
        if self.profile_mode == "full":
            metrics_dir = eval_dir / "hardware"
            metrics_ok, metrics_cmd = self._run_msprof_pass(
                test_file=test_file,
                output_dir=metrics_dir,
                flags=self.HARDWARE_FLAGS,
                timeout=timeout,
                pass_name="hardware",
            )
            if not metrics_ok:
                print(
                    "[Executor] WARNING: hardware profile failed; "
                    "the timing result remains valid and will be returned."
                )

        self._extract_compact_profile(
            timing_dir=timing_dir,
            metrics_dir=metrics_dir,
            eval_dir=eval_dir,
            timing=execution_time,
            timing_source=source,
            timing_samples=sample_count,
            commands={"timing": timing_cmd, "hardware": metrics_cmd},
            metrics_success=metrics_ok,
        )
        print(f"[Executor] Target kernel time: {execution_time:.3f} us ({source}, n={sample_count})")
        print(f"[Executor] Compact profile: {eval_dir / 'profile_summary.json'}")
        return execution_time

    def evaluate(self, code: str, timeout: int = 1200) -> EvaluationResult:
        """Evaluate one candidate while preserving the original return contract."""
        print(f"\n[{ICONS['microscope']}] [Executor] Starting evaluation...")
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_"))
        try:
            kernel_file = temp_dir / f"{self.kernel_name}.py"
            kernel_file.write_text(code, encoding="utf-8")

            test_content = self.test_code_path.read_text(encoding="utf-8")
            kernel_name_re = re.escape(self.kernel_name)
            module_re = rf"(?:kernel|{kernel_name_re}(?:_\d+)?(?:\.[\w.]+)?)"
            from_pattern = rf"^from\s+{module_re}\s+import"
            import_pattern = rf"^import\s+{module_re}\b(\s+as\s+\w+)?"

            modified_test = re.sub(
                from_pattern,
                f"from {self.kernel_name} import",
                test_content,
                flags=re.MULTILINE,
            )
            modified_test = re.sub(
                import_pattern,
                rf"import {self.kernel_name}\1",
                modified_test,
                flags=re.MULTILINE,
            )

            test_file = temp_dir / f"test_{self.kernel_name}.py"
            test_file.write_text(modified_test, encoding="utf-8")
            print(f"       └─ Candidate code length: {len(code)} characters")

            current_time = self._run_msprof(test_file, timeout=timeout)
        except Exception as exc:
            current_time = None
            self.last_profile_error = f"Evaluation setup failed: {exc}"
            print(f"[Executor] ERROR: {self.last_profile_error}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if current_time is None:
            return EvaluationResult(
                success=False,
                execution_time=0.0,
                speedup=0.0,
                fitness=0.0,
                error=self.last_profile_error or "Performance test failed",
            )

        print(f"[{ICONS['check']}] [Executor] Timing pass completed")
        print(f"       └─ Baseline time: {self.baseline_time:.3f} us")
        print(f"       └─ Optimized time: {current_time:.3f} us")

        if current_time > 0 and self.baseline_time > 0:
            raw_speedup = self.baseline_time / current_time - 1.0
            speedup = max(raw_speedup, 0.0)
            fitness = min(speedup, 2.0)
            if raw_speedup < 0:
                print(f"[{ICONS['warning']}] Candidate is slower than baseline; speedup clipped to 0")
            elif raw_speedup > 2.0:
                print(f"[{ICONS['trophy']}] Speedup exceeds 2.0; fitness capped at 2.0")
            competition_score = fitness * 100.0
            print(f"[{ICONS['target']}] Final speedup: {speedup:.4f}")
            print(f"       └─ Fitness: {fitness:.4f}; competition score: {competition_score:.1f}/200")
        else:
            speedup = 0.0
            fitness = 0.0
            print(f"[{ICONS['warning']}] Invalid baseline or execution time; score set to 0")

        return EvaluationResult(
            success=True,
            execution_time=current_time,
            speedup=speedup,
            fitness=fitness,
            error=None,
        )
