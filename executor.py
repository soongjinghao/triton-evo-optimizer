#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Triton Executor using ordinary application-level MSProf task-time profiling.

Verified command form:
  msprof --output=DIR --task-time=l0 --ai-core=off --ascendcl=off \
      --runtime-api=off python3 TEST_SCRIPT

The test script remains responsible for correctness assertions.
"""

import csv
import json
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import EAConfig


ICONS = {
    "rocket": "🚀", "timer": "⏱️", "check": "✅", "cross": "❌",
    "gear": "⚙️", "chart": "📊", "warning": "⚠️", "target": "🎯",
    "zap": "⚡", "trophy": "🏆", "microscope": "🔬",
}


@dataclass
class EvaluationResult:
    """Evaluation result; original fields are kept for caller compatibility."""
    success: bool
    execution_time: float  # Median kernel task duration, microseconds.
    speedup: float
    fitness: float
    error: Optional[str] = None
    measurements: List[float] = field(default_factory=list)
    measurement_stats: Dict[str, float] = field(default_factory=dict)
    hardware_metrics: List[Dict[str, Any]] = field(default_factory=list)
    profile_dir: Optional[str] = None


class TritonExecutor:
    """Evaluate candidate Triton code with ordinary MSProf task-time capture.

    profile_repeats is the number of independent MSProf sessions. Each session
    runs the whole test file once; all matching target-kernel tasks are retained.
    """

    def __init__(
        self,
        baseline_time: float,
        test_code_path: str,
        config: EAConfig,
        kernel_name: str = "kernel",
        work_dir: Optional[Path] = None,
        profile_repeats: int = 1,
        msprof_executable: str = "msprof",
    ):
        self.baseline_time = float(baseline_time)
        self.test_code_path = Path(test_code_path)
        self.config = config
        self.kernel_name = kernel_name
        self.work_dir = Path(work_dir) if work_dir is not None else Path(".")
        self.performance_dir = self.work_dir / "performance"
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        self.profile_repeats = max(1, int(profile_repeats))
        self.msprof_executable = msprof_executable
        self.last_profile_error: Optional[str] = None

        print(f"\n[{ICONS['rocket']}] [Executor] Initializing Triton executor...")
        print(f"       └─ Operator name: {kernel_name}")
        print(f"       └─ Working directory: {self.work_dir}")
        print(f"       └─ Baseline time: {self.baseline_time:.2f}us (read from JSON)")
        print(f"       └─ Test file: {self.test_code_path}")
        print(f"       └─ MSProf repeats: {self.profile_repeats}")
        print("       └─ Profiling mode: task-time=l0 (AI-core counters disabled)")

    # -------------------------- CSV helpers --------------------------

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            value = str(value).strip().replace(",", "")
            if not value or value.upper() in {"N/A", "NA", "NONE"}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        number = TritonExecutor._safe_float(value)
        return int(number) if number is not None else None

    @staticmethod
    def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = []
            for raw in reader:
                row = {}
                for key, value in raw.items():
                    if key is not None:
                        row[key.strip().lstrip("\ufeff")] = (
                            value.strip() if isinstance(value, str) else ""
                        )
                rows.append(row)
        return columns, rows

    @staticmethod
    def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
        files = [p for p in directory.glob(pattern) if p.is_file()]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _find_latest_prof_dir(self, root: Path) -> Optional[Path]:
        prof_dirs = [p for p in root.rglob("PROF_*") if p.is_dir()] if root.exists() else []
        return max(prof_dirs, key=lambda p: p.stat().st_mtime) if prof_dirs else None

    @staticmethod
    def _find_profiler_output_dir(prof_dir: Path) -> Optional[Path]:
        expected = prof_dir / "mindstudio_profiler_output"
        if expected.is_dir():
            return expected
        task_csvs = list(prof_dir.rglob("task_time_*.csv"))
        return task_csvs[0].parent if task_csvs else None

    def _parse_op_summary(
        self, path: Optional[Path]
    ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
        """Index op_summary rows by device/task/stream for joining."""
        if path is None or not path.exists():
            return {}
        try:
            _, rows = self._read_csv(path)
        except Exception:
            return {}

        indexed = {}
        for row in rows:
            key = (row.get("Device_id", ""), row.get("Task ID", ""), row.get("Stream ID", ""))
            indexed[key] = {
                "op_name": row.get("Op Name"),
                "op_type": row.get("OP Type"),
                "task_type": row.get("Task Type"),
                "task_wait_time_us": self._safe_float(row.get("Task Wait Time(us)")),
                "block_num": self._safe_int(row.get("Block Num")),
                "mix_block_num": self._safe_int(row.get("Mix Block Num")),
                "input_shapes": row.get("Input Shapes"),
                "input_data_types": row.get("Input Data Types"),
                "output_shapes": row.get("Output Shapes"),
                "output_data_types": row.get("Output Data Types"),
            }
        return indexed

    def _parse_profile_run(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        prof_dir = self._find_latest_prof_dir(run_dir)
        if prof_dir is None:
            self.last_profile_error = f"No PROF_* directory found under {run_dir}"
            return None

        output_dir = self._find_profiler_output_dir(prof_dir)
        if output_dir is None:
            self.last_profile_error = f"mindstudio_profiler_output not found under {prof_dir}"
            return None

        task_csv = self._latest_file(output_dir, "task_time_*.csv")
        if task_csv is None:
            self.last_profile_error = f"task_time_*.csv not found under {output_dir}"
            return None
        summary_csv = self._latest_file(output_dir, "op_summary_*.csv")

        try:
            _, task_rows = self._read_csv(task_csv)
        except Exception as exc:
            self.last_profile_error = f"Failed to read {task_csv}: {exc}"
            return None

        summary_index = self._parse_op_summary(summary_csv)
        target = self.kernel_name.strip().lower()
        matched = []
        for row in task_rows:
            observed_name = row.get("kernel_name", "").strip()
            if observed_name.lower() != target:
                continue
            duration = self._safe_float(row.get("task_time(us)"))
            if duration is None or duration <= 0:
                continue

            device_id = row.get("Device_id", "")
            task_id = row.get("task_id", "")
            stream_id = row.get("stream_id", "")
            summary = summary_index.get((device_id, task_id, stream_id), {})
            record = {
                "device_id": self._safe_int(device_id),
                "kernel_name": observed_name,
                "kernel_type": row.get("kernel_type"),
                "stream_id": self._safe_int(stream_id),
                "task_id": self._safe_int(task_id),
                "task_time_us": duration,
                "task_start_us": self._safe_float(row.get("task_start(us)")),
                "task_stop_us": self._safe_float(row.get("task_stop(us)")),
                "profile_run_dir": str(run_dir),
                "prof_dir": str(prof_dir),
                "task_time_csv": str(task_csv),
                "op_summary_csv": str(summary_csv) if summary_csv else None,
            }
            record.update(summary)
            matched.append(record)

        if not matched:
            self.last_profile_error = (
                f"No task_time rows matched kernel_name={self.kernel_name!r} in {task_csv}"
            )
            return None

        return {
            "prof_dir": str(prof_dir),
            "output_dir": str(output_dir),
            "task_time_csv": str(task_csv),
            "op_summary_csv": str(summary_csv) if summary_csv else None,
            "task_records": matched,
        }

    # ----------------------- ordinary msprof -------------------------

    def _run_msprof(self, test_file: Path, timeout: int = 300) -> Optional[Dict[str, Any]]:
        """Run the verified ordinary-msprof task-time command, not `msprof op`."""
        self.last_profile_error = None
        result_root = self.performance_dir / self.kernel_name
        result_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evaluation_dir = result_root / f"eval_{stamp}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        all_records: List[Dict[str, Any]] = []
        # One comparable latency sample per independent profiling session.
        # The first matching launch is selected because a test script may
        # contain additional correctness-only calls with different options.
        selected_records: List[Dict[str, Any]] = []
        run_summaries = []
        for index in range(self.profile_repeats):
            run_dir = evaluation_dir / f"repeat_{index + 1:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                self.msprof_executable,
                f"--output={run_dir}",
                "--task-time=l0",
                "--ai-core=off",
                "--ascendcl=off",
                "--runtime-api=off",
                "python3",
                str(test_file.resolve()),
            ]
            print(
                f"[Executor] MSProf run {index + 1}/{self.profile_repeats}:",
                " ".join(command),
            )
            log_file = run_dir / "msprof.log"
            try:
                with log_file.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        text=True, timeout=timeout,
                    )
            except subprocess.TimeoutExpired:
                self.last_profile_error = f"MSProf timed out after {timeout}s; see {log_file}"
                return None
            except Exception as exc:
                self.last_profile_error = f"Failed to launch MSProf: {exc}; see {log_file}"
                return None

            if process.returncode != 0:
                self.last_profile_error = (
                    f"MSProf/test process exited with code {process.returncode}; see {log_file}"
                )
                return None

            parsed = self._parse_profile_run(run_dir)
            if parsed is None:
                return None
            run_records = parsed["task_records"]
            for record in run_records:
                record["profile_repeat"] = index + 1
            all_records.extend(run_records)
            selected_records.append(run_records[0])
            run_summaries.append({
                "repeat": index + 1,
                "prof_dir": parsed["prof_dir"],
                "task_time_csv": parsed["task_time_csv"],
                "op_summary_csv": parsed["op_summary_csv"],
                "matched_task_count": len(run_records),
                "selected_task_id": run_records[0].get("task_id"),
                "selected_task_time_us": run_records[0].get("task_time_us"),
            })

        # Use the first matching target-kernel launch from each independent
        # session as the latency sample. Keep every matching launch in the raw
        # records file for later analysis.
        values = [float(r["task_time_us"]) for r in selected_records]
        if not values:
            self.last_profile_error = f"No valid measurements found for {self.kernel_name}"
            return None

        stats = {
            "count": float(len(values)),
            "mean_us": float(statistics.mean(values)),
            "median_us": float(statistics.median(values)),
            "std_us": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "min_us": float(min(values)),
            "max_us": float(max(values)),
        }

        records_path = evaluation_dir / "kernel_task_records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = {
            "kernel_name": self.kernel_name,
            "profiling_method": "ordinary_msprof_task_time_l0",
            "profile_repeats": self.profile_repeats,
            "execution_time_definition": (
                "median of the first matching target-kernel task from each "
                "independent MSProf session"
            ),
            "measurement_policy": "first_matching_kernel_task_per_session",
            "execution_time_us": stats["median_us"],
            "measurement_stats": stats,
            "measurements_us": values,
            "all_matched_task_times_us": [
                float(record["task_time_us"]) for record in all_records
            ],
            "selected_measurement_count": len(selected_records),
            "matched_task_count": len(all_records),
            "run_summaries": run_summaries,
            "kernel_task_records": str(records_path),
            "evaluation_dir": str(evaluation_dir),
        }
        summary_path = evaluation_dir / "profile_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        print(f"[{ICONS['chart']}] [Executor] MSProf task-time results:")
        print(f"       └─ Selected latency samples: {len(values)}")
        print(f"       └─ All matched kernel tasks retained: {len(all_records)}")
        print(f"       └─ Median: {stats['median_us']:.3f} us")
        print(f"       └─ Mean:   {stats['mean_us']:.3f} us")
        print(f"       └─ Std:    {stats['std_us']:.3f} us")
        print(f"       └─ Min/Max: {stats['min_us']:.3f} / {stats['max_us']:.3f} us")
        print(f"       └─ Raw records: {records_path}")
        print(f"       └─ Summary: {summary_path}")

        return {
            "execution_time_us": stats["median_us"],
            "measurements": values,
            "measurement_stats": stats,
            "hardware_metrics": all_records,
            "profile_dir": str(evaluation_dir),
        }

    # -------------------------- evaluation ---------------------------

    def evaluate(self, code: str, timeout: int = 1200) -> EvaluationResult:
        """Evaluate candidate; correctness assertions run inside test_code_path."""
        print(f"\n[{ICONS['microscope']}] [Executor] Starting evaluation of optimized code...")
        print(f"[{ICONS['gear']}] [Executor] Step 1/3: Preparing test environment...")
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_"))
        profile_data = None

        try:
            kernel_file = temp_dir / f"{self.kernel_name}.py"
            kernel_file.write_text(code, encoding="utf-8")
            test_content = self.test_code_path.read_text(encoding="utf-8")

            # Redirect imports in the test to the candidate module in temp_dir.
            kernel_name_re = re.escape(self.kernel_name)
            module_re = rf"(?:kernel|{kernel_name_re}(?:_\d+)?(?:\.[\w.]+)?)"
            from_pattern = rf"^from\s+{module_re}\s+import"
            import_pattern = rf"^import\s+{module_re}\b(\s+as\s+\w+)?"
            modified_test = re.sub(
                from_pattern, f"from {self.kernel_name} import",
                test_content, flags=re.MULTILINE,
            )
            modified_test = re.sub(
                import_pattern, rf"import {self.kernel_name}\1",
                modified_test, flags=re.MULTILINE,
            )

            test_file = temp_dir / f"test_{self.kernel_name}.py"
            test_file.write_text(modified_test, encoding="utf-8")
            print(f"       └─ Candidate code length: {len(code)} characters")
            print(f"       └─ Temporary candidate: {kernel_file}")
            print(f"       └─ Temporary test: {test_file}")

            print(
                f"[{ICONS['gear']}] [Executor] Step 2/3: "
                "Running ordinary MSProf task-time test..."
            )
            profile_data = self._run_msprof(test_file, timeout=timeout)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if profile_data is None:
            error = self.last_profile_error or "MSProf profiling failed"
            print(f"[{ICONS['cross']}] [Executor] Performance test failed!")
            print(f"       └─ Reason: {error}")
            return EvaluationResult(
                success=False, execution_time=0.0, speedup=0.0,
                fitness=0.0, error=error,
            )

        current_time = float(profile_data["execution_time_us"])
        print(f"[{ICONS['check']}] [Executor] Step 3/3: Calculating speedup...")
        print(f"       └─ Baseline time: {self.baseline_time:.3f} us")
        print(f"       └─ Candidate time (median): {current_time:.3f} us")

        if current_time > 0 and self.baseline_time > 0:
            raw_speedup = self.baseline_time / current_time - 1.0
            speedup = max(raw_speedup, 0.0)
            fitness = min(speedup, 2.0)
            print(f"       └─ Raw speedup: {raw_speedup:.4f}")
            if raw_speedup < 0:
                print(f"[{ICONS['warning']}] Candidate slower than baseline; speedup set to 0")
            elif raw_speedup > 2.0:
                print(f"[{ICONS['trophy']}] Speedup exceeds score cap (2.0)")
            else:
                print(f"[{ICONS['zap']}] Speedup valid")
            print(f"[{ICONS['target']}] [Executor] Evaluation complete!")
            print(f"       └─ Final speedup: {speedup:.4f}")
            print(f"       └─ Fitness score: {fitness:.4f} (max 2.0)")
            print(f"       └─ Competition score: {fitness * 100:.1f}/200 points")
        else:
            print(f"[{ICONS['warning']}] Invalid execution or baseline time; score set to 0")
            speedup = 0.0
            fitness = 0.0

        return EvaluationResult(
            success=True,
            execution_time=current_time,
            speedup=speedup,
            fitness=fitness,
            error=None,
            measurements=profile_data["measurements"],
            measurement_stats=profile_data["measurement_stats"],
            hardware_metrics=profile_data["hardware_metrics"],
            profile_dir=profile_data["profile_dir"],
        )
