#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton Executor Module - Encapsulates Evaluation Interface (msprof version, JSON baseline)
High-Throughput Thread-Safe & Isolated Version
"""

import os
import re
import sys
import json
import subprocess
import csv
import tempfile
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from config import EAConfig


# Icon definitions
ICONS = {
    'rocket': '🚀',
    'timer': '⏱️',
    'check': '✅',
    'cross': '❌',
    'gear': '⚙️',
    'chart': '📊',
    'sparkle': '✨',
    'warning': '⚠️',
    'bulb': '💡',
    'stopwatch': '⏱️',
    'target': '🎯',
    'zap': '⚡',
    'trophy': '🏆',
    'microscope': '🔬',
    'repeat': '🔄',
    'save': '💾'
}


@dataclass
class EvaluationResult:
    """Evaluation result"""
    success: bool
    execution_time: float  # Unit: microseconds (us)
    speedup: float
    fitness: float
    error: Optional[str] = None


class TritonExecutor:
    """
    Triton Code Execution and Performance Evaluator
    """

    def __init__(self, 
                 baseline_time: float,
                 test_code_path: str,
                 config: EAConfig,
                 kernel_name: str = "kernel",
                 work_dir: Optional[Path] = None):
        self.baseline_time = baseline_time  # Use the passed-in baseline directly
        self.test_code_path = Path(test_code_path)
        self.config = config
        self.kernel_name = kernel_name
        self.work_dir = work_dir or Path(".")
        self.performance_dir = self.work_dir / "performance"
        self.performance_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{ICONS['rocket']}] [Executor] Initializing Triton executor...")
        print(f"       └─ Operator name: {kernel_name}")
        print(f"       └─ Working directory: {self.work_dir}")
        print(f"       └─ Baseline time: {baseline_time:.2f}us (read from JSON)")
        print(f"       └─ Test file: {self.test_code_path}")

    def _find_latest_opprof_dir(self, result_dir: Path) -> Optional[Path]:
        """Find the latest OPPROF_* directory"""
        if not result_dir.exists():
            return None

        opprof_dirs = [
            d for d in result_dir.iterdir() 
            if d.is_dir() and d.name.startswith("OPPROF_")
        ]

        if not opprof_dirs:
            return None

        return max(opprof_dirs, key=lambda d: d.stat().st_mtime)

    def _parse_op_basic_info(self, result_dir: Path) -> Optional[float]:
        """兼容新旧 msprof 输出格式，读取 Task Duration(us)"""

        opprof_dir = self._find_latest_opprof_dir(result_dir)

        if not opprof_dir:
            return None

        # 递归查找：
        # 旧格式：OPPROF_xxx/OpBasicInfo.csv
        # 新格式：OPPROF_xxx/device0/kernel/0/OpBasicInfo_xxx.csv
        csv_files = list(opprof_dir.rglob("OpBasicInfo*.csv"))

        if not csv_files:
            return None

        # 取最新生成的 csv
        csv_path = max(
            csv_files,
            key=lambda p: p.stat().st_mtime
        )

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        duration = float(
                            row.get("Task Duration(us)", 0)
                        )

                        if duration > 0:
                            return duration

                    except (ValueError, TypeError):
                        continue

        except Exception:
            pass

        return None

    # 🛠️ 修改点 1：增加 device_id 参数，引入进程级 custom_env 隔离
    def _run_msprof(self, test_file: Path, timeout: int = 300, result_dir: Optional[Path] = None, device_id: int = 0) -> Optional[float]:
        """
        Run msprof op command to get performance data
        """
        if result_dir is None:
            result_dir = self.performance_dir / self.kernel_name
        result_dir.mkdir(parents=True, exist_ok=True)

        test_script_abs = str(test_file.resolve())

        # 用当前解释器跑测试脚本，确保 torch/triton 可用
        # （直接写 `python3` 可能命中系统 python3，没装 torch）
        python_bin = sys.executable or "python3"

        # 构建子进程独立的环境变量，彻底消除线程间对 ASCEND_DEVICE_ID 的锁竞争
        custom_env = os.environ.copy()
        custom_env['ASCEND_DEVICE_ID'] = str(device_id)
        custom_env['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
        custom_env['TRITON_CACHE_DIR'] = f"/tmp/triton_cache_dev_{device_id}"

        # Build msprof command
        cmd_str = (
            f'msprof op '
            f'--output={result_dir} '
            f'--application="{python_bin} {test_script_abs}" '
            f'--kernel-name="{self.kernel_name}" '
            f'--aic-metrics=MemoryDetail,Occupancy,PipeUtilization,Roofline'
        )

        print(cmd_str)

        log_file = result_dir / "get_prof.log"

        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd_str,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    env=custom_env
                )

            if result.returncode != 0:
                print(f"msprof fail")
                return None

            return self._parse_op_basic_info(result_dir)

        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

    def evaluate(self, code: str, timeout: int = 1200, device_id: int = 0) -> EvaluationResult:
        """
        Evaluate a single operator code - [Core Interface]
        """
        print(f"\n[{ICONS['microscope']}] [Executor] Starting evaluation on device {device_id}...")

        # 🛠️ 修改点 2：取消 os.environ['ASCEND_DEVICE_ID'] = str(device_id)，改由 _run_msprof 内局部控制

        # 构建设备隔离的输出目录，避免多线程并发下文件覆盖冲突
        device_result_dir = self.performance_dir / f"{self.kernel_name}_dev_{device_id}" / self.kernel_name
        device_result_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Create temporary environment
        print(f"[{ICONS['gear']}] [Executor] Step 1/3: Preparing test environment...")
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_dev_{device_id}_"))

        # Write code
        kernel_file = temp_dir / f"{self.kernel_name}.py"
        with open(kernel_file, 'w', encoding='utf-8') as f:
            f.write(code)

        # Copy test file and modify import
        with open(self.test_code_path, 'r', encoding='utf-8') as f:
            test_content = f.read()

        # Rewrite import statements in the test file so they point to the
        # generated kernel module (self.kernel_name) regardless of which
        # placeholder form the dataset originally used:
        #
        #   1. the generic placeholder "kernel"          → from kernel import ...
        #   2. the dataset module name "{kernel_name}"   → from {kernel_name} import ...
        #   3. a numbered variant    "{kernel_name}_1"   → from {kernel_name}_1 import ...
        #
        # All variants are rewritten to "from {kernel_name} import".
        # The same logic applies to the "import kernel / import {kernel_name}"
        # form (with optional "as alias" preserved so the test body keeps working).

        kernel_name = re.escape(self.kernel_name)

        # Handle "from X import Y" patterns
        from_pattern = rf'from\s+({kernel_name}|kernel)\s*(?:_\d+)?\s+import'
        modified_test = re.sub(
            from_pattern,
            f'from {self.kernel_name} import',
            test_content,
            flags=re.MULTILINE
        )

        # Handle "import X" patterns (with optional "as alias" preserved)
        import_pattern = rf'import\s+({kernel_name}|kernel)\b((?:\s+as\s+\w+)?)'
        modified_test = re.sub(
            import_pattern,
            lambda m: f'import {self.kernel_name}{m.group(2)}',
            modified_test,
            flags=re.MULTILINE
        )

        test_file = temp_dir / f"test_{self.kernel_name}.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(modified_test)

        print(f"       └─ Code length: {len(code)} characters")

        # Step 2: Run performance test
        print(f"[{ICONS['gear']}] [Executor] Step 2/3: Running performance test on device {device_id}...")
        
        # 🛠️ 修改点 3：将 device_id 正确传入 _run_msprof
        current_time = self._run_msprof(test_file, timeout=timeout, result_dir=device_result_dir, device_id=device_id)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        if current_time is None:
            print(f"[{ICONS['cross']}] [Executor] Performance test failed!")
            return EvaluationResult(
                success=False,
                execution_time=0.0,
                speedup=0.0,
                fitness=0.0,
                error="Performance test failed"
            )

        # Step 3: Calculate speedup
        print(f"[{ICONS['check']}] [Executor] Step 3/3: Calculating speedup...")
        print(f"       └─ Baseline time: {self.baseline_time:.2f}us")
        print(f"       └─ Optimized time: {current_time:.2f}us")

        if current_time > 0 and self.baseline_time > 0:
            raw_speedup = self.baseline_time / current_time - 1
            speedup = max(raw_speedup, 0.0)
            fitness = min(speedup, 2.0)

            print(f"       └─ Raw speedup: {raw_speedup:.4f}")

            if raw_speedup < 0:
                print(f"[{ICONS['warning']}]       └─ Warning: Optimized slower than baseline, speedup set to 0")
            elif raw_speedup > 2.0:
                print(f"[{ICONS['trophy']}]       └─ Speedup exceeds upper limit (2.0), calculated as 2.0")
            else:
                print(f"[{ICONS['zap']}]       └─ Speedup valid, counted in score")

            print(f"[{ICONS['target']}] [Executor] Evaluation complete!")
            print(f"       └─ Final speedup: {speedup:.4f}")
            print(f"       └─ Fitness score: {fitness:.4f} (max 2.0)")

            competition_score = fitness * 100
            print(f"       └─ Competition score: {competition_score:.1f}/200 points")

        else:
            print(f"[{ICONS['warning']}] [Executor] Execution time invalid, score set to 0")
            speedup = 0.0
            fitness = 0.0

        return EvaluationResult(
            success=True,
            execution_time=current_time,
            speedup=speedup,
            fitness=fitness,
            error=None
        )

    def measure_repeat(self, code: str, repeats: int = 5, device_id: int = 0,
                       timeout: int = 600) -> EvaluationResult:
        """串行、单设备重复测量，取中位数作为协议化延迟（阶段 A/C 专用）。

        与搜索过程中的 evaluate 使用完全相同的 msprof 命令与计时口径，
        仅增加重复次数以抑制短 Kernel 的测量抖动。
        """
        samples = []
        last_err = None
        for _ in range(max(1, repeats)):
            res = self.evaluate(code, timeout=timeout, device_id=device_id)
            if res.success and res.execution_time > 0:
                samples.append(res.execution_time)
            else:
                last_err = res.error or "measurement failed"

        if not samples:
            return EvaluationResult(False, 0.0, 0.0, 0.0, error=last_err)

        samples.sort()
        median = samples[len(samples) // 2] if len(samples) % 2 else \
            (samples[len(samples) // 2 - 1] + samples[len(samples) // 2]) / 2

        return EvaluationResult(
            success=True,
            execution_time=median,
            speedup=self.baseline_time / median if median > 0 else 0.0,
            fitness=0.0,
            error=None
        )