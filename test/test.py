#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Operator Tester
=========================

从 executor.py 中抽离出来的 msprof 测试逻辑，独立运行。
把要测的算子代码整个粘贴到 test/test.txt 里（纯代码，不带任何声明），
然后通过命令行 `--kernel <算子名>` 指定算子名，本脚本会：

  1. 从 test/test.txt 读出算子代码（纯代码，不带任何声明行）；
  2. 用命令行 `--kernel` 指定的算子名，把代码写到临时目录下的 `{算子名}.py`；
  3. 在 ../datasets/{算子名}/ 下找 test_{算子名}_1.py（找不到再试 _2/_3/无编号），
     把里面 `from {算子名} import` / `import {算子名}` 等导入改写到临时目录的
     `{算子名}.py`（与 executor.py 的 evaluate() 逻辑一致）；
  4. 用 msprof op 跑改写后的测试文件，解析 Task Duration(us) 并打印。

用法:
    cd /workspace/Agent/test
    python test.py --kernel _act_quant_kernel    # 必填：指定算子名
    python test.py --kernel softplus --device 1  # 指定 NPU 设备号
    python test.py --kernel softplus --timeout 600  # 单算子超时 600 秒
    python test.py --kernel _act_quant_kernel --txt /path/to/code.py  # 用别的代码文件

test/test.txt 格式:
    纯算子代码，直接粘贴即可（不需要 # kernel 声明行）。
    算子名由命令行 --kernel 提供。
"""

import os
import re
import sys
import csv
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict


# Icon definitions (与 executor.py 保持一致)
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
    'target': '🎯',
    'zap': '⚡',
    'trophy': '🏆',
    'microscope': '🔬',
}


class OperatorTester:
    """
    独立算子测试器：封装从 executor.py 抽离的 msprof 调用、import 重写与结果解析逻辑。

    与 executor.py 中 TritonExecutor 的区别：
      - 不依赖 EAConfig / baseline_time / speedup 计算
      - 待测代码来自 test/test.txt，测试文件来自 datasets/{kernel_name}/
      - import 重写、临时目录、msprof 调用与 executor.evaluate() 保持一致
    """

    def __init__(self,
                 work_dir: Optional[Path] = None,
                 datasets_dir: Optional[Path] = None,
                 device_id: int = 0,
                 timeout: int = 300):
        """
        Args:
            work_dir: 工作目录，msprof 性能数据会输出到其下的 performance/
            datasets_dir: 算子数据集根目录（默认 ../datasets）
            device_id: NPU 设备号
            timeout: msprof 单次执行超时（秒）
        """
        self.work_dir = work_dir or Path(__file__).resolve().parent
        self.datasets_dir = datasets_dir or (self.work_dir.parent / "datasets")
        self.device_id = device_id
        self.timeout = timeout

        self.performance_dir = self.work_dir / "performance"
        self.performance_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{ICONS['rocket']}] [Tester] 初始化算子测试器...")
        print(f"       └─ 工作目录: {self.work_dir}")
        print(f"       └─ 数据集目录: {self.datasets_dir}")
        print(f"       └─ NPU 设备: {self.device_id}")
        print(f"       └─ 超时: {self.timeout}s")

    # ------------------------------------------------------------------
    # 测试文件查找（逻辑参考 main.py 的 find_test_file）
    # ------------------------------------------------------------------
    def find_test_file(self, kernel_name: str) -> Path:
        """
        在 datasets/{kernel_name}/ 下按优先级查找测试文件:
          1. test_{kernel_name}_1.py
          2. test_{kernel_name}_2.py
          3. test_{kernel_name}_3.py
          4. test_{kernel_name}.py
        """
        kernel_dir = self.datasets_dir / kernel_name
        if not kernel_dir.exists():
            raise FileNotFoundError(
                f"算子目录不存在: {kernel_dir}"
            )

        for test_case_id in range(1, 4):
            test_file = kernel_dir / f"test_{kernel_name}_{test_case_id}.py"
            if test_file.exists():
                return test_file

        test_file = kernel_dir / f"test_{kernel_name}.py"
        if test_file.exists():
            return test_file

        raise FileNotFoundError(
            f"在 {kernel_dir} 下找不到测试文件 "
            f"(test_{kernel_name}_1.py / test_{kernel_name}_2.py / "
            f"test_{kernel_name}_3.py / test_{kernel_name}.py)"
        )

    # ------------------------------------------------------------------
    # import 重写（从 executor.py.evaluate() 抽离，逻辑保持一致）
    # ------------------------------------------------------------------
    @staticmethod
    def _rewrite_test_imports(test_content: str, kernel_name: str) -> str:
        """
        把测试文件里指向占位模块的 import 改写到 {kernel_name}：
          from kernel import ...        -> from {kernel_name} import ...
          from {kernel_name} import ... -> from {kernel_name} import ...
          from {kernel_name}_1 import ..-> from {kernel_name} import ...
          import kernel [as alias]      -> import {kernel_name} [as alias]
          import {kernel_name} [as alias] -> import {kernel_name} [as alias]
        """
        kn = re.escape(kernel_name)

        # Handle "from X import Y" patterns
        from_pattern = rf'from\s+({kn}|kernel)\s*(?:_\d+)?\s+import'
        modified_test = re.sub(
            from_pattern,
            f'from {kernel_name} import',
            test_content,
            flags=re.MULTILINE
        )

        # Handle "import X" patterns (with optional "as alias" preserved)
        import_pattern = rf'import\s+({kn}|kernel)\b((?:\s+as\s+\w+)?)'
        modified_test = re.sub(
            import_pattern,
            lambda m: f'import {kernel_name}{m.group(2)}',
            modified_test,
            flags=re.MULTILINE
        )

        return modified_test

    # ------------------------------------------------------------------
    # msprof 相关（从 executor.py 抽离，保持核心逻辑一致）
    # ------------------------------------------------------------------
    def _find_latest_opprof_dir(self, result_dir: Path) -> Optional[Path]:
        """查找最新的 OPPROF_* 目录"""
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

    def _run_msprof(self,
                    test_file: Path,
                    kernel_name: str,
                    result_dir: Optional[Path] = None) -> Optional[float]:
        """
        运行 msprof op 命令获取性能数据
        （从 executor.py._run_msprof 抽离，使用实例配置的 device_id 与 timeout）
        """
        if result_dir is None:
            result_dir = self.performance_dir / kernel_name
        result_dir.mkdir(parents=True, exist_ok=True)

        test_script_abs = str(test_file.resolve())

        # 用当前解释器跑测试脚本，确保 torch/triton 可用
        # （直接写 `python3` 可能命中系统 python3，没装 torch）
        python_bin = sys.executable or "python3"

        # 构建子进程独立的环境变量，避免多进程下 ASCEND_DEVICE_ID 冲突
        custom_env = os.environ.copy()
        custom_env['ASCEND_DEVICE_ID'] = str(self.device_id)
        custom_env['ASCEND_RT_VISIBLE_DEVICES'] = str(self.device_id)
        custom_env['TRITON_CACHE_DIR'] = f"/tmp/triton_cache_dev_{self.device_id}"

        # Build msprof command
        cmd_str = (
            f'msprof op '
            f'--output={result_dir} '
            f'--application="{python_bin} {test_script_abs}" '
            f'--kernel-name="{kernel_name}" '
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
                    timeout=self.timeout,
                    env=custom_env
                )

            if result.returncode != 0:
                print(f"msprof fail (returncode={result.returncode})")
                self._peek_log_errors(log_file)
                return None

            duration = self._parse_op_basic_info(result_dir)
            if duration is None:
                # msprof 退出码为 0 但没解析到 Task Duration，
                # 多半是测试脚本自身抛异常（如 torch 未装、kernel 跑挂）。
                print("msprof 未返回有效 Task Duration，日志关键行：")
                self._peek_log_errors(log_file)
                print(f"(完整日志: {log_file})")
            return duration

        except subprocess.TimeoutExpired:
            print(f"msprof timeout ({self.timeout}s)")
            self._peek_log_errors(log_file)
            return None
        except Exception as e:
            print(f"msprof exception: {e}")
            return None

    @staticmethod
    def _peek_log_errors(log_file: Path, max_lines: int = 20) -> None:
        """打印 msprof 日志里看起来像错误/异常的前若干行，方便定位根因。"""
        if not log_file.exists():
            return
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.read().splitlines()
        except Exception:
            return

        keywords = ("Traceback", "Error", "ERROR", "Exception",
                    "ModuleNotFound", "ImportError", "failed", "WARN")
        hits = [ln for ln in lines if any(k in ln for k in keywords)]
        if not hits:
            hits = lines[-max_lines:]
        else:
            hits = hits[:max_lines]

        print(f"--- 日志关键行 ({log_file.name}) ---")
        for ln in hits:
            print(f"    {ln}")
        print("---")

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def test_code(self, code: str, kernel_name: str) -> Dict:
        """
        测试一段算子代码：
          1. 写到临时目录下的 {kernel_name}.py
          2. 找到对应的测试文件并重写 import
          3. 用 msprof 跑测试文件
          4. 返回 Task Duration(us)
        """
        print(f"\n{'=' * 60}")
        print(f"[{ICONS['microscope']}] [Tester] 测试算子: {kernel_name}")
        print(f"{'=' * 60}")

        result = {
            "kernel_name": kernel_name,
            "test_file": None,
            "success": False,
            "task_duration_us": None,
            "error": None,
        }

        # Step 1: 准备临时环境，写 kernel 代码 + 改写后的测试文件
        print(f"[{ICONS['gear']}] Step 1/3: 准备测试环境...")
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{kernel_name}_test_"))

        try:
            kernel_file = temp_dir / f"{kernel_name}.py"
            with open(kernel_file, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"       └─ 算子代码 -> {kernel_file} ({len(code)} 字符)")

            # 找测试文件
            try:
                test_file = self.find_test_file(kernel_name)
                result["test_file"] = str(test_file)
            except FileNotFoundError as e:
                result["error"] = str(e)
                print(f"[{ICONS['cross']}]       └─ {e}")
                return result

            with open(test_file, 'r', encoding='utf-8') as f:
                test_content = f.read()

            # 重写 import（与 executor.py.evaluate() 一致）
            modified_test = self._rewrite_test_imports(test_content, kernel_name)

            test_file_tmp = temp_dir / f"test_{kernel_name}.py"
            with open(test_file_tmp, 'w', encoding='utf-8') as f:
                f.write(modified_test)
            print(f"       └─ 测试文件 -> {test_file_tmp}")
            print(f"       └─ (源: {test_file})")

            # Step 2: 运行 msprof
            print(f"[{ICONS['gear']}] Step 2/3: 运行 msprof (device={self.device_id})...")
            duration = self._run_msprof(test_file_tmp, kernel_name)

            if duration is None:
                result["error"] = "msprof 未返回有效 Task Duration"
                print(f"[{ICONS['cross']}] Step 2: 性能测试失败")
                return result

            # Step 3: 汇报
            print(f"[{ICONS['check']}] Step 3/3: Task Duration = {duration:.2f} us")
            result["success"] = True
            result["task_duration_us"] = duration
            return result

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# test.txt 读取
# ----------------------------------------------------------------------
def read_code(txt_path: Path) -> str:
    """读取 test.txt 的纯算子代码内容（不做任何解析，算子名由命令行 --kernel 提供）。"""
    if not txt_path.exists():
        raise FileNotFoundError(f"代码文件不存在: {txt_path}")

    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


# ----------------------------------------------------------------------
# 报告打印
# ----------------------------------------------------------------------
def print_report(result: Dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{ICONS['chart']}] 测试结果")
    print(f"{'=' * 60}")
    print(f"算子名     : {result['kernel_name']}")
    print(f"测试文件   : {result['test_file'] or 'N/A'}")
    if result['success']:
        print(f"状态       : {ICONS['check']} 成功")
        print(f"Task Duration : {result['task_duration_us']:.2f} us")
    else:
        print(f"状态       : {ICONS['cross']} 失败")
        print(f"错误       : {result['error'] or '未知'}")


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="读取 test/test.txt 里的算子代码，用 msprof 跑对应测试（算子名由 --kernel 指定）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test.py --kernel _act_quant_kernel   # 必填：指定算子名
  python test.py --kernel softplus --device 1
  python test.py --kernel softplus --timeout 600
  python test.py --kernel softplus --txt /path/to/code.py

test/test.txt 格式:
  纯算子代码，直接粘贴即可（不需要 # kernel 声明行）。
        """
    )

    parser.add_argument(
        "--txt", type=str, default=None,
        help="算子代码文件路径（默认为脚本同目录下的 test.txt）"
    )
    parser.add_argument(
        "--datasets-dir", type=str, default=None,
        help="算子数据集根目录（默认为 ../datasets）"
    )
    parser.add_argument(
        "--kernel", type=str, required=True,
        help="算子名（必填），脚本会到 ../datasets/{算子名}/ 下找 test_{算子名}_1.py"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="NPU 设备号（默认 0）"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="msprof 执行超时（秒，默认 300）"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    here = Path(__file__).resolve().parent

    # 解析路径
    txt_path = Path(args.txt) if args.txt else (here / "test.txt")
    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else (here.parent / "datasets")

    # 读取算子代码（纯代码，算子名来自 --kernel）
    try:
        code = read_code(txt_path)
    except FileNotFoundError as e:
        print(f"[{ICONS['cross']}] {e}")
        return

    kernel_name = args.kernel

    print(f"[{ICONS['bulb']}] 代码来源: {txt_path}")
    print(f"[{ICONS['bulb']}] 算子名  : {kernel_name}")
    print(f"[{ICONS['bulb']}] 代码长度: {len(code)} 字符")

    # 初始化测试器并执行
    tester = OperatorTester(
        work_dir=here,
        datasets_dir=datasets_dir,
        device_id=args.device,
        timeout=args.timeout,
    )

    result = tester.test_code(code, kernel_name)
    print_report(result)


if __name__ == "__main__":
    main()
