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
import time
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
    # 编译阶段单独出参（任务③'）。未启用预编译门时保持 None（未知），
    # 不用 False 冒充，避免把"没测"当成"没过"。
    compile_pass: Optional[bool] = None
    compile_time_ms: Optional[float] = None
    compile_error_type: Optional[str] = None
    compile_error: Optional[str] = None


# ---------------------------------------------------------------------------
# 编译可行性检查（compile-only gate）
# ---------------------------------------------------------------------------
# 目的：在不做正确性校验、不跑 msprof 的前提下，单独触发一次 Triton-Ascend 的
# JIT 编译，从而把"编译失败"与"正确性失败"区分开。
#
# 实现方式（2026-10-17 在真机 triton 3.2.0 / torch_npu 上实测）：
#   * triton.runtime.jit.JITFunction **没有**独立的 compile 方法，只有
#     run / warmup / preload / cache_key / create_binder 等；编译只能通过
#     warmup（内部即 run(..., warmup=True)）或真实 launch 触发。
#   * warmup 会真正走完 NPU 编译并返回 CompiledKernel（带 metadata / asm），
#     但**不会** launch kernel，因此不需要正确性数据、也不产生 msprof 采集开销。
#   * 编译期错误（未定义符号、类型不匹配、非法 constexpr 等）都会以
#     triton.CompilationError 抛出，可被捕获。
#   * 实测耗时：首次编译 ~1.7s，命中 Triton 缓存 ~0s；但子进程冷启动
#     （import torch + torch_npu）约 7.8s，占一次完整评测（~16.5s）的 47%，
#     因此**逐候选单独预编译并不划算**，只在需要区分失败原因或批量编译时使用。
#
# 这里用 sitecustomize 注入的方式把"真实 launch"改写为"只编译"，
# 好处是不需要手写 input_spec —— 直接复用测试文件已有的调用参数。
_COMPILE_ONLY_PATCH = '''
import triton
from triton.runtime.jit import JITFunction

_orig_run = JITFunction.run
_N_ATTEMPT = [0]   # 触发编译的次数
_N_OK = [0]        # 其中真正编译成功的次数（用于区分"编译失败"与"编译成功后脚本报错"）


def _run(self, *args, **kwargs):
    # warmup=True 即"只编译不 launch"，交给原实现
    kwargs["warmup"] = True
    _N_ATTEMPT[0] += 1
    out = _orig_run(self, *args, **kwargs)
    _N_OK[0] += 1      # 只有没抛异常才算编译成功
    return out


JITFunction.run = _run

# 编译阶段 kernel 不会真正执行，输出张量未初始化；
# 关掉正确性断言，避免把"未初始化输出"误报成编译失败。
try:
    import torch
    torch.testing.assert_close = lambda *a, **k: None
except Exception:
    pass

import atexit, json, os, sys
def _dump():
    try:
        p = os.environ.get("_EA_COMPILE_COUNT_OUT")
        if p:
            with open(p, "w") as f:
                json.dump({"n_attempt": _N_ATTEMPT[0], "n_ok": _N_OK[0]}, f)
    except Exception:
        pass
atexit.register(_dump)
'''


# 只有这些才是"编译本身失败"；其余是编译成功之后测试脚本自身的问题，
# 不能记到编译门头上（否则会低估编译门通过率、也分不清拦截的到底是什么）。
COMPILE_ERROR_TYPES = ("CompilationError", "OutOfResources", "SyntaxError",
                       "NameError")

# traceback 里出现这些路径/字样 ⇒ 异常是在 Triton 的编译阶段抛出的。
# 用"是否进入 Triton 内部编译路径"来判定，比逐个枚举错误信息稳得多：
# 实测遇到的编译期错误形态除了显式的 CompilationError，还有
#   * `unsupported AST node type: Lambda`（前端不支持的语法节点）
#   * `AttributeError: module 'triton.language' has no attribute 'inv'`
#     （tl.* 名字不存在，在 jit.py 的 visit_Attribute 里抛）
# 后者按消息文本分类会被误判成 Unknown，但它在 traceback 里带着
# triton/runtime/jit.py —— 编译阶段才会进的代码路径。
_COMPILE_STAGE_MARKERS = (
    "triton/compiler/", "triton/runtime/jit.py", "triton/language/",
    "triton/backends/", "triton/_C", "triton.compiler.errors",
    "unsupported AST node type", "CompilationError",
)


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

        # 【修复】旧实现三种失败一律返回 None，上层只能记一句
        # "Performance test failed"，于是"10/120 候选失败"无法定位原因
        # （是超时？算错？还是设备异常？）。这里把失败类型显式分出来。
        self._last_failure = None

        def _classify(log_path):
            try:
                txt = log_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return "perf_test_failed", ""
            low = txt.lower()
            for kw, kind in (("assertionerror", "accuracy_failed"),
                             ("not allclose", "accuracy_failed"),
                             ("allclose", "accuracy_failed"),
                             ("mismatch", "accuracy_failed"),
                             ("valueerror: shape", "accuracy_failed"),
                             ("out of bounds", "accuracy_failed"),
                             ("timeout", "timeout"),
                             ("compilationerror", "compile_error"),
                             ("triton.compiler", "compile_error"),
                             ("outofresources", "resource_exhausted"),
                             ("acl", "device_error")):
                if kw in low:
                    for line in txt.splitlines():
                        if kw in line.lower():
                            return kind, line.strip()[:200]
                    return kind, ""
            return "perf_test_failed", (txt.strip().splitlines() or [""])[-1][:200]

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
                print(f"msprof fail (rc={result.returncode})")
                kind, detail = _classify(log_file)
                self._last_failure = (kind, f"rc={result.returncode} {detail}".strip())
                return None

            return self._parse_op_basic_info(result_dir)

        except subprocess.TimeoutExpired:
            self._last_failure = ("timeout", f">{timeout}s")
            return None
        except Exception as e:
            self._last_failure = ("executor_error", f"{type(e).__name__}: {str(e)[:150]}")
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
        temp_dir, test_file = self._prepare_test_dir(code, device_id)
        print(f"       └─ Code length: {len(code)} characters")

        # Step 1.5（可选，默认关闭）：编译可行性门。
        # 开启后"编译失败"与"正确性/测量失败"才真正分离；关闭时 compile_pass 为 None
        # （未知），不用 False 冒充，以免低估编译门价值。
        cinfo = None
        if getattr(self.config, "pre_compile_check", False):
            cinfo = self.compile_check(code, timeout=timeout, device_id=device_id)
            if not cinfo["compile_pass"]:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"[{ICONS['cross']}] [Executor] 编译门拒绝："
                      f"{cinfo['error_type']}: {str(cinfo['error_message'])[:200]}")
                return EvaluationResult(
                    success=False, execution_time=0.0, speedup=0.0, fitness=0.0,
                    error=f"compile failed: {cinfo['error_type']}",
                    compile_pass=False,
                    compile_time_ms=cinfo["compile_time_ms"],
                    compile_error_type=cinfo["error_type"],
                    compile_error=cinfo["error_message"],
                )

        # Step 2: Run performance test
        print(f"[{ICONS['gear']}] [Executor] Step 2/3: Running performance test on device {device_id}...")
        
        # 🛠️ 修改点 3：将 device_id 正确传入 _run_msprof
        current_time = self._run_msprof(test_file, timeout=timeout, result_dir=device_result_dir, device_id=device_id)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        if current_time is None:
            kind, detail = getattr(self, "_last_failure", None) or (None, None)
            print(f"[{ICONS['cross']}] [Executor] Performance test failed!"
                  + (f" [{kind}] {detail}" if kind else ""))
            return EvaluationResult(
                success=False,
                execution_time=0.0,
                speedup=0.0,
                fitness=0.0,
                error=f"perf_test_failed:{kind or 'unknown'}"
                      + (f":{detail}" if detail else ""),
                # 走到这里说明编译门已通过（或没开），失败发生在正确性/测量阶段
                compile_pass=(True if cinfo else None),
                compile_time_ms=(cinfo or {}).get("compile_time_ms"),
                compile_error_type=(cinfo or {}).get("error_type"),
                compile_error=(cinfo or {}).get("error_message"),
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
            error=None,
            compile_pass=(True if cinfo else None),
            compile_time_ms=(cinfo or {}).get("compile_time_ms"),
            compile_error_type=(cinfo or {}).get("error_type"),
            compile_error=(cinfo or {}).get("error_message"),
        )

    # ------------------------------------------------------------------ #
    # 测试环境准备（evaluate 与 compile_check 共用）
    # ------------------------------------------------------------------ #
    def _prepare_test_dir(self, code: str, device_id: int = 0):
        """把待测内核与改写后的测试文件写进一个临时目录，返回 (temp_dir, test_file)。

        原来这段逻辑内联在 evaluate 里，compile_check 需要完全相同的产物
        （同样的 import 改写、同样的 torch_npu 注册补丁），否则"编译门通过但
        正式评测报 ImportError"这类不一致会非常难排查，因此抽成公共方法。
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_dev_{device_id}_"))

        # Write code
        kernel_file = temp_dir / f"{self.kernel_name}.py"
        with open(kernel_file, 'w', encoding='utf-8') as f:
            f.write(code)

        with open(self.test_code_path, 'r', encoding='utf-8') as f:
            test_content = f.read()

        # 把测试文件里的 import 统一改写到生成的内核模块名，兼容三种占位形式：
        #   1. 通用占位 "kernel"             → from kernel import ...
        #   2. 数据集模块名 "{kernel_name}"   → from {kernel_name} import ...
        #   3. 带编号变体 "{kernel_name}_1"   → from {kernel_name}_1 import ...
        kernel_name = re.escape(self.kernel_name)
        from_pattern = rf'from\s+({kernel_name}|kernel)\s*(?:_\d+)?\s+import'
        modified_test = re.sub(from_pattern, f'from {self.kernel_name} import',
                               test_content, flags=re.MULTILINE)
        import_pattern = rf'import\s+({kernel_name}|kernel)\b((?:\s+as\s+\w+)?)'
        modified_test = re.sub(
            import_pattern,
            lambda m: f'import {self.kernel_name}{m.group(2)}',
            modified_test, flags=re.MULTILINE)

        # 【修复】子进程不会自动注册 NPU 后端，缺这一行 torch.device("npu") 会抛
        # RuntimeError: Expected one of cpu, cuda, ... at start of device string: npu
        if not re.search(r'^\s*import\s+torch_npu\b', modified_test, flags=re.MULTILINE):
            modified_test = ("import torch_npu  # noqa: F401  确保子进程注册 NPU 后端\n"
                             + modified_test)

        test_file = temp_dir / f"test_{self.kernel_name}.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(modified_test)
        return temp_dir, test_file

    # ------------------------------------------------------------------ #
    # 编译可行性检查
    # ------------------------------------------------------------------ #
    def compile_check(self, code: str, timeout: int = 600, device_id: int = 0) -> dict:
        """只做 Triton-Ascend JIT 编译，不做正确性校验、不跑 msprof。

        返回：
            {
              "compile_pass": bool,
              "compile_time_ms": float,
              "error_type": str | None,     # 见 COMPILE_ERROR_TYPES / PostCompile* / NoKernelLaunch
              "error_message": str | None,
              "n_attempt": int,             # 触发编译的 launch 次数
              "n_ok": int,                  # 其中编译成功的次数
            }

        判定规则（针对评审"必须确认至少触发一次预期 JIT 编译、且要区分编译错误与
        编译后脚本异常"两条意见）：
          1. n_attempt == 0            → **未知**：没观察到 JIT 编译，既不能算通过
                                         也不能算失败（NoKernelLaunch 只是"没测到"）
          2. n_ok < n_attempt          → **确认失败**：有 launch 在编译阶段抛错
          3. 全部编译成功但 rc != 0    → **确认成功**：编译没问题，是测试脚本后续出错
                                         compile_pass=True，error_type 记 PostCompile:<类别>
          4. 全部编译成功且 rc == 0    → **确认成功**

        结果里除 compile_pass 外还有 `status` 字段，取值 pass / fail / unknown：
        "未触发编译""批量子进程异常""超时"都归 unknown，由调用方按预置规则处理，
        绝不能当成已确认的编译失败直接过滤掉。

        实现要点与实测结论见文件头 `_COMPILE_ONLY_PATCH` 的说明：
        JITFunction 没有独立的 compile 方法，只能通过 warmup（即 run(warmup=True)）
        触发；warmup 会真正产出 CompiledKernel 但不 launch kernel。
        """
        import sys as _sys
        temp_dir, test_file = self._prepare_test_dir(code, device_id)
        patch_dir = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_compilegate_"))
        count_out = patch_dir / "n_launch.txt"
        try:
            (patch_dir / "sitecustomize.py").write_text(_COMPILE_ONLY_PATCH,
                                                        encoding="utf-8")
            env = os.environ.copy()
            env['ASCEND_DEVICE_ID'] = str(device_id)
            env['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
            env['TRITON_CACHE_DIR'] = f"/tmp/triton_cache_dev_{device_id}"
            env['PYTHONPATH'] = f"{patch_dir}{os.pathsep}{temp_dir}" + \
                                (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else "")
            env['_EA_COMPILE_COUNT_OUT'] = str(count_out)

            t0 = time.time()
            try:
                proc = subprocess.run([_sys.executable, str(test_file)],
                                      capture_output=True, text=True,
                                      timeout=timeout, env=env, cwd=str(temp_dir))
                rc, err = proc.returncode, (proc.stderr or "")
            except subprocess.TimeoutExpired:
                rc, err = -9, f"TimeoutExpired after {timeout}s"
            elapsed_ms = (time.time() - t0) * 1000.0

            n_attempt = n_ok = 0
            try:
                _c = json.loads(count_out.read_text().strip())
                n_attempt, n_ok = int(_c.get("n_attempt", 0)), int(_c.get("n_ok", 0))
            except Exception:
                n_attempt = n_ok = 0

            verdict = self._judge_compile_result(n_attempt, n_ok, rc, err)
            verdict["compile_time_ms"] = round(elapsed_ms, 1)
            return verdict
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(patch_dir, ignore_errors=True)

    # 批量编译驱动：在**同一个子进程**里连续编译多个候选，把 python+torch_npu
    # 的冷启动（实测 ~7.8s）摊薄到整批上。逐候选单独起进程时，冷启动占 9.8s 里的
    # 绝大部分，这正是"先编译 40 个再评测 20 个反而更慢"的根因。
    _BATCH_DRIVER = r'''
import json, os, runpy, sys, time, traceback

out_path, kname, cands = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
try:
    import sitecustomize as _sc
except Exception:
    _sc = sys.modules.get("sitecustomize")

res = []
for c in cands:
    d, idx = c["dir"], c["i"]
    sys.path.insert(0, d)
    try:
        os.chdir(d)
        sys.modules.pop(kname, None)
        if _sc is not None:
            _sc._N_ATTEMPT[0] = 0
            _sc._N_OK[0] = 0
        err = None
        t0 = time.time()
        try:
            runpy.run_path(os.path.join(d, "test_%s.py" % kname), run_name="__main__")
        except BaseException as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-2000:]
        el = (time.time() - t0) * 1000.0
        res.append({
            "index": idx,
            "n_attempt": int(_sc._N_ATTEMPT[0]) if _sc else 0,
            "n_ok": int(_sc._N_OK[0]) if _sc else 0,
            "error": err,
            "compile_time_ms": round(el, 1),
        })
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass

with open(out_path, "w") as f:
    json.dump(res, f)
'''

    def compile_check_batch(self, codes, timeout: int = 1800, device_id: int = 0):
        """一个进程编译整批候选，返回与 compile_check 同 schema 的结果列表。

        与逐个调用 compile_check 的区别：冷启动只付一次。
        实测参考：单个候选 ~9.8s（其中冷启动 ~7.8s）；批量 N 个 ≈ 7.8 + N × 2.0s。
        """
        import sys as _sys
        import json as _json
        if not codes:
            return []

        tmp_root = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_cgbatch_"))
        patch_dir = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_cgpatch_"))
        try:
            # 测试文件的改写与候选内容无关，用第一个候选生成一次即可
            one_dir, one_test = self._prepare_test_dir(codes[0], device_id)
            test_src = Path(one_test).read_text(encoding="utf-8")
            shutil.rmtree(one_dir, ignore_errors=True)

            (patch_dir / "sitecustomize.py").write_text(_COMPILE_ONLY_PATCH,
                                                        encoding="utf-8")
            cands = []
            for i, code in enumerate(codes):
                d = tmp_root / f"c{i}"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{self.kernel_name}.py").write_text(code, encoding="utf-8")
                (d / f"test_{self.kernel_name}.py").write_text(test_src,
                                                               encoding="utf-8")
                cands.append({"i": i, "dir": str(d)})

            driver = tmp_root / "_driver.py"
            driver.write_text(self._BATCH_DRIVER, encoding="utf-8")
            out_json = tmp_root / "_out.json"

            env = os.environ.copy()
            env['ASCEND_DEVICE_ID'] = str(device_id)
            env['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
            env['TRITON_CACHE_DIR'] = f"/tmp/triton_cache_dev_{device_id}"
            env['PYTHONPATH'] = f"{patch_dir}" + \
                (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else "")

            t0 = time.time()
            try:
                subprocess.run([_sys.executable, str(driver), str(out_json),
                                self.kernel_name, _json.dumps(cands)],
                               capture_output=True, text=True, timeout=timeout,
                               env=env, cwd=str(tmp_root))
            except subprocess.TimeoutExpired:
                pass
            batch_ms = round((time.time() - t0) * 1000.0, 1)

            try:
                raw = _json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                raw = []

            by_idx = {r.get("index"): r for r in raw}
            out = []
            for i in range(len(codes)):
                r = by_idx.get(i)
                if r is None:
                    # 批量子进程异常/超时 → 该候选**未被确认**不能编译，记未知。
                    # 旧实现把它记成 False 直接过滤，等于把"没测"当成"没过"。
                    out.append({"compile_pass": None, "status": "unknown",
                                "compile_time_ms": 0.0,
                                "error_type": "BatchAborted",
                                "error_message": ("批量编译子进程未产出该候选的结果"
                                                  "（异常或超时），编译可行性未知。"),
                                "n_attempt": 0, "n_ok": 0})
                    continue
                n_att, n_ok = int(r.get("n_attempt", 0)), int(r.get("n_ok", 0))
                err = r.get("error")
                v = self._judge_compile_result(n_att, n_ok, 0 if not err else 1, err)
                v["compile_time_ms"] = r.get("compile_time_ms", 0.0)
                out.append(v)
            for r in out:
                r["batch_wall_ms"] = batch_ms
            return out
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
            shutil.rmtree(patch_dir, ignore_errors=True)

    @classmethod
    def _judge_compile_result(cls, n_attempt, n_ok, rc, err):
        """把 (编译尝试数, 编译成功数, 子进程退出码, stderr) 判成一条编译结论。

        抽成纯函数是为了能离线单测 —— 这三条规则正是评审要求补上的：
          1. n_attempt == 0  → 没触发 JIT 编译，不能算"编译通过"（防假阳性）
          2. n_ok < n_attempt → 有 launch 在编译阶段抛错 → 真编译失败
          3. 全编译成功但 rc != 0 → 编译没问题，是测试脚本后续出错 → 记 PostCompile
        """
        # 未知：一次 JIT 编译都没观察到。这**不等于**编译失败，只是没测到。
        # 超时同样是"没测出来"，不是"测出来不行"
        if rc == -9:
            return {"compile_pass": None, "status": "unknown",
                    "error_type": "Timeout",
                    "error_message": (err or "compile check timeout").strip()[-2000:],
                    "n_attempt": n_attempt, "n_ok": n_ok}
        if n_attempt == 0:
            return {"compile_pass": None, "status": "unknown",
                    "error_type": "NoKernelLaunch",
                    "error_message": ("测试脚本未触发任何 Triton JIT 编译；"
                                      "无法确认编译可行性，记为未知而非失败。"),
                    "n_attempt": 0, "n_ok": 0}
        if n_ok < n_attempt:
            return {"compile_pass": False, "status": "fail",
                    "error_type": cls._classify_compile_error(err or ""),
                    "error_message": (err or "").strip()[-2000:],
                    "n_attempt": n_attempt, "n_ok": n_ok}
        etype = None
        if rc != 0:
            etype = "PostCompile:" + cls._classify_compile_error(err or "")
        return {"compile_pass": True, "status": "pass", "error_type": etype,
                "error_message": ((err or "").strip()[-2000:] if rc != 0 else None),
                "n_attempt": n_attempt, "n_ok": n_ok}

    @staticmethod
    def _classify_compile_error(stderr: str) -> str:
        """把子进程 stderr 归类成稳定的错误类型，便于统计"编译门拦住的是什么"。

        注意：编译阶段 kernel 不执行，输出张量未初始化，测试文件里基于输出值的
        断言可能以 AssertionError / RuntimeError 形式抛出。这类**不是**编译错误，
        归类为 NonCompileError，避免把编译门的成功率算低。
        """
        s = stderr or ""
        if "TimeoutExpired" in s:
            return "Timeout"
        # 先按"是否进入 Triton 编译路径"判定（覆盖 CompilationError 及各类
        # 前端/后端 lowering 期异常），再退到具体消息文本
        for marker in _COMPILE_STAGE_MARKERS:
            if marker in s:
                return "CompilationError"
        if "OutOfResources" in s or "out of resource" in s.lower():
            return "OutOfResources"
        if "SyntaxError" in s or "IndentationError" in s:
            return "SyntaxError"
        if "NameError" in s:
            return "NameError"
        if "AssertionError" in s:
            return "AssertionError(non-compile)"
        if "ImportError" in s or "ModuleNotFoundError" in s:
            return "ImportError"
        if "RuntimeError" in s:
            return "RuntimeError"
        return "Unknown"

    def measure_repeat(self, code: str, repeats: int = 5, device_id: int = 0,
                       timeout: int = 600) -> EvaluationResult:
        """串行、单设备重复测量，取中位数作为协议化延迟（阶段 A/C 专用）。

        与搜索过程中的 evaluate 使用完全相同的 msprof 命令与计时口径，
        仅增加重复次数以抑制短 Kernel 的测量抖动。
        """
        samples = []
        last_err = None
        consecutive_failures = 0
        for _ in range(max(1, repeats)):
            res = self.evaluate(code, timeout=timeout, device_id=device_id)
            if res.success and res.execution_time > 0:
                samples.append(res.execution_time)
                consecutive_failures = 0
            else:
                last_err = res.error or "measurement failed"
                consecutive_failures += 1
                # 编译/正确性失败是确定性的，重复测量不会改变结果，
                # 连续失败即提前退出，避免对不可用种子浪费全部 repeats 次评测。
                if consecutive_failures >= 2:
                    break

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