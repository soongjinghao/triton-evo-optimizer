#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编译可行性门（compile gate）基准测量。

回答优化方案「任务③」必须先回答的三个问题：

  Q1. Triton-Ascend 是否真有一条"只编译、不做正确性、不跑 msprof"的路径？
      → 是：JITFunction.warmup（内部 run(warmup=True)）。本脚本直接调用
        executor.compile_check()，它就是这条路径的封装。
  Q2. 编译门的拦截率有多高？拦住的都是什么错误？
      → 报 compile_pass 比例与 error_type 分布。若拦截率极低，说明编译门
        不值得常开（实测参考：整体失败率 0.9%）。
  Q3. 编译本身花多久？首次 vs 命中 Triton 缓存差多少？
      → 报 compile_time_ms 的分位数，并与一次完整评测的耗时对照。

用法（需要 NPU）:
    python experiments/benchmark_compile_gate.py --kernels _act_quant_kernel --limit 20
    python experiments/benchmark_compile_gate.py --kernels K1 K2 K3 --limit 20 --repeat 2

--limit 控制每个算子的候选数；--repeat 2 用于测"首次编译 vs 缓存命中"的差值
（第二次跑同一批代码会命中 TRITON_CACHE_DIR）。
"""

import argparse
import glob
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def find_candidates_in_codes(codes_dir, kernel, limit):
    """从历史候选代码库（experiments/dataset/codes/*.py）里筛出该算子的候选。

    datasets/ 每个算子只有 2 个文件，凑不出"20 个真实候选"的批量编译基准；
    而 codes/ 里存着上万份历史生成的候选代码（按 candidate_id 命名）。
    这里按"代码里出现该算子函数名"来筛，能拿到足够多的**真实**候选。
    """
    out = []
    for p in sorted(glob.glob(str(Path(codes_dir) / "*.py"))):
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if kernel in txt:
            out.append(p)
        if limit and len(out) >= limit:
            break
    return out


def find_candidates(dataset_dir, kernel, limit):
    """从数据集目录里挑出该算子的候选代码文件。"""
    base = Path(dataset_dir)
    pats = [str(base / f"**/{kernel}*.py"), str(base / f"**/{kernel}/**/*.py")]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    # 排除测试文件本身，只留内核实现
    files = [f for f in files if not os.path.basename(f).startswith("test")]
    seen, uniq = set(), []
    for f in sorted(files):
        rp = os.path.realpath(f)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(f)
    return uniq[:limit] if limit else uniq


def find_test_file(dataset_dir, kernel):
    cands = glob.glob(str(Path(dataset_dir) / "**" / f"test*{kernel}*.py"), recursive=True)
    cands += glob.glob(str(Path(dataset_dir) / kernel / f"test*{kernel}*.py"), recursive=True)
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser(description="编译可行性门基准（需要 NPU）")
    ap.add_argument("--kernels", nargs="+", required=True)
    ap.add_argument("--dataset-dir", default=str(ROOT / "datasets"))
    ap.add_argument("--codes-dir", default=None,
                    help="历史候选代码库目录；指定后从里面按算子名筛候选（可凑够 20 个）")
    ap.add_argument("--limit", type=int, default=20, help="每个算子的候选数")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每个候选重复编译次数（2 可测缓存命中收益）")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--batch", action="store_true",
                    help="同进程批量编译（摊薄冷启动），并给出与逐个编译的对照")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    from config import EAConfig
    from executor import TritonExecutor

    cfg = EAConfig()
    cfg.pre_compile_check = False  # 本脚本直接调 compile_check，不走 evaluate

    all_rows = []
    for kernel in a.kernels:
        test_file = find_test_file(a.dataset_dir, kernel)
        if not test_file:
            print(f"⚠️  {kernel}: 未找到测试文件，跳过")
            continue
        if a.codes_dir:
            files = find_candidates_in_codes(a.codes_dir, kernel, a.limit)
            src_desc = f"codes 库（{a.codes_dir}）"
        else:
            files = find_candidates(a.dataset_dir, kernel, a.limit)
            src_desc = f"datasets（{a.dataset_dir}）"
        if not files:
            print(f"⚠️  {kernel}: 未找到候选代码，跳过")
            continue
        print(f"  候选来源: {src_desc}，命中 {len(files)} 个")
        ex = TritonExecutor(baseline_time=1.0, test_code_path=test_file,
                            config=cfg, kernel_name=kernel,
                            work_dir=Path("/tmp/compile_gate_bench"))
        codes = [Path(f).read_text(encoding="utf-8") for f in files]
        names = [os.path.basename(f) for f in files]

        if a.batch:
            print(f"\n=== {kernel}: {len(codes)} 个候选 · 同进程批量编译 ===")
            t0 = time.time()
            infos = ex.compile_check_batch(codes, timeout=a.timeout,
                                           device_id=a.device_id)
            wall = time.time() - t0
            for nm, info in zip(names, infos):
                info.update({"kernel": kernel, "file": nm, "repeat": 0,
                             "wall_ms": round(wall / max(1, len(codes)) * 1000, 1)})
                all_rows.append(info)
                print(f"  {nm[:40]:40s} pass={str(info['compile_pass']):5s} "
                      f"{info['compile_time_ms']:8.1f}ms "
                      f"attempt={info['n_attempt']} ok={info['n_ok']} "
                      f"{info['error_type'] or ''}")
            print(f"  → 批量总墙钟 {wall:.1f}s，折合 {wall / max(1, len(codes)):.2f}s/候选"
                  f"（逐个编译实测 ≈ 9.8s/候选）")
            all_rows.append({"_batch_summary": True, "kernel": kernel,
                             "n": len(codes), "wall_s": round(wall, 1),
                             "per_candidate_s": round(wall / max(1, len(codes)), 2)})
            continue

        print(f"\n=== {kernel}: {len(files)} 个候选 × {a.repeat} 次 ===")
        for f, code in zip(files, codes):
            for r in range(a.repeat):
                t0 = time.time()
                info = ex.compile_check(code, timeout=a.timeout, device_id=a.device_id)
                info.update({"kernel": kernel, "file": os.path.basename(f),
                             "repeat": r, "wall_ms": round((time.time() - t0) * 1000, 1)})
                all_rows.append(info)
                print(f"  {os.path.basename(f)[:40]:40s} #{r} "
                      f"pass={str(info['compile_pass']):5s} "
                      f"{info['compile_time_ms']:8.1f}ms "
                      f"attempt={info['n_attempt']} ok={info['n_ok']} "
                      f"{info['error_type'] or ''}")

    if not all_rows:
        print("❌ 没有任何编译结果。")
        return 1

    # ---------- 汇总 ----------
    rows = [r for r in all_rows if not r.get("_batch_summary")]
    n = len(rows)
    n_pass = sum(1 for r in rows if r["compile_pass"])
    times = [r["compile_time_ms"] for r in rows]
    first = [r["compile_time_ms"] for r in rows if r.get("repeat", 0) == 0]
    cached = [r["compile_time_ms"] for r in rows if r.get("repeat", 0) > 0]

    print("\n" + "=" * 72)
    print(f"总编译次数 {n}   通过 {n_pass}   拦截 {n - n_pass} "
          f"(拦截率 {(n - n_pass) / n * 100:.1f}%)")
    print(f"compile_time_ms: p50={_pct(times, .5):.0f}  p90={_pct(times, .9):.0f}  "
          f"min={min(times):.0f}  max={max(times):.0f}  mean={st.mean(times):.0f}")
    if first and cached:
        print(f"首次编译 p50={_pct(first, .5):.0f}ms   缓存命中 p50={_pct(cached, .5):.0f}ms   "
              f"→ 缓存节省 {_pct(first, .5) - _pct(cached, .5):.0f}ms")
    errs = {}
    for r in rows:
        if not r["compile_pass"]:
            errs[r["error_type"]] = errs.get(r["error_type"], 0) + 1
    if errs:
        print("拦截错误类型分布:")
        for k, v in sorted(errs.items(), key=lambda x: -x[1]):
            print(f"    {k:<28s} {v}")
    else:
        print("拦截错误类型分布: 无（全部编译通过）")

    print("\n判读参考（2026-10-17 真机实测，同一算子）:")
    print("    python+torch+torch_npu 冷启动 ~7.8s")
    print("    只编译                ~9.8s（净编译 ≈ 2.0s）")
    print("    msprof 完整评测       ~16.5s")
    print("  → 冷启动占一次评测的 ~47%，**逐候选单独预编译并不划算**；")
    print("    编译门的正确用法是批量预编译 + 为候选池补位，而不是省时间。")
    print("=" * 72)

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(all_rows, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"✅ JSON 已导出: {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
