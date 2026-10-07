#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.3 的重做实验（方案甲）：只测**第 0 代**候选质量，不走完整进化流程。

为什么要重做
------------
3.3 原先用端到端 GM(S) 度量 Profiling 证据与知识库检索的作用，但该指标经过
两代进化与精英保留后已被稀释，且 15 个算子的几何平均被个别极端算子主导。
而这两个组件**只作用于第 0 代候选生成**，因此这里把度量下沉到组件真正起作用的
那一层，直接比较各配置第 0 代候选相对最快种子的加速比

    S0 = T_seed / T_candidate

与原有做法的两点关键差异
------------------------
  1. 每个候选用 measure_repeat 多次测量取中位数，量测抖动由单次 ±15%
     降到约 ±2.5%（该数值由 mutation_probe 的无操作对照组实测得到）；
  2. 样本量由"每配置 15 个最终值"提高到约 430 个候选，
     按算子配对后足以检出 5% 级别的差异。

两个阶段
--------
  A（纯测量）：a6 / b3 / b4 / a7 的第 0 代候选代码已存盘于
     experiments/results/<cfg>/<kernel>__r0_gen0/*.py，直接重测，不需要 LLM。
  B（生成+测量）：full 当初跑的是 50 算子主实验、未存盘 gen0，需要重新生成。
     注意：不能用 stage_search，它会把结果写回 results/full 而覆盖 3.1 主实验；
     这里改为 build_ea 后直接取 ea.gen0_snapshot，不写任何既有目录。

用法
----
    source set_env/set_api_local.sh
    python experiments/gen0_remeasure_33.py --phase A          # 纯测量（先跑这个）
    python experiments/gen0_remeasure_33.py --phase B          # 补跑 full 的 gen0
    python experiments/gen0_remeasure_33.py --phase A --limit 5  # 小规模标定
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from experiments.harness import build_ea                       # noqa: E402
from experiments import common                                  # noqa: E402
from experiments.analyze_logs import K15                        # noqa: E402

OUT = ROOT / "experiments" / "results" / "gen0_quality_33.json"
SAVE_DIR = ROOT / "experiments" / "results" / "gen0_probe"

EXISTING = ("a6", "b3", "b4", "a7")     # 已存盘 gen0 的配置


def t_seed_us(kernel):
    m = common.MANIFEST_DIR / f"{kernel}.json"
    if not m.exists():
        return None
    return json.loads(m.read_text(encoding="utf-8")).get("t_seed_us")


def gen0_files(cfg):
    """已存盘的第 0 代候选代码。"""
    out = []
    for d in sorted((common.RESULTS_DIR / cfg).glob("*_gen0")):
        kernel = d.name.split("__")[0]
        if kernel not in K15:
            continue
        for p in sorted(d.glob("*.py")):
            out.append((kernel, p))
    return out


def load_state():
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"samples": [], "parents": []}


def save_state(st):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def make_executor(kernel, method="full"):
    """为某个算子构造一次 executor，供该算子的全部候选复用。

    两点收益：
      1. 省掉每个候选都重建 EA 的开销（加载向量库等），单候选由约 45s 降到约 25s；
      2. 同一算子的所有候选、所有配置共用同一个 executor，
         测量协议完全一致，跨配置比较更公平。
    """
    ea, cfg = build_ea(kernel, method, run_id=0, seed=0, log=False)
    return ea.executor


def measure_one(executor, code, repeats):
    """测量单份第 0 代候选代码，返回其稳定延迟。"""
    return executor.measure_repeat(code, repeats=repeats, device_id=0)


def phase_a(repeats, limit):
    """重测已存盘的第 0 代候选（不需要 LLM）。"""
    st = load_state()
    done = {(s["config"], s["kernel"], s["file"]) for s in st["samples"]}
    exec_cache = {}
    for cfg in EXISTING:
        files = gen0_files(cfg)
        if limit:
            files = files[:limit]
        print(f"\n=== 配置 {cfg}: {len(files)} 个第 0 代候选 ===")
        for i, (kernel, p) in enumerate(files, 1):
            if (cfg, kernel, p.name) in done:
                print(f"  [{i}/{len(files)}] {kernel}/{p.name} 已有结果，跳过")
                continue
            if kernel not in exec_cache:
                exec_cache[kernel] = make_executor(kernel)
            ts = t_seed_us(kernel)
            t0 = time.time()
            try:
                res = measure_one(exec_cache[kernel], common.read_code(p), repeats)
                st["samples"].append({
                    "config": cfg, "kernel": kernel, "file": p.name,
                    "success": bool(res.success), "latency_us": res.execution_time,
                    "s0": (ts / res.execution_time) if (res.success and res.execution_time and ts) else None,
                    "error": (res.error or "")[:200],
                    "repeats": repeats, "seconds": round(time.time() - t0, 1),
                })
                v = st["samples"][-1]
                print(f"  [{i}/{len(files)}] {kernel}/{p.name} "
                      f"{'%.4f' % v['latency_us'] if v['latency_us'] else 'FAILED'} us "
                      f"S0={v['s0']:.4f}" if v["s0"] else
                      f"  [{i}/{len(files)}] {kernel}/{p.name} 失败")
            except Exception as e:
                st["samples"].append({"config": cfg, "kernel": kernel, "file": p.name,
                                      "success": False, "latency_us": None, "s0": None,
                                      "error": repr(e)[:200], "repeats": repeats,
                                      "seconds": round(time.time() - t0, 1)})
                print(f"  [{i}/{len(files)}] {kernel}/{p.name} 异常 {e}")
            save_state(st)      # 逐条落盘，中断不丢数据
    return st


def phase_b(repeats, limit):
    """补跑 full 的第 0 代：只生成 gen0，不进化、不写既有结果目录。"""
    st = load_state()
    done = {(s["kernel"], s["file"]) for s in st["samples"] if s["config"] == "full"}
    kernels = sorted(K15)
    if limit:
        kernels = kernels[:limit]
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for ki, kernel in enumerate(kernels, 1):
        print(f"\n=== [{ki}/{len(kernels)}] full / {kernel} ===")
        try:
            # seed 必须与原 3.3 消融一致（seed=0），否则 FULL 的第 0 代
            # 与其他配置不在同一随机条件下，跨配置比较失去可比性。
            ea, cfg = build_ea(kernel, "full", run_id=0, seed=0,
                               log=False, max_generations=0)
            seed_codes = [common.read_code(p)
                          for p in common.seed_code_paths(kernel)]
            ea.run(seed_codes)
            snap = list(getattr(ea, "gen0_snapshot", []))
            print(f"  第 0 代候选 {len(snap)} 个")
            ts = t_seed_us(kernel)
            for i, ind in enumerate(snap):
                if not ind.code:
                    continue
                fname = f"{kernel}__{i}.py"
                if (kernel, fname) in done:
                    print(f"    #{i} 已有结果，跳过")
                    continue
                (SAVE_DIR / fname).write_text(ind.code, encoding="utf-8")
                t0 = time.time()
                res = ea.executor.measure_repeat(ind.code, repeats=repeats, device_id=0)
                st["samples"].append({
                    "config": "full", "kernel": kernel, "file": fname,
                    "success": bool(res.success), "latency_us": res.execution_time,
                    "s0": (ts / res.execution_time) if (res.success and res.execution_time and ts) else None,
                    "error": (res.error or "")[:200],
                    "repeats": repeats, "seconds": round(time.time() - t0, 1),
                })
                v = st["samples"][-1]
                print(f"    #{i} {('%.4f us S0=%.4f' % (v['latency_us'], v['s0'])) if v['s0'] else 'FAILED'}")
                save_state(st)
        except Exception as e:
            print(f"  [错误] {kernel}: {e}")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("A", "B"), required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="小规模标定用")
    a = ap.parse_args()

    t0 = time.time()
    fn = phase_a if a.phase == "A" else phase_b
    st = fn(a.repeats, a.limit)
    save_state(st)
    n = len(st["samples"])
    print(f"\n完成：样本 {n} 个，用时 {(time.time()-t0)/3600:.2f} 小时")
    print(f"结果: {OUT}")


if __name__ == "__main__":
    main()
