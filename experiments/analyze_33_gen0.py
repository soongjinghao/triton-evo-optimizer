#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.3 的**第 0 代候选质量**分析（方案乙，不需要 NPU，只读现有日志）。

为什么需要它
------------
3.3 原先只用端到端 GM(S) 度量 Profiling 证据与知识库检索的作用，但该指标
存在两个层次问题：

  1. 这两个组件**只作用于第 0 代候选生成**，而 GM(S) 是经过两代进化与精英
     保留之后的结果，信号已被稀释；
  2. GM(S) 在 15 个算子上取几何平均，被个别极端算子主导（最高 > 22、最低 < 1）。

因此这里把指标下沉到组件实际作用的那一层：直接比较各配置**第 0 代候选**
相对最快种子的加速比 S0 = T_seed / T_candidate，并按算子做配对检验。

⚠️ 口径说明（写论文时必须交代）
------------------------------
本分析使用的是搜索过程中的**单次实测延迟**（eval 事件里的 latency_us），
未做重复测量，因此含约 ±15% 的量测抖动（该数值由 mutation_probe 的无操作
对照组实测得到）。故本表用于说明"组件作用在第 0 代"这一层次事实，
其数值差异本身不足以单独支撑显著性结论。

用法
----
    python experiments/analyze_33_gen0.py
"""
import collections
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import common                       # noqa: E402
from experiments.analyze_logs import K15             # noqa: E402

OUT = ROOT / "experiments" / "results" / "table6c_gen0_quality.json"

# 与 3.3 正文一致的配置顺序与名称
CONFIGS = [
    ("a6", "A6"), ("b3", "B3"), ("b4", "B4"), ("a7", "A7"), ("full", "FULL"),
]
DESC = {
    "a6": "关闭 Profiling 证据，保留受约束 RAG",
    "b3": "启用 Profiling，不使用知识库",
    "b4": "启用 Profiling，普通语义 Top-k 检索",
    "a7": "同时关闭 Profiling 证据与知识库",
    "full": "全部启用（Profiling + 混合重排 + Guard）",
}


def t_seed_us(kernel):
    m = common.MANIFEST_DIR / f"{kernel}.json"
    if not m.exists():
        return None
    return json.loads(m.read_text(encoding="utf-8")).get("t_seed_us")


def gen0_latencies(cfg):
    """读取某配置在 K15 上全部第 0 代候选的实测延迟。"""
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(str(common.LOGS_DIR / f"{cfg}__*__r0.jsonl"))):
        kernel = os.path.basename(f).split("__")[1]
        if kernel not in K15:
            continue
        for line in open(f, encoding="utf-8"):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "eval" and \
                    e.get("operation") == "gen0_strategy_guided":
                out[kernel].append(e.get("latency_us"))
    return out


def gm(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def build():
    from scipy.stats import wilcoxon
    data = {c: gen0_latencies(c) for c, _ in CONFIGS}
    kernels = sorted(set.intersection(*[set(data[c]) for c, _ in CONFIGS]))

    rows, per_kernel_best = [], {}
    for cfg, label in CONFIGS:
        all_s, n_cand, n_valid, best = [], 0, 0, {}
        for k in kernels:
            ts = t_seed_us(k)
            lats = [x for x in data[cfg][k] if x]
            n_cand += len(data[cfg][k])
            n_valid += len(lats)
            if not (lats and ts):
                continue
            s = [ts / v for v in lats]
            all_s.extend(s)
            best[k] = max(s)
        per_kernel_best[cfg] = best
        rows.append({
            "config": cfg, "label": label, "desc": DESC[cfg],
            "n_candidates": n_cand, "n_valid": n_valid,
            "valid_rate": n_valid / n_cand if n_cand else 0.0,
            "gm_s0": gm(all_s),
            "median_s0": statistics.median(all_s) if all_s else float("nan"),
            "p_better": sum(1 for x in all_s if x > 1) / len(all_s) if all_s else 0.0,
            "gm_best_s0": gm(list(best.values())),
            "n_kernels": len(best),
        })

    ck = sorted(set.intersection(*[set(per_kernel_best[c]) for c, _ in CONFIGS]))
    pairs = []
    for cfg, label in CONFIGS:
        if cfg == "full":
            continue
        x = [per_kernel_best["full"][k] for k in ck]
        y = [per_kernel_best[cfg][k] for k in ck]
        w = sum(1 for p, q in zip(x, y) if p > q)
        try:
            stat, p = wilcoxon(x, y)
        except Exception:
            stat, p = float("nan"), float("nan")
        pairs.append({"ref": "full", "other": cfg, "label": label,
                      "n_pairs": len(ck), "full_wins": w,
                      "W": stat, "p": p,
                      "significant": bool(p < 0.05)})

    return {"kernels": kernels, "rows": rows, "pairwise": pairs,
            "note": "S0 = T_seed / T_candidate，使用搜索过程中的单次实测延迟（含约 ±15% 抖动）"}


def main():
    res = build()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 96)
    print("表6c  第 0 代候选质量（相对最快种子 T_seed 的加速比 S0）")
    print("=" * 96)
    print(f"{'配置':8s}{'候选':>6s}{'有效':>6s}{'有效率':>8s}{'GM(S0)':>9s}"
          f"{'中位S0':>9s}{'P(S0>1)':>9s}{'每算子最优S0(GM)':>16s}")
    for r in res["rows"]:
        print(f"{r['label']:8s}{r['n_candidates']:>6d}{r['n_valid']:>6d}"
              f"{r['valid_rate']*100:>7.1f}%{r['gm_s0']:>9.4f}"
              f"{r['median_s0']:>9.4f}{r['p_better']*100:>8.1f}%"
              f"{r['gm_best_s0']:>16.4f}")

    print("\n配对检验（每算子取该配置第 0 代最优候选的 S0，FULL vs 各对照）")
    print(f"{'对比':16s}{'配对':>8s}{'FULL胜':>9s}{'W':>8s}{'p':>9s}   判定")
    for q in res["pairwise"]:
        judge = "显著" if q["significant"] else "不显著"
        print(f"FULL vs {q['label']:6s}{q['n_pairs']:>8d}"
              f"{q['full_wins']:>6d}/{q['n_pairs']}{q['W']:>8.1f}"
              f"{q['p']:>9.4f}   {judge}")
    print(f"\n已保存: {OUT}")


if __name__ == "__main__":
    main()
