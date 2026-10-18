#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线候选预筛选实验的结果汇总（纯读文件，不占用 NPU）。

产出论文 3.5 需要的几张表：
  * 逐算子对照：三组各自的 T_best 与 S
  * 汇总：GM(S)、中位数 S、相对 baseline 的胜率
  * 成对差异：predictor − random、predictor − baseline（按算子配对）
  * 跨 seed 波动：同一算子多次重复的极差
  * 成本：LLM 调用数 / Token / 墙钟（缺失时标“—”，不当 0）

口径与 3.1 一致：同一算子先对多次重复取中位数，再跨算子取几何平均。

用法
----
    cd /workspace/Agent
    python experiments/summarize_filter_online.py                 # seed0
    python experiments/summarize_filter_online.py --seeds 0 1 2   # 3 次重复
    python experiments/summarize_filter_online.py --md            # 额外输出 Markdown

退出码恒为 0（汇总不判定成败，判定请用 audit_filter_online.py）。
"""

import sys
import json
import math
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import common                                    # noqa: E402
from experiments.splits import load_split                         # noqa: E402

OUT_DIR = ROOT / "experiments" / "results" / "filter_online"
DEFAULT_CONFIGS = ["baseline", "predictor", "random"]
LABEL = {"baseline": "无预筛选", "predictor": "模型筛选",
         "random": "随机筛选", "fulleval": "全评诊断"}


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def gm(xs):
    vs = [v for v in xs if v and v > 0]
    if not vs:
        return None
    return math.exp(sum(math.log(v) for v in vs) / len(vs))


def load(cfg, kernel, seed):
    p = OUT_DIR / cfg / f"{kernel}__r{seed}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not (rec.get("success") and rec.get("t_best_us")):
        return None
    return rec


def main():
    ap = argparse.ArgumentParser(description="在线候选预筛选结果汇总")
    ap.add_argument("--config", "-c", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--kernels", "-k", nargs="+", default=None)
    ap.add_argument("--seeds", "-s", type=int, nargs="+", default=(0,))
    ap.add_argument("--md", action="store_true", help="额外写出 Markdown 到 results/")
    args = ap.parse_args()

    split = load_split("dev15")
    kernels = args.kernels or split["test_kernels"]

    # per[cfg][kernel] = [S...]；base[kernel] = T_base
    per = {c: defaultdict(list) for c in args.config}
    tbest = {c: defaultdict(list) for c in args.config}
    cost = {c: {"tokens": [], "seconds": [], "npu": []} for c in args.config}
    t_base = {}

    for k in kernels:
        man = common.MANIFEST_DIR / f"{k}.json"
        if man.exists():
            try:
                t_base[k] = json.loads(man.read_text(encoding="utf-8")).get("t_base_us")
            except Exception:
                t_base[k] = None
        for c in args.config:
            for s in args.seeds:
                rec = load(c, k, s)
                if not rec:
                    continue
                per[c][k].append(rec.get("S"))
                tbest[c][k].append(rec["t_best_us"])
                a = rec.get("audit") or {}
                if a.get("llm_tokens"):
                    cost[c]["tokens"].append(a["llm_tokens"])
                if rec.get("seconds"):
                    cost[c]["seconds"].append(rec["seconds"])
                if a.get("npu_calls") is not None:
                    cost[c]["npu"].append(a["npu_calls"])

    def med_S(c, k):
        return median([v for v in per[c][k] if v])

    # 只统计三组都有效的算子
    usable = [k for k in kernels
              if all(med_S(c, k) for c in args.config)]

    print("=" * 88)
    print(" 在线候选预筛选：逐算子对照")
    print(f" 算子 {len(usable)}/{len(kernels)} 参与统计 | seed {list(args.seeds)}")
    print("=" * 88)
    head = f"{'算子':<34}" + "".join(f"{LABEL.get(c, c):>13}" for c in args.config)
    print(head + f"{'预测-随机':>12}")
    print("-" * 88)
    diff_pr, diff_pb = [], []
    for k in kernels:
        vals = {c: med_S(c, k) for c in args.config}
        line = f"{k:<34}"
        for c in args.config:
            line += f"{vals[c]:>13.4f}" if vals[c] else f"{'—':>13}"
        d = None
        if vals.get("predictor") and vals.get("random"):
            d = vals["predictor"] - vals["random"]
            diff_pr.append(d)
            line += f"{d:>+12.4f}"
        else:
            line += f"{'—':>12}"
        if vals.get("predictor") and vals.get("baseline"):
            diff_pb.append(vals["predictor"] - vals["baseline"])
        print(line)
    print("-" * 88)

    # ---------------- 汇总 ----------------
    print("\n 汇总（跨算子几何平均；每算子先取重复中位数）")
    print(f"{'配置':<14}{'GM(S)':>10}{'中位数S':>10}{'参与算子':>10}"
          f"{'NPU/算子':>10}{'墙钟/算子(s)':>14}{'Token/算子':>12}")
    summary = []
    for c in args.config:
        ss = [med_S(c, k) for k in usable]
        g = gm([v for v in ss if v])
        md = median([v for v in ss if v])
        npu = median(cost[c]["npu"]) if cost[c]["npu"] else None
        sec = median(cost[c]["seconds"]) if cost[c]["seconds"] else None
        tok = median(cost[c]["tokens"]) if cost[c]["tokens"] else None
        summary.append({"config": c, "label": LABEL.get(c, c),
                        "gm_s": g, "median_s": md, "n_kernel": len(usable),
                        "npu_per_kernel": npu, "seconds_per_kernel": sec,
                        "tokens_per_kernel": tok})
        f = lambda v, d=4: f"{v:.{d}f}" if v else "—"      # noqa: E731
        print(f"{LABEL.get(c, c):<14}{f(g):>10}{f(md):>10}{len(usable):>10}"
              f"{f(npu, 1):>10}{f(sec, 0):>14}{f(tok, 0):>12}")

    # ---------------- 成对差异 ----------------
    if diff_pr:
        win = sum(1 for d in diff_pr if d > 0)
        print(f"\n 成对差异（按算子配对，共 {len(diff_pr)} 对）")
        print(f"  模型筛选 − 随机筛选：均值 {sum(diff_pr)/len(diff_pr):+.4f}  "
              f"中位数 {median(diff_pr):+.4f}  胜率 {win}/{len(diff_pr)}")
    if diff_pb:
        win = sum(1 for d in diff_pb if d > 0)
        print(f"  模型筛选 − 无预筛选：均值 {sum(diff_pb)/len(diff_pb):+.4f}  "
              f"中位数 {median(diff_pb):+.4f}  胜率 {win}/{len(diff_pb)}")

    # ---------------- 跨 seed 波动 ----------------
    if len(args.seeds) > 1:
        print(f"\n 跨 seed 波动（同一算子多次重复的极差 / 中位数）")
        print(f"{'配置':<14}{'波动中位数':>12}{'波动最大':>12}{'最大波动算子':>34}")
        for c in args.config:
            spreads = {}
            for k in kernels:
                vs = [v for v in per[c][k] if v]
                if len(vs) >= 2:
                    m = median(vs) or 1.0
                    spreads[k] = (max(vs) - min(vs)) / m
            if not spreads:
                continue
            worst = max(spreads, key=spreads.get)
            print(f"{LABEL.get(c, c):<14}{median(list(spreads.values())):>12.4f}"
                  f"{max(spreads.values()):>12.4f}{worst:>34}")

    # ---------------- 落盘 ----------------
    common.ensure_dirs()
    out = common.RESULTS_DIR / "filter_online_summary.json"
    out.write_text(json.dumps({
        "seeds": list(args.seeds),
        "kernels": kernels,
        "usable_kernels": usable,
        "split": split["name"],
        "split_version": split.get("version"),
        "per_kernel": {c: {k: per[c][k] for k in kernels} for c in args.config},
        "summary": summary,
        "pairwise": {
            "predictor_minus_random": diff_pr,
            "predictor_minus_baseline": diff_pb,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[summarize] 已写出 {out}")

    if args.md:
        md = ["### 在线候选预筛选：三组对照\n",
              "| 配置 | GM(S) | 中位数 S | 参与算子数 | NPU 评测/算子 |",
              "|---|---|---|---|---|"]
        for s in summary:
            g = f"{s['gm_s']:.4f}" if s["gm_s"] else "—"
            m = f"{s['median_s']:.4f}" if s["median_s"] else "—"
            n = f"{s['npu_per_kernel']:.1f}" if s["npu_per_kernel"] else "—"
            md.append(f"| {s['label']} | {g} | {m} | {s['n_kernel']} | {n} |")
        p = common.RESULTS_DIR / "filter_online_summary.md"
        p.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[summarize] 已写出 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
