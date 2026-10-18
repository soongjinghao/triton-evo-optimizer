#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 fulleval 日志算"候选预筛选"的在线（真机）代级指标。

原理
----
fulleval 配置（filter_evaluate_all=True）在一次运行中同时记录了：
  * score 行：每代全部候选的模型打分与 selected 标记（即"若启用预筛选会选 top-k"）；
  * eval  行：每代全部候选的真机评测结果（child_latency / improved）。
两者通过 candidate_id 同 run 内一一对应，因此可直接算出：
  精确率 = 被选中的候选中真改进的比例
  召回率 = 被选中且真改进的候选 / 全部真改进候选
这与表10"预筛选"行的在线指标口径一致，且不需要跨运行匹配候选。

指标说明
--------
  随机基线 = 池内真改进比例。随机挑 k 个的期望精确率就是它，
  因此「模型精确率 − 随机基线」才是模型排序带来的净增益。
  池内 AUC = 用模型打分对"是否真改进"排序的能力（0.5 = 无判别力）。

⚠️ 必须限定算子范围：日志目录里可能残留旧版本运行留下的日志
（旧模型 / 旧分代编号），混入后结果不可信。默认只统计 --kernels 指定的算子。

用法
----
    cd /workspace/Agent
    # 限定本轮跑的 5 个算子（推荐）
    python experiments/compute_online_pr.py --kernels _act_quant_kernel \
        _dequantize_k_cache_fast_kernel _fwd_kernel_ep_gather \
        _pack_seq_kernel _selective_scan_update_kernel

    # 只统计最近 N 小时内更新的日志（避免混入旧结果）
    python experiments/compute_online_pr.py --since-hours 6

    python experiments/compute_online_pr.py --all      # 全部日志（不推荐）
"""
import json
import glob
import os
import sys
import time
import argparse
from pathlib import Path

# 本脚本位于 experiments/ 下，日志目录是其同级目录
LOGS = str(Path(__file__).resolve().parent / "logs")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def auc(pairs):
    """pairs = [(score, label)]，label ∈ {0,1}；返回 Mann-Whitney AUC。"""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    # 秩和法，O(n log n)
    allp = sorted([(s, y) for s, y in pairs])
    n = len(allp)
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and allp[j + 1][0] == allp[i][0]:
            j += 1
        avg = (i + j) / 2 + 1          # 1-based 平均秩，处理并列
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1
    sum_pos = sum(ranks[t] for t in range(n) if allp[t][1] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def analyze(path):
    rows = load(path)
    score = {r["candidate_id"]: r for r in rows if "event" not in r}
    ev = {r["candidate_id"]: r for r in rows if r.get("event") == "eval"}

    sel = sel_imp = imp = n_scored = 0
    pairs = []
    per_gen = {}
    for cid, s in score.items():
        e = ev.get(cid)
        if not e:
            continue
        n_scored += 1
        y = 1 if e.get("improved") else 0
        pairs.append((s.get("score", 0.0), y))
        g = s.get("gen")
        d = per_gen.setdefault(g, {"n": 0, "sel": 0, "imp": 0, "sel_imp": 0,
                                   "pairs": []})
        d["n"] += 1
        d["imp"] += y
        d["pairs"].append((s.get("score", 0.0), y))
        if s.get("selected"):
            sel += 1
            d["sel"] += 1
            if y:
                sel_imp += 1
                d["sel_imp"] += 1
        if y:
            imp += 1

    if sel == 0:
        return None
    prec = sel_imp / sel
    rec = sel_imp / imp if imp else 0.0
    rand = imp / n_scored if n_scored else 0.0      # 随机挑选的期望精确率
    return {
        "sel": sel, "sel_imp": sel_imp, "imp": imp, "n_scored": n_scored,
        "prec": prec, "rec": rec, "random": rand, "gain": prec - rand,
        "auc": auc(pairs),
        "per_gen": {g: {"n": d["n"], "sel": d["sel"], "imp": d["imp"],
                        "sel_imp": d["sel_imp"], "auc": auc(d["pairs"])}
                    for g, d in per_gen.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="在线代级指标（P@k / 随机基线 / 池内 AUC）")
    ap.add_argument("--config", default="fulleval",
                    help="实验配置名（日志文件名中段），默认 fulleval；"
                         "分析冒烟请用 --config predictor/random 等实际配置名")
    ap.add_argument("--kernels", nargs="+", default=None,
                    help="只统计这些算子（推荐，避免混入旧日志）")
    ap.add_argument("--since-hours", type=float, default=None,
                    help="只统计最近 N 小时内更新过的日志")
    ap.add_argument("--all", action="store_true",
                    help="统计全部日志（不推荐，可能混入旧版本结果）")
    a = ap.parse_args()

    # 【修复】旧实现把 glob 硬编码成 filter_scores__fulleval__*，
    # 于是 --config smoke/predictor 根本读不到对应日志，只会返回旧的 fulleval 结果，
    # 且不会报错 —— 极易把上一轮实验的数字当成新一轮的结论。
    files = sorted(glob.glob(f"{LOGS}/filter_scores__{a.config}__*.jsonl"))
    if not files:
        print(f"❌ 没有 filter_scores__{a.config}__*.jsonl 日志；"
              f"可用配置：{sorted({os.path.basename(p).split('__')[1] for p in glob.glob(f'{LOGS}/filter_scores__*.jsonl')})}",
              file=sys.stderr)
        return 1
    picked = []
    for p in files:
        name = os.path.basename(p).split("__")[2]
        if a.kernels:
            if name not in a.kernels:
                continue
        elif a.since_hours is not None:
            if time.time() - os.path.getmtime(p) > a.since_hours * 3600:
                continue
        elif not a.all:
            print("❌ 未指定范围。请用 --kernels / --since-hours / --all 之一，"
                  "避免混入旧版本日志。", file=sys.stderr)
            return 2
        picked.append((name, p))

    if not picked:
        print("❌ 没有符合条件的日志", file=sys.stderr)
        return 1

    T = {"sel": 0, "sel_imp": 0, "imp": 0, "n_scored": 0}
    all_pairs = []
    rows_out = []
    print(f"各算子（config={a.config}，同 run 内）：")
    for name, p in picked:
        r = analyze(p)
        if not r:
            continue
        rows_out.append((name, r))
        for k in T:
            T[k] += r[k]
        # pooled AUC 需要逐算子的 pairs，重新取一次
        rows = load(p)
        score = {x["candidate_id"]: x for x in rows if "event" not in x}
        ev = {x["candidate_id"]: x for x in rows if x.get("event") == "eval"}
        for cid, s in score.items():
            e = ev.get(cid)
            if e:
                all_pairs.append((s.get("score", 0.0),
                                  1 if e.get("improved") else 0))
        print(f"  {name:42s} 选中={r['sel']:2d} 选中且改进={r['sel_imp']:2d} "
              f"总改进={r['imp']:2d} 精确率={r['prec']:.3f} 召回率={r['rec']:.3f} "
              f"随机基线={r['random']:.3f} 增益={r['gain']:+.3f} "
              f"AUC={r['auc']:.3f}" if r['auc'] is not None else "AUC=—")

    if T["sel"] == 0:
        return 1
    pooled_p = T["sel_imp"] / T["sel"]
    pooled_r = T["sel_imp"] / T["imp"] if T["imp"] else 0.0
    pooled_rand = T["imp"] / T["n_scored"] if T["n_scored"] else 0.0
    pooled_auc = auc(all_pairs)
    print(f"\n汇总（pooled，{len(rows_out)} 个算子）：")
    print(f"  精确率  P@4      = {pooled_p:.4f}")
    print(f"  召回率  R@4      = {pooled_r:.4f}")
    print(f"  随机基线（期望P）= {pooled_rand:.4f}")
    print(f"  相对随机增益     = {pooled_p - pooled_rand:+.4f}")
    print(f"  池内 AUC         = {pooled_auc:.4f}"
          if pooled_auc is not None else "  池内 AUC         = —")
    print(f"  (选中={T['sel']} 选中且改进={T['sel_imp']} "
          f"总改进={T['imp']} 池内候选={T['n_scored']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
