#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选池分布与筛选空间分析（0 NPU，纯离线）。

回答优化方案「任务①」的核心问题：

  Q1. 一个 20 候选池里到底有多少可筛选的空间？
      → 用 Random@k / 模型@k / Oracle@k 三方对照量化上限与当前位置。
  Q2. 这个空间在不同算子、不同 operation（crossover / mutation）上是否一致？
      → 逐算子、逐 operation 拆解，判断预算该往哪儿投。
  Q3. 池内延迟分布有多"宽"？筛选是否可能带来显著收益？
      → 报 P90/P10、CV、最优/最差加速比。
  Q4. 父代基线缺失有多严重？（2026-10-17 实测：fulleval 5 算子 37.5%）
      → 报 parent_latency_source 分布，并支持 --strict-source 只统计可信档。

设计要点
--------
1. **必须用真实候选池**：每个 (kernel, gen) 的池是同一次运行里真实生成的那一批，
   绝不跨算子/跨代合并（合并会伪造出一个"分布更宽"的假池，把结论带偏）。
2. **Random@k 用蒙特卡洛**而不是"池内改进率"近似：二者在 k 远小于池大小时相等，
   但报 Random 的方差能让"模型 − Random"的差是否显著有直观参照。
3. **GM 的稳健性**：best-of-pool 的 GM 会被单个极端候选绑架（实测 _act_quant_kernel
   gen2 出现 S=12.4 的离群点，把 10 池 GM 从 1.0149 抬到 1.2058）。因此除了全量 GM，
   额外报"去掉贡献最大的那一池后的 GM"（leave-one-pool-out 下界）。

用法
----
    # 只分析指定算子（推荐，避免混入旧版本日志）
    python experiments/analyze_candidate_distribution.py \
        --kernels _act_quant_kernel _pack_seq_kernel --cfg fulleval

    # 严格口径：只统计父代基线来自直接父代（parent/dominant）的候选
    python experiments/analyze_candidate_distribution.py --cfg fulleval --strict-source

    # 导出 JSON 供评审核对
    python experiments/analyze_candidate_distribution.py --cfg fulleval --json-out out.json
"""

import argparse
import glob
import json
import math
import os
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parent / "logs"

# 只有这两档的父代基线是"直接父代的真实实测值"（与 evolutionary_algorithm 保持一致）
TRUSTED_PARENT_SOURCES = ("parent", "dominant")
MONTE_CARLO = 20000


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def _pos(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def discover(cfg=None, kernels=None, since_hours=None, allow_all=False):
    """发现待分析的打分日志。

    与 compute_online_pr.py 一样**拒绝无界 glob**：目录里同时存在多代实验结果，
    无差别读取会把旧模型、旧代号的结果混进来。
    """
    files = sorted(glob.glob(str(LOGS / "filter_scores__*__*.jsonl")))
    picked = []
    for p in files:
        name = os.path.basename(p)
        parts = name.split("__")
        if len(parts) < 4:
            continue
        f_cfg, kernel = parts[1], parts[2]
        if cfg and f_cfg != cfg:
            continue
        if kernels and kernel not in kernels:
            continue
        if since_hours is not None and not cfg and not kernels:
            if time.time() - os.path.getmtime(p) > since_hours * 3600:
                continue
        picked.append((f_cfg, kernel, p))
    if not picked and not allow_all:
        return None
    return picked


# --------------------------------------------------------------------------- #
# 池构建
# --------------------------------------------------------------------------- #
def load_online_sources(cfg, kernel):
    """从真机事件日志 filter_online__{cfg}__{kernel}__r*.jsonl 取父代基线来源。

    打分日志（filter_scores）在修复前**没有** parent_latency_source 字段，
    但真机事件日志一直有（parent / dominant / seed / none）。若只看打分日志，
    严格口径对旧日志会全部放行，于是"静默回退到种子"的那批候选又被当成可信档，
    Q4 的结论和修复前一样失真。这里按 candidate_id 把真机日志的来源标记并上来。
    """
    out = {}
    for p in sorted(glob.glob(str(LOGS / f"filter_online__{cfg}__{kernel}__r*.jsonl"))):
        for r in _load_rows(p):
            if r.get("event") == "eval":
                out.setdefault(r["candidate_id"], r.get("parent_latency_source"))
    return out


def build_pools(score_path, strict_source=False, min_pool=5, online_src=None):
    """把一个算子的打分日志切成 (gen -> 候选池)。

    每个池 = 该代真实生成的全部候选；只有同时有打分行与真值行的候选才进入统计。
    """
    rows = _load_rows(score_path)
    score = {r["candidate_id"]: r for r in rows if "event" not in r}
    ev = {r["candidate_id"]: r for r in rows if r.get("event") == "eval"}
    online_src = online_src or {}

    by_gen = defaultdict(list)
    for cid, s in score.items():
        e = ev.get(cid)
        if e is None:
            continue
        src = e.get("parent_latency_source") or online_src.get(cid)
        comparable = e.get("comparable")
        if comparable is None:  # 旧日志没有该字段，用父代延迟是否为有限值推断
            comparable = _pos(e.get("parent_latency")) is not None
        if strict_source and src is not None and src not in TRUSTED_PARENT_SOURCES:
            continue
        if strict_source and src is None and not comparable:
            continue
        child = _pos(e.get("child_latency"))
        if child is None:
            continue
        by_gen[s.get("gen")].append({
            "cid": cid,
            "speedup": (_pos(e.get("parent_latency")) / child)
                       if _pos(e.get("parent_latency")) else None,
            "latency": child,
            "improved": 1 if e.get("improved") else 0,
            "score": s.get("score"),
            "selected": bool(s.get("selected")),
            "operation": s.get("operation") or "none",
            "mutation_type": s.get("mutation_type") or "none",
            "source": src,
        })

    pools = {g: c for g, c in sorted(by_gen.items(), key=lambda x: (x[0] is None, x[0]))
             if len(c) >= min_pool}
    # 父代基线来源分布（全量，不受 strict 过滤影响，用于报 Q4）
    src_dist = defaultdict(int)
    for cid, e in ev.items():
        s = e.get("parent_latency_source") or online_src.get(cid)
        if s is None:
            s = "unknown(旧日志)" if _pos(e.get("parent_latency")) else "missing"
        src_dist[s] += 1
    return pools, dict(src_dist), len(ev)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _mc_random_best(speedups, k, n_mc=MONTE_CARLO):
    """随机抽 k 个，返回 (best 的期望, best 的 GM 期望)。"""
    if len(speedups) <= k:
        return max(speedups), max(speedups)
    rng = random.Random(20261017)
    bests = []
    for _ in range(n_mc):
        bests.append(max(rng.sample(speedups, k)))
    return sum(bests) / len(bests), gm(bests)


def _mc_random_prec(labels, k, n_mc=MONTE_CARLO):
    rng = random.Random(20261017)
    tot = 0
    n = len(labels)
    if n < k:
        return None
    for _ in range(n_mc):
        tot += sum(rng.sample(labels, k))
    return tot / n_mc / k


def analyze_pool(kernel, gen, pool, k=4, n_mc=MONTE_CARLO):
    labels = [c["improved"] for c in pool]
    n = len(pool)
    sp = [c["speedup"] for c in pool if c["speedup"]]
    lats = sorted(c["latency"] for c in pool)

    kk = min(k, n)
    sel = [c for c in pool if c["selected"]] or \
          sorted(pool, key=lambda c: -(c["score"] if c["score"] is not None else -1e9))[:kk]
    sel = sel[:kk]
    model_prec = (sum(c["improved"] for c in sel) / len(sel)) if sel else None
    # Oracle：真值排序取前 kk，命中数 = min(池内改进数, kk)
    n_imp = sum(labels)
    oracle_prec = min(n_imp, kk) / kk
    rand_prec = _mc_random_prec(labels, kk, n_mc)

    rand_best, rand_best_gm = _mc_random_best(sp, kk, n_mc) if sp else (None, None)
    p10 = lats[max(0, int(0.10 * (n - 1)))]
    p90 = lats[min(n - 1, int(0.90 * (n - 1)))]

    return {
        "kernel": kernel, "gen": gen, "n": n, "k": kk,
        "n_improved": n_imp,
        "imp_ratio": n_imp / n,
        "model_P@k": model_prec,
        "oracle_P@k": oracle_prec,
        "random_P@k": rand_prec,
        "random_best_S": rand_best,
        "oracle_best_S": max(sp) if sp else None,
        "pool_gm_S": gm(sp),
        "lat_p90_p10": (p90 / p10) if p10 else None,
        "lat_cv": (st.pstdev(lats) / (sum(lats) / n)) if n > 1 else None,
        "op_mix": dict(_tally(c["operation"] for c in pool)),
    }


def _tally(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="候选池分布与筛选空间分析（0 NPU）")
    ap.add_argument("--cfg", default=None, help="只分析该配置的日志（如 fulleval / smoke）")
    ap.add_argument("--kernels", nargs="+", default=None, help="只分析这些算子")
    ap.add_argument("--since-hours", type=float, default=None, help="只分析最近 N 小时更新的日志")
    ap.add_argument("--all", action="store_true", help="不过滤（不推荐，可能混入旧版本结果）")
    ap.add_argument("--gen", nargs="+", type=int, default=None, help="只看这些代（默认全部）")
    ap.add_argument("--k", type=int, default=4, help="Top-k，默认 4（每代送评测数）")
    ap.add_argument("--strict-source", action="store_true",
                    help="只统计父代基线来自直接父代（parent/dominant）的候选")
    ap.add_argument("--json-out", default=None, help="把结果导出为 JSON")
    a = ap.parse_args()

    if not a.cfg and not a.kernels and a.since_hours is None and not a.all:
        print("❌ 未指定范围。请用 --cfg / --kernels / --since-hours / --all 之一，"
              "避免混入旧版本日志。", file=sys.stderr)
        return 2

    picked = discover(cfg=a.cfg, kernels=a.kernels, since_hours=a.since_hours,
                      allow_all=a.all)
    if not picked:
        print("❌ 没有匹配到任何 filter_scores 日志。", file=sys.stderr)
        return 1

    print("=" * 96)
    print("候选池分布与筛选空间分析（0 NPU）")
    print(f"  范围: cfg={a.cfg or '*'} kernels={a.kernels or '*'}  "
          f"严格父代口径={a.strict_source}  Top-k={a.k}")
    print("=" * 96)

    per_pool, src_all, op_rows = [], defaultdict(int), []
    n_ev_total = 0
    for cfg, kernel, path in picked:
        pools, src_dist, n_ev = build_pools(
            path, strict_source=a.strict_source,
            online_src=load_online_sources(cfg, kernel))
        n_ev_total += n_ev
        for s, c in src_dist.items():
            src_all[s] += c
        for g, pool in pools.items():
            if a.gen is not None and g not in a.gen:
                continue
            r = analyze_pool(kernel, g, pool, k=a.k)
            per_pool.append(r)
            # 逐 operation 拆解
            for op in sorted({c["operation"] for c in pool}):
                sub = [c for c in pool if c["operation"] == op]
                if len(sub) < 3:
                    continue
                op_rows.append(analyze_pool(kernel, g, sub, k=min(a.k, len(sub))))
                op_rows[-1]["operation"] = op

    if not per_pool:
        print("❌ 没有可用的候选池（可能池太小或真值行缺失）。", file=sys.stderr)
        return 1

    # ---------- Q4 父代基线来源 ----------
    print("\n【Q4】父代基线来源分布（全部 eval 行，共 %d 条）" % sum(src_all.values()))
    tot_src = sum(src_all.values()) or 1
    for s, c in sorted(src_all.items(), key=lambda x: -x[1]):
        flag = "" if s in TRUSTED_PARENT_SOURCES else "   ← 回退基线，improved 语义非『比父代快』"
        print(f"    {s:<18s} {c:5d}  ({c / tot_src * 100:5.1f}%){flag}")

    # ---------- Q1/Q3 逐池 ----------
    print("\n【Q1/Q3】逐候选池（池 = 同一次运行里真实生成的那一批，未跨代/跨算子合并）")
    hdr = (f"{'kernel':30s} {'g':>2s} {'n':>3s} {'imp':>4s} "
           f"{'Rand':>6s} {'Model':>6s} {'Oracle':>7s} "
           f"{'RandBS':>7s} {'OracBS':>7s} {'poolGM':>7s} {'P90/P10':>8s} {'CV':>6s}")
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for r in per_pool:
        f = lambda v, w, p=4: ("%*.4f" % (w, v)) if v is not None else " " * w
        print(f"  {r['kernel'][:30]:30s} {str(r['gen']):>2s} {r['n']:>3d} {r['n_improved']:>4d} "
              f"{f(r['random_P@k'],6,3)} {f(r['model_P@k'],6,3)} {f(r['oracle_P@k'],7,3)} "
              f"{f(r['random_best_S'],7)} {f(r['oracle_best_S'],7)} {f(r['pool_gm_S'],7)} "
              f"{f(r['lat_p90_p10'],8,3)} {f(r['lat_cv'],6,4)}")

    # ---------- 汇总 ----------
    def _avg(key):
        xs = [r[key] for r in per_pool if r[key] is not None]
        return sum(xs) / len(xs) if xs else None

    rand, model, oracle = _avg("random_P@k"), _avg("model_P@k"), _avg("oracle_P@k")
    print("\n【汇总】P@k 三方对照（%d 个池的均值）" % len(per_pool))
    print(f"    Random@k = {rand:.4f}    模型@k = {model:.4f}    Oracle@k = {oracle:.4f}")
    if rand is not None and oracle is not None and oracle > rand:
        gain = (model - rand) if model is not None else 0.0
        print(f"    理论空间 (Oracle − Random) = {oracle - rand:+.4f}")
        print(f"    模型已捕获 (Model − Random) = {gain:+.4f}  "
              f"→ 占理论空间 {gain / (oracle - rand) * 100:.1f}%")
    else:
        print("    ⚠️ Oracle 与 Random 无差距：该批池内几乎没有筛选空间。")

    # GM 稳健性：leave-one-pool-out
    gms = [(r["kernel"], r["gen"], r["pool_gm_S"]) for r in per_pool if r["pool_gm_S"]]
    if len(gms) >= 2:
        allgm = gm([x[2] for x in gms])
        worst = min(gms, key=lambda x: x[2])
        best = max(gms, key=lambda x: x[2])
        loo = gm([x[2] for x in gms if x is not best])
        print(f"\n【稳健性】池内加速比 GM = {allgm:.4f}（{len(gms)} 池）")
        print(f"    去掉贡献最大的池（{best[0][:24]} g{best[1]}，GM={best[2]:.4f}）后 = {loo:.4f}")
        print(f"    最低池 {worst[0][:24]} g{worst[1]} GM={worst[2]:.4f}"
              f"  → 单池极值对 GM 的影响 = {allgm - loo:+.4f}")
        if abs(allgm - loo) > 0.05:
            print("    ⚠️ GM 对单个池高度敏感，不要单独用 GM 做主指标。")

    # ---------- Q2 逐 operation ----------
    if op_rows:
        print("\n【Q2】按 operation 拆解（池内子集，样本少，仅作方向性参考）")
        agg = defaultdict(list)
        for r in op_rows:
            agg[r["operation"]].append(r)
        print(f"  {'operation':22s} {'子集数':>6s} {'平均n':>6s} {'imp率':>7s} "
              f"{'Rand':>7s} {'Model':>7s} {'Oracle':>7s}")
        for op, rs in sorted(agg.items()):
            m = lambda k: (sum(r[k] for r in rs if r[k] is not None) /
                           max(1, len([r for r in rs if r[k] is not None])))
            print(f"  {op:22s} {len(rs):>6d} {sum(r['n'] for r in rs) / len(rs):>6.1f} "
                  f"{m('imp_ratio'):>7.3f} {m('random_P@k'):>7.3f} "
                  f"{m('model_P@k'):>7.3f} {m('oracle_P@k'):>7.3f}")

    print("\n" + "=" * 96)
    print("判读建议")
    print("  * 若 Oracle − Random 很小（<0.05）：该批算子/该批池没有筛选价值，")
    print("    应把预算投向改进样本稀疏的算子（看逐池表里 imp 小的那些行）。")
    print("  * 若模型 ≈ Random 而 Oracle 明显更高：排序模型无效，先修特征/标签口径，")
    print("    不要靠扩大算子数量去堆显著性。")
    print("  * 若 Q4 里回退档占比高：improved 标签混了『比种子快』，"
          "用 --strict-source 重跑本脚本再下结论。")
    print("=" * 96)

    if a.json_out:
        out = {"scope": {"cfg": a.cfg, "kernels": a.kernels,
                         "strict_source": a.strict_source, "k": a.k},
               "parent_source_dist": dict(src_all),
               "pools": per_pool,
               "by_operation": op_rows,
               "summary": {"random_P@k": rand, "model_P@k": model, "oracle_P@k": oracle}}
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"\n✅ JSON 已导出: {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
