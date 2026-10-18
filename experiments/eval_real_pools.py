#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**真实候选池**上评估候选预筛选模型的 Top-k 收益（0 NPU）。

为什么需要这个脚本
------------------
评审指出：`retrain_candidate_filter.py` 的 `_pool_p_at_k()` 只按 (kernel, gen)
分组，会把多次运行/多批生成的候选并成一个假池。历史快照上 (kernel, gen) 分组的
中位组内条数与线上"每代约 20 选 4"完全对不上，P@4、随机基线、新旧模型对比
全部失真（P@4=0.5583 / 0.5083 即由此而来，不得再引用）。

本脚本改用**在线日志里真实存在的候选池**：fulleval 配置下每代的 20 个候选
全部送评，因此 `filter_online__fulleval__{kernel}__r*.jsonl` 里同一池的
20 条 eval 记录就是线上那一次真实的"20 选 4"。

分组键（v8.2 修正）
------------------
评审指出：`pools[(kernel, gen)] = members` 在多随机种子实验下会被后一个
运行覆盖。现在分组键统一为：

  * 事件里有 `pool_id` → 直接用它（唯一标识一次运行里真实一起生成的那批）；
  * 旧日志没有 → 退化成 `{kernel}|r{run_id}|g{gen}`（**含 run_id**，不再丢运行维度）。

逐池统计、池内 AUC、聚类 Bootstrap、符号翻转置换检验全部按这个键组织，
算子簇由 `pool_kernel`（池 → 算子）提供，不再从键里猜。

口径要点
--------
1. **完整池 / 不完整池分开报**：只有池内全部候选都有可信标签才进正式 P@4。
2. **p 值用零假设检验**：置信区间用按算子聚类 Bootstrap；显著性用
   以算子为单位的符号翻转置换检验（簇数 ≤20 时精确枚举 2^C）。
3. **实际延迟收益**：除 P@4 外，增加 Random / 各模型 / Oracle 选出的 4 个候选里
   **最快实际延迟**的比较（P@4 只数"改进个数"，看不出改进幅度）。

用法
----
    python experiments/eval_real_pools.py \
        --config fulleval --kernels K1 K2 ... --run-ids 905 \
        --old experiments/dataset/candidate_filter_model__dev15.joblib \
        --new experiments/dataset/candidate_filter_model__v7.joblib
"""

import argparse
import collections
import glob
import itertools
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_features_shared import (      # noqa: E402
    load_codes, compute_code_features, normalize_pids)

LOGS = ROOT / "experiments" / "logs"
NUMERIC = ["log_parent", "gen", "pop_size", "tau",
           "r_block", "r_warps", "r_stages", "sim",
           "r_ops", "r_mask", "ops_abs", "block_abs"]
CATEGORICAL = ["operation", "mutation_type"]
DEFAULT_OLD = ROOT / "experiments" / "dataset" / "candidate_filter_model__dev15.joblib"
DEFAULT_NEW = ROOT / "experiments" / "dataset" / "candidate_filter_model__v7.joblib"
# 与 evolutionary_algorithm.TRUSTED_PARENT_SOURCES 保持一致
TRUSTED_PARENT_SOURCES = ("parent", "dominant", "backbone")


def _pos(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _pool_key_of(r, kernel):
    """池的唯一分组键：有 pool_id 就用它，否则退化成 {kernel}|r{run_id}|g{gen}。

    退化形式**必须带 run_id** —— 否则同一个算子的不同随机种子会互相覆盖。
    """
    pid = r.get("pool_id")
    if pid:
        return str(pid)
    return f"{kernel}|r{r.get('run_id', 0)}|g{r.get('gen')}"


def load_real_pools(config, kernels, min_pool, max_pool, run_ids=None):
    """从真机事件日志里取真实候选池。

    返回 (pools, pool_kernel, skipped)。

    【修正】旧实现用 `pools[(kernel, gen)] = members`，多随机种子/多次运行时
    后一个运行会**覆盖**前一个。现在：
      * 事件带 pool_id → 以 pool_id 为键（唯一）；
      * 否则 → `{kernel}|r{run_id}|g{gen}`（带上 run_id，至少不丢运行维度）。
    """
    pools = collections.OrderedDict()
    pool_kernel = {}
    skipped = collections.Counter()
    for k in kernels:
        files = sorted(glob.glob(str(LOGS / f"filter_online__{config}__{k}__r*.jsonl")))
        for p in files:
            rid = Path(p).stem.split("__r")[-1]
            if run_ids and rid not in run_ids:
                continue
            rows = []
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "eval" and r.get("gen", 0) >= 1:
                    r.setdefault("run_id", rid)
                    rows.append(r)
            by_pool = collections.defaultdict(list)
            for r in rows:
                by_pool[_pool_key_of(r, k)].append(r)
            for key in sorted(by_pool):
                members = by_pool[key]
                if len(members) < min_pool:
                    skipped["too_small"] += 1
                    continue
                if len(members) > max_pool:
                    skipped["too_large_suspect_merged"] += 1
                    continue
                if key in pools:                      # 理论上不会再撞，兜底保护
                    skipped["duplicate_pool_key"] += 1
                    continue
                pools[key] = members
                pool_kernel[key] = k
    return pools, pool_kernel, skipped


def ref_parent_id(r):
    """解析该候选真正的**参考父代** ID。

    优先级：reference_parent_id → backbone_id → dominant_parent → parent_ids[0]
    前三个是 EA 明确记录的「训练标签真正参照的那位**已评测**父代」；
    只有旧日志才会退化到 parent_ids[0]，而交叉场景下它未必是主干父代。
    """
    for key in ("reference_parent_id", "backbone_id", "dominant_parent"):
        v = r.get(key)
        if v:
            return str(v), key
    pids = normalize_pids(r.get("parent_ids"))
    if pids:
        return pids[0], "parent_ids[0]"
    return None, "none"


def build_df(pools, pool_kernel, codes, tau_default):
    """把池内候选转成特征表，并判定每个池是否"完整可打分"。"""
    recs = []
    pool_info = {}
    n_dropped = 0
    n_parent_code_missing = 0      # 只统计**保留下来**的样本
    n_tau_imputed = 0
    ref_src = collections.Counter()

    for key, members in pools.items():
        kernel = pool_kernel[key]
        n_labeled = 0
        n_trusted = 0
        for r in members:
            cid = str(r.get("candidate_id"))
            ch_code = codes.get(cid, "")
            pl = _pos(r.get("parent_latency"))
            cl = _pos(r.get("latency_us"))
            if pl is None or cl is None:
                n_dropped += 1
                continue
            n_labeled += 1
            src = r.get("parent_latency_source")
            if src is None or src in TRUSTED_PARENT_SOURCES:
                n_trusted += 1

            pid, psrc = ref_parent_id(r)
            ref_src[psrc] += 1
            pcode = codes.get(pid) if pid else None
            if pcode is None:
                n_parent_code_missing += 1
            cf = compute_code_features(ch_code, pcode)

            tau = _pos(r.get("tau"))
            if tau is None:
                tau = float(tau_default)
                n_tau_imputed += 1

            recs.append(dict(
                pool_key=key, kernel=kernel, gen=r.get("gen"),
                run_id=r.get("run_id"),
                candidate_id=cid,
                parent_latency=pl, child_latency=cl,
                y=1 if pl > cl else 0,
                log_parent=math.log1p(pl),
                gen_f=float(r.get("gen") or 0),
                pop_size=float(r.get("pop_size") or 6),
                tau=float(tau),
                operation=r.get("operation") or "none",
                mutation_type=r.get("mutation_type") or "none",
                parent_latency_source=src,
                reference_parent_id=pid,
                ref_parent_source=psrc,
                parent_code_missing=bool(cf.get("parent_code_missing")),
                child_code_sha256=r.get("child_code_sha256"),
                **{k: cf[k] for k in NUMERIC if k in cf},
            ))
        pool_info[key] = dict(n_raw=len(members), n_labeled=n_labeled,
                              n_trusted=n_trusted,
                              complete=(n_labeled == len(members)))

    df = pd.DataFrame(recs)
    if df.empty:
        return df, pool_info, dict(n_dropped=n_dropped, n_parent_code_missing=0,
                                   n_tau_imputed=n_tau_imputed, ref_src=ref_src)
    df["pool_complete"] = [pool_info[k]["complete"] for k in df["pool_key"]]
    return df, pool_info, dict(n_dropped=n_dropped,
                               n_parent_code_missing=n_parent_code_missing,
                               n_tau_imputed=n_tau_imputed,
                               ref_src=ref_src)


def per_pool(df, scores, k=4):
    """逐池 P@k。df 只含本次要统计的池（完整池 / 不完整池分开调用）。"""
    if df.empty:
        return {}
    d = df.copy()
    d["s"] = scores
    out = {}
    for key, g in d.groupby("pool_key", sort=False):
        if len(g) < k:
            continue
        gg = g.sort_values("s", ascending=False)
        out[key] = (float((gg["y"].to_numpy()[:k] == 1).mean()),
                    float(g["y"].mean()), len(g))
    return out


def pool_auc(df, scores):
    """池内 AUC（逐池内的正负对排序正确率）。"""
    aucs = []
    for _, g in df.groupby("pool_key", sort=False):
        idx = g.index.to_numpy()
        yy = g["y"].to_numpy()
        if len(set(yy)) < 2:
            continue
        pos, neg = idx[yy == 1], idx[yy == 0]
        s = np.asarray(scores)
        aucs.append(float(np.mean([(s[i] > s[j]) + 0.5 * (s[i] == s[j])
                                   for i in pos for j in neg])))
    return float(np.mean(aucs)) if aucs else None


def cluster_bootstrap(pa, pb, pool_kernel, n_boot=20000, seed=20261017):
    """按算子整簇重抽的 Bootstrap —— **只用来给置信区间**。

    Bootstrap 重抽出来的是观测分布而非零假设分布，
    因此不能用它算 p 值（评审指出）。p 值请用 cluster_signflip_test。
    """
    keys = sorted(set(pa) & set(pb))
    if not keys:
        return None
    kernels = sorted({pool_kernel[k] for k in keys})
    arr = {kk: np.array([pa[k][0] - pb[k][0] for k in keys if pool_kernel[k] == kk])
           for kk in kernels}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(kernels), size=len(kernels))
        diffs[i] = np.concatenate([arr[kernels[j]] for j in pick]).mean()
    obs = float(np.concatenate([arr[kk] for kk in kernels]).mean())
    return {"mean_diff": obs,
            "ci95": (float(np.percentile(diffs, 2.5)),
                     float(np.percentile(diffs, 97.5))),
            "n_pools": len(keys), "n_clusters": len(kernels)}


def cluster_signflip_test(pa, pb, pool_kernel, n_perm=None, seed=20261017):
    """以算子为单位的**符号翻转置换检验**（零假设：两个模型无差异）。

    1. 逐池算配对差 d = P@k(A) − P@k(B)；
    2. 按算子聚成簇级均值（同一算子的多个代/多次运行不是独立样本）；
    3. 零假设下每个簇的符号 ± 等概率 → 翻转簇符号重算均值；
    4. 簇数 C ≤ 20 时枚举全部 2^C 个符号组合（精确检验），否则随机置换。

    注意：C 很小时分辨力受限于 2/2^C（C=3 → 最小 p=0.25；C=5 → 0.0625）。
    """
    keys = sorted(set(pa) & set(pb))
    if not keys:
        return None
    kernels = sorted({pool_kernel[k] for k in keys})
    cluster = np.array([np.mean([pa[k][0] - pb[k][0]
                                 for k in keys if pool_kernel[k] == kk])
                        for kk in kernels])
    obs = float(cluster.mean())
    C = len(cluster)
    if C == 0 or not np.any(cluster):
        return {"obs": obs, "p_value": 1.0, "n_clusters": C,
                "n_perm": 0, "exact": False, "min_p": None}
    exact = (n_perm is None) and (C <= 20)
    if exact:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=C)))
        means = signs.dot(cluster) / C
        p = float(np.mean(np.abs(means) >= abs(obs) - 1e-12))
        n_used = int(len(signs))
    else:
        n = int(n_perm or 20000)
        rng = np.random.default_rng(seed)
        s = rng.choice([-1.0, 1.0], size=(n, C))
        means = s.dot(cluster) / C
        p = float((np.sum(np.abs(means) >= abs(obs) - 1e-12) + 1) / (n + 1))
        n_used = n
    return {"obs": obs, "p_value": p, "n_clusters": C,
            "n_perm": n_used, "exact": exact,
            "min_p": (2.0 / 2 ** C) if exact else None}


def latency_gain(df, models, k, pool_kernel, base, n_mc=3000, seed=20261017):
    """实际延迟收益：Random / 各模型 / Oracle 选出的 k 个里**最快**的那一个。

    P@4 只数"选中的 k 个里有几个比父代快"，看不出改进幅度，也看不出
    "选中 4 个里最好的是不是真的好"。这里补上以延迟为单位的比较：
      * random_best：随机抽 k 个的最快延迟（蒙特卡洛期望）
      * {tag}_best：该模型 Top-k 里的最快延迟
      * oracle_best：全池最快（真值上界）
      * regret = {tag}_best − oracle_best（越小越好，0 = 选中了全池最优）
    """
    if df.empty or not models:
        return None
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby("pool_key", sort=False):
        if len(g) < k:
            continue
        lat = g["child_latency"].to_numpy(dtype=float)
        n = len(lat)
        draws = min(int(n_mc), 5000)
        rb = float(np.mean([lat[rng.choice(n, k, replace=False)].min()
                            for _ in range(draws)]))
        row = {"pool": key, "kernel": pool_kernel.get(key, "?"), "n": n,
               "random_best": rb, "oracle_best": float(lat.min())}
        for tag, m in models.items():
            s = m.predict_proba(g[NUMERIC + CATEGORICAL])[:, 1]
            top = np.argsort(-np.asarray(s))[:k]
            row[f"{tag}_best"] = float(lat[top].min())
        rows.append(row)
    if not rows:
        return None
    out = {"per_pool": rows, "n_mc": draws}
    # 不同算子的绝对延迟差几个数量级（1.36us vs 22.8us），直接平均没有意义；
    # 归一化指标（每层相对 Oracle 的倍数）才是可跨算子比较的量。
    agg = {}
    for tag in list(models) + ["random", "oracle"]:
        col = f"{tag}_best"
        vals = np.array([r[col] for r in rows], dtype=float)
        ratio = vals / np.array([r["oracle_best"] for r in rows], dtype=float)
        agg[tag] = {"best_latency_mean": float(vals.mean()),
                    "regret_vs_oracle": float((vals - np.array(
                        [r["oracle_best"] for r in rows])).mean()),
                    "ratio_vs_oracle": float(ratio.mean()),
                    "ratio_geomean": float(np.exp(np.mean(np.log(ratio))))}
    out["agg"] = agg
    # 模型 vs base 的配对检验：用**归一化后的比值**（越小越好）
    tests = {}
    for tag in models:
        if tag == base or base not in models:
            continue
        pa = {r["pool"]: (r[f"{tag}_best"] / r["oracle_best"], 0.0, r["n"])
              for r in rows}
        pb = {r["pool"]: (r[f"{base}_best"] / r["oracle_best"], 0.0, r["n"])
              for r in rows}
        tests[tag] = cluster_signflip_test(pa, pb, pool_kernel)
    out["tests"] = tests
    return out


def report_section(title, df_sub, models, k, base, pool_kernel,
                   raw_sizes=None, note="", latency=True):
    """在给定的池子集上出一份完整报告。"""
    print("\n" + "-" * 88)
    print(title)
    if note:
        for ln in note.strip().splitlines():
            print(f"  {ln}")
    print("-" * 88)
    if df_sub.empty:
        print("  （该集合为空）")
        return {}
    X = df_sub[NUMERIC + CATEGORICAL]
    res = {}
    print(f"  {'模型':10s} {'P@%d' % k:>8s} {'随机基线':>10s} {'增益':>9s} {'池内AUC':>8s}")
    for tag, m in models.items():
        s = m.predict_proba(X)[:, 1]
        pp = per_pool(df_sub, s, k)
        if not pp:
            print(f"  {tag:10s} —— 没有大小 ≥ {k} 的池")
            continue
        p_at_k = float(np.mean([v[0] for v in pp.values()]))
        rand = float(np.mean([v[1] for v in pp.values()]))
        auc = pool_auc(df_sub, s)
        res[tag] = {"scores": s, "pools": pp, "p_at_k": p_at_k,
                    "random": rand, "auc": auc}
        print(f"  {tag:10s} {p_at_k:>8.4f} {rand:>10.4f} {p_at_k - rand:>+9.4f} "
              f"{('%.4f' % auc if auc is not None else '—'):>8s}")
    if not res:
        return {}

    first = list(res)[0]
    raw_sizes = raw_sizes or {}
    print(f"\n  逐池明细（{k} 选 {k}；n = 可用于打分 / 池内原始）")
    print(f"    {'pool':44s} {'n':>8s} {'改进':>4s} " +
          "".join(f"{t[:11] + '_P@%d' % k:>14s}" for t in res))
    for key in sorted(res[first]["pools"]):
        n = res[first]["pools"][key][2]
        raw = int(raw_sizes.get(key, n))
        imp = int(round(res[first]["pools"][key][1] * n))
        line = f"    {key[:44]:44s} {f'{n}/{raw}':>8s} {imp:>4d} "
        for t in res:
            line += f"{res[t]['pools'][key][0]:>14.3f}"
        print(line)

    b = base if base in res else first
    tests = {}
    for tag in res:
        if tag == b:
            continue
        cb = cluster_bootstrap(res[tag]["pools"], res[b]["pools"], pool_kernel)
        pt = cluster_signflip_test(res[tag]["pools"], res[b]["pools"], pool_kernel)
        tests[tag] = {"bootstrap": cb, "signflip": pt}
        if not cb or not pt:
            continue
        desc = ("精确枚举" if pt["exact"]
                else ("差异恒为 0" if not pt["n_perm"]
                      else f"{pt['n_perm']} 次随机置换"))
        print(f"\n  {tag} − {b}：")
        print(f"    均值差          = {cb['mean_diff']:+.4f} "
              f"（{cb['n_pools']} 池 / {cb['n_clusters']} 个算子簇）")
        print(f"    95% CI（Bootstrap，按算子聚类） = "
              f"[{cb['ci95'][0]:+.4f}, {cb['ci95'][1]:+.4f}]")
        extra = ""
        if pt.get("min_p"):
            extra = f"（C={pt['n_clusters']} 时最小可达 p={pt['min_p']:.4f}）"
        print(f"    p（符号翻转置换，{desc}）= {pt['p_value']:.4f} → "
              f"{'差异不显著' if pt['p_value'] >= 0.05 else '差异显著'} {extra}")

    out = {"models": {t: {"p_at_k": res[t]["p_at_k"],
                          "random_baseline": res[t]["random"],
                          "pool_auc": res[t]["auc"]} for t in res},
           "tests": tests}

    if latency:
        lg = latency_gain(df_sub, models, k, pool_kernel, b)
        if lg:
            print(f"\n  实际延迟收益（每池选 {k} 个里**最快**的那个；越小越好）")
            print(f"    {'策略':12s} {'最快延迟均值':>14s} {'/Oracle(算术)':>14s} "
                  f"{'/Oracle(几何)':>14s}")
            for tag in ["random"] + list(models) + ["oracle"]:
                a = lg["agg"].get(tag)
                if not a:
                    continue
                print(f"    {tag:12s} {a['best_latency_mean']:>14.4f} "
                      f"{a['ratio_vs_oracle']:>14.4f} {a['ratio_geomean']:>14.4f}")
            print("    注：不同算子绝对延迟差数量级，跨算子只能看 /Oracle 比值；")
            print(f"    random 为 {lg['n_mc']} 次蒙特卡洛抽样期望。")
            for tag, pt in (lg["tests"] or {}).items():
                if not pt:
                    continue
                print(f"    {tag} − {b} 的 (/Oracle) 差 = {pt['obs']:+.4f}，"
                      f"p = {pt['p_value']:.4f} → "
                      f"{'不显著' if pt['p_value'] >= 0.05 else '显著'}")
            out["latency_gain"] = {k_: v for k_, v in lg["agg"].items()}
            out["latency_tests"] = lg["tests"]
    return out


def main():
    ap = argparse.ArgumentParser(description="真实候选池上的 Top-k 评估（0 NPU）")
    ap.add_argument("--config", default="fulleval")
    ap.add_argument("--kernels", nargs="+", required=True)
    ap.add_argument("--old", default=str(DEFAULT_OLD))
    ap.add_argument("--new", default=str(DEFAULT_NEW))
    ap.add_argument("--extra", nargs="*", default=[],
                    help="额外模型，格式 tag=path，可重复")
    ap.add_argument("--base", default="old")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-pool", type=int, default=15)
    ap.add_argument("--max-pool", type=int, default=25)
    ap.add_argument("--run-ids", nargs="*", default=None,
                    help="只取指定 run_id 的日志")
    ap.add_argument("--tau-default", type=float, default=0.90,
                    help="事件里没有 tau 时用的值；默认 0.90 = "
                         "EAConfig.seed_diversity_threshold 的默认值")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    codes = load_codes()
    pools, pool_kernel, skipped = load_real_pools(
        a.config, a.kernels, a.min_pool, a.max_pool,
        set(a.run_ids) if a.run_ids else None)
    if not pools:
        print("❌ 没有找到大小符合的候选池。")
        return 1

    sizes = [len(v) for v in pools.values()]
    print("=" * 88)
    print("真实候选池上的 Top-k 评估（0 NPU）")
    print(f"  配置 {a.config}，算子 {len(a.kernels)} 个"
          + (f"，run_id {sorted(a.run_ids)}" if a.run_ids else ""))
    print(f"  认定为真实池的数量: {len(pools)}，池大小 中位 "
          f"{sorted(sizes)[len(sizes) // 2]}（min {min(sizes)} / max {max(sizes)}）")
    if skipped:
        print(f"  跳过: {dict(skipped)}")
    print("=" * 88)

    df, pool_info, st = build_df(pools, pool_kernel, codes, a.tau_default)
    n_raw = sum(v["n_raw"] for v in pool_info.values())
    complete = [k for k, v in pool_info.items() if v["complete"]]
    partial = [k for k, v in pool_info.items() if not v["complete"]]
    us = sorted(v["n_labeled"] for v in pool_info.values())
    n_runs = len({v for v in (df["run_id"] if not df.empty else [])})

    print(f"\n池内候选总数 {n_raw}，可用于打分的 {len(df)} "
          f"（丢弃 {st['n_dropped']}，{st['n_dropped'] / max(1, n_raw) * 100:.1f}%："
          f"缺 parent_latency 或 latency_us，标签无法定义）")
    print(f"  每池可用大小 中位 {us[len(us) // 2]}（min {us[0]} / max {us[-1]}）")
    print(f"\n【完整池判定】只有池内**全部 {min(sizes)} 个候选都有可信标签**才是完整池")
    print(f"  完整池 {len(complete)} 个 → 可算正式 P@{a.k}")
    print(f"  不完整池 {len(partial)} 个 → 只能作诊断，不能当作线上 "
          f"{min(sizes)} 选 {a.k} 的收益")
    print(f"  完整且父代延迟来源全部可信的池: "
          f"{sum(1 for v in pool_info.values() if v['complete'] and v['n_trusted'] == v['n_raw'])} 个")
    if df.empty:
        print("\n❌ 没有任何可打分候选。")
        return 1

    print(f"\n正样本率 {df['y'].mean():.1%}（可用于打分的 {len(df)} 条）")
    print(f"父代源码缺失（8 个代码特征退化）: {st['n_parent_code_missing']} / {len(df)} "
          f"({st['n_parent_code_missing'] / max(1, len(df)) * 100:.1f}%)  "
          f"← 分母为过滤后的样本数")
    print(f"参考父代来源: {dict(st['ref_src'])}")
    print(f"tau 缺失被填补: {st['n_tau_imputed']} / {len(df)} "
          f"（填补值 {a.tau_default}）")

    model_specs = [("old", a.old), ("new", a.new)]
    for spec in a.extra:
        if "=" in spec:
            t, pp = spec.split("=", 1)
            model_specs.append((t, pp))
    models = {}
    for tag, path in model_specs:
        if Path(path).is_file():
            models[tag] = joblib.load(path)
        else:
            print(f"⚠️ 找不到模型 {path}")
    if not models:
        return 1

    df_complete = df[df["pool_complete"]].copy()
    df_partial = df[~df["pool_complete"]].copy()
    raw_sizes = {k: v["n_raw"] for k, v in pool_info.items()}

    out = {"config": a.config, "k": a.k, "n_pools": len(pools),
           "pool_sizes": sizes, "n_raw": n_raw, "n_scored": int(len(df)),
           "n_dropped": int(st["n_dropped"]),
           "parent_code_missing": int(st["n_parent_code_missing"]),
           "ref_parent_source": {k: int(v) for k, v in st["ref_src"].items()},
           "tau_default": a.tau_default, "tau_imputed": int(st["n_tau_imputed"]),
           "n_complete_pools": len(complete), "n_partial_pools": len(partial),
           "base": a.base, "run_ids": sorted(a.run_ids) if a.run_ids else None}

    r_c = report_section(
        f"【A】完整池上的正式 P@{a.k}（{len(complete)} 池）", df_complete, models,
        a.k, a.base, pool_kernel, raw_sizes,
        note="这是唯一可以引用的 Top-k 收益：池内每个候选都有可信标签，"
             f"与线上 {min(sizes)} 选 {a.k} 场景一致。" if complete else
             "⚠️ 本批数据里没有完整池，正式 P@k **无法计算**。")
    r_p = report_section(
        f"【B】不完整池上的诊断性 P@{a.k}（{len(partial)} 池，不可引用）",
        df_partial, models, a.k, a.base, pool_kernel, raw_sizes,
        note="每个池只保留了部分有标签的候选，实际是『在剩余候选里选 "
             f"{a.k}』，难度低于真实 {min(sizes)} 选 {a.k}，"
             "因此这里的数字**不能**当作线上收益。")
    out["complete_pools"], out["partial_pools"] = r_c, r_p

    oracle = []
    for _, g in (df_complete.groupby("pool_key", sort=False)
                 if not df_complete.empty else []):
        if len(g) >= a.k:
            oracle.append(min(a.k, int(g["y"].sum())) / a.k)
    if oracle:
        print(f"\nOracle（完整池上用真值选，理论上界）P@{a.k} = "
              f"{float(np.mean(oracle)):.4f}")
        out["oracle_complete"] = float(np.mean(oracle))

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                               default=str), encoding="utf-8")
        print(f"\n✅ JSON: {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
