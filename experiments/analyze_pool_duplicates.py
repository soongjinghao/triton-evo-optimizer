#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选池重复率与计时稳定性分析（0 NPU）。

为什么需要这个脚本
------------------
评审在 120 条新样本上直接统计后发现：

    每代生成 20 个候选，但平均每池只有约 9.5 种不同的源码（重复率约 52.5%），
    其中 24/120 的子代源码哈希与参考父代完全相同。

如果同一候选池反复生成近乎相同的实现，那么"再增加训练样本"或"延长进化代数"
都只是在重复消耗 LLM 与 NPU 资源。因此**先做去重与同代码复测诊断**，
比立刻扩到 3000 条训练样本更划算。

它回答
------
1. 每池有多少**不同**的源码？重复率、最大重复次数？
2. 有多少子代的源码与**参考父代完全相同**（即"生成"其实没改任何东西）？
3. 近重复（token Jaccard ≥ 阈值）占比？
4. **同一份源码**在同一池内测出的延迟差多少？（>10% 需要核对启动配置，
   不能先归因为噪声）
5. 池内真正改善"历史最优"的候选有几个？（衡量有效多样性，而不只是字符串差异）

用法
----
    python experiments/analyze_pool_duplicates.py \
        --samples experiments/dataset/samples_new_3op.jsonl \
        --json-out experiments/results/pool_duplicates_3op.json

只做第 4 项的复测建议（需要人工/真机确认，本脚本不跑 NPU）：
    对 `--spread-thresh` 命中的源码，固定输入与启动配置重复计时，
    判断差异来自测量波动还是实际运行配置不同。
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_features_shared import (      # noqa: E402
    load_codes, RE_TOK)

DEFAULT_CODES = ROOT / "experiments" / "dataset" / "codes"
DEFAULT_SAMPLES = ROOT / "experiments" / "dataset" / "samples_new_3op.jsonl"


def _pos(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _tokens(code):
    return set(RE_TOK.findall(code or ""))


def jaccard(a, b):
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    ap = argparse.ArgumentParser(description="候选池重复率与计时稳定性分析（0 NPU）")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--codes-dir", default=str(DEFAULT_CODES))
    ap.add_argument("--near-thresh", type=float, default=0.99,
                    help="近重复判定阈值（token Jaccard）。注意：同一算子的候选"
                         "共享大量骨架 token，该值天生偏高（0.95 时几乎全部命中），"
                         "只能与'完全相同哈希'一起读，不能单独当重复率用")
    ap.add_argument("--spread-thresh", type=float, default=0.10,
                    help="同一源码的延迟相对极差超过此值即需核对（默认 10%）")
    ap.add_argument("--max-examples", type=int, default=5)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.samples, encoding="utf-8") if l.strip()]
    codes = load_codes(a.codes_dir)

    pools = collections.defaultdict(list)
    for r in rows:
        key = r.get("pool_id") or f"{r.get('kernel')}|r{r.get('run_id')}|g{r.get('gen')}"
        pools[key].append(r)

    print("=" * 100)
    print("候选池重复率与计时稳定性分析（0 NPU）")
    print(f"  样本 {a.samples}: {len(rows)} 条")
    print(f"  代码库: {len(codes)} 条；候选池 {len(pools)} 个")
    print(f"  近重复阈值 Jaccard ≥ {a.near_thresh}；延迟极差阈值 > {a.spread_thresh:.0%}")
    print("=" * 100)

    per_pool = []
    for key in sorted(pools):
        mem = pools[key]
        n = len(mem)
        hashes = [str(r.get("child_code_sha256") or "") for r in mem]
        cnt = collections.Counter(hashes)
        n_distinct = len([h for h in cnt if h])
        # 与参考父代相同的源码数量
        same_as_parent = sum(
            1 for r in mem
            if r.get("child_code_sha256") and r.get("parent_code_sha256")
            and str(r["child_code_sha256"]) == str(r["parent_code_sha256"]))

        # 近重复：按 union-find 合并 Jaccard ≥ 阈值的源码
        texts = {h: codes.get(str(r.get("candidate_id")), "")
                 for h, r in zip(hashes, mem)}
        parent_of = {}
        def find(x):
            parent_of.setdefault(x, x)
            while parent_of[x] != x:
                parent_of[x] = parent_of[parent_of[x]]
                x = parent_of[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent_of[rx] = ry
        uniq = sorted({h for h in hashes if h})
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                if jaccard(texts.get(uniq[i], ""), texts.get(uniq[j], "")) >= a.near_thresh:
                    union(uniq[i], uniq[j])
        clusters = collections.defaultdict(int)
        for h in hashes:
            if h:
                clusters[find(h)] += 1
        n_near_dup = sum(v for v in clusters.values() if v > 1)

        # 同一源码的延迟一致性
        by_hash = collections.defaultdict(list)
        for r in mem:
            cl = _pos(r.get("child_latency"))
            if r.get("child_code_sha256") and cl:
                by_hash[str(r["child_code_sha256"])].append(cl)
        spreads = []
        for h, vs in by_hash.items():
            if len(vs) >= 2:
                spreads.append(((max(vs) - min(vs)) / min(vs), h, vs))
        spreads.sort(reverse=True)
        n_spread_bad = sum(1 for s, _, _ in spreads if s > a.spread_thresh)

        # 真正改善：相对池内父代最优 / 相对种子
        pars = [_pos(r.get("parent_latency")) for r in mem]
        pars = [p for p in pars if p]
        best_par = min(pars) if pars else None
        seeds = [_pos(r.get("seed_time")) for r in mem]
        seeds = [s for s in seeds if s]
        best_seed = min(seeds) if seeds else None
        beat_par = sum(1 for r in mem
                       if best_par and _pos(r.get("child_latency"))
                       and _pos(r["child_latency"]) < best_par)
        beat_seed = sum(1 for r in mem
                        if best_seed and _pos(r.get("child_latency"))
                        and _pos(r["child_latency"]) < best_seed)

        per_pool.append(dict(
            pool=key, n=n,
            n_distinct=n_distinct,
            dup_rate=1.0 - n_distinct / max(1, n),
            max_multiplicity=max(cnt.values()) if cnt else 0,
            same_as_parent=same_as_parent,
            near_dup=n_near_dup,
            near_dup_rate=n_near_dup / max(1, n),
            n_multi_measured=len(spreads),
            n_spread_bad=n_spread_bad,
            max_spread=(spreads[0][0] if spreads else 0.0),
            beat_best_parent=beat_par,
            beat_seed=beat_seed,
            worst_spreads=[dict(spread=round(s, 4), sha=h[:12],
                                latencies=[round(v, 4) for v in vs])
                           for s, h, vs in spreads[:a.max_examples]],
        ))

    print(f"\n{'pool':46s} {'n':>3s} {'不同源码':>8s} {'重复率':>8s} "
          f"{'最大重复':>8s} {'=父代':>6s} {'近重复':>7s} "
          f"{'同码>10%':>8s} {'胜父代最优':>10s} {'胜种子':>6s}")
    for p in per_pool:
        print(f"{p['pool'][:46]:46s} {p['n']:3d} {p['n_distinct']:8d} "
              f"{p['dup_rate']:8.1%} {p['max_multiplicity']:8d} "
              f"{p['same_as_parent']:6d} {p['near_dup']:7d} "
              f"{p['n_spread_bad']:8d} {p['beat_best_parent']:10d} "
              f"{p['beat_seed']:6d}")

    tot = sum(p["n"] for p in per_pool)
    dis = sum(p["n_distinct"] for p in per_pool)
    same_par = sum(p["same_as_parent"] for p in per_pool)
    near = sum(p["near_dup"] for p in per_pool)
    beat_par = sum(p["beat_best_parent"] for p in per_pool)
    print("\n" + "-" * 100)
    print("汇总")
    print(f"  候选总数 {tot}，不同源码 {dis} → 池内重复率 "
          f"{1 - dis / max(1, tot):.1%}")
    print(f"  每池平均不同源码数 {dis / max(1, len(per_pool)):.1f} / "
          f"{tot / max(1, len(per_pool)):.1f}")
    print(f"  与参考父代源码完全相同: {same_par} / {tot} "
          f"({same_par / max(1, tot):.1%})")
    print(f"  近重复（Jaccard ≥ {a.near_thresh}）: {near} / {tot} "
          f"({near / max(1, tot):.1%})")
    print("    ⚠️ token Jaccard 被共享骨架抬高（阈值 0.95 时几乎全部命中），仅作参考；")
    print("      真正衡量『生成没产出新东西』的是上面两个**精确哈希**指标。")
    print(f"  真正胜过池内父代最优的候选: {beat_par} / {tot} "
          f"({beat_par / max(1, tot):.1%})")

    bad = [(p, s) for p in per_pool for s in p["worst_spreads"]
           if s["spread"] > a.spread_thresh]
    print(f"\n  同一源码延迟相对极差 > {a.spread_thresh:.0%} 的组: {len(bad)} 组")
    for p, s in bad[:a.max_examples]:
        print(f"    {p['pool'][:40]:40s} sha={s['sha']} "
              f"极差 {s['spread']:.1%}  实测 {s['latencies']}")
    if bad:
        print("    → 归因为噪声之前，先固定输入与启动配置重复计时：")
        print("      若复测后仍有差异，说明同一份源码可能对应不同的外部启动配置。")
    print("-" * 100)

    if a.json_out:
        out = {"samples": str(a.samples), "n_samples": len(rows),
               "n_pools": len(per_pool),
               "near_thresh": a.near_thresh, "spread_thresh": a.spread_thresh,
               "summary": {"n_candidates": tot, "n_distinct": dis,
                           "dup_rate": 1 - dis / max(1, tot),
                           "same_as_parent": same_par, "near_dup": near,
                           "beat_best_parent": beat_par},
               "per_pool": per_pool}
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"✅ JSON: {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
