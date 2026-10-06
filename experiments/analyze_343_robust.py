#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.4.3 变异策略 —— 稳健性分析

相比 analyze_logs.analyze_343，本脚本额外提供：
  表A  样本溯源：说明每个样本为何被纳入/排除（可审计，避免"选择性汇报"质疑）
  表B  变异增益主指标 + bootstrap 95% CI（判断单步增益是否显著 ≠ 1）
  表C  最终加速比 GM(S) + Friedman 检验 + leave-one-out 敏感度
  表D  per-kernel 明细（暴露是否被单个算子绑架）
"""
import json
import math
import random
import sys
from collections import defaultdict

sys.path.insert(0, '/workspace/Agent')
from experiments import common
from experiments.analyze_logs import load_events, gm, gm_s_for, K15

GROUP = ["mut_uniform", "mut_aggressive", "mut_adaptive"]
LABEL = {"mut_uniform": "均匀变异", "mut_aggressive": "激进变异", "mut_adaptive": "自适应变异"}
random.seed(20261005)


def boot_ci(vals, fn, B=5000, alpha=0.05):
    """bootstrap 百分位置信区间"""
    if len(vals) < 3:
        return float('nan'), float('nan')
    s = sorted(fn([vals[random.randrange(len(vals))] for _ in vals]) for _ in range(B))
    return s[int(alpha / 2 * B)], s[int(1 - alpha / 2 * B)]


def wilson_ci(k, n, z=1.96):
    """Wilson 比例置信区间（小样本比正态近似稳健）"""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def per_kernel_S(cfg):
    """每算子加速比：同一算子多次 run 先取中位数"""
    per = defaultdict(list)
    for f in (common.RESULTS_DIR / cfg).glob("*__r*.remeasure.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not (r.get("success") and r.get("t_best_us")):
            continue
        # 与论文其余表格保持同一口径：只统计 K15 这 15 个规范算子
        if K15 and r.get("kernel") not in K15:
            continue
        man = common.MANIFEST_DIR / f"{r['kernel']}.json"
        if not man.exists():
            continue
        try:
            tb = json.loads(man.read_text(encoding="utf-8")).get("t_base_us")
        except (json.JSONDecodeError, OSError):
            continue
        if tb:
            per[r["kernel"]].append(tb / r["t_best_us"])
    out = {}
    for k, v in per.items():
        v.sort()
        out[k] = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
    return out


def friedman(data, cfgs, kern):
    """Friedman 检验 + Kendall's W。

    并列值必须取平均秩（mid-rank），否则会把完全相同的加速比强行分出名次，
    人为放大/缩小统计量。同时按标准公式做平局校正。
    df = k-1 = 2 时卡方生存函数恰为 exp(-x/2)，p 值精确。
    """
    R = defaultdict(float)
    tie_sum = 0
    n, kk = len(kern), len(cfgs)
    for k in kern:
        v = [data[c][k] for c in cfgs]
        order = sorted(range(kk), key=lambda i: -v[i])
        ranks = [0.0] * kk
        i = 0
        while i < kk:
            j = i
            while j + 1 < kk and v[order[j + 1]] == v[order[i]]:
                j += 1
            width = j - i + 1
            avg = (i + 1 + j + 1) / 2.0          # 占据名次 i+1..j+1，取平均
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            if width > 1:
                tie_sum += width ** 3 - width    # 平局校正项 Σ(t³-t)
            i = j + 1
        for idx, r in enumerate(ranks):
            R[cfgs[idx]] += r

    chi2 = 12 / (n * kk * (kk + 1)) * sum(R[c] ** 2 for c in cfgs) - 3 * n * (kk + 1)
    if tie_sum:
        denom = 1 - tie_sum / (n * kk * (kk * kk - 1))
        if denom > 0:
            chi2 /= denom
    p = math.exp(-chi2 / 2) if kk == 3 else None
    return chi2, p, R, chi2 / (n * (kk - 1)) if kk > 1 else float('nan')


def main():
    ev = {}
    for cfg in GROUP:
        ev[cfg] = [e for e in load_events(cfg)
                   if e.get("event") == "eval" and e.get("operation") == "mutation"
                   and (not K15 or e.get("kernel") in K15)]

    # ---------- 表A 样本溯源 ----------
    print("\n" + "=" * 78)
    print("表A  样本溯源（说明每个变异样本为何能/不能用于计算增益）")
    print("=" * 78)
    print(f"{'配置':<14}{'变异总数':>8}{'测得耗时':>10}{'父代耗时=父代':>14}"
          f"{'=显性':>8}{'=种子':>8}{'缺失(父代评估失败)':>18}")
    print("-" * 78)
    gains = {}
    for cfg in GROUP:
        src = defaultdict(int)
        for e in ev[cfg]:
            s = e.get("parent_latency_source") or (
                "parent" if e.get("parent_latency") else "legacy/none")
            src[s] += 1
        valid = sum(1 for e in ev[cfg] if e.get("latency_us"))
        gains[cfg] = [e["parent_latency"] / e["latency_us"] for e in ev[cfg]
                      if e.get("parent_latency") and e.get("latency_us")
                      and (e.get("parent_latency_source") in (None, "parent", "dominant"))]
        print(f"{cfg:<14}{len(ev[cfg]):>8}{valid:>10}{src['parent']:>14}"
              f"{src['dominant']:>8}{src['seed']:>8}{src['none'] + src['legacy/none']:>18}")

    # ---------- 表B 变异增益 ----------
    print("\n" + "=" * 78)
    print("表B  变异单步增益（主指标）  g = 父代耗时 / 子代耗时")
    print("=" * 78)
    print(f"{'配置':<14}{'n':>5}{'GM(g)':>9}{'GM 95%CI':>20}{'显著≠1':>8}"
          f"{'成功率':>8}{'成功率95%CI':>18}")
    print("-" * 78)
    rows = []
    for cfg in GROUP:
        g = gains[cfg]
        m = gm(g)
        lo, hi = boot_ci(g, gm)
        k = sum(1 for x in g if x > 1)
        slo, shi = wilson_ci(k, len(g))
        sig = "是" if not (lo <= 1.0 <= hi) else "否"
        rows.append(dict(config=cfg, n=len(g), gm_g=m, ci_lo=lo, ci_hi=hi,
                         success_rate=k / len(g) if g else 0, sr_lo=slo, sr_hi=shi,
                         significant=sig))
        print(f"{cfg:<14}{len(g):>5}{m:>9.4f}  [{lo:6.3f}, {hi:6.3f}]{sig:>8}"
              f"{k / len(g):>7.1%}  [{slo:5.1%}, {shi:5.1%}]")
    print("\n  '显著≠1' = 该变异策略的单步增益是否显著区别于 1（有益/有害）")
    print("  CI 重叠 = 三种策略无法区分")

    # ---------- 表C 最终加速比 + Friedman ----------
    print("\n" + "=" * 78)
    print("表C  最终加速比 GM(S) 与显著性")
    print("=" * 78)
    S = {c: per_kernel_S(c) for c in GROUP}
    kern = sorted(set.intersection(*[set(S[c]) for c in GROUP]))
    print(f"  共同算子数: {len(kern)}")
    print(f"\n{'配置':<14}{'n':>5}{'GM(S)':>9}")
    print("-" * 78)
    gmS = {}
    for c in GROUP:
        gmS[c] = gm([S[c][k] for k in kern])
        print(f"{c:<14}{len(kern):>5}{gmS[c]:>9.4f}")
    chi2, p, R, W = friedman(S, GROUP, kern)
    sig = "显著" if (p is not None and p < 0.05) else "不显著"
    print(f"\n  Friedman: chi2 = {chi2:.4f},  p = {p:.4f}  ({sig})")
    print(f"  Kendall's W = {W:.4f}  (0=无一致性, 1=完全一致)")
    for c in GROUP:
        print(f"    秩和 {c:<14} {R[c]:.1f}")

    # ---------- leave-one-out ----------
    print("\n  敏感度：逐个剔除算子后 GM(S) 排名是否翻转")
    print(f"  {'被剔除的算子':<44}{'剔除后最优':>14}{'是否翻转':>10}")
    print("  " + "-" * 68)
    base_best = max(GROUP, key=lambda c: gmS[c])
    flips = []
    for drop in kern:
        ks = [k for k in kern if k != drop]
        sub = {c: gm([S[c][k] for k in ks]) for c in GROUP}
        best = max(GROUP, key=lambda c: sub[c])
        if best != base_best:
            flips.append(drop)
            print(f"  {drop:<44}{best:>14}{'★翻转':>10}")
    if not flips:
        print(f"  （无翻转：剔除任一算子后最优仍为 {base_best}）")
    else:
        print(f"\n  ⚠ {len(flips)}/{len(kern)} 个算子会导致最优者翻转 —— 结论不稳健")

    # ---------- 表D per-kernel ----------
    print("\n" + "=" * 78)
    print("表D  各算子加速比明细")
    print("=" * 78)
    print(f"{'算子':<42}" + "".join(f"{LABEL[c]:>12}" for c in GROUP))
    print("-" * 78)
    for k in kern:
        print(f"{k:<42}" + "".join(f"{S[c][k]:>12.3f}" for c in GROUP))

    # ---------- 落盘 ----------
    out = dict(
        provenance={c: len(ev[c]) for c in GROUP},
        gains=rows,
        gm_s=gmS,
        friedman=dict(chi2=chi2, p=p, kendall_w=W, significant=p is not None and p < 0.05),
        leave_one_out_flips=flips,
        n_kernels=len(kern),
    )
    p_out = common.RESULTS_DIR / "table9_mutation_robust.json"
    p_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[analyze] 已写出 {p_out}")


if __name__ == "__main__":
    main()
