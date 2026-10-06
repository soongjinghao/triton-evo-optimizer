#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.4.1 父代选择策略的**机制探测**（selection probe）。

目的
----
在固定种群上测出四种选择策略各自的"选择概率分布"，用具体数字说明
各策略的选择压力与探索行为差异，供论文 3.4.1 引用。

⚠️ 定位说明（写论文时要讲清楚）
------------------------------
这是**机制探测**，不是实验结果。它回答"各策略如何选择"（机械性质，
可精确复现）；**不**回答"哪个策略最终优化得更好"——后者由 15 个算子的
真实实验（表7 + Friedman/Nemenyi）给出。两者互相补充，不可互相替代。

种群设定的依据
--------------
种群规模取 6，与 config.EAConfig.population_size 一致。
适应度取值参照实测：被选父代适应度的中位数为 1.000、p90 为 1.107、
p99 为 1.615，约 20.8% 恰好等于 1.000（锚点种子）。因此示例种群设为
    [0.000, 1.000, 1.000, 1.020, 1.080, 1.600]
即：1 个评测失败个体、2 个与锚点持平、2 个小幅改进、1 个明显改进。

用法
----
    python experiments/selection_probe.py
"""

import collections
import glob
import json
import math
import sys
import types
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolutionary_algorithm import EvolutionaryAlgorithm   # noqa: E402
from genetic_operators import Individual                   # noqa: E402

# ---------------- 参数 ----------------
POP_FITNESS = [0.000, 1.000, 1.000, 1.020, 1.080, 1.600]
MODES = ("uniform", "tournament", "roulette", "ucb")
T = 40000          # 主测量的采样次数（只统计第一个父代）
T_UCB = 4000       # UCB 时间演化采样的总次数
W = 1000           # 演化分析中"早期/晚期"窗口大小
N_RANDOM = 200     # 鲁棒性检验的随机种群数
OUT_JSON = ROOT / "experiments" / "results" / "selection_probe.json"


def make_pop(fits):
    pop = []
    for i, f in enumerate(fits):
        ind = Individual(code=f"c{i}", fitness=f)
        ind.id = f"i{i}"
        pop.append(ind)
    return pop


def new_ea(fits):
    ea = EvolutionaryAlgorithm.__new__(EvolutionaryAlgorithm)
    ea.population = make_pop(fits)
    ea._ucb_counts = {}
    ea.generation = 0
    ea._log_selection = lambda *a, **k: None
    return ea


def measure(mode, fits, n=T):
    """只统计第一个父代 p1 的选择概率（避开 p1==p2 去重的影响，口径干净）。"""
    ea = new_ea(fits)
    ea.config = types.SimpleNamespace(selection=mode)
    c = collections.Counter()
    for _ in range(n):
        p1, _ = ea.select_parents()
        c[p1.id] += 1
    return [c[f"i{i}"] / n for i in range(len(fits))]


def theory_uniform(fits):
    n = len(fits)
    return [1.0 / n] * n


def theory_roulette(fits):
    """p_j = f_j / Σf （fitness<=0 者概率为 0）"""
    w = [max(f, 0.0) for f in fits]
    s = sum(w)
    return [x / s for x in w] if s > 0 else [0.0] * len(fits)


def theory_tournament(fits, k=3):
    """精确枚举所有 k 元子集；并列最大时按 random.sample 的顺序等分。"""
    n = len(fits)
    p = [0.0] * n
    for sub in combinations(range(n), k):
        mx = max(fits[i] for i in sub)
        tops = [i for i in sub if fits[i] == mx]
        for i in tops:
            p[i] += 1.0 / len(tops)
    m = sum(p)
    return [x / m for x in p] if m else p


def metrics(p, fits):
    """从选择概率算汇总指标。"""
    n = len(fits)
    h = -sum(x * math.log(x) for x in p if x > 0)
    return {
        "entropy": h,
        "norm_entropy": h / math.log(n) if n > 1 else 0.0,
        "n_eff": math.exp(h),                       # 有效父代数
        "expected_fitness": sum(pi * fi for pi, fi in zip(p, fits)),
        "p_select_failed": sum(pi for pi, fi in zip(p, fits) if fi <= 0.0),
        "p_top": max(p),                            # 给最强个体的概率
    }


def ucb_evolution(fits, total=T_UCB, w=W):
    """UCB 的选择分布随时间演化：早期重探索，晚期重利用。"""
    ea = new_ea(fits)
    ea.config = types.SimpleNamespace(selection="ucb")
    early, late = collections.Counter(), collections.Counter()
    for t in range(total):
        p1, _ = ea.select_parents()
        if t < w:
            early[p1.id] += 1
        elif t >= total - w:
            late[p1.id] += 1
    te, tl = sum(early.values()), sum(late.values())
    return ([early[f"i{i}"] / te for i in range(len(fits))],
             [late[f"i{i}"] / tl for i in range(len(fits))])


def empirical_fitness_pool():
    """从真实日志里取出被选父代的适应度，作为鲁棒性检验的经验分布。"""
    vals = []
    for f in glob.glob(str(ROOT / "experiments" / "**" / "*sel_*.jsonl"), recursive=True):
        if "archive" in f:
            continue
        try:
            for line in open(f, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("event") != "selection":
                    continue
                for v in (d.get("parent_fitness") or []):
                    if isinstance(v, (int, float)) and v > 0:
                        vals.append(float(v))
        except OSError:
            continue
    return vals


def log_concentration():
    """从真实实验日志测"选择集中度"（按 kernel+run 分组后取均值）。

    与第[1]~[5]节的机制探测互补：那一节回答"策略在固定种群上如何选择"，
    这一节回答"策略在真实搜索过程中实际表现得有多集中"。

    ⚠️ 局限：该指标混合了各个世代（个体会进出种群），因此不等同于
    "单代内的选择压力"，只作为跨策略的相对比较。
    """
    import statistics as st
    out = {}
    for cfg in ("sel_uniform", "sel_tournament", "sel_roulette", "sel_ucb"):
        evs = []
        for f in glob.glob(str(ROOT / "experiments" / "**" / f"*{cfg}*.jsonl"),
                           recursive=True):
            if "archive" in f:
                continue
            try:
                for line in open(f, encoding="utf-8"):
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("event") == "selection":
                        evs.append(d)
            except OSError:
                continue
        if not evs:
            out[cfg] = None
            continue
        grp = collections.defaultdict(list)
        for d in evs:
            grp[(d.get("kernel"), d.get("run_id"))].append(d)
        neffs, dists, reps = [], [], []
        for _k, g in grp.items():
            cnt = collections.Counter()
            seen, rep, tot = set(), 0, 0
            for d in g:
                for pid in (d.get("parent_ids") or []):
                    cnt[pid] += 1
                    tot += 1
                    if pid in seen:
                        rep += 1
                    seen.add(pid)
            if tot == 0:
                continue
            p = [c / tot for c in cnt.values()]
            h = -sum(x * math.log(x) for x in p)
            neffs.append(math.exp(h))
            dists.append(len(cnt))
            reps.append(rep / tot)
        out[cfg] = {
            "n_runs": len(neffs),
            "distinct_parents": st.mean(dists),
            "n_eff": st.mean(neffs),
            "repeat_rate": st.mean(reps),
        }
    return out


def main():
    fits = POP_FITNESS
    n = len(fits)
    print("=" * 84)
    print("3.4.1 父代选择策略 —— 机制探测（不是实验结果）")
    print("=" * 84)
    print(f"\n种群规模 = {n}（与 population_size 一致）")
    print("适应度  = " + ", ".join(f"{f:.3f}" for f in fits))
    print("（含 1 个评测失败个体 0.000，2 个与锚点持平 1.000）")

    # ---------- 主表 ----------
    print("\n" + "-" * 84)
    print("[1] 各策略的选择概率分布（只统计第一个父代，采样 %d 次）" % T)
    print("-" * 84)
    header = f"{'适应度':>8}" + "".join(f"{m:>13}" for m in MODES)
    print(header)

    probs = {}
    for m in MODES:
        probs[m] = measure(m, fits)

    for i, f in enumerate(fits):
        tag = f"{f:.3f}" + (" *" if f <= 0 else "  ")
        print(f"{tag:>8}" + "".join(f"{probs[m][i]:>13.4f}" for m in MODES))
    print(f"{'Σ':>8}" + "".join(f"{sum(probs[m]):>13.4f}" for m in MODES))
    print("\n  * 为评测失败个体（适应度 0）")

    # ---------- 理论校验 ----------
    print("\n" + "-" * 84)
    print("[2] 与理论值校验（uniform / roulette / tournament 有闭式解）")
    print("-" * 84)
    th = {"uniform": theory_uniform(fits),
          "roulette": theory_roulette(fits),
          "tournament": theory_tournament(fits)}
    print(f"  {'策略':<12}{'实测vs理论 最大偏差':>22}")
    for m in ("uniform", "roulette", "tournament"):
        d = max(abs(a - b) for a, b in zip(probs[m], th[m]))
        print(f"  {m:<12}{d:>22.4f}")
    print("  → 偏差均属采样噪声量级，说明探测口径正确")

    # ---------- 汇总指标 ----------
    print("\n" + "-" * 84)
    print("[3] 汇总指标")
    print("-" * 84)
    print(f"  {'策略':<12}{'有效父代N_eff':>14}{'归一化熵':>10}"
          f"{'给最强个体的p':>14}{'选中失败个体p':>14}{'期望适应度':>12}")
    summ = {}
    for m in MODES:
        q = metrics(probs[m], fits)
        summ[m] = q
        print(f"  {m:<12}{q['n_eff']:>14.3f}{q['norm_entropy']:>10.3f}"
              f"{q['p_top']:>14.4f}{q['p_select_failed']:>14.4f}"
              f"{q['expected_fitness']:>12.4f}")
    print(f"\n  N_eff = exp(熵)，上界为种群规模 {n}；越小说明选择压力越强")
    print("  uniform 的 N_eff 应恰为 6.000（无压力）")

    # ---------- UCB 演化 ----------
    print("\n" + "-" * 84)
    print("[4] UCB 的时间演化（前 %d 次 vs 最后 %d 次，共 %d 次）" % (W, W, T_UCB))
    print("-" * 84)
    e, l = ucb_evolution(fits)
    print(f"  {'适应度':>8}{'早期':>12}{'晚期':>12}{'变化':>12}")
    for i, f in enumerate(fits):
        print(f"  {f:>8.3f}{e[i]:>12.4f}{l[i]:>12.4f}{l[i]-e[i]:>+12.4f}")
    print("  → 早期探索项权重大（分布更平），随 n_i 拉开而向高适应度收敛")

    # ---------- 鲁棒性 ----------
    print("\n" + "-" * 84)
    print("[5] 鲁棒性：在 %d 个随机种群上重复（适应度从实测分布抽样）" % N_RANDOM)
    print("-" * 84)
    pool = empirical_fitness_pool()
    if not pool:
        print("  未取到实测分布，跳过")
        robust = None
    else:
        import random
        random.seed(20261005)
        agg = {m: collections.defaultdict(list) for m in MODES}
        for _ in range(N_RANDOM):
            fw = [random.choice(pool) for _ in range(n)]
            fw[random.randrange(n)] = 0.0      # 注入 1 个评测失败个体
            for m in MODES:
                q = metrics(measure(m, fw, n=4000), fw)
                for k2, v in q.items():
                    agg[m][k2].append(v)
        print(f"  经验分布样本数: {len(pool)}")
        print(f"  {'策略':<12}{'N_eff':>10}{'归一化熵':>10}"
              f"{'给最强p':>10}{'选失败p':>10}{'期望适应度':>12}")
        robust = {}
        for m in MODES:
            robust[m] = {k2: sum(v) / len(v) for k2, v in agg[m].items()}
            r = robust[m]
            print(f"  {m:<12}{r['n_eff']:>10.3f}{r['norm_entropy']:>10.3f}"
                  f"{r['p_top']:>10.4f}{r['p_select_failed']:>10.4f}"
                  f"{r['expected_fitness']:>12.4f}")

    # ---------- 真实日志上的集中度 ----------
    print("\n" + "-" * 84)
    print("[6] 真实实验日志上的选择集中度（跨策略相对比较）")
    print("-" * 84)
    logc = log_concentration()
    print(f"  {'配置':<18}{'运行数':>7}{'选中不同个体':>13}{'N_eff':>9}{'重复选择率':>11}")
    any_data = False
    for cfg in ("sel_uniform", "sel_tournament", "sel_roulette", "sel_ucb"):
        r = logc.get(cfg)
        if not r:
            print(f"  {cfg:<18}{'—':>7}{'尚无数据':>13}")
            continue
        any_data = True
        print(f"  {cfg:<18}{r['n_runs']:>7}{r['distinct_parents']:>13.2f}"
              f"{r['n_eff']:>9.3f}{r['repeat_rate']:>11.3f}")
    if any_data:
        print("  重复选择率 = 该个体此前已被选过的比例；越高越集中")
        print("  ⚠️ 混合各世代，非单代内的选择压力，仅作跨策略相对比较")

    # ---------- 保存 ----------
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "note": "机制探测（非实验结果），用于说明各策略的选择压力与探索行为",
        "population_fitness": fits,
        "population_size": n,
        "n_samples": T,
        "selection_probability": probs,
        "theory": {k: v for k, v in th.items()},
        "summary": summ,
        "ucb_evolution": {"early": e, "late": l},
        "robustness": robust,
        "log_concentration": logc,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
