#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.4.1 父代选择策略：多策略统计检验（Friedman + Nemenyi）。

为什么要换检验方法
------------------
策略数由 2 增至 4 后，若继续逐对做 Wilcoxon 符号秩检验，比较次数由 1 次增至
C(4,2) = 6 次，Bonferroni 阈值降至 0.05/6 = 0.0083，原有的 p = 0.026 将不再显著。
因此改用算法对比的标准方案（Demsar, 2006）：

  1) Friedman 检验：先整体检验"各策略之间是否存在差异"；
  2) Nemenyi 事后检验：整体显著后再做两两比较，用临界差值 CD 判读。

同时预先声明主比较为「UCB vs 轮盘赌」（本文要验证的假设：显式补偿探索是否优于
纯适应度加权）。主比较是事前指定的单一假设，不承担多重比较校正。

口径
----
与 analyze_logs.gm_s_for 保持一致：每个算子先取重复测量的中位数作为该算子的 S，
再跨算子取几何平均得到 GM(S)。Friedman 在"算子 × 策略"的配对矩阵上进行。

用法
----
    python experiments/stats_341.py
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import common                       # noqa: E402
from experiments.ablation_components import CONFIG_341  # noqa: E402

ALPHA = 0.05
OUT_JSON = ROOT / "experiments" / "results" / "stats_341.json"

# 主比较（事前指定，不做多重比较校正）
PRIMARY = ("sel_ucb", "sel_roulette")

# Nemenyi 临界值表（α=0.05, df=∞），用于 studentized_range 不可用时的兜底
Q_TABLE_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def per_kernel_s(cfg):
    """复算某配置的逐算子 S：与 analyze_logs.gm_s_for 完全同口径。"""
    per = defaultdict(list)
    cdir = common.RESULTS_DIR / cfg
    if not cdir.exists():
        return {}
    for f in sorted(cdir.glob("*__r*.remeasure.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not (rec.get("success") and rec.get("t_best_us")):
            continue
        man = common.MANIFEST_DIR / f"{rec['kernel']}.json"
        if not man.exists():
            continue
        try:
            t_base = json.loads(man.read_text(encoding="utf-8")).get("t_base_us")
        except (json.JSONDecodeError, OSError):
            continue
        if t_base:
            per[rec["kernel"]].append(t_base / rec["t_best_us"])

    out = {}
    for k, vs in per.items():
        vs.sort()
        out[k] = vs[len(vs) // 2]      # 取重复测量的中位数
    return out


def q_crit(k, alpha=ALPHA):
    """Nemenyi 用的学生化极差临界值 q(α, k, df=∞)。"""
    try:
        return float(studentized_range.ppf(1 - alpha, k, np.inf))
    except Exception:
        return Q_TABLE_05.get(k, 2.569)


def main():
    cfgs = list(CONFIG_341)
    data = {c: per_kernel_s(c) for c in cfgs}

    print("=" * 78)
    print("3.4.1 父代选择策略 —— 多策略统计检验")
    print("=" * 78)

    print("\n[1] 数据完整性（逐算子配对，缺失算子将被剔除）")
    print(f"    {'配置':<18}{'有结果算子数':>14}")
    for c in cfgs:
        print(f"    {c:<18}{len(data[c]):>14}")

    avail = [c for c in cfgs if data[c]]
    missing = [c for c in cfgs if not data[c]]
    if missing:
        print(f"\n    ⚠️ 尚无结果，暂不参与检验: {', '.join(missing)}")
    if len(avail) < 2:
        print("\n    可用配置不足 2 个，无法检验。")
        return

    kernels = sorted(set.intersection(*[set(data[c]) for c in avail]))
    if not kernels:
        print("\n    各配置之间没有共同算子，无法配对。")
        return

    M = np.array([[data[c][k] for c in avail] for k in kernels], dtype=float)
    n, k = M.shape

    print(f"\n    配对算子数 n = {n}，参与策略数 k = {len(avail)}")

    # ---------- GM(S) ----------
    print("\n[2] GM(S)（每算子取重复中位数，再跨算子取几何平均）")
    print(f"    {'配置':<18}{'GM(S)':>10}")
    gms = {}
    for j, c in enumerate(avail):
        gms[c] = common.geometric_mean(list(M[:, j]))
        print(f"    {c:<18}{gms[c]:>10.4f}")

    if k < 3:
        print("\n    策略数 < 3，Friedman/Nemenyi 不适用（改用 Wilcoxon 即可）。")
        _save({"n_kernels": n, "configs": avail, "gm_s": gms})
        return

    # ---------- Friedman ----------
    stat, p_fr = friedmanchisquare(*[M[:, j] for j in range(k)])
    kendall_w = stat / (n * (k - 1))
    print("\n[3] Friedman 检验（整体）")
    print(f"    chi2 = {stat:.4f},  p = {p_fr:.4f}")
    print(f"    Kendall's W = {kendall_w:.4f}  (0=无一致性, 1=完全一致)")
    overall = "各策略间存在显著差异" if p_fr < ALPHA else "未检出各策略间的整体差异"
    print(f"    → {overall}（α = {ALPHA}）")

    # ---------- Nemenyi ----------
    ranks = np.apply_along_axis(lambda row: rankdata(-row), 1, M)  # 1 = 最优
    R = ranks.mean(axis=0)
    qa = q_crit(k)
    cd = qa * np.sqrt(k * (k + 1) / (6.0 * n))

    print("\n[4] Nemenyi 事后检验")
    print(f"    平均秩（越小越优）:")
    for j, c in enumerate(avail):
        print(f"      {c:<18}{R[j]:>8.3f}")
    print(f"    临界差值 CD = {qa:.3f} × sqrt({k}×{k + 1}/(6×{n})) = {cd:.3f}")
    print(f"\n    {'比较':<40}{'|ΔR|':>8}{'q':>8}{'p':>10}{'结论':>8}")
    print("    " + "-" * 74)

    pairs = []
    for i, j in combinations(range(k), 2):
        diff = abs(R[i] - R[j])
        q_obs = diff / np.sqrt(k * (k + 1) / (6.0 * n))
        try:
            pv = float(studentized_range.sf(q_obs, k, np.inf))
        except Exception:
            pv = float("nan")
        sig = diff > cd
        pairs.append({
            "a": avail[i], "b": avail[j], "mean_rank_a": float(R[i]),
            "mean_rank_b": float(R[j]), "abs_diff_rank": float(diff),
            "q": float(q_obs), "p": pv, "significant": bool(sig),
        })
        print(f"    {avail[i] + ' vs ' + avail[j]:<40}{diff:>8.3f}{q_obs:>8.3f}"
              f"{pv:>10.4f}{'显著' if sig else '不显著':>8}")

    # ---------- 主比较（事前指定，不校正） ----------
    print("\n[5] 主比较（事前指定：UCB vs 轮盘赌，Wilcoxon 符号秩，不校正）")
    if all(c in avail for c in PRIMARY):
        ia, ib = avail.index(PRIMARY[0]), avail.index(PRIMARY[1])
        x, y = M[:, ia], M[:, ib]
        wins = int((x > y).sum())
        try:
            w_stat, w_p = wilcoxon(x, y)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        print(f"    {PRIMARY[0]} vs {PRIMARY[1]}: {wins} 胜 {n - wins} 负, "
              f"W = {w_stat}, p = {w_p:.4f}")
        primary = {"a": PRIMARY[0], "b": PRIMARY[1], "wins": wins,
                   "losses": n - wins, "W": float(w_stat), "p": float(w_p)}
    else:
        print(f"    缺少 {' 或 '.join(PRIMARY)} 的结果，暂无法比较。")
        primary = None

    _save({
        "n_kernels": n, "configs": avail, "gm_s": gms,
        "friedman": {"chi2": float(stat), "p": float(p_fr), "kendall_w": float(kendall_w)},
        "nemenyi": {"q_crit": float(qa), "cd": float(cd),
                    "mean_ranks": {c: float(R[j]) for j, c in enumerate(avail)},
                    "pairs": pairs},
        "primary": primary,
    })


def _save(obj):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
