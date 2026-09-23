"""3.2 种子相似度阈值敏感性实验。

三个子命令：
  survey  零成本普查：全 50 个 Kernel 的种子相似度分布、各 tau 的种子保留率、
          相邻设置之间的决策变化 Kernel 数 ΔN，并给出对 tau 敏感的 Kernel 子集
  run     对指定 Kernel 跑一组 tau 配置的搜索 + 复测
  table   汇总所有 tau_* 结果，输出表 5（种子保留率 / D_G0 / 有效候选率 / GM(S)）

用法示例：
  python experiments/ablation_tau.py survey
  python experiments/ablation_tau.py run --tau 0.85 --budget 5
  python experiments/ablation_tau.py table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from experiments import common

# 六种设置：0 表示不筛选（内置转换为阈值 1.01）
TAU_CONFIGS = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95]
SENSITIVE_RANGE = (0.70, 0.95)


def tau_tag(tau: float) -> str:
    return "none" if tau <= 0 else str(tau)


# ------------------------------- survey -------------------------------
def survey():
    kernels = common.list_kernels()
    sims = {}
    for k in kernels:
        s = common.seed_similarity(k)
        if s is not None:
            sims[k] = s

    if not sims:
        print("未找到任何含两个种子的 Kernel")
        return

    print(f"=== {len(sims)} 个 Kernel 的种子源码相似度 ===")
    for k, s in sorted(sims.items(), key=lambda x: x[1]):
        print(f"  {s:.3f}  {k}")

    print(f"\n=== 各设置的过滤情况（基于全部 {len(sims)} 个 Kernel）===")
    print(f"{'设置':<12}{'被过滤Kernel数':>14}{'种子保留率':>12}{'与上档ΔN':>12}")
    prev_filtered = None
    for tau in TAU_CONFIGS:
        if tau <= 0:
            filtered = 0
            label = "不筛选"
        else:
            filtered = sum(1 for s in sims.values() if s >= tau)
            label = f"tau={tau}"
        retention = (2 * len(sims) - filtered) / (2 * len(sims))
        delta = "—" if prev_filtered is None else str(abs(filtered - prev_filtered))
        print(f"{label:<12}{filtered:>14}{retention * 100:>11.1f}%{delta:>12}")
        prev_filtered = filtered

    sensitive = [k for k, s in sims.items()
                 if SENSITIVE_RANGE[0] <= s < SENSITIVE_RANGE[1]]
    # 相似度低于下界的样本在任何候选 tau 下都不会触发过滤，
    # 相似度不低于上界的样本在任何候选 tau 下都会被过滤，两者均不敏感
    below = [k for k, s in sims.items() if s < SENSITIVE_RANGE[0]]
    above = [k for k, s in sims.items() if s >= SENSITIVE_RANGE[1]]

    print(f"\n不敏感样本：相似度 < {SENSITIVE_RANGE[0]} 的 {len(below)} 个"
          f"（任何 tau 均不过滤），相似度 >= {SENSITIVE_RANGE[1]} 的 {len(above)} 个"
          f"（任何 tau 均过滤），合计 {len(below) + len(above)} 个")
    print(f"\n对 tau 敏感的 Kernel（相似度落在 "
          f"[{SENSITIVE_RANGE[0]}, {SENSITIVE_RANGE[1]})）：{len(sensitive)} 个")
    for k in sorted(sensitive, key=lambda x: sims[x]):
        print(f"  {sims[k]:.3f}  {k}")

    out = common.RESULTS_DIR / "tau_survey.json"
    common.ensure_dirs()
    out.write_text(json.dumps({
        "similarity": sims,
        "sensitive_kernels": sorted(sensitive),
        "insensitive_below": sorted(below),
        "insensitive_above": sorted(above),
        "tau_configs": TAU_CONFIGS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[survey] 已保存 {out}")


# ------------------------------- run -------------------------------
def run(tau: float, kernels, budget: int, run_id: int, seed: int,
        repeats: int, device: int, method: str = "full", skip_done: bool = True):
    from experiments.pipeline import stage_search, stage_remeasure

    common.ensure_dirs()
    tag = tau_tag(tau)
    # 用 tau_{tag} 作为配置名（同时决定结果目录与日志前缀），
    # 避免与 3.1 的 full 日志同名而互相覆盖
    cfg = f"tau_{tag}"
    for k in kernels:
        done = common.RESULTS_DIR / cfg / f"{k}__r{run_id}.remeasure.json"
        if skip_done and done.exists():
            print(f"[skip] tau={tag} / {k} / r{run_id} 已完成")
            continue
        print(f"\n===== run: {k} | tau={tau} =====")
        try:
            stage_search(k, cfg, run_id=run_id, seed=seed, budget=budget,
                         tau=tau, save_gen0=True)
        except Exception as e:
            print(f"[run] {k} 搜索失败: {type(e).__name__}: {e}")
            continue
        try:
            stage_remeasure(k, cfg, run_id=run_id, repeats=repeats,
                            device_id=device, tau=tau)
        except Exception as e:
            print(f"[run] {k} 复测失败: {type(e).__name__}: {e}")


# ------------------------------- table -------------------------------
def delta_n_table(sims):
    """相邻设置之间决策发生变化的 Kernel 数 ΔN（表5 区分度列）。"""
    out, prev = {}, None
    for tau in TAU_CONFIGS:
        filtered = 0 if tau <= 0 else sum(1 for s in sims.values() if s >= tau)
        out[tau] = None if prev is None else abs(filtered - prev)
        prev = filtered
    return out


def table(kernels=None):
    rows = []
    kernels = kernels or common.list_kernels()
    sims = {k: s for k in kernels
            if (s := common.seed_similarity(k)) is not None}
    deltas = delta_n_table(sims)
    retentions = {tau: ((2 * len(sims) - (0 if tau <= 0 else sum(
        1 for v in sims.values() if v >= tau))) / (2 * len(sims)) if sims else 0.0)
        for tau in TAU_CONFIGS}

    for tau in TAU_CONFIGS:
        tag = tau_tag(tau)
        tdir = common.RESULTS_DIR / f"tau_{tag}"
        if not tdir.exists():
            continue

        searches = sorted(tdir.glob("*__r*.json"))
        searches = [p for p in searches if not p.name.endswith(".remeasure.json")]
        remeas = sorted(tdir.glob("*__r*.remeasure.json"))

        per_kernel = {}
        for p in searches:
            rec = json.loads(p.read_text(encoding="utf-8"))
            per_kernel.setdefault(rec["kernel"], []).append(rec)

        d_g0s, retained, totals = [], 0, 0
        for k, recs in per_kernel.items():
            vals = [r.get("d_g0", 0.0) for r in recs]
            vals.sort()
            mid = vals[len(vals) // 2] if vals else 0.0
            d_g0s.append(mid)
            for r in recs:
                retained += r.get("n_seeds_retained", 0)
                totals += r.get("n_seeds_total", 0)

        n_valid = sum(r.get("n_valid", 0) for recs in per_kernel.values() for r in recs)
        n_eval = sum(r.get("n_eval", 0) for recs in per_kernel.values() for r in recs)

        # GM(S)：每 Kernel 取多次重复的中位数，再跨 Kernel 取几何平均
        ss = []
        for p in remeas:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if not (rec.get("success") and rec.get("t_best_us")):
                continue
            man = common.MANIFEST_DIR / f"{rec['kernel']}.json"
            if not man.exists():
                continue
            t_base = json.loads(man.read_text(encoding="utf-8")).get("t_base_us")
            if t_base:
                ss.append(t_base / rec["t_best_us"])

        rows.append({
            "setting": "不筛选" if tau <= 0 else f"{tau:.2f}",
            "n_kernels": len(per_kernel),
            # 种子保留率：基于全部算子的相似度分布直接统计（零成本，覆盖 50 个算子），
            # 与实际跑过哪些算子无关，以保证与 ΔN 列口径一致
            "retention": retentions.get(tau, 0.0),
            "delta_n": deltas.get(tau),
            "d_g0": sum(d_g0s) / len(d_g0s) if d_g0s else 0.0,
            "valid_rate": n_valid / n_eval if n_eval else 0.0,
            "gm_s": common.geometric_mean(ss) if ss else None,
            "n_for_gm": len(ss),
        })

    if not rows:
        print("未找到任何 tau_* 实验结果")
        return

    print(f"\n表5 种子相似度阈值敏感性")
    print(f"{'阈值tau':<10}{'种子保留率':>12}{'ΔN':>8}{'D_G0':>10}"
          f"{'有效候选率':>12}{'GM(S)':>10}{'N_used':>8}")
    print("-" * 72)
    for r in rows:
        gm = f"{r['gm_s']:.4f}" if r["gm_s"] else "—"
        dn = "—" if r["delta_n"] is None else str(r["delta_n"])
        print(f"{r['setting']:<10}{r['retention'] * 100:>11.1f}%{dn:>8}"
              f"{r['d_g0']:>10.4f}{r['valid_rate'] * 100:>11.1f}%{gm:>10}"
              f"{r['n_for_gm']:>8}")

    common.ensure_dirs()
    (common.RESULTS_DIR / "table5.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[table] 已写出 {common.RESULTS_DIR / 'table5.json'}")


def main():
    ap = argparse.ArgumentParser(description="3.2 种子相似度阈值敏感性实验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("survey", help="零成本普查相似度分布与种子保留率")

    rp = sub.add_parser("run", help="跑一组 tau 配置的搜索与复测")
    rp.add_argument("--tau", type=float, required=True, help="0 表示不筛选")
    rp.add_argument("--kernels", "-k", nargs="+", default=None,
                    help="默认使用对 tau 敏感的 Kernel 子集")
    rp.add_argument("--method", "-m", default="full")
    rp.add_argument("--budget", type=int, default=None,
                    help="NPU 评测预算；设为 5 可只跑到第 0 代（便宜层）")
    rp.add_argument("--run-id", "-r", type=int, default=0)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--repeats", type=int, default=5)
    rp.add_argument("--device", type=int, default=0)
    rp.add_argument("--no-skip", action="store_true", help="已完成的也重跑")

    tp = sub.add_parser("table", help="汇总输出表5")
    tp.add_argument("--kernels", "-k", nargs="+", default=None)

    args = ap.parse_args()

    if args.cmd == "survey":
        survey()
    elif args.cmd == "table":
        table(args.kernels)
    else:
        kernels = args.kernels
        if not kernels:
            sp = common.RESULTS_DIR / "tau_survey.json"
            if sp.exists():
                kernels = json.loads(sp.read_text(encoding="utf-8"))["sensitive_kernels"]
            if not kernels:
                kernels = common.PILOT_KERNELS
        run(args.tau, kernels, args.budget, args.run_id, args.seed,
            args.repeats, args.device, args.method,
            skip_done=not args.no_skip)


if __name__ == "__main__":
    main()
