"""从 JSONL 事件日志统计 3.3 / 3.4 的各项指标。

用法：
  python experiments/analyze_logs.py 3.3      # 有效候选率（表6）
  python experiments/analyze_logs.py 3.4.1    # 被选父代平均适应度（表7）
  python experiments/analyze_logs.py 3.4.2    # 交叉指标（表8）
  python experiments/analyze_logs.py 3.4.3    # 变异指标（表9）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict

from experiments import common
from experiments.ablation_components import GROUPS


def load_events(config_name):
    """读取某配置的全部事件（跨 Kernel、跨重复）。"""
    events = []
    for f in sorted(common.LOGS_DIR.glob(f"{config_name}__*__r*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def gm(values):
    return common.geometric_mean([v for v in values if v and v > 0])


def gm_s_for(config_name):
    """从复测结果算该配置的 GM(S)：每算子先取重复中位数，再跨算子取几何平均。"""
    per_kernel = defaultdict(list)
    cdir = common.RESULTS_DIR / config_name
    if not cdir.exists():
        return None, 0
    for f in sorted(cdir.glob("*__r*.remeasure.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not (rec.get("success") and rec.get("t_best_us")):
            continue
        man = common.MANIFEST_DIR / f"{rec['kernel']}.json"
        if not man.exists():
            continue
        t_base = json.loads(man.read_text(encoding="utf-8")).get("t_base_us")
        if t_base:
            per_kernel[rec["kernel"]].append(t_base / rec["t_best_us"])

    meds = []
    for vs in per_kernel.values():
        vs.sort()
        meds.append(vs[len(vs) // 2])
    return common.geometric_mean(meds), len(meds)


# ---------------- 3.3：有效候选率 ----------------
def analyze_33():
    rows = []
    for cfg in GROUPS["3.3"]:
        evs = load_events(cfg)
        if not evs:
            rows.append({"config": cfg, "note": "无数据"})
            continue
        n_eval = sum(1 for e in evs if e.get("event") == "eval")
        n_valid = sum(1 for e in evs if e.get("event") == "eval" and e.get("latency_us"))
        n_reject = sum(1 for e in evs if e.get("event") == "reject")
        gms, nk = gm_s_for(cfg)
        rows.append({
            "config": cfg,
            "n_eval": n_eval,
            "n_valid": n_valid,
            "n_ast_reject": n_reject,
            "valid_rate": n_valid / n_eval if n_eval else 0.0,
            "gm_s": gms,
            "n_kernels": nk,
        })

    print("\n表6 Profiling 证据与知识库检索策略消融（3.3）")
    print(f"{'配置':<10}{'送评测数':>10}{'有效数':>10}{'AST拒绝':>10}{'有效候选率':>12}{'GM(S)':>10}")
    print("-" * 64)
    for r in rows:
        if "note" in r:
            print(f"{r['config']:<10}{'—':>10}{'—':>10}{'—':>10}{'无数据':>12}{'—':>10}")
            continue
        sv = f"{r['gm_s']:.4f}" if r["gm_s"] else "—"
        print(f"{r['config']:<10}{r['n_eval']:>10}{r['n_valid']:>10}"
              f"{r['n_ast_reject']:>10}{r['valid_rate'] * 100:>11.1f}%{sv:>10}")
    _save("table6_valid_rate.json", rows)


# ---------------- 3.4.1：被选父代平均适应度 ----------------
def analyze_341():
    rows = []
    for cfg in GROUPS["3.4.1"]:
        evs = load_events(cfg)
        fits = [f for e in evs if e.get("event") == "selection"
                for f in (e.get("parent_fitness") or [])]
        gms, n = gm_s_for(cfg)
        rows.append({
            "config": cfg,
            "selection_mode": cfg.replace("sel_", ""),
            "n_selections": len([e for e in evs if e.get("event") == "selection"]),
            "mean_parent_fitness": sum(fits) / len(fits) if fits else None,
            "gm_s": gms,
            "n_kernels": n,
        })

    print("\n表7 父代选择策略对比（3.4.1）")
    print(f"{'配置':<18}{'选择事件数':>12}{'被选父代平均适应度':>20}{'GM(S)':>10}{'算子数':>8}")
    print("-" * 70)
    for r in rows:
        mf = f"{r['mean_parent_fitness']:.4f}" if r["mean_parent_fitness"] else "—"
        gv = f"{r['gm_s']:.4f}" if r["gm_s"] else "—"
        print(f"{r['config']:<18}{r['n_selections']:>12}{mf:>20}{gv:>10}{r['n_kernels']:>8}")
    _save("table7_selection.json", rows)


# ---------------- 3.4.2：交叉指标 ----------------
def analyze_342():
    rows = []
    for cfg in GROUPS["3.4.2"]:
        evs = [e for e in load_events(cfg)
               if e.get("event") == "eval" and e.get("operation") == "crossover"]
        total = len(evs)
        valid = [e for e in evs if e.get("latency_us")]
        better = [e for e in valid
                  if e.get("parent_latency") and e["latency_us"] < e["parent_latency"]]
        g = [e["parent_latency"] / e["latency_us"] for e in valid if e.get("parent_latency")]

        rows.append({
            "config": cfg,
            "n_crossover": total,
            "n_valid": len(valid),
            "valid_rate": len(valid) / total if total else 0.0,
            "n_better": len(better),
            "better_ratio": len(better) / len(valid) if valid else 0.0,
            "gm_g_cross": gm(g),
            "n_for_gm": len(g),
            "gm_s": gm_s_for(cfg)[0],
        })

    print("\n表8 交叉策略对比（3.4.2）")
    print(f"{'配置':<22}{'子代总数':>10}{'有效子代率':>12}{'优于主干比例':>14}{'GM(G_cross)':>14}{'GM(S)':>10}")
    print("-" * 84)
    for r in rows:
        gv = f"{r['gm_g_cross']:.4f}" if r["gm_g_cross"] else "—"
        sv = f"{r['gm_s']:.4f}" if r["gm_s"] else "—"
        print(f"{r['config']:<22}{r['n_crossover']:>10}{r['valid_rate'] * 100:>11.1f}%"
              f"{r['better_ratio'] * 100:>13.1f}%{gv:>14}{sv:>10}")
    _save("table8_crossover.json", rows)


# ---------------- 3.4.3：变异指标 ----------------
def analyze_343():
    rows = []
    for cfg in GROUPS["3.4.3"]:
        evs = [e for e in load_events(cfg)
               if e.get("event") == "eval" and e.get("operation") == "mutation"]
        total = len(evs)
        valid = [e for e in evs if e.get("latency_us")]
        pairs = [(e["parent_latency"], e["latency_us"]) for e in valid if e.get("parent_latency")]
        better = [p / c for p, c in pairs if c < p]
        g = [p / c for p, c in pairs]

        rows.append({
            "config": cfg,
            "n_mutation": total,
            "n_valid": len(valid),
            "valid_rate": len(valid) / total if total else 0.0,
            "n_better": len(better),
            "success_rate": len(better) / len(pairs) if pairs else 0.0,
            "gm_g_mut": gm(g),
            "n_for_gm": len(g),
            "gm_s": gm_s_for(cfg)[0],
        })

    print("\n表9 三种变异策略总体对比（3.4.3）")
    print(f"{'配置':<18}{'变异总数':>10}{'有效变异率':>12}{'优化成功率':>12}{'GM(G_mut)':>12}{'GM(S)':>10}")
    print("-" * 76)
    for r in rows:
        gv = f"{r['gm_g_mut']:.4f}" if r["gm_g_mut"] else "—"
        sv = f"{r['gm_s']:.4f}" if r["gm_s"] else "—"
        print(f"{r['config']:<18}{r['n_mutation']:>10}{r['valid_rate'] * 100:>11.1f}%"
              f"{r['success_rate'] * 100:>11.1f}%{gv:>12}{sv:>10}")

    # 补充材料：按 mutation_type 细分
    detail = defaultdict(lambda: {"total": 0, "valid": 0, "better": 0, "g": []})
    for cfg in GROUPS["3.4.3"]:
        for e in load_events(cfg):
            if e.get("event") != "eval" or e.get("operation") != "mutation":
                continue
            mt = e.get("mutation_type") or "unknown"
            d = detail[(cfg, mt)]
            d["total"] += 1
            if e.get("latency_us"):
                d["valid"] += 1
                if e.get("parent_latency"):
                    ratio = e["parent_latency"] / e["latency_us"]
                    d["g"].append(ratio)
                    if ratio > 1:
                        d["better"] += 1

    print("\n补充材料：按变异类型细分")
    print(f"{'配置':<18}{'变异类型':<22}{'调用':>6}{'有效':>6}{'成功率':>10}{'GM(G_mut)':>12}")
    print("-" * 76)
    for (cfg, mt), d in sorted(detail.items()):
        sr = d["better"] / len(d["g"]) if d["g"] else 0.0
        gv = f"{gm(d['g']):.4f}" if d["g"] else "—"
        print(f"{cfg:<18}{mt:<22}{d['total']:>6}{d['valid']:>6}{sr * 100:>9.1f}%{gv:>12}")

    _save("table9_mutation.json", rows)


def _save(name, rows):
    common.ensure_dirs()
    p = common.RESULTS_DIR / name
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[analyze] 已写出 {p}")


def main():
    ap = argparse.ArgumentParser(description="从日志统计 3.3 / 3.4 指标")
    ap.add_argument("target", choices=["3.3", "3.4.1", "3.4.2", "3.4.3"])
    args = ap.parse_args()

    {"3.3": analyze_33, "3.4.1": analyze_341,
     "3.4.2": analyze_342, "3.4.3": analyze_343}[args.target]()


if __name__ == "__main__":
    main()
