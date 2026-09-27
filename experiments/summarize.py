"""3.1 主实验聚合：按 Kernel 计算 F(k*) 与 S(k*)，输出 GM(F) / GM(S)。

聚合规则（对应 3.1 正文）：
  1. 同一方法多次独立重复时，先在 Kernel 内部对重复结果取中位数；
  2. F = T_seed / T_best，S = T_base / T_best；
  3. 几何平均仅在三种方法均有有效候选的 Kernel 子集上计算，并报告 N_used。

额外产出（供论文表4 使用）：
  - 中位数：几何平均易被极端值主导，中位数作为稳健性对照；
  - 按参考实现延迟分组（短/中/长）的分组统计：
    用于说明方法在不同规模 Kernel 上的收益差异；
  - Markdown 表格（results/table4.md）可直接粘贴进论文。

数学关系（可用作论文注脚）：
  GM(S) = GM(T_base/T_seed) × GM(F)
  即 GM(S) 比 GM(F) 多一个仅由数据集决定的常数因子。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json

from experiments import common

# 分组阈值（单位 us）：参考实现延迟
SHORT_MAX = 5.0
MID_MAX = 100.0

GROUP_LABEL = {
    'short': f'短 (<{SHORT_MAX:g}μs)',
    'mid': f'中 ({SHORT_MAX:g}~{MID_MAX:g}μs)',
    'long': f'长 (≥{MID_MAX:g}μs)',
}


def load_manifest(kernel):
    p = common.MANIFEST_DIR / f"{kernel}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_remeasures(kernel, method):
    out = []
    for p in sorted((common.RESULTS_DIR / method).glob(f"{kernel}__r*.remeasure.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("success") and rec.get("t_best_us"):
            out.append(rec)
    return out


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def group_of(t_base):
    """按参考实现延迟分组。"""
    if t_base is None:
        return None
    if t_base < SHORT_MAX:
        return 'short'
    if t_base < MID_MAX:
        return 'mid'
    return 'long'


def main(kernels=None):
    kernels = kernels or common.list_kernels()
    rows = []
    per_kernel = {}

    for kernel in kernels:
        man = load_manifest(kernel)
        if not man or not man.get("t_base_us") or not man.get("t_seed_us"):
            print(f"[Summarize] 跳过 {kernel}：manifest 缺失")
            continue

        t_base, t_seed = man["t_base_us"], man["t_seed_us"]
        row = {"kernel": kernel, "t_base": t_base, "t_seed": t_seed,
               "group": group_of(t_base)}
        per_kernel[kernel] = {}

        for method in common.METHODS:
            recs = load_remeasures(kernel, method)
            t_best = median([r["t_best_us"] for r in recs]) if recs else None
            if t_best:
                row[f"F_{method}"] = t_seed / t_best
                row[f"S_{method}"] = t_base / t_best
                per_kernel[kernel][method] = (t_seed / t_best, t_base / t_best)
            else:
                row[f"F_{method}"] = None
                row[f"S_{method}"] = None
                per_kernel[kernel][method] = None
        rows.append(row)

    usable = [k for k, v in per_kernel.items()
              if all(v.get(m) for m in common.METHODS)]
    n_used = len(usable)

    # ---------------- 逐 Kernel 明细 ----------------
    print(f"\n{'=' * 78}")
    print(f"3.1 主实验结果（参与统计 Kernel 数 = {n_used}/{len(rows)}）")
    print(f"{'=' * 78}")
    header = f"{'Kernel':<28}{'T_base':>10}{'T_seed':>10}"
    for m in common.METHODS:
        header += f"{'F_' + m:>12}{'S_' + m:>12}"
    print(header)
    print("-" * 78)
    for r in rows:
        line = f"{r['kernel']:<28}{r['t_base']:>10.2f}{r['t_seed']:>10.2f}"
        for m in common.METHODS:
            f, s = r.get(f"F_{m}"), r.get(f"S_{m}")
            line += f"{f:>12.4f}" if f else f"{'—':>12}"
            line += f"{s:>12.4f}" if s else f"{'—':>12}"
        print(line)
    print("-" * 78)

    def llm_calls_per_kernel(method):
        """该方法的每算子平均大模型调用数（表4 成本列）。"""
        d = common.RESULTS_DIR / method
        if not d.exists():
            return None
        vals = []
        for f in d.glob("*__r*.json"):
            if f.name.endswith(".remeasure.json"):
                continue
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if "llm_calls" in rec and rec.get("kernel") in per_kernel:
                vals.append(rec["llm_calls"])
        return sum(vals) / len(vals) if vals else None

    # ---------------- 表4 主表 ----------------
    summary = [{"method": "B0", "GM(F)": None, "GM(S)": 1.0,
                "median(F)": None, "median(S)": 1.0,
                "N_used": n_used, "LLM调用数/算子": None}]
    for m in common.METHODS:
        fs = [per_kernel[k][m][0] for k in usable]
        ss = [per_kernel[k][m][1] for k in usable]
        summary.append({
            "method": {"b1": "B1", "b2": "B2", "full": "FULL"}[m],
            "GM(F)": common.geometric_mean(fs),
            "GM(S)": common.geometric_mean(ss),
            "median(F)": median(fs),
            "median(S)": median(ss),
            "N_used": n_used,
            "LLM调用数/算子": llm_calls_per_kernel(m),
        })

    print(f"\n表4 整体优化效果（N_used={n_used}）")
    print(f"{'方法':<8}{'GM(F)':>10}{'GM(S)':>10}{'中位数F':>10}{'中位数S':>10}"
          f"{'参与算子数':>10}{'LLM调用数/算子':>14}")
    for s in summary:
        gmf = f"{s['GM(F)']:.4f}" if s.get("GM(F)") else "—"
        gms = f"{s['GM(S)']:.4f}" if s.get("GM(S)") else "—"
        mdf = f"{s['median(F)']:.4f}" if s.get("median(F)") else "—"
        mds = f"{s['median(S)']:.4f}" if s.get("median(S)") else "—"
        lc = f"{s['LLM调用数/算子']:.1f}" if s["LLM调用数/算子"] else "—"
        print(f"{s['method']:<8}{gmf:>10}{gms:>10}{mdf:>10}"
              f"{mds:>10}{s['N_used']:>10}{lc:>14}")

    # ---------------- 分组统计 ----------------
    # 分组：组别由各 Kernel 的 T_base 决定
    kg = {r['kernel']: r['group'] for r in rows}
    grouped = {'short': [], 'mid': [], 'long': []}
    for k in usable:
        g = kg.get(k)
        if g in grouped:
            grouped[g].append(k)

    print(f"\n按参考实现延迟分组的统计（各组的 GM(S) 与中位数）")
    print(f"{'组别':<18}{'Kernel数':>8}{'B1_GM(S)':>11}{'B2_GM(S)':>11}"
          f"{'FULL_GM(S)':>12}{'FULL_中位数S':>13}")
    group_rows = []
    for g in ('short', 'mid', 'long'):
        ks = grouped[g]
        if not ks:
            continue
        gvals = {m: [per_kernel[k][m][1] for k in ks] for m in common.METHODS}
        line = f"{GROUP_LABEL[g]:<18}{len(ks):>8}"
        for m in common.METHODS:
            line += f"{common.geometric_mean(gvals[m]):>11.4f}" \
                if m != 'full' else f"{common.geometric_mean(gvals[m]):>12.4f}"
        med_full = median(gvals['full'])
        line += f"{med_full:>13.4f}" if med_full else f"{'—':>13}"
        print(line)
        group_rows.append({
            "group": g, "label": GROUP_LABEL[g], "n_kernel": len(ks),
            "GM(S)_b1": common.geometric_mean(gvals['b1']),
            "GM(S)_b2": common.geometric_mean(gvals['b2']),
            "GM(S)_full": common.geometric_mean(gvals['full']),
            "median(S)_full": median(gvals['full']),
        })

    # ---------------- 落盘 ----------------
    common.ensure_dirs()
    with open(common.RESULTS_DIR / "per_kernel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["kernel"])
        w.writeheader()
        w.writerows(rows)
    (common.RESULTS_DIR / "table4.json").write_text(
        json.dumps({"summary": summary, "groups": group_rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown（可直接粘进论文）
    md = []
    md.append("### 表4 整体优化效果\n")
    md.append("| 方法 | GM(F) | GM(S) | 中位数F | 中位数S | 参与 Kernel 数 | LLM 调用数/算子 |")
    md.append("|---|---|---|---|---|---|---|")
    for s in summary:
        gmf = f"{s['GM(F)']:.4f}" if s.get("GM(F)") else "—"
        gms = f"{s['GM(S)']:.4f}" if s.get("GM(S)") else "—"
        mdf = f"{s['median(F)']:.4f}" if s.get("median(F)") else "—"
        mds = f"{s['median(S)']:.4f}" if s.get("median(S)") else "—"
        lc = f"{s['LLM调用数/算子']:.1f}" if s["LLM调用数/算子"] else "—"
        md.append(f"| {s['method']} | {gmf} | {gms} | {mdf} | "
                  f"{mds} | {s['N_used']} | {lc} |")
    md.append("\n### 表4b 按 Kernel 规模分组的加速比 GM(S)\n")
    md.append("| 组别 | Kernel 数 | B1 | B2 | FULL | FULL 中位数 |")
    md.append("|---|---|---|---|---|---|")
    for g in group_rows:
        md.append(f"| {g['label']} | {g['n_kernel']} | {g['GM(S)_b1']:.4f} | "
                  f"{g['GM(S)_b2']:.4f} | {g['GM(S)_full']:.4f} | {g['median(S)_full']:.4f} |")
    (common.RESULTS_DIR / "table4.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n[Summarize] 已写出 per_kernel.csv / table4.json / table4.md")
    return summary


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="汇总 3.1 主实验结果")
    _ap.add_argument("--kernels", "-k", nargs="+", default=None,
                     help="指定要汇总的 Kernel，默认汇总全部已有 manifest 的 Kernel")
    _args = _ap.parse_args()
    main(_args.kernels)
