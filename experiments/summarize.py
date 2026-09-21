"""3.1 主实验聚合：按 Kernel 计算 F(k*) 与 S(k*)，并输出 GM(F) / GM(S)。

聚合规则（对应 3.1 正文）：
  1. 同一方法多次独立重复时，先在 Kernel 内部对重复结果取中位数；
  2. F = T_seed / T_best，S = T_base / T_best；
  3. 几何平均仅在三种方法均有有效候选的 Kernel 子集上计算，并报告 N_used。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json

from experiments import common


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


def main(kernels=None):
    # 默认汇总全部已有 manifest 的 Kernel，缺失数据的会被自动跳过
    kernels = kernels or common.list_kernels()
    rows = []
    per_kernel = {}

    for kernel in kernels:
        man = load_manifest(kernel)
        if not man or not man.get("t_base_us") or not man.get("t_seed_us"):
            print(f"[Summarize] 跳过 {kernel}：manifest 缺失")
            continue

        t_base, t_seed = man["t_base_us"], man["t_seed_us"]
        row = {"kernel": kernel, "t_base": t_base, "t_seed": t_seed}
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

    # 三种方法均有有效候选的 Kernel 子集
    usable = [k for k, v in per_kernel.items()
              if all(v.get(m) for m in common.METHODS)]
    n_used = len(usable)

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

    summary = [{"method": "B0", "GM(F)": None, "GM(S)": 1.0,
                "N_used": n_used, "LLM调用数/算子": None}]
    for m in common.METHODS:
        fs = [per_kernel[k][m][0] for k in usable]
        ss = [per_kernel[k][m][1] for k in usable]
        summary.append({
            "method": {"b1": "B1", "b2": "B2", "full": "FULL"}[m],
            "GM(F)": common.geometric_mean(fs),
            "GM(S)": common.geometric_mean(ss),
            "N_used": n_used,
            "LLM调用数/算子": llm_calls_per_kernel(m),
        })

    print(f"\n表4 整体优化效果（几何平均，N_used={n_used}）")
    print(f"{'方法':<8}{'GM(F)':>12}{'GM(S)':>12}{'参与算子数':>10}{'LLM调用数/算子':>14}")
    for s in summary:
        gmf = f"{s['GM(F)']:.4f}" if s["GM(F)"] else "—"
        lc = f"{s['LLM调用数/算子']:.1f}" if s["LLM调用数/算子"] else "—"
        print(f"{s['method']:<8}{gmf:>12}{s['GM(S)']:>12.4f}"
              f"{s['N_used']:>10}{lc:>14}")

    # 落盘
    common.ensure_dirs()
    with open(common.RESULTS_DIR / "per_kernel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["kernel"])
        w.writeheader()
        w.writerows(rows)
    (common.RESULTS_DIR / "table4.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Summarize] 已写出 {common.RESULTS_DIR / 'per_kernel.csv'} 与 table4.json")
    return summary


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="汇总 3.1 主实验结果")
    _ap.add_argument("--kernels", "-k", nargs="+", default=None,
                     help="指定要汇总的 Kernel，默认汇总全部已有 manifest 的 Kernel")
    _args = _ap.parse_args()
    main(_args.kernels)
