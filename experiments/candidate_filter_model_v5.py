#!/usr/bin/env python3
"""表11 预筛选模型 v5：评测前可得的静态特征对比。

本轮回答的问题：把特征换成"子代与父代的静态特征"是否更好？
以及父代 Profiling 是否值得加入。

四组输入对比：
  1. 基线（数值 + 父代历史）
  2. + 子代静态结构特征
  3. + 父子代静态差异（diff）特征      <- 本轮重点
  4. + 父代 Profiling（合法但已验证增益有限，作对照）

特征合法性：全部取自候选/父代源码与搜索状态，评测前即可获得。
不使用子代 child_profile（需 NPU profiling，评测后才产生）。

评估协议：仅保留候选池 > 13 的 (kernel, method) 组（否则 top-13 等于全选，
指标失真）；按组的 GroupKFold 5 折。
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "experiments" / "dataset" / "samples.jsonl"
CODES = ROOT / "experiments" / "dataset" / "codes"
OUT_JSON = ROOT / "experiments" / "results" / "table11_candidate_filter_v5.json"
OUT_MD = ROOT / "experiments" / "results" / "table11_candidate_filter_v5.md"
BUDGET = 13

STRUCT_PATTERNS = {
    "n_load": r"tl\.load",
    "n_store": r"tl\.store",
    "n_mask": r"mask",
    "n_block": r"BLOCK",
    "n_constexpr": r"tl\.constexpr",
    "n_arange": r"tl\..arange",
    "n_program_id": r"tl\.program_id",
    "n_where": r"tl\.where",
    "n_multiple_of": r"tl\.multiple_of",
    "n_max_contiguous": r"tl\.max_contiguous",
    "n_if": r"\bif\b",
    "n_for": r"\bfor\b",
    "n_sum": r"tl\.sum",
    "n_max": r"tl\.max",
    "n_atomic": r"tl\.atomic",
    "n_dot": r"tl\.dot",
}
PARAM_PATTERNS = {
    "num_warps": r"num_warps\s*=\s*(\d+)",
    "num_stages": r"num_stages\s*=\s*(\d+)",
}


def struct_feats(code):
    """从源码抽取结构计数与规模特征。"""
    if not code:
        return {k: 0 for k in STRUCT_PATTERNS} | {
            "code_len": 0, "code_lines": 0}
    f = {k: len(re.findall(p, code)) for k, p in STRUCT_PATTERNS.items()}
    f["code_len"] = len(code)
    f["code_lines"] = code.count("\n") + 1
    for name, pat in PARAM_PATTERNS.items():
        m = re.findall(pat, code)
        f[name] = float(m[-1]) if m else np.nan
    return f


def jaccard(a, b):
    """源码 token 集合相似度，衡量父子代改动幅度。"""
    if not a or not b:
        return np.nan
    ta = set(re.findall(r"[A-Za-z_]\w*", a))
    tb = set(re.findall(r"[A-Za-z_]\w*", b))
    if not ta or not tb:
        return np.nan
    return len(ta & tb) / len(ta | tb)


def load_codes():
    m = {}
    for p in CODES.rglob("*.py"):
        try:
            m[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    return m


def main():
    rows = [json.loads(l) for l in open(SAMPLES, encoding="utf-8") if l.strip()]
    # 每个候选被评测时的 profiling（父代查自己当年那条记录 -> 评测前可得）
    prof = {}
    for r in rows:
        cid, cp = r.get("candidate_id"), r.get("child_profile")
        if cid and isinstance(cp, dict) and cp:
            prof[cid] = cp

    code_map = load_codes()
    print(f"[数据] 候选代码库: {len(code_map)} 份")

    df = pd.DataFrame(rows)
    df = df[df["run_id"].isin([0, "0"])].copy()
    for c in ("tau", "gen", "pop_size", "parent_latency", "child_latency"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["parent_latency"] > 0) & (df["child_latency"] > 0)].copy()
    df = df[df["child_latency"] / df["parent_latency"] <= 10].copy()
    df["y"] = (df["child_latency"] < df["parent_latency"]).astype(int)
    df["log_parent"] = np.log1p(df["parent_latency"])
    df["tau"] = df["tau"].fillna(-1.0)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["parent_key"] = df["kernel"].astype(str) + "|" + df["parent_ids"].astype(str)
    df["parent_tried"] = df.groupby("parent_key").cumcount()
    past = df.groupby("parent_key")["y"]
    df["parent_prior_sum"] = past.cumsum() - df["y"]
    df["parent_prior_rate"] = np.where(
        df["parent_tried"] > 0,
        df["parent_prior_sum"] / df["parent_tried"].replace(0, np.nan),
        np.nan).astype(float)
    df["parent_prior_rate"] = df["parent_prior_rate"].fillna(-1.0)

    # ---- 子代 / 父代 静态特征 ----
    child_codes = [code_map.get(str(c), "") for c in df["candidate_id"]]

    def parent_code(pid):
        if isinstance(pid, list):
            for p in pid:
                if str(p) in code_map:
                    return code_map[str(p)]
            return ""
        return code_map.get(str(pid), "") if isinstance(pid, str) else ""

    parent_codes = [parent_code(p) for p in df["parent_ids"]]
    have_child = sum(1 for c in child_codes if c)
    have_parent = sum(1 for c in parent_codes if c)
    print(f"[数据] 子代代码可得 {have_child}/{len(df)}；"
          f"父代代码可得 {have_parent}/{len(df)} "
          f"({have_parent/len(df):.1%})")

    cs = pd.DataFrame([struct_feats(c) for c in child_codes], index=df.index)
    ps_ = pd.DataFrame([struct_feats(c) for c in parent_codes], index=df.index)

    struct_cols = list(STRUCT_PATTERNS.keys()) + ["code_len", "code_lines"]
    for c in struct_cols:
        df["c_" + c] = cs[c]
        df["p_" + c] = ps_[c]
        df["d_" + c] = df["c_" + c] - df["p_" + c]
    for name in PARAM_PATTERNS:
        df["c_" + name] = cs[name]
        df["p_" + name] = ps_[name]
        df["d_" + name] = df["c_" + name] - df["p_" + name]

    df["sim"] = [jaccard(a, b) for a, b in zip(child_codes, parent_codes)]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["len_ratio"] = np.where(df["p_code_len"] > 0,
                                   df["c_code_len"] / df["p_code_len"], np.nan)

    # 父代 Profiling（合法：父代已评测过）
    def parent_prof(pid):
        if isinstance(pid, list):
            for p in pid:
                if p in prof:
                    return prof[p]
            return {}
        return prof.get(pid, {}) if isinstance(pid, str) else {}

    ppl = [parent_prof(p) for p in df["parent_ids"]]
    for col in ("vec", "scalar", "mte2", "mte3"):
        df["pp_" + col] = pd.to_numeric([d.get(col) for d in ppl], errors="coerce")
    df["pp_block_dim"] = pd.to_numeric([d.get("block_dim") for d in ppl],
                                       errors="coerce")

    # ---- 只保留真正有筛选空间的组 ----
    cnt = df.groupby(["kernel", "method"])["y"].transform("size")
    big = df[cnt > BUDGET].copy()
    big["grp"] = big["kernel"].astype(str) + "|" + big["method"].astype(str)
    y = big["y"].to_numpy()
    grp = big["grp"].to_numpy()
    print(f"[评估] 候选池>{BUDGET} 的组: {big['grp'].nunique()} 组 / {len(big)} 条，"
          f"正例率 {y.mean():.1%}")

    BASE = ["log_parent", "gen", "pop_size", "tau",
            "parent_tried", "parent_prior_rate"]
    CHILD_S = ["c_" + c for c in struct_cols] + ["c_num_warps", "c_num_stages"]
    DIFF = (["d_" + c for c in struct_cols]
            + ["d_num_warps", "d_num_stages", "sim", "len_ratio"])
    PPROF = ["pp_vec", "pp_scalar", "pp_mte2", "pp_mte3", "pp_block_dim"]

    configs = [
        ("1. 基线（数值+历史）", BASE),
        ("2. +子代静态结构", BASE + CHILD_S),
        ("3. +父子静态差异(diff)", BASE + CHILD_S + DIFF),
        ("4. +父代Profiling(对照)", BASE + CHILD_S + DIFF + PPROF),
    ]

    def pk(yt, sc, gg, k=BUDGET):
        t = pd.DataFrame({"y": yt, "s": sc, "g": gg})
        h = sel = pos = 0
        for _, g in t.groupby("g"):
            top = g.nlargest(min(k, len(g)), "s")
            h += int(top["y"].sum())
            sel += len(top)
            pos += int(g["y"].sum())
        return (h / sel if sel else 0.0), (h / pos if pos else 0.0)

    results = []
    for name, cols in configs:
        X = big[cols].to_numpy(dtype=float)
        aucs, prs, pat, rat = [], [], [], []
        for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
            m = Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler()),
                          ("c", LogisticRegression(max_iter=3000))])
            m.fit(X[tr], y[tr])
            s = m.predict_proba(X[te])[:, 1]
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], s))
            prs.append(average_precision_score(y[te], s))
            p, r = pk(y[te], s, grp[te])
            pat.append(p)
            rat.append(r)
        res = {"variant": name, "auc": float(np.mean(aucs)),
               "auc_std": float(np.std(aucs)), "pr_auc": float(np.mean(prs)),
               "p_at_13": float(np.mean(pat)), "r_at_13": float(np.mean(rat))}
        results.append(res)
        print(f"  {name:<24} AUC={res['auc']:.4f}±{res['auc_std']:.4f}  "
              f"PR-AUC={res['pr_auc']:.4f}  P@13={res['p_at_13']:.4f}  "
              f"R@13={res['r_at_13']:.4f}")

    rng = np.random.default_rng(0)
    rp = []
    for _ in range(300):
        h = sel = 0
        for _, g in pd.DataFrame({"y": y, "g": grp}).groupby("g"):
            n = len(g)
            take = min(BUDGET, n)
            idx = rng.choice(n, size=take, replace=False)
            h += int(g["y"].to_numpy()[idx].sum())
            sel += take
        rp.append(h / sel)
    rand13 = float(np.mean(rp))
    print(f"\n[随机基线] P@13={rand13:.4f} ± {np.std(rp):.4f}")

    OUT_JSON.write_text(json.dumps({
        "n_samples": int(len(big)), "n_groups": int(big["grp"].nunique()),
        "positive_rate": round(float(y.mean()), 4), "budget": BUDGET,
        "random_baseline_p13": round(rand13, 4),
        "note": "仅评测前可得特征；不含子代 child_profile",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 表11（v5：静态特征对比）候选预筛选离线评估", "",
             f"- 样本：{len(big)} 条 / {big['grp'].nunique()} 组（仅候选池>{BUDGET}）",
             f"- 正样本比例：{y.mean():.1%}",
             "- 特征：全部为评测前可得（源码静态特征 + 搜索状态 + 父代历史）",
             "- 不含子代 child_profile（评测后才产生）", "",
             "| 输入特征 | AUC | PR-AUC | P@13 | R@13 |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['variant']} | {r['auc']:.4f} | {r['pr_auc']:.4f} | "
                     f"{r['p_at_13']:.4f} | {r['r_at_13']:.4f} |")
    lines.append(f"| 随机排序基线 | — | — | {rand13:.4f} | — |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[输出] {OUT_MD}")


if __name__ == "__main__":
    main()
