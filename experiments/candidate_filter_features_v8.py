#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.5 候选预筛选：第三轮特征工程（纯 CPU，不占 NPU）。

前两轮结论（v6/v7）
------------------
父子比值、代码相似度、父代未截断 speedup、rel_seed 等候选级特征，
在逻辑回归下 AUC 均未超过「仅数值特征」基线（0.6259）。但前两轮只用
逻辑回归，且未充分尝试非线性模型。本轮：

1. 整合前两轮的全部评测前可得特征；
2. 新增 4 个候选级特征（父子算子种类变化 / token 差异 / 计算强度比值 /
   是否改变 BLOCK 大小），全部评测前可得；
3. 同时评估 **逻辑回归** 与 **梯度提升树（HistGradientBoosting）** 两类学习器，
   用新三指标（AUC / PR-AUC / PR-AUC÷正样本率(lift) / 池内可区分度）对比。

主指标沿用 AUC（稳健、跨模型可比）；lift 用于消除正样本率基准。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_model_v5 import load_codes, SAMPLES  # noqa: E402
from experiments.candidate_filter_features_v6 import (                 # noqa: E402
    load_df, add_parent_speedup, add_code_features, CAT,
    ratio, jaccard, struct_feats, tie_rate)
from experiments.candidate_filter_features_v7 import add_v7_features   # noqa: E402

from sklearn.compose import ColumnTransformer                            # noqa: E402
from sklearn.impute import SimpleImputer                                 # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier               # noqa: E402
from sklearn.linear_model import LogisticRegression                      # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score        # noqa: E402
from sklearn.model_selection import GroupKFold                           # noqa: E402
from sklearn.pipeline import Pipeline                                    # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler          # noqa: E402

OUT = ROOT / "experiments" / "results" / "candidate_filter_features_v8.json"

RE_TL_CALL = re.compile(r"tl\.[a-z_]+")
RE_TOK = re.compile(r"[A-Za-z_]\w*")


# ---------------- v8 新增特征 ----------------
def _pick_parent(code_map, pids, pl, lat_of):
    if isinstance(pids, str):
        try:
            pids = json.loads(pids.replace("'", '"'))
        except Exception:
            pids = [pids]
    cands = [str(p) for p in (pids or []) if str(p) in code_map]
    if not cands:
        return ""
    if pl and len(cands) > 1:
        best, bd = None, None
        for c in cands:
            if c in lat_of:
                d = abs(lat_of[c] - pl)
                if bd is None or d < bd:
                    best, bd = c, d
        if best is not None and bd < 1e-6:
            return code_map[best]
    return code_map[cands[0]]


def add_v8_features(df):
    """第三轮新增：父子算子种类变化 / token 差异 / 计算强度比值 / 块是否改变。"""
    code_map = load_codes()
    lat_of = {}
    for cid, cl in zip(df["candidate_id"], df["child_latency"]):
        if isinstance(cl, (int, float)) and not np.isnan(cl):
            lat_of[str(cid)] = float(cl)
    recs = []
    for cid, pids, pl in zip(df["candidate_id"], df["parent_ids"],
                             df["parent_latency"]):
        ch = struct_feats(code_map.get(str(cid), ""))
        pc = struct_feats(_pick_parent(code_map, pids, pl, lat_of))
        cc, cp = code_map.get(str(cid), ""), _pick_parent(code_map, pids, pl, lat_of)
        ops_c = set(RE_TL_CALL.findall(cc))
        ops_p = set(RE_TL_CALL.findall(cp))
        changed_ops = len(ops_c.symmetric_difference(ops_p))
        tok_c, tok_p = set(RE_TOK.findall(cc)), set(RE_TOK.findall(cp))
        diff = len(tok_c.symmetric_difference(tok_p))
        denom = len(tok_c | tok_p) + 1
        diff_tokens = diff / denom
        ci_c = (ch["n_dot"] + 1) / (ch["n_load"] + ch["n_store"] + 1)
        ci_p = (pc["n_dot"] + 1) / (pc["n_load"] + pc["n_store"] + 1)
        r_compute = ratio(ci_c, ci_p)
        block_changed = 1.0 if ch["block"] != pc["block"] else 0.0
        recs.append(dict(changed_ops=float(changed_ops),
                         diff_tokens=float(diff_tokens),
                         r_compute=float(r_compute),
                         block_changed=block_changed))
    for k in recs[0]:
        df[k] = [r[k] for r in recs]
    print(f"[特征] v8 新增：changed_ops/diff_tokens/r_compute/block_changed 已加")
    return df


# ---------------- 特征集 ----------------
BASE = ["log_parent", "gen", "pop_size", "tau"]
SPEEDUP = ["log_parent_speedup"]
CFG = ["r_block", "r_warps", "r_stages"]
CODE = ["r_load", "r_store", "r_dot", "r_sum", "r_lines", "sim"]
V7 = ["rel_seed", "r_ops", "r_mask", "ops_abs", "block_abs"]
V8 = ["changed_ops", "diff_tokens", "r_compute", "block_changed"]

VARIANTS = [
    ("A 仅数值(基线)", BASE),
    ("B +speedup", BASE + SPEEDUP),
    ("C +CFG", BASE + SPEEDUP + CFG),
    ("D +CODE", BASE + SPEEDUP + CFG + CODE),
    ("E +V7", BASE + SPEEDUP + CFG + CODE + V7),
    ("F +V8", BASE + SPEEDUP + CFG + CODE + V7 + V8),
]


def make_pipe(num_cols, kind):
    num = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("s", StandardScaler())]) if kind == "lr" else \
        Pipeline([("i", SimpleImputer(strategy="median"))])  # 树对尺度不敏感
    clf = (LogisticRegression(max_iter=3000, class_weight="balanced")
           if kind == "lr" else
           HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0))
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", num, num_cols),
            ("cat", Pipeline([("i", SimpleImputer(strategy="constant",
                                                  fill_value="none")),
                              ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT),
        ])),
        ("clf", clf),
    ])


def evaluate(df, num_cols, kind, positive_rate):
    X = df[num_cols + CAT]
    y = df["y"].to_numpy()
    groups = df["kernel"].to_numpy()
    aucs, prs, lifts, dips = [], [], [], []
    for tr, te in GroupKFold(5).split(X, y, groups):
        pipe = make_pipe(num_cols, kind)
        pipe.fit(X.iloc[tr], y[tr])
        s = pipe.predict_proba(X.iloc[te])[:, 1]
        yt, gt = y[te], groups[te]
        aucs.append(roc_auc_score(yt, s))
        pra = average_precision_score(yt, s)
        prs.append(pra)
        lifts.append(pra / positive_rate)
        pooled = df.iloc[te].assign(score=s)
        vals = [tie_rate(sub["score"].to_numpy())
                for _, sub in pooled.groupby(["kernel", "gen"]) if len(sub) >= 5]
        dips.append(float(np.mean(vals)) if vals else np.nan)
    return dict(auc=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                pr_auc=float(np.mean(prs)),
                pr_lift=float(np.mean(lifts)),
                distinct_in_pool=float(np.nanmean(dips)))


def main():
    df = add_v8_features(
        add_v7_features(add_code_features(add_parent_speedup(load_df()))))
    positive_rate = float(df["y"].mean())
    print(f"\n样本 {len(df)}；正样本率 {positive_rate:.4f}；算子 {df['kernel'].nunique()}")

    out = {}
    for kind in ("lr", "gbm"):
        print("\n" + "=" * 96)
        print(f"学习器 = {kind.upper()}；指标 AUC / PR-AUC / PR-AUC/P(lift) / 池内可区分度")
        print("=" * 96)
        print(f"  {'变体':30s}{'AUC':>16s}{'PR-AUC':>12s}{'lift':>16s}"
              f"{'池内可区分度':>14s}")
        rows = []
        for name, cols in VARIANTS:
            r = evaluate(df, cols, kind, positive_rate)
            r["variant"], r["numeric_features"] = name, cols
            rows.append(r)
            print(f"  {name:30s}{r['auc']:.4f}±{r['auc_std']:.3f}"
                  f"{r['pr_auc']:>12.4f}{r['pr_lift']:>16.3f}"
                  f"{r['distinct_in_pool']:>13.1%}")
        best = max(rows, key=lambda r: r["auc"])
        print(f"  -> {kind.upper()} 最优(AUC): {best['variant']} "
              f"AUC={best['auc']:.4f} lift={best['pr_lift']:.3f}")
        out[kind] = rows
    out["positive_rate"] = positive_rate
    out["protocol"] = "按算子 5 折 GroupKFold；对外指标仅 AUC / PR-AUC / 池内可区分度"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {OUT}")


if __name__ == "__main__":
    main()
