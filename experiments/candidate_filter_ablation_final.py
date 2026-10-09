#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.5 候选预筛选：论文式特征消融（基于最终模型口径，纯 CPU，不占 NPU）。

复用 candidate_filter_model_final 的 load_df / add_pc_features，
保证父子代码相对特征的构造口径与在线最终模型完全一致。

消融路径严格对应正文 3.5.2 特征式：
  仅数值特征(6) → + 配置数值父子比(3) → + 代码 token Jaccard 相似度(1)
  → + 算子/mask 调用数父子比(2) → + 子代算子/块规模绝对值(2)
  = 增强父子特征（14 项，最终模型）
每个中间配置均同时评估逻辑回归(LR)与梯度提升树(GBM)，用于模型选型对比。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_model_final import (  # noqa: E402
    load_df, add_pc_features)
from experiments.candidate_filter_features_v6 import tie_rate  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

OUT = ROOT / "experiments" / "results" / "candidate_filter_ablation_final.json"

CAT = ["operation", "mutation_type"]


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
        yt = y[te]
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
    df = add_pc_features(load_df())
    positive_rate = float(df["y"].mean())
    print(f"样本 {len(df)}；正样本率 {positive_rate:.4f}；算子 {df['kernel'].nunique()}")

    BASE = ["log_parent", "gen", "pop_size", "tau"]
    CFG = ["r_block", "r_warps", "r_stages"]
    SIM = ["sim"]
    OPM = ["r_ops", "r_mask"]
    ABS = ["ops_abs", "block_abs"]

    steps = [
        ("仅数值特征（基线）", BASE),
        ("+ 配置数值父子比（r_block/r_warps/r_stages）", BASE + CFG),
        ("+ 代码 token Jaccard 相似度 sim", BASE + CFG + SIM),
        ("+ 算子与 mask 调用数父子比（r_ops/r_mask）", BASE + CFG + SIM + OPM),
        ("+ 子代算子/块规模绝对值（增强父子特征，最终模型）",
         BASE + CFG + SIM + OPM + ABS),
    ]

    out = {"positive_rate": positive_rate,
           "protocol": "按算子 5 折 GroupKFold；指标 AUC / PR-AUC / PR-AUC÷P(lift) / 池内可区分度",
           "results": []}
    print("\n" + "=" * 104)
    print(f"{'配置':54s}{'模型':>8s}{'AUC':>10s}{'PR-AUC':>10s}"
          f"{'lift':>8s}{'可区分度':>10s}")
    print("-" * 104)
    for name, cols in steps:
        for kind in ("lr", "gbm"):
            r = evaluate(df, cols, kind, positive_rate)
            r.update(variant=name, model=kind, numeric_features=cols)
            out["results"].append(r)
            print(f"{name:54s}{kind:>8s}{r['auc']:.4f}±{r['auc_std']:.3f}"
                  f"{r['pr_auc']:>10.4f}{r['pr_lift']:>8.3f}"
                  f"{r['distinct_in_pool']:>9.1%}")

    # 参照：随机排序基线 / 无预筛选
    y = df["y"].to_numpy()
    groups = df["kernel"].to_numpy()
    rng = np.random.default_rng(0)
    aucs, prs, lifts, dips = [], [], [], []
    for tr, te in GroupKFold(5).split(np.zeros((len(df), 1)), y, groups):
        s = rng.random(len(te))
        yt = y[te]
        aucs.append(roc_auc_score(yt, s))
        pra = average_precision_score(yt, s)
        prs.append(pra)
        lifts.append(pra / positive_rate)
        pooled = df.iloc[te].assign(score=s)
        vals = [tie_rate(sub["score"].to_numpy())
                for _, sub in pooled.groupby(["kernel", "gen"]) if len(sub) >= 5]
        dips.append(float(np.mean(vals)) if vals else np.nan)
    out["random_baseline"] = dict(
        auc=float(np.mean(aucs)), pr_auc=float(np.mean(prs)),
        pr_lift=float(np.mean(lifts)), distinct_in_pool=float(np.nanmean(dips)))
    out["no_prefilter"] = dict(
        auc=None, pr_auc=positive_rate, pr_lift=1.0, distinct_in_pool=None)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[保存] {OUT}")


if __name__ == "__main__":
    main()
