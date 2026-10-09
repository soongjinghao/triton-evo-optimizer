#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练并保存 3.6 的**最终**候选预筛选模型，供在线（真机）实验调用。

模型选择
--------
按 3.6.5 的权衡：在合法特征（评测前可得）中，
  - 「仅数值特征」P@13 最高（0.4699），且系数可直接读出、可解释；
  - 「数值+策略」P@4 略高（0.4589 vs 0.4335），但引入 300 维 TF-IDF 文本特征，
    无法给出直观解释，部署成本更高，且差距在有限样本下不显著。
因此选用 **仅数值特征的逻辑回归**，与论文 3.6.3 / 3.6.5 的论述一致。

⚠️ 关于系数的重要说明
----------------------
本模型是**预测性**模型，不是因果模型。离线系数显示
「structure_rewrite」为正，但 3.4.3 的机制探测（在同一父代上随机抽取变异类型）
表明三种类型的改进率并无差异（44.7% / 46.3% / 47.1%，p ≈ 0.87）。
差异来自训练数据中的**选择效应**：变异类型是依父代水平分配的
（见 adaptive 的分级规则），因此类型与结果的关联混有「哪类父代被选中」。
预筛选只需要预测性，故该系数仍可用；但不得解释为
「结构重写更能带来改进」。

输入特征（全部评测前可得）
--------------------------
  log_parent = log1p(父代延迟)、gen（代数）、pop_size、tau、
  operation（mutation / crossover / gen0_*）、mutation_type

用法
----
    python experiments/candidate_filter_model_final.py
"""
import json
import sys
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
# 复用 v6 的代码特征抽取，保证训练口径与离线评估完全一致
sys.path.insert(0, str(ROOT))
from experiments.candidate_filter_features_v6 import (           # noqa: E402
    struct_feats, jaccard, ratio, load_codes, tie_rate)

SAMPLES = ROOT / "experiments" / "dataset" / "samples.jsonl"
MODEL_OUT = ROOT / "experiments" / "dataset" / "candidate_filter_model.joblib"
REPORT_OUT = ROOT / "experiments" / "results" / "candidate_filter_final.json"

# H 基础(8) + V7 强区分特征(r_ops/r_mask/ops_abs/block_abs，实测对可区分度贡献最大)；
# 剔除 rel_seed(v7 实测零区分贡献且在线需额外取 seed_time) 与 speedup(已证有害)
NUMERIC = ["log_parent", "gen", "pop_size", "tau",
           "r_block", "r_warps", "r_stages", "sim",
           "r_ops", "r_mask", "ops_abs", "block_abs"]
CATEGORICAL = ["operation", "mutation_type"]


def load_df():
    rows = [json.loads(l) for l in open(SAMPLES, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    raw = len(df)
    df = df[df["run_id"] == 0].copy()
    ratio = df["child_latency"] / df["parent_latency"]
    df = df[ratio <= 10].copy()
    df["mutation_type"] = df["mutation_type"].fillna("none").astype(str)
    for col in ("tau", "gen", "pop_size", "parent_latency", "child_latency"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tau"] = df["tau"].fillna(-1.0)
    df = df[(df["parent_latency"] > 0) & (df["child_latency"] > 0)].copy()
    df["y"] = (df["child_latency"] < df["parent_latency"]).astype(int)
    df["log_parent"] = np.log1p(df["parent_latency"])
    print(f"[数据] 原始 {raw} → 清洗后 {len(df)}；"
          f"正样本 {df['y'].mean():.1%}；算子 {df['kernel'].nunique()}")
    return df


def build_prep():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="none")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def make_pipe():
    return Pipeline([("prep", build_prep()),
                     ("clf", LogisticRegression(max_iter=2000,
                                                class_weight="balanced"))])


def precision_at_k(y, scores, k):
    order = np.argsort(-np.asarray(scores))[:k]
    return float(np.mean(np.asarray(y)[order])) if len(order) else float("nan")


# ---------------- 在线一致的父子代码特征（父代取 parent_ids[0]） ----------------
RE_TL_CALL = re.compile(r"tl\.[a-z_]+")
RE_MASK = re.compile(r"mask")


def _parent_code(code_map, pids):
    """取主干父代(parent_ids[0])代码；gen0 无父代返回 None（调用方回退到子代自身）。"""
    if isinstance(pids, str):
        try:
            pids = json.loads(pids.replace("'", '"'))
        except Exception:
            pids = [pids]
    if pids:
        pid = str(pids[0])
        if pid in code_map:
            return code_map[pid]
    return None


def add_pc_features(df):
    """构造父子代码相对特征，口径与在线 _filter_features 完全一致：
    父代取 parent_ids[0]（主干父代），gen0 无父代时回退为子代自身（比值=0、sim=1）。"""
    code_map = load_codes()
    cache = {}

    def f(cid):
        cid = str(cid)
        if cid not in cache:
            cache[cid] = code_map.get(cid, "")
        return cache[cid]

    recs = []
    for cid, pids in zip(df["candidate_id"], df["parent_ids"]):
        ch = f(cid)
        chf = struct_feats(ch)
        pc = _parent_code(code_map, pids)
        if pc is not None:
            pcf = struct_feats(pc)
            c_ops, p_ops = len(RE_TL_CALL.findall(ch)), len(RE_TL_CALL.findall(pc))
            c_mask, p_mask = len(RE_MASK.findall(ch)), len(RE_MASK.findall(pc))
            recs.append(dict(
                r_block=ratio(chf["block"], pcf["block"]),
                r_warps=ratio(chf["warps"], pcf["warps"]),
                r_stages=ratio(chf["stages"], pcf["stages"]),
                sim=jaccard(ch, pc),
                r_ops=ratio(c_ops, p_ops),
                r_mask=ratio(c_mask, p_mask),
                ops_abs=np.log1p(c_ops),
                block_abs=np.log1p(chf["block"]),
            ))
        else:  # gen0：父代=子代自身 → 比值特征为 0、sim=1
            c_ops = len(RE_TL_CALL.findall(ch))
            c_mask = len(RE_MASK.findall(ch))
            recs.append(dict(
                r_block=0.0, r_warps=0.0, r_stages=0.0, sim=1.0,
                r_ops=0.0, r_mask=0.0,
                ops_abs=np.log1p(c_ops), block_abs=np.log1p(chf["block"]),
            ))
    for k in recs[0]:
        df[k] = [r[k] for r in recs]
    return df


def main():
    df = load_df()
    # 父子代码相对 + V7 强区分特征（口径与在线一致：父代取 parent_ids[0]）
    df = add_pc_features(df)
    X = df[NUMERIC + CATEGORICAL]
    y = df["y"].to_numpy()
    groups = df["kernel"].to_numpy()

    # ---- 与表12 同协议的 5 折评估（仅用于复核，不用于选模型） ----
    gkf = GroupKFold(n_splits=5)
    aucs, p4, p13, base4, base13, praucs = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipe()
        pipe.fit(X.iloc[tr], y[tr])
        s = pipe.predict_proba(X.iloc[te])[:, 1]
        yt = y[te]
        aucs.append(roc_auc_score(yt, s))
        praucs.append(average_precision_score(yt, s))
        p4.append(precision_at_k(yt, s, 4))
        p13.append(precision_at_k(yt, s, 13))
        rs = rng.random(len(yt))
        base4.append(precision_at_k(yt, rs, 4))
        base13.append(precision_at_k(yt, rs, 13))
    print(f"\n[复核] 按算子 5 折：AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"P@4={np.mean(p4):.4f}（随机 {np.mean(base4):.4f}）  "
          f"P@13={np.mean(p13):.4f}（随机 {np.mean(base13):.4f}）")

    # ---- 在全量数据上拟合最终模型并保存 ----
    pipe = make_pipe()
    pipe.fit(X, y)
    names = (NUMERIC + list(pipe.named_steps["prep"].named_transformers_["cat"]
                            .named_steps["o"].get_feature_names_out(CATEGORICAL)))
    coef = pipe.named_steps["clf"].coef_[0]

    print("\n=== 模型系数（正 = 更可能优于父代，仅供预测，不作因果解释）===")
    coefs = {}
    for n, c in sorted(zip(names, coef), key=lambda x: -abs(x[1])):
        coefs[n] = float(c)
        print(f"  {n:44s} {c:+.4f}  (OR={np.exp(c):.3f})")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    # 池内可区分度：同一 (算子, 代数) 池内不同分值数 / 候选数（H 核心收益：解决同分打平）
    pool_scores = pipe.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]
    pooled = df.assign(score=pool_scores)
    vals = []
    for _, sub in pooled.groupby(["kernel", "gen"]):
        if len(sub) >= 5:
            vals.append(tie_rate(sub["score"].to_numpy()))
    distinct_in_pool = float(np.nanmean(vals)) if vals else float("nan")

    REPORT_OUT.write_text(json.dumps({
        "variant": "H基础+V7强特征（去 speedup/rel_seed）",
        "n_samples": int(len(df)),
        "n_kernels": int(df["kernel"].nunique()),
        "positive_rate": float(df["y"].mean()),
        "cv_auc": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
        "cv_prauc": float(np.mean(praucs)),
        "cv_lift": float(np.mean(praucs) / df["y"].mean()),
        "cv_p4": float(np.mean(p4)), "cv_p4_random": float(np.mean(base4)),
        "cv_p13": float(np.mean(p13)), "cv_p13_random": float(np.mean(base13)),
        "distinct_in_pool": distinct_in_pool,
        "coefficients": coefs,
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
        "caveat": "预测性模型；类型系数含选择效应，不得作因果解释（见 3.4.3 机制探测）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[保存] 模型 {MODEL_OUT}")
    print(f"[保存] 报告 {REPORT_OUT}")


if __name__ == "__main__":
    main()
