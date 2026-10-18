#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**一致的父子特征定义**下重新训练候选预筛选模型（0 NPU）。

评审意见："先审计历史训练样本，修复能够可靠恢复的父子关系，再在固定 35/15
算子隔离方案下重新训练。"

与旧训练（candidate_filter_model_final.py）的三点区别
----------------------------------------------------
1. **训练集只留"特征可信"的样本**（由 audit_training_samples.py 产出）：
   * 剔除 gen0（在线从不筛选 gen0，且无真实父代、代码特征天然退化）
   * 剔除父代代码不可恢复的样本（否则 8 个代码特征是常数，是噪声不是缺失）
   * 剔除 dev15 的 15 个留出算子
2. **特征实现与在线同源**：两侧都调
   experiments.candidate_filter_features_shared.compute_code_features，
   不再各写一份（旧版在线是退化实现，见 CHANGELOG）。
3. **评估更严格**：除了 GroupKFold CV，还在**从未参与训练的 15 个留出算子**上
   直接打分，并与旧模型 `candidate_filter_model__dev15.joblib` 在同一批行上对比。

用法
----
    python experiments/retrain_candidate_filter.py \
        --clean experiments/dataset/samples_clean_v7.jsonl \
        --split experiments/splits/dev15.json \
        --out experiments/dataset/candidate_filter_model__v7.joblib
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_features_shared import (      # noqa: E402
    load_codes, compute_code_features, normalize_pids)

NUMERIC = ["log_parent", "gen", "pop_size", "tau",
           "r_block", "r_warps", "r_stages", "sim",
           "r_ops", "r_mask", "ops_abs", "block_abs"]
CATEGORICAL = ["operation", "mutation_type"]

DEFAULT_CLEAN = ROOT / "experiments" / "dataset" / "samples_clean_v7.jsonl"
DEFAULT_SPLIT = ROOT / "experiments" / "splits" / "dev15.json"
DEFAULT_OUT = ROOT / "experiments" / "dataset" / "candidate_filter_model__v7.joblib"
OLD_MODEL = ROOT / "experiments" / "dataset" / "candidate_filter_model__dev15.joblib"
REPORT_OUT = ROOT / "experiments" / "results" / "candidate_filter_v7.json"


def _sha256(p):
    p = Path(p)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _pos(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) and f > 0 else None


def build_df(path, codes):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    for col in ("tau", "gen", "pop_size", "parent_latency", "child_latency"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tau"] = df["tau"].fillna(-1.0)
    df["mutation_type"] = df["mutation_type"].fillna("none").astype(str)
    df["operation"] = df["operation"].fillna("none").astype(str)
    df["y"] = (df["child_latency"] < df["parent_latency"]).astype(int)
    df["log_parent"] = np.log1p(df["parent_latency"])

    # 父子代码特征：与在线同源的唯一实现
    recs = []
    for cid, pids in zip(df["candidate_id"], df["parent_ids"]):
        p = normalize_pids(pids)
        pc = codes.get(p[0]) if p else None
        recs.append(compute_code_features(codes.get(str(cid), ""), pc))
    for k in recs[0]:
        df[k] = [r[k] for r in recs]
    return df


def make_pipe(class_weight="balanced"):
    cw = None if str(class_weight).lower() in ("none", "null", "") else class_weight
    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="none")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(max_iter=2000,
                                                class_weight=cw))])


def pooled_p_at_k(df, scores, k=4):
    """按真实候选池（同 kernel+gen）分组的 Top-k 精确率，再对池平均。"""
    o = _pool_p_at_k(df, scores, k)
    if not o:
        return None, None, 0
    v = [x[0] for x in o.values()]
    r = [x[1] for x in o.values()]
    return float(np.mean(v)), float(np.mean(r)), len(o)


FAKE_POOL_WARNED = [False]


def warn_fake_pool():
    """旧快照没有 pool_id，只能退化为 (kernel, gen) 分组 —— 必须显式告警。"""
    if FAKE_POOL_WARNED[0]:
        return
    FAKE_POOL_WARNED[0] = True
    print("\n  ⚠️  本快照无真实候选池标识（pool_id），P@k 只能按 (kernel, gen) 近似分组。\n"
          "     这类分组可能把多次运行的候选并成假池，P@k/随机基线/模型对比都会失真。\n"
          "     真实 20 选 4 请用 experiments/eval_real_pools.py（读在线日志里的真实池）。")


def _pool_p_at_k(d, scores, k=4):
    """返回 {pool_key: (Top-k 精确率, 池内随机基线)}。"""
    d = d.copy()
    d["s"] = scores
    out = {}
    key_cols = ["pool_id"] if "pool_id" in d.columns and d["pool_id"].notna().any() \
        else ["kernel", "gen"]
    for key, g in d.groupby(key_cols, sort=False):
        if len(g) < k:
            continue
        gg = g.sort_values("s", ascending=False)
        out[key] = (float((gg["y"].to_numpy()[:k] == 1).mean()),
                    float(g["y"].mean()))
    return out


def paired_perm_test(pa, pb, n_perm=20000, seed=20261017):
    """两个模型在同一批池上的 P@k 配对比较：均值差 + 置换检验双尾 p。

    P@4 只在 30 个池上平均，方差很大；不做检验就报"新模型更好/更差"是不负责任的。
    """
    keys = sorted(set(pa) & set(pb))
    if len(keys) < 2:
        return None
    a = np.array([pa[k][0] for k in keys])
    b = np.array([pb[k][0] for k in keys])
    d0 = a - b
    diff = float(d0.mean())
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        sgn = np.where(rng.integers(0, 2, size=len(d0)) == 0, 1.0, -1.0)
        if abs((d0 * sgn).mean()) >= abs(diff) - 1e-15:
            cnt += 1
    return {"n_pools": len(keys), "mean_diff": diff,
            "p_value": (cnt + 1) / (n_perm + 1)}


def main():
    ap = argparse.ArgumentParser(description="一致父子特征下的候选预筛选模型重训（0 NPU）")
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--raw", default=str(ROOT / "experiments" / "dataset"
                                        / "samples_frozen_20261016.jsonl"),
                    help="原始冻结快照，用于构造留出评估集")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gen-max", type=int, default=0,
                    help=">0 时只用 gen<=gen-max 的样本训练（线上主要服务 Gen1/Gen2）")
    ap.add_argument("--class-weight", default="balanced",
                    help="balanced / none（P@k 关注排序顶端，类别权重会影响顶端校准）")
    a = ap.parse_args()

    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    test_kernels = set(split.get("test_kernels") or split.get("test") or [])
    codes = load_codes()

    tr_all = build_df(a.clean, codes)
    tr = tr_all[tr_all["gen"] <= a.gen_max].copy() if a.gen_max > 0 else tr_all
    if a.gen_max > 0:
        print(f"  [子集] 只用 gen<= {a.gen_max} 的样本：{len(tr)} / {len(tr_all)}")
    assert not (set(tr["kernel"]) & test_kernels), "训练集混入了留出算子！"
    warn_fake_pool()
    print("=" * 84)
    print("候选预筛选模型重训（0 NPU）")
    print(f"  训练集 {a.clean}: {len(tr)} 条 / {tr['kernel'].nunique()} 个算子"
          f"   正样本率 {tr['y'].mean():.1%}")
    print(f"  留出算子 {len(test_kernels)} 个（训练集中出现 0 个）")
    print("=" * 84)

    # ---------------- 1. GroupKFold CV（按算子分组） ----------------
    pipe = make_pipe(a.class_weight)
    X = tr[NUMERIC + CATEGORICAL]
    y = tr["y"].to_numpy()
    groups = tr["kernel"].to_numpy()
    n_folds = min(a.folds, tr["kernel"].nunique())
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tri, tei in gkf.split(X, y, groups):
        p = make_pipe(a.class_weight)
        p.fit(X.iloc[tri], y[tri])
        s = p.predict_proba(X.iloc[tei])[:, 1]
        if len(set(y[tei])) > 1:
            aucs.append(roc_auc_score(y[tei], s))
    cv_auc = float(np.mean(aucs)) if aucs else None
    print(f"\n[1] GroupKFold CV AUC（{n_folds} 折，按算子分组）= "
          f"{cv_auc:.4f}" if cv_auc else "[1] CV 无法计算")

    pipe.fit(X, y)

    # ---------------- 2. 留出 15 算子评估 ----------------
    raw = build_df(a.raw, codes)
    hold = raw[raw["kernel"].isin(test_kernels)].copy()
    # 留出集同样只保留"特征可信"的行：有父代、父代代码可取、非 gen0
    hold = hold[~hold["operation"].astype(str).str.startswith("gen0")]
    hold_ok = []
    for i, (cid, pids) in enumerate(zip(hold["candidate_id"], hold["parent_ids"])):
        p = normalize_pids(pids)
        if p and p[0] in codes:
            hold_ok.append(i)
    hold = hold.iloc[hold_ok]
    print(f"[2] 留出集 {len(hold)} 条 / {hold['kernel'].nunique()} 个算子"
          f"   正样本率 {hold['y'].mean():.1%}")

    new_s = pipe.predict_proba(hold[NUMERIC + CATEGORICAL])[:, 1]
    new_auc = roc_auc_score(hold["y"], new_s) if len(set(hold["y"])) > 1 else None
    new_p, new_r, n_pools = pooled_p_at_k(hold, new_s, k=4)

    # ---------------- 3. 与旧模型在同一批行上对比 ----------------
    old_auc = old_p = old_r = None
    old_s = None
    paired = None
    if Path(OLD_MODEL).is_file():
        try:
            old = joblib.load(OLD_MODEL)
            old_s = old.predict_proba(hold[NUMERIC + CATEGORICAL])[:, 1]
            old_auc = roc_auc_score(hold["y"], old_s) if len(set(hold["y"])) > 1 else None
            old_p, old_r, _ = pooled_p_at_k(hold, old_s, k=4)
        except Exception as e:
            print(f"  ⚠️ 旧模型无法在留出集上打分: {e}")

    print("\n[3] 新旧模型在**同一批留出行**上的对比（15 个算子，从未参与训练）")
    print(f"    {'':12s} {'AUC':>8s} {'P@4':>8s} {'随机基线':>10s} {'增益':>8s}")
    for name, auc_, p4, rb in (("旧 dev15", old_auc, old_p, old_r),
                               ("新 v7", new_auc, new_p, new_r)):
        if auc_ is None:
            print(f"    {name:12s} {'—':>8s}")
            continue
        g = (p4 - rb) if (p4 is not None and rb is not None) else None
        print(f"    {name:12s} {auc_:>8.4f} {p4:>8.4f} {rb:>10.4f} "
              f"{(f'{g:+.4f}' if g is not None else '—'):>8s}")
    print(f"    （P@4 按真实候选池分组，共 {n_pools} 个池）")

    if old_s is not None:
        pt = paired_perm_test(_pool_p_at_k(hold, new_s, 4),
                              _pool_p_at_k(hold, old_s, 4))
        if pt:
            verdict = "差异不显著" if pt["p_value"] >= 0.05 else "差异显著"
            print(f"\n    配对检验（{pt['n_pools']} 个池，同一批池逐池配对）："
                  f"新 − 旧 = {pt['mean_diff']:+.4f}，"
                  f"置换检验 p = {pt['p_value']:.4f} → {verdict}")
            paired = pt

    # ---------------- 4. 系数 ----------------
    names = list(NUMERIC) + list(
        pipe.named_steps["prep"].named_transformers_["cat"]
        .named_steps["o"].get_feature_names_out(CATEGORICAL))
    coef = pipe.named_steps["clf"].coef_[0]
    top = sorted(zip(names, coef), key=lambda x: -abs(x[1]))[:10]
    print("\n[4] 系数（前 10，按绝对值）")
    for n_, c_ in top:
        print(f"    {n_:38s} {c_:+.4f}")

    # ---------------- 保存 ----------------
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, a.out)
    rep = {
        "train_samples": a.clean,
        "train_sha256": _sha256(a.clean),
        "split": a.split,
        "split_sha256": _sha256(a.split),
        "n_train": int(len(tr)),
        "n_train_kernels": int(tr["kernel"].nunique()),
        "positive_rate_train": float(tr["y"].mean()),
        "feature_impl": "experiments/candidate_filter_features_shared",
        "cv_auc_groupkfold": cv_auc,
        "cv_folds": n_folds,
        "class_weight": a.class_weight,
        "holdout": {
            "n_kernels": int(hold["kernel"].nunique()),
            "n_rows": int(len(hold)),
            "positive_rate": float(hold["y"].mean()),
            "new_auc": new_auc, "new_p_at_4": new_p, "random_baseline": new_r,
            "old_auc": old_auc, "old_p_at_4": old_p,
            "paired_perm_test": paired,
        },
        "coefficients": {n_: float(c_) for n_, c_ in top},
        "model_out": a.out,
        "model_sha256": _sha256(a.out),
    }
    Path(REPORT_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_OUT).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"\n✅ 模型: {a.out}\n✅ 报告: {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
