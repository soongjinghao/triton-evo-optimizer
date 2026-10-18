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
import os
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
# 父子代码特征的**唯一**实现：在线侧 evolutionary_algorithm 也用它
from experiments.candidate_filter_features_shared import (       # noqa: E402
    compute_code_features)

SAMPLES = ROOT / "experiments" / "dataset" / "samples.jsonl"
MODEL_OUT = ROOT / "experiments" / "dataset" / "candidate_filter_model.joblib"
REPORT_OUT = ROOT / "experiments" / "results" / "candidate_filter_final.json"
# 冻结快照文件名（由 run_phase.py freeze 生成）
DEFAULT_SNAPSHOT_NAME = "samples_frozen_20261016.jsonl"


def _sha256_file(p) -> str:
    """文件 SHA256，用于把模型/数据版本固化进结果，便于恢复时核对。"""
    import hashlib
    p = Path(p)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_samples_path() -> Path:
    """默认训练数据路径：**优先**用冻结快照，不存在才回退到动态 samples.jsonl。

    【修改】训练数据会随实验推进被 collect_data 持续追加。若默认读动态文件，
    "离线表11 的评估"与"在线实验用的模型"就可能基于不同样本集，论文自洽性受损。
    改成优先快照后，所有未显式传路径的调用方（含离线消融脚本）都自动可复现。
    """
    snap = SAMPLES.parent / os.getenv("EA_DATASET_SNAPSHOT", DEFAULT_SNAPSHOT_NAME)
    if snap.is_file():
        return snap
    print(f"⚠️ 未找到冻结快照 {snap.name}，回退到动态数据 {SAMPLES.name}；"
          f"请先运行 python experiments/run_phase.py freeze")
    return SAMPLES

# H 基础(8) + V7 强区分特征(r_ops/r_mask/ops_abs/block_abs，实测对可区分度贡献最大)；
# 剔除 rel_seed(v7 实测零区分贡献且在线需额外取 seed_time) 与 speedup(已证有害)
NUMERIC = ["log_parent", "gen", "pop_size", "tau",
           "r_block", "r_warps", "r_stages", "sim",
           "r_ops", "r_mask", "ops_abs", "block_abs"]
CATEGORICAL = ["operation", "mutation_type"]


def load_df(path=None):
    """读取并清洗训练样本。

    【修改】旧实现固定读全局 SAMPLES（动态增长的 samples.jsonl），
    调用方无法指定数据版本。现在支持显式传路径：
        load_df(Path(".../samples_frozen_20261016.jsonl"))
    不传时走 default_samples_path()（优先冻结快照）。
    """
    src = Path(path) if path else default_samples_path()
    if not src.is_file():
        raise FileNotFoundError(f"训练数据不存在：{src}")
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
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
    print(f"[数据] {src.name}：原始 {raw} → 清洗后 {len(df)}；"
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
    """构造父子代码相对特征。

    【统一】旧实现在这里用 v6 的 struct_feats/ratio/jaccard 自己算一遍，
    而在线侧用 evolutionary_algorithm 的另一份实现 —— 两份代码靠人工保持一致。
    现在两边共同调用 experiments.candidate_filter_features_shared.compute_code_features，
    从机制上消除训练/推理特征漂移（评审要求"使用同一套父子特征"）。
    """
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
        pc = _parent_code(code_map, pids)
        recs.append(compute_code_features(ch, pc))
    for k in recs[0]:
        df[k] = [r[k] for r in recs]
    return df


def precision_at_k_pooled(y, scores, pool_ids, k):
    """按真实候选池（同 kernel+gen+run 同批生成）分组，组内 Top-k 精确率再对池平均。

    修正点：原离线 P@4 在整折测试集上取前 4，与线上『每个算子每一代 20 候选取 4』口径不一致；
    这里改用真实候选池分组，更贴近线上决策。
    """
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores),
                     "pool": list(pool_ids)})
    ps = []
    for _, sub in d.groupby("pool"):
        if len(sub) < k:
            continue
        order = sub["s"].values.argsort()[::-1][:k]
        ps.append(float(sub["y"].values[order].mean()))
    return float(np.nanmean(ps)) if ps else float("nan")


def train_and_save(train_df, model_out, report_out, train_kernels=None,
                   exclude_kernels=None, dataset_path=None):
    """在 train_df（已按 kernel 分组）上做 5 折复核 + 全量训练，并保存模型与元信息。

    dataset_path：显式指定本次训练所用的数据文件（应是冻结快照），
    会连同其 SHA256 写入 meta，供在线实验恢复时核对版本。
    """
    global SAMPLES
    if dataset_path:
        SAMPLES = Path(dataset_path)
    X = train_df[NUMERIC + CATEGORICAL]
    y = train_df["y"].to_numpy()
    groups = train_df["kernel"].to_numpy()
    pools = (train_df["kernel"].astype(str) + "|" + train_df["gen"].astype(str)
             + "|" + train_df["run_id"].astype(str)).to_numpy()

    gkf = GroupKFold(n_splits=5)
    aucs, p4, p13, base4, base13, praucs, pp4 = [], [], [], [], [], [], []
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
        pp4.append(precision_at_k_pooled(yt, s, pools[te], 4))
        rs = rng.random(len(yt))
        base4.append(precision_at_k(yt, rs, 4))
        base13.append(precision_at_k(yt, rs, 13))
    print(f"\n[复核] 按算子 5 折：AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"P@4(整折)={np.mean(p4):.4f}（随机 {np.mean(base4):.4f}）  "
          f"P@4(按池)={np.mean(pp4):.4f}（随机 {np.mean(base4):.4f}）  "
          f"P@13={np.mean(p13):.4f}（随机 {np.mean(base13):.4f}）")

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

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    # 池内可区分度：同一 (算子, 代数) 池内不同分值数 / 候选数（H 核心收益：解决同分打平）
    pool_scores = pipe.predict_proba(train_df[NUMERIC + CATEGORICAL])[:, 1]
    pooled = train_df.assign(score=pool_scores)
    vals = []
    for _, sub in pooled.groupby(["kernel", "gen"]):
        if len(sub) >= 5:
            vals.append(tie_rate(sub["score"].to_numpy()))
    distinct_in_pool = float(np.nanmean(vals)) if vals else float("nan")

    # 【修改】train_kernels 必须落成**显式名单**。
    # 旧实现在没传 --train-kernels 时写 "ALL"，而在线实验的防泄漏检查
    # 要求 train_kernels 不是 "ALL"，于是该检查永远失效。这里从数据反推真实名单。
    tk = sorted(train_kernels) if train_kernels \
        else sorted(set(train_df["kernel"].astype(str)))
    meta = {
        "model_out": str(model_out),
        "train_kernels": tk,
        "n_train_kernels": len(tk),
        "exclude_kernels": sorted(exclude_kernels) if exclude_kernels else [],
        "n_samples": int(len(train_df)),
        "n_kernels": int(train_df["kernel"].nunique()),
        "positive_rate": float(train_df["y"].mean()),
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
        # 数据/模型版本固化：恢复实验时据此核对，避免复用"修复前"的旧结果
        "dataset_snapshot": str(SAMPLES),
        "dataset_sha256": _sha256_file(SAMPLES),
        "model_sha256": _sha256_file(model_out),
    }
    (model_out.parent / (model_out.stem + ".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    REPORT_OUT = Path(report_out)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps({
        "variant": "H基础+V7强特征（去 speedup/rel_seed）",
        "model_out": str(model_out),
        "train_kernels": meta["train_kernels"],
        "exclude_kernels": meta["exclude_kernels"],
        "n_samples": int(len(train_df)),
        "n_kernels": int(train_df["kernel"].nunique()),
        "positive_rate": float(train_df["y"].mean()),
        "cv_auc": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
        "cv_prauc": float(np.mean(praucs)),
        "cv_lift": float(np.mean(praucs) / train_df["y"].mean()),
        "cv_p4_flat": float(np.mean(p4)), "cv_p4_flat_random": float(np.mean(base4)),
        "cv_p4_pooled": float(np.mean(pp4)),
        "cv_p13": float(np.mean(p13)), "cv_p13_random": float(np.mean(base13)),
        "distinct_in_pool": distinct_in_pool,
        "coefficients": coefs,
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
        "caveat": "预测性模型；类型系数含选择效应，不得作因果解释（见 3.4.3 机制探测）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] 模型 {model_out}")
    print(f"[保存] 元信息 {model_out.stem}.meta.json")
    print(f"[保存] 报告 {REPORT_OUT}")


def main():
    global SAMPLES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-kernels", nargs="+", default=None,
                    help="仅用这些算子训练；与 --exclude-kernels 互斥")
    ap.add_argument("--exclude-kernels", nargs="+", default=None,
                    help="排除这些（测试）算子后再训练；与 --train-kernels 互斥")
    ap.add_argument("--model-out", default=None,
                    help="模型保存路径；默认按训练算子集合哈希命名，避免覆盖")
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--samples", default=None,
                    help="训练数据路径；默认用冻结快照（见 default_samples_path）")
    a = ap.parse_args()

    if a.train_kernels and a.exclude_kernels:
        raise SystemExit("❌ --train-kernels 与 --exclude-kernels 互斥")

    SAMPLES = Path(a.samples) if a.samples else default_samples_path()
    df = load_df(SAMPLES)

    if a.train_kernels:
        tk = set(a.train_kernels)
        before = len(df)
        df = df[df["kernel"].isin(tk)].copy()
        print(f"[子集] 仅训练算子 {len(tk)} 个 → 样本 {before} → {len(df)}")
    elif a.exclude_kernels:
        ek = set(a.exclude_kernels)
        before = len(df)
        df = df[~df["kernel"].isin(ek)].copy()
        print(f"[子集] 排除测试算子 {len(ek)} 个 → 样本 {before} → {len(df)}")

    df = add_pc_features(df)

    model_out = a.model_out
    if model_out is None:
        import hashlib
        key = ",".join(sorted(df["kernel"].unique())) if len(df) else "ALL"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        model_out = MODEL_OUT.parent / f"candidate_filter_model__{h}.joblib"
    report_out = a.report_out or REPORT_OUT
    train_and_save(df, model_out, report_out,
                   train_kernels=(a.train_kernels or None),
                   exclude_kernels=(a.exclude_kernels or None))


if __name__ == "__main__":
    main()
