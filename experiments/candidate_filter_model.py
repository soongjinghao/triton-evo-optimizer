"""3.6 候选预筛选机制：离线训练与评估。

用法：
    python experiments/candidate_filter_model.py

说明：
    本脚本只读取已采集的 experiments/dataset/samples.jsonl，
    不发起任何 LLM 调用，也不使用 NPU，因此可在无设备环境下运行。

评估协议：
    按 kernel 做 GroupKFold，避免同一算子的样本同时出现在训练集与测试集。
    除 AUC、PR-AUC 外，重点报告 Precision@K，因为每个算子的 NPU 评测预算为 13。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "experiments" / "dataset" / "samples.jsonl"
OUT_JSON = ROOT / "experiments" / "results" / "table11_candidate_filter.json"
OUT_MD = ROOT / "experiments" / "results" / "table11_candidate_filter.md"

# 3.1 主实验统一预算：每个算子 13 次 NPU 评测
BUDGET = 13


def load_clean():
    """读取样本并做保守清洗。"""
    rows = []
    with SAMPLES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    raw = len(df)

    # 只保留正式 run 0，剔除调试重复
    df = df[df["run_id"] == 0].copy()
    after_run = len(df)

    # 剔除异常评测：子代比父代慢 10 倍以上，多为编译/执行退化样本
    ratio = df["child_latency"] / df["parent_latency"]
    df = df[ratio <= 10].copy()
    after_outlier = len(df)

    df["strategy_text"] = df["applied_strategy"].fillna("").astype(str)
    df["profile_text"] = df["child_profile"].fillna("").astype(str)
    df["mutation_type"] = df["mutation_type"].fillna("none").astype(str)
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce").fillna(-1.0)
    df["gen"] = pd.to_numeric(df["gen"], errors="coerce").fillna(0)
    df["pop_size"] = pd.to_numeric(df["pop_size"], errors="coerce").fillna(0)
    df["parent_latency"] = pd.to_numeric(df["parent_latency"], errors="coerce")
    df["child_latency"] = pd.to_numeric(df["child_latency"], errors="coerce")
    df = df[df["parent_latency"] > 0].copy()
    df = df[df["child_latency"] > 0].copy()

    # 标签：子代延迟低于父代，即该候选带来改进
    df["y"] = (df["child_latency"] < df["parent_latency"]).astype(int)
    df["log_parent"] = np.log1p(df["parent_latency"])

    print(f"[数据] 原始样本 {raw}")
    print(f"[数据] 保留正式 run0 后 {after_run}")
    print(f"[数据] 剔除异常评测后 {after_outlier}")
    print(f"[数据] 最终建模样本 {len(df)}，正样本 {int(df['y'].sum())} "
          f"({df['y'].mean():.1%})，算子数 {df['kernel'].nunique()}")
    return df


def build_preprocessor():
    """数值/类别/文本三类特征的统一预处理。"""
    numeric = ["log_parent", "gen", "pop_size", "tau"]
    categorical = ["operation", "mutation_type"]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="none")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
        ("strategy", TfidfVectorizer(max_features=300, min_df=2,
                                     ngram_range=(1, 2), lowercase=True),
         "strategy_text"),
        ("profile", TfidfVectorizer(max_features=400, min_df=2,
                                    ngram_range=(1, 1), lowercase=True,
                                    token_pattern=r"[A-Za-z_][A-Za-z0-9_]+"),
         "profile_text"),
    ])


def to_dense(X):
    """梯度提升模型要求稠密输入；特征规模约 700 维，转稠密开销可接受。"""
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def make_models():
    return {
        "逻辑回归": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "梯度提升": Pipeline([
            ("dense", FunctionTransformer(to_dense, accept_sparse=True)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_depth=6,
                random_state=0,
            )),
        ]),
    }


def build_variant(name):
    """构造不同特征组合，用于特征消融。"""
    numeric = ["log_parent", "gen", "pop_size", "tau"]
    categorical = ["operation", "mutation_type"]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="none")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    strategy_vec = TfidfVectorizer(max_features=300, min_df=2,
                                   ngram_range=(1, 2), lowercase=True)
    profile_vec = TfidfVectorizer(max_features=400, min_df=2,
                                  ngram_range=(1, 1), lowercase=True,
                                  token_pattern=r"[A-Za-z_][A-Za-z0-9_]+")

    if name == "仅数值特征":
        cols = [("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical)]
    elif name == "仅策略文本":
        cols = [("strategy", strategy_vec, "strategy_text")]
    elif name == "仅 Profiling 文本":
        cols = [("profile", profile_vec, "profile_text")]
    elif name == "数值+策略":
        cols = [("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical),
                ("strategy", strategy_vec, "strategy_text")]
    else:
        cols = [("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical),
                ("strategy", strategy_vec, "strategy_text"),
                ("profile", profile_vec, "profile_text")]
    return ColumnTransformer(cols)


def precision_at_k(df_eval, scores, k):
    """按 kernel 组内取分数最高的至多 k 个候选，统计其中真正例比例。"""
    tmp = df_eval.copy()
    tmp["score"] = scores
    hits, selected, possible = 0, 0, 0
    for _, g in tmp.groupby("kernel"):
        top = g.nlargest(min(k, len(g)), "score")
        hits += int(top["y"].sum())
        selected += len(top)
        possible += int(g["y"].sum())
    precision = hits / selected if selected else 0.0
    recall = hits / possible if possible else 0.0
    return precision, recall, selected


def main():
    df = load_clean()
    X_all = df[["log_parent", "gen", "pop_size", "tau", "operation",
                "mutation_type", "strategy_text", "profile_text"]]
    y = df["y"].to_numpy()
    groups = df["kernel"].to_numpy()

    variants = ["全部特征", "仅数值特征", "仅策略文本", "仅 Profiling 文本",
                "数值+策略"]
    models = make_models()
    results = {}
    for vname in variants:
        X = X_all[["log_parent", "gen", "pop_size", "tau", "operation",
                   "mutation_type", "strategy_text", "profile_text"]]
        for name, clf in ({"逻辑回归": models["逻辑回归"]}.items()
                          if vname != "全部特征" else models.items()):
            pipe = Pipeline([("prep", build_variant(vname)), ("clf", clf)])
            key = name if vname == "全部特征" else f"{name}（{vname}）"
            run_fold(pipe, X, y, groups, df, results, key, variant=vname)
            continue

    # 随机基线：多次种子，给出均值与标准差
    rng_base = np.random.default_rng(0)
    rand_stats = {}
    base_pos = float(y.mean())
    for k in (4, 8, 13):
        ps = []
        for _ in range(200):
            hits = selected = 0
            for _, g in df.groupby("kernel"):
                n = len(g)
                take = min(k, n)
                idx = rng_base.choice(n, size=take, replace=False)
                hits += int(g["y"].to_numpy()[idx].sum())
                selected += take
            ps.append(hits / selected)
        rand_stats[k] = {
            "mean": float(np.mean(ps)),
            "std": float(np.std(ps)),
        }
    print("\n[随机基线]")
    for k, s in rand_stats.items():
        print(f"    Precision@{k:<2}={s['mean']:.4f} ± {s['std']:.4f}")

    base_rate = base_pos
    payload = {
        "n_samples": int(len(df)),
        "n_kernels": int(df["kernel"].nunique()),
        "positive_rate": round(base_rate, 4),
        "budget": BUDGET,
        "protocol": "按 kernel 的 GroupKFold 5 折，无 NPU、无 LLM 调用的离线评估",
        "models": results,
        "random_baseline": rand_stats,
        "full_eval_baseline": {
            "precision": round(base_rate, 4),
            "recall": 1.0,
            "note": "不做预筛选、13 次评测全量执行时的候选有效率",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    lines = ["# 表11 候选预筛选模型离线评估结果", ""]
    lines.append(f"- 样本数：{len(df)}；算子数：{df['kernel'].nunique()}")
    lines.append(f"- 正样本比例：{base_rate:.1%}")
    lines.append(f"- 评估协议：{payload['protocol']}")
    lines.append("")
    lines.append("| 配置 | 模型 | AUC | PR-AUC | P@4 | P@8 | P@13 | R@13 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, r in results.items():
        p = r["precision_at_k"]
        lines.append(
            f"| {r.get('variant', '全部特征')} | {r['model']} | {r['auc']:.4f} | "
            f"{r['pr_auc']:.4f} | {p[4]:.4f} | {p[8]:.4f} | {p[13]:.4f} | "
            f"{r['recall_at_k'][13]:.4f} |")
    lines.append(
        f"| — | 随机排序基线 | — | — | {rand_stats[4]['mean']:.4f} | "
        f"{rand_stats[8]['mean']:.4f} | {rand_stats[13]['mean']:.4f} | "
        f"{rand_stats[13]['mean'] / base_rate:.4f} |")
    lines.append(
        f"| — | 无预筛选（3.1 FULL） | — | — | {base_rate:.4f} | "
        f"{base_rate:.4f} | {base_rate:.4f} | 1.0000 |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[输出] {OUT_JSON}")
    print(f"[输出] {OUT_MD}")


def run_fold(pipe, X, y, groups, df, results, key, variant="全部特征"):
    """按 kernel 分组交叉验证，并把结果写入 results[key]。"""
    cv = GroupKFold(n_splits=5)
    aucs, aps = [], []
    pk = {4: [], 8: [], 13: []}
    rk = {4: [], 8: [], 13: []}

    for train_idx, test_idx in cv.split(X, y, groups):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        scores = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        aucs.append(roc_auc_score(y_test, scores))
        aps.append(average_precision_score(y_test, scores))
        df_eval = df.iloc[test_idx][["kernel", "y"]].copy()
        for k in (4, 8, 13):
            p, r, _ = precision_at_k(df_eval, scores, k)
            pk[k].append(p)
            rk[k].append(r)

    if not aucs:
        return
    results[key] = {
        "variant": variant,
        "model": key.split("（")[0],
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_folds": [round(float(v), 4) for v in aucs],
        "pr_auc": float(np.mean(aps)),
        "precision_at_k": {k: float(np.mean(v)) for k, v in pk.items()},
        "recall_at_k": {k: float(np.mean(v)) for k, v in rk.items()},
    }
    print(f"\n[{key}] AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"PR-AUC={np.mean(aps):.4f}")
    for k in (4, 8, 13):
        print(f"    Precision@{k:<2}={np.mean(pk[k]):.4f}  "
              f"Recall@{k:<2}={np.mean(rk[k]):.4f}")


if __name__ == "__main__":
    main()
