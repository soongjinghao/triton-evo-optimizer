"""3.6 候选预筛选机制：严格版离线模型。

与初版的区别：
    1. 只使用评测前可得特征，剔除评测后才产生的 child_profile；
    2. 增加候选代码文本特征，从 experiments/dataset/codes 读取；
    3. 同时评估“仅数值特征”“数值+代码”“数值+策略+代码”三种输入。

用法：
    python experiments/candidate_filter_model_v2.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "experiments" / "dataset" / "samples.jsonl"
CODES = ROOT / "experiments" / "dataset" / "codes"
OUT_JSON = ROOT / "experiments" / "results" / "table11_candidate_filter_v2.json"
OUT_MD = ROOT / "experiments" / "results" / "table11_candidate_filter_v2.md"

BUDGET = 13


def load_clean():
    rows = []
    with SAMPLES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    raw = len(df)
    df = df[df["run_id"] == 0].copy()
    ratio = df["child_latency"] / df["parent_latency"]
    df = df[ratio <= 10].copy()

    df["strategy_text"] = df["applied_strategy"].fillna("").astype(str)
    df["mutation_type"] = df["mutation_type"].fillna("none").astype(str)
    for col in ("tau", "gen", "pop_size", "parent_latency", "child_latency"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tau"] = df["tau"].fillna(-1.0)
    df = df[(df["parent_latency"] > 0) & (df["child_latency"] > 0)].copy()

    # 读取候选代码；缺失时以空串代替，保证特征矩阵可构造
    code_map = {}
    if CODES.exists():
        for p in CODES.rglob("*.py"):
            code_map[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
    df["cand_key"] = df["candidate_id"].astype(str)
    df["code_text"] = df["cand_key"].map(code_map).fillna("")
    df["has_code"] = (df["code_text"].str.len() > 0).astype(int)

    df["y"] = (df["child_latency"] < df["parent_latency"]).astype(int)
    df["log_parent"] = np.log1p(df["parent_latency"])

    print(f"[数据] 原始 {raw} -> 正式 run0 且清洗后 {len(df)}")
    print(f"[数据] 正样本 {int(df['y'].sum())} ({df['y'].mean():.1%})，"
          f"算子数 {df['kernel'].nunique()}")
    print(f"[数据] 命中代码文件 {int(df['has_code'].sum())}/{len(df)}")
    return df


def build_variant(name):
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
    # 代码按 token 切分，重点捕捉 tl.load / tl.store / mask / multiple_of 等模式
    code_vec = TfidfVectorizer(max_features=600, min_df=2,
                               ngram_range=(1, 1), lowercase=True,
                               token_pattern=r"[A-Za-z_][A-Za-z0-9_]*")

    if name == "仅数值特征":
        cols = [("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical)]
    elif name == "数值+策略":
        cols = [("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical),
                ("strategy", strategy_vec, "strategy_text")]
    else:
        cols = [("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical),
                ("strategy", strategy_vec, "strategy_text"),
                ("code", code_vec, "code_text")]
    return ColumnTransformer(cols)


def precision_at_k(df_eval, scores, k):
    tmp = df_eval.copy()
    tmp["score"] = scores
    hits, selected, possible = 0, 0, 0
    for _, g in tmp.groupby("kernel"):
        top = g.nlargest(min(k, len(g)), "score")
        hits += int(top["y"].sum())
        selected += len(top)
        possible += int(g["y"].sum())
    return (hits / selected if selected else 0.0,
            hits / possible if possible else 0.0)


def run_fold(pipe, X, y, groups, df, results, key, variant):
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
            p, r = precision_at_k(df_eval, scores, k)
            pk[k].append(p)
            rk[k].append(r)
    if not aucs:
        return
    results[key] = {
        "variant": variant,
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "pr_auc": float(np.mean(aps)),
        "precision_at_k": {k: float(np.mean(v)) for k, v in pk.items()},
        "recall_at_k": {k: float(np.mean(v)) for k, v in rk.items()},
    }
    print(f"\n[{key}] AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"PR-AUC={np.mean(aps):.4f}")
    for k in (4, 8, 13):
        print(f"    Precision@{k:<2}={np.mean(pk[k]):.4f}  "
              f"Recall@{k:<2}={np.mean(rk[k]):.4f}")


def main():
    df = load_clean()
    X = df[["log_parent", "gen", "pop_size", "tau", "operation",
            "mutation_type", "strategy_text", "code_text"]]
    y = df["y"].to_numpy()
    groups = df["kernel"].to_numpy()

    results = {}
    for vname in ("仅数值特征", "数值+策略", "数值+策略+代码"):
        pipe = Pipeline([
            ("prep", build_variant(vname)),
            ("clf", LogisticRegression(max_iter=2000,
                                       class_weight="balanced")),
        ])
        run_fold(pipe, X, y, groups, df, results,
                 f"逻辑回归（{vname}）", vname)

    rng = np.random.default_rng(0)
    rand_stats = {}
    for k in (4, 8, 13):
        ps = []
        for _ in range(200):
            hits = selected = 0
            for _, g in df.groupby("kernel"):
                n = len(g)
                take = min(k, n)
                idx = rng.choice(n, size=take, replace=False)
                hits += int(g["y"].to_numpy()[idx].sum())
                selected += take
            ps.append(hits / selected)
        rand_stats[k] = {"mean": float(np.mean(ps)),
                         "std": float(np.std(ps))}
    print("\n[随机基线]")
    for k, s in rand_stats.items():
        print(f"    Precision@{k:<2}={s['mean']:.4f} ± {s['std']:.4f}")

    base_rate = float(y.mean())
    payload = {
        "n_samples": int(len(df)),
        "n_kernels": int(df["kernel"].nunique()),
        "positive_rate": round(base_rate, 4),
        "budget": BUDGET,
        "protocol": ("按 kernel 的 GroupKFold 5 折；仅使用评测前可得特征，"
                     "剔除评测后才产生的 child_profile"),
        "models": results,
        "random_baseline": rand_stats,
        "full_eval_baseline": {
            "precision": round(base_rate, 4),
            "recall": 1.0,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    lines = ["# 表11（严格版）候选预筛选模型离线评估结果", ""]
    lines.append(f"- 样本数：{len(df)}；算子数：{df['kernel'].nunique()}")
    lines.append(f"- 正样本比例：{base_rate:.1%}")
    lines.append(f"- 评估协议：{payload['protocol']}")
    lines.append("")
    lines.append("| 输入特征 | AUC | PR-AUC | P@4 | P@8 | P@13 | R@13 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, r in results.items():
        p = r["precision_at_k"]
        lines.append(f"| {r['variant']} | {r['auc']:.4f} | {r['pr_auc']:.4f} | "
                     f"{p[4]:.4f} | {p[8]:.4f} | {p[13]:.4f} | "
                     f"{r['recall_at_k'][13]:.4f} |")
    lines.append(f"| 随机排序基线 | — | — | {rand_stats[4]['mean']:.4f} | "
                 f"{rand_stats[8]['mean']:.4f} | {rand_stats[13]['mean']:.4f} | "
                 f"{rand_stats[13]['mean'] / base_rate:.4f} |")
    lines.append(f"| 无预筛选（3.1 FULL） | — | — | {base_rate:.4f} | "
                 f"{base_rate:.4f} | {base_rate:.4f} | 1.0000 |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[输出] {OUT_JSON}")
    print(f"[输出] {OUT_MD}")


if __name__ == "__main__":
    main()
