"""3.6 候选预筛选机制：增强特征版。

在严格版基础上新增两类特征：
    1. 代码结构特征：tl.load / tl.store / mask / BLOCK / 常量等计数；
    2. 父代历史统计：该父代已尝试次数与历史成功率，只用当前折之前的信息。

用法：
    python experiments/candidate_filter_model_v3.py
"""

import json
import re
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
OUT_JSON = ROOT / "experiments" / "results" / "table11_candidate_filter_v3.json"
OUT_MD = ROOT / "experiments" / "results" / "table11_candidate_filter_v3.md"

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


def code_struct(code: str):
    """从候选代码抽取结构计数特征。"""
    if not code:
        return {k: 0 for k in STRUCT_PATTERNS} | {
            "code_len": 0, "code_lines": 0}
    feats = {k: len(re.findall(p, code)) for k, p in STRUCT_PATTERNS.items()}
    feats["code_len"] = len(code)
    feats["code_lines"] = code.count("\n") + 1
    return feats


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

    code_map = {}
    if CODES.exists():
        for p in CODES.rglob("*.py"):
            code_map[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
    df["code_text"] = df["candidate_id"].astype(str).map(code_map).fillna("")

    struct = pd.DataFrame([code_struct(c) for c in df["code_text"]],
                          index=df.index)
    df = pd.concat([df, struct], axis=1)

    df["y"] = (df["child_latency"] < df["parent_latency"]).astype(int)
    df["log_parent"] = np.log1p(df["parent_latency"])

    # 父代历史统计：按时间顺序累计，避免用到未来样本
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["parent_key"] = df["kernel"].astype(str) + "|" + df["parent_ids"].astype(str)
    df["parent_tried"] = df.groupby("parent_key").cumcount()
    past = df.groupby("parent_key")["y"]
    df["parent_prior_sum"] = past.cumsum() - df["y"]
    df["parent_prior_rate"] = np.where(
        df["parent_tried"] > 0,
        df["parent_prior_sum"] / df["parent_tried"].replace(0, np.nan),
        np.nan,
    )
    df["parent_prior_rate"] = df["parent_prior_rate"].fillna(-1.0)

    print(f"[数据] 原始 {raw} -> 清洗后 {len(df)}")
    print(f"[数据] 正样本 {int(df['y'].sum())} ({df['y'].mean():.1%})，"
          f"算子数 {df['kernel'].nunique()}")
    return df


def build_variant(name):
    numeric_base = ["log_parent", "gen", "pop_size", "tau"]
    structural = sorted(STRUCT_PATTERNS.keys()) + ["code_len", "code_lines"]
    history = ["parent_tried", "parent_prior_rate"]
    categorical = ["operation", "mutation_type"]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="none")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    code_vec = TfidfVectorizer(max_features=600, min_df=2,
                               ngram_range=(1, 1), lowercase=True,
                               token_pattern=r"[A-Za-z_][A-Za-z0-9_]*")

    if name == "数值+结构+历史":
        cols = [("num", numeric_pipe, numeric_base + structural + history),
                ("cat", categorical_pipe, categorical)]
    elif name == "全部（结构+历史+代码）":
        cols = [("num", numeric_pipe, numeric_base + structural + history),
                ("cat", categorical_pipe, categorical),
                ("code", code_vec, "code_text")]
    elif name == "数值+结构":
        cols = [("num", numeric_pipe, numeric_base + structural),
                ("cat", categorical_pipe, categorical)]
    else:  # 数值+历史
        cols = [("num", numeric_pipe, numeric_base + history),
                ("cat", categorical_pipe, categorical)]
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
    numeric_base = ["log_parent", "gen", "pop_size", "tau"]
    structural = sorted(STRUCT_PATTERNS.keys()) + ["code_len", "code_lines"]
    history = ["parent_tried", "parent_prior_rate"]
    X = df[numeric_base + structural + history +
           ["operation", "mutation_type", "code_text"]]
    y = df["y"].to_numpy()
    groups = df["kernel"].to_numpy()

    results = {}
    for vname in ("数值+结构", "数值+历史", "数值+结构+历史",
                  "全部（结构+历史+代码）"):
        pipe = Pipeline([
            ("prep", build_variant(vname)),
            ("clf", LogisticRegression(max_iter=3000,
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
        "protocol": ("按 kernel 的 GroupKFold 5 折；仅使用评测前可得特征；"
                     "父代历史统计按时间顺序累计，避免未来信息泄漏"),
        "models": results,
        "random_baseline": rand_stats,
        "full_eval_baseline": {"precision": round(base_rate, 4),
                               "recall": 1.0},
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    lines = ["# 表11（增强特征版）候选预筛选模型离线评估结果", ""]
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
