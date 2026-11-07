#!/usr/bin/env python3
"""Train and compare candidate-ranking models on the latest CPU-only dataset."""

import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_features_shared import (  # noqa: E402
    CODE_FEATURE_KEYS, compute_code_features, load_codes, normalize_pids,
)

DATA_PATH = ROOT / "experiments" / "dataset" / "samples.jsonl"
SPLIT_PATH = ROOT / "experiments" / "splits" / "dev15.json"
OUTPUT_DIR = ROOT / "experiments" / "results" / "candidate_filter_latest_20261027"
MODEL_PATH = ROOT / "experiments" / "dataset" / "candidate_filter_model__latest35.joblib"
REPORT_PATH = OUTPUT_DIR / "evaluation.json"
OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"

BASE_NUMERIC = ["log_parent", "gen", "pop_size", "tau"]
FULL_NUMERIC = BASE_NUMERIC + list(CODE_FEATURE_KEYS)
CATEGORICAL = ["operation", "mutation_type"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from exc
            rows.append(row)
    return rows


def make_dataset(path, train_kernels, holdout_kernels):
    codes = load_codes()
    rows = load_rows(path)
    selected = []
    excluded = {
        "wrong_split": 0,
        "gen0_or_non_evolutionary": 0,
        "invalid_latency": 0,
        "extreme_latency_ratio": 0,
        "missing_source_code": 0,
    }

    for row in rows:
        kernel = str(row.get("kernel", ""))
        if kernel not in train_kernels and kernel not in holdout_kernels:
            excluded["wrong_split"] += 1
            continue
        operation = str(row.get("operation") or "none")
        try:
            generation = int(row.get("gen") or 0)
            parent_latency = float(row["parent_latency"])
            child_latency = float(row["child_latency"])
        except (KeyError, TypeError, ValueError):
            excluded["invalid_latency"] += 1
            continue
        if generation <= 0 or operation not in {"mutation", "crossover"}:
            excluded["gen0_or_non_evolutionary"] += 1
            continue
        if (not math.isfinite(parent_latency) or not math.isfinite(child_latency)
                or parent_latency <= 0 or child_latency <= 0):
            excluded["invalid_latency"] += 1
            continue
        if child_latency / parent_latency > 10:
            excluded["extreme_latency_ratio"] += 1
            continue

        child_code = codes.get(str(row.get("candidate_id")))
        parent_ids = normalize_pids(row.get("parent_ids"))
        parent_code = next((codes[pid] for pid in parent_ids if pid in codes), None)
        if not child_code or parent_code is None:
            excluded["missing_source_code"] += 1
            continue

        features = compute_code_features(child_code, parent_code)
        if features["parent_code_missing"]:
            excluded["missing_source_code"] += 1
            continue
        sample = {
            "kernel": kernel,
            "method": str(row.get("method") or "none"),
            "run_id": str(row.get("run_id") or "none"),
            "gen": generation,
            "candidate_id": str(row.get("candidate_id") or ""),
            "parent_latency": parent_latency,
            "child_latency": child_latency,
            "y": int(child_latency < parent_latency),
            "log_parent": math.log1p(parent_latency),
            "pop_size": float(row.get("pop_size") or 0),
            "tau": float(row.get("tau")) if row.get("tau") is not None else -1.0,
            "operation": operation,
            "mutation_type": str(row.get("mutation_type") or "none"),
            **{name: features[name] for name in CODE_FEATURE_KEYS},
        }
        sample["pool"] = "|".join(
            (sample["kernel"], sample["method"], sample["run_id"], str(generation)))
        selected.append(sample)

    frame = pd.DataFrame(selected)
    if frame.empty:
        raise RuntimeError("No eligible samples remain after filtering.")
    if frame[["parent_latency", "child_latency", *FULL_NUMERIC]].isna().any().any():
        raise RuntimeError("Non-finite or missing numeric feature values remain.")
    return frame, excluded


def make_pipeline(model_name, params, numeric_features):
    dense_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler() if model_name.startswith("lr_") else "passthrough"),
    ])
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="none")),
                ("encode", dense_onehot),
            ]), CATEGORICAL),
        ],
        sparse_threshold=0,
    )
    if model_name.startswith("lr_"):
        classifier = LogisticRegression(max_iter=3000, **params)
    elif model_name.startswith("extra_trees_"):
        classifier = ExtraTreesClassifier(
            n_estimators=300, n_jobs=1, random_state=20261027, **params)
    elif model_name.startswith("random_forest_"):
        classifier = RandomForestClassifier(
            n_estimators=300, n_jobs=1, random_state=20261027, **params)
    elif model_name.startswith("hist_gb_"):
        classifier = HistGradientBoostingClassifier(
            max_iter=200, random_state=20261027, **params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


def candidate_specs():
    specs = []
    for feature_name, cols in (
        ("base", BASE_NUMERIC),
        ("online_full", FULL_NUMERIC),
    ):
        for c in (0.1, 1.0, 10.0):
            specs.append((
                f"lr_{feature_name}_C{c:g}_balanced", feature_name, cols,
                {"C": c, "class_weight": "balanced"},
            ))
            specs.append((
                f"lr_{feature_name}_C{c:g}_unweighted", feature_name, cols,
                {"C": c, "class_weight": None},
            ))
        for leaf in (2, 5, 10):
            specs.append((
                f"extra_trees_{feature_name}_leaf{leaf}", feature_name, cols,
                {"min_samples_leaf": leaf, "max_features": "sqrt",
                 "class_weight": "balanced"},
            ))
            specs.append((
                f"random_forest_{feature_name}_leaf{leaf}", feature_name, cols,
                {"min_samples_leaf": leaf, "max_features": "sqrt",
                 "class_weight": "balanced_subsample"},
            ))
        for leaf in (7, 15):
            specs.append((
                f"hist_gb_{feature_name}_leaf{leaf}", feature_name, cols,
                {"max_leaf_nodes": leaf, "learning_rate": 0.05,
                 "l2_regularization": 1.0, "min_samples_leaf": 15},
            ))
    return specs


def ranking_metrics(frame, scores):
    scored = frame[["pool", "y"]].copy()
    scored["score"] = scores
    p4_large, r4_large, baseline_large = [], [], []
    p13_large, r13_large = [], []
    p4_any, baseline_any = [], []
    pool_counts = {"all_ge_4": 0, "expanded_gt_13": 0}
    for _, pool in scored.groupby("pool", sort=False):
        labels = pool["y"].to_numpy()
        ranking = np.argsort(-pool["score"].to_numpy())
        if len(pool) >= 4:
            pool_counts["all_ge_4"] += 1
            p4_any.append(float(labels[ranking[:4]].mean()))
            baseline_any.append(float(labels.mean()))
        if len(pool) > 13:
            pool_counts["expanded_gt_13"] += 1
            selected4 = labels[ranking[:4]]
            selected13 = labels[ranking[:13]]
            positives = int(labels.sum())
            p4_large.append(float(selected4.mean()))
            r4_large.append(float(selected4.sum() / positives) if positives else 0.0)
            p13_large.append(float(selected13.mean()))
            r13_large.append(float(selected13.sum() / positives) if positives else 0.0)
            baseline_large.append(float(labels.mean()))

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    return {
        "pool_count_ge_4": pool_counts["all_ge_4"],
        "pool_count_gt_13": pool_counts["expanded_gt_13"],
        "p_at_4_all_pools_ge_4": mean_or_none(p4_any),
        "random_p_at_4_all_pools_ge_4": mean_or_none(baseline_any),
        "p_at_4_expanded_pools_gt_13": mean_or_none(p4_large),
        "random_p_at_4_expanded_pools_gt_13": mean_or_none(baseline_large),
        "r_at_4_expanded_pools_gt_13": mean_or_none(r4_large),
        "p_at_13_expanded_pools_gt_13": mean_or_none(p13_large),
        "r_at_13_expanded_pools_gt_13": mean_or_none(r13_large),
    }


def evaluate_cv(frame, spec):
    model_name, feature_name, numeric_features, params = spec
    x_cols = numeric_features + CATEGORICAL
    groups = frame["kernel"].to_numpy()
    predictions = np.full(len(frame), np.nan)
    fold_auc = []
    splitter = GroupKFold(n_splits=5)

    for train_idx, test_idx in splitter.split(frame[x_cols], frame["y"], groups):
        model = make_pipeline(model_name, params, numeric_features)
        model.fit(frame.iloc[train_idx][x_cols], frame.iloc[train_idx]["y"])
        prediction = model.predict_proba(frame.iloc[test_idx][x_cols])[:, 1]
        predictions[test_idx] = prediction
        if len(np.unique(frame.iloc[test_idx]["y"])) == 2:
            fold_auc.append(float(roc_auc_score(
                frame.iloc[test_idx]["y"], prediction)))

    if np.isnan(predictions).any():
        raise RuntimeError(f"{model_name}: OOF prediction was not filled.")
    result = {
        "model": model_name,
        "feature_set": feature_name,
        "n_samples": len(frame),
        "roc_auc_oof": float(roc_auc_score(frame["y"], predictions)),
        "roc_auc_fold_mean": float(np.mean(fold_auc)),
        "roc_auc_fold_std": float(np.std(fold_auc)),
        "pr_auc_oof": float(average_precision_score(frame["y"], predictions)),
        **ranking_metrics(frame, predictions),
    }
    return result, predictions


def main():
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_kernels = set(split["train_kernels"])
    holdout_kernels = set(split["test_kernels"])
    if train_kernels & holdout_kernels or len(train_kernels) != 35 or len(holdout_kernels) != 15:
        raise RuntimeError("Expected a disjoint 35/15 kernel split.")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    frame, exclusions = make_dataset(DATA_PATH, train_kernels, holdout_kernels)
    train = frame[frame["kernel"].isin(train_kernels)].reset_index(drop=True)
    holdout = frame[frame["kernel"].isin(holdout_kernels)].reset_index(drop=True)
    if train["kernel"].nunique() != 35 or holdout["kernel"].nunique() != 15:
        raise RuntimeError("The filtered data does not cover all 35/15 split kernels.")

    print(f"Latest dataset: {DATA_PATH} ({sha256_file(DATA_PATH)})")
    print(f"Eligible train: {len(train)} rows / {train['kernel'].nunique()} kernels; "
          f"positive rate={train['y'].mean():.3f}")
    print(f"Eligible holdout: {len(holdout)} rows / {holdout['kernel'].nunique()} kernels; "
          f"positive rate={holdout['y'].mean():.3f}")
    print(f"Excluded rows: {exclusions}")

    results = []
    predictions_by_model = {}
    for spec in candidate_specs():
        result, predictions = evaluate_cv(train, spec)
        results.append(result)
        predictions_by_model[result["model"]] = (spec, predictions)
        print(
            f"{result['model']:42s} AUC={result['roc_auc_oof']:.4f} "
            f"PR-AUC={result['pr_auc_oof']:.4f} "
            f"P@4(pool>13)={result['p_at_4_expanded_pools_gt_13']}"
        )

    eligible = [r for r in results if r["pool_count_gt_13"] > 0]
    if eligible:
        selection_rule = (
            "Highest OOF P@4 averaged over reconstructed training candidate "
            "pools with more than 13 candidates; tie-break by OOF PR-AUC then "
            "ROC-AUC."
        )
        best = max(
            eligible,
            key=lambda r: (
                r["p_at_4_expanded_pools_gt_13"],
                r["pr_auc_oof"],
                r["roc_auc_oof"],
            ),
        )
    else:
        selection_rule = (
            "No reconstructed training candidate pool has more than 13 "
            "candidates, so select by OOF PR-AUC (primary, due to class "
            "imbalance), then ROC-AUC; P@4 is not used for model selection."
        )
        best = max(
            results,
            key=lambda r: (r["pr_auc_oof"], r["roc_auc_oof"]),
        )
    best_spec, oof_predictions = predictions_by_model[best["model"]]
    best_name, best_feature_set, best_numeric, best_params = best_spec
    final_model = make_pipeline(best_name, best_params, best_numeric)
    x_cols = best_numeric + CATEGORICAL
    final_model.fit(train[x_cols], train["y"])

    holdout_predictions = final_model.predict_proba(holdout[x_cols])[:, 1]
    holdout_metrics = {
        "n_samples": len(holdout),
        "n_kernels": int(holdout["kernel"].nunique()),
        "positive_rate": float(holdout["y"].mean()),
        "roc_auc": float(roc_auc_score(holdout["y"], holdout_predictions)),
        "pr_auc": float(average_precision_score(holdout["y"], holdout_predictions)),
        **ranking_metrics(holdout, holdout_predictions),
        "interpretation": (
            "Exploratory holdout diagnostic only: this same 15-kernel split has "
            "been inspected in earlier model experiments, so it is not a pristine "
            "one-time test set."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    model_hash = sha256_file(MODEL_PATH)
    report = {
        "schema": "candidate-filter-latest-35-15-v1",
        "dataset": str(DATA_PATH),
        "dataset_sha256": sha256_file(DATA_PATH),
        "split": str(SPLIT_PATH),
        "split_sha256": sha256_file(SPLIT_PATH),
        "split_version": split.get("version"),
        "n_raw_rows": sum(1 for line in DATA_PATH.open(encoding="utf-8") if line.strip()),
        "training_protocol": (
            "Latest JSONL; restrict to the fixed 35 training kernels; use only "
            "generation>0 mutation/crossover rows with valid latency ratio<=10 "
            "and recoverable child/parent source; select models using 5-fold "
            "GroupKFold by kernel."
        ),
        "feature_availability": (
            "Only search-time parent/state metadata and child/parent source "
            "features; no child latency, correctness outcome, profile, label, or "
            "post-evaluation field is used as an input."
        ),
        "excluded_row_counts": exclusions,
        "n_train_rows": len(train),
        "n_train_kernels": int(train["kernel"].nunique()),
        "train_positive_count": int(train["y"].sum()),
        "train_negative_count": int((1 - train["y"]).sum()),
        "holdout_rows_for_diagnostic": len(holdout),
        "candidate_models": results,
        "selection_rule": selection_rule,
        "selected_model": best,
        "selected_params": best_params,
        "selected_feature_set": best_feature_set,
        "holdout_diagnostic": holdout_metrics,
        "model_path": str(MODEL_PATH),
        "model_sha256": model_hash,
        "model_is_trained_on": sorted(train_kernels),
        "model_excludes": sorted(holdout_kernels),
        "caveats": [
            "The 35 training kernels determine model selection; do not use holdout metrics to pick a different model.",
            "The 15-kernel holdout has prior experimental use and is exploratory, not a pristine test.",
            "The filtered training data contains no reconstructed pool larger than 13 candidates; P@4 could not be used to select the model.",
            "Pool IDs are reconstructed as kernel|method|run_id|generation because the source JSONL has no explicit pool_id.",
            "The saved model is trained only on the 35 training kernels; it can be used to evaluate the 15 held-out kernels without training leakage.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    meta = {
        "model_out": str(MODEL_PATH),
        "model_sha256": model_hash,
        "dataset_snapshot": str(DATA_PATH),
        "dataset_sha256": report["dataset_sha256"],
        "split": str(SPLIT_PATH),
        "split_sha256": report["split_sha256"],
        "train_kernels": sorted(train_kernels),
        "exclude_kernels": sorted(holdout_kernels),
        "n_train_kernels": len(train_kernels),
        "n_samples": len(train),
        "positive_count": int(train["y"].sum()),
        "negative_count": int((1 - train["y"]).sum()),
        "features": {"numeric": best_numeric, "categorical": CATEGORICAL},
        "selected_model": best_name,
        "selected_params": best_params,
        "feature_set": best_feature_set,
    }
    MODEL_PATH.with_name(MODEL_PATH.stem + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    oof = train[["kernel", "method", "run_id", "gen", "candidate_id",
                 "parent_latency", "child_latency", "y", "pool"]].copy()
    oof["score"] = oof_predictions
    oof.to_csv(OOF_PATH, index=False)
    print(f"\nSelected: {best_name} / {best_feature_set}")
    print(f"CV metrics: AUC={best['roc_auc_oof']:.4f}, "
          f"PR-AUC={best['pr_auc_oof']:.4f}, "
          f"P@4 expanded pools={best['p_at_4_expanded_pools_gt_13']}")
    print(f"Holdout diagnostic: AUC={holdout_metrics['roc_auc']:.4f}, "
          f"PR-AUC={holdout_metrics['pr_auc']:.4f}, "
          f"P@4 expanded pools={holdout_metrics['p_at_4_expanded_pools_gt_13']}")
    print(f"Model: {MODEL_PATH} ({model_hash})")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
