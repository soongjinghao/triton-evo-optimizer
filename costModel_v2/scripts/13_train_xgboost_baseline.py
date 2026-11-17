#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CostModel V2 - first XGBoost baseline.

Run from Agent project root:
    python costModel_v2/scripts/13_train_xgboost_baseline.py

The model predicts log_speedup_vs_seed. Validation uses GroupKFold by
operator_name, so candidates from one operator cannot appear in both the
training and validation folds.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except Exception as exc:
    raise SystemExit(
        "[train] ERROR: xgboost is not installed in this Python environment.\n"
        "Install with: python -m pip install xgboost\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

try:
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as exc:
    raise SystemExit(
        "[train] ERROR: scikit-learn is required.\n"
        "Install with: python -m pip install scikit-learn\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )


ROOT = Path(__file__).resolve().parents[2]

# IDs, labels, grouping keys and execution metadata are never model inputs.
EXCLUDE_COLUMNS = {
    "candidate_id",
    "operator_name",
    "selected_role",
    "label_latency_us",
    "log_latency_us",
    "seed_latency_us",
    "relative_latency_vs_seed",
    "speedup_vs_seed",
    "log_speedup_vs_seed",
    "measurement_count",
    "device_id",
    "compile_status",
    "source_sha256",
}


def spearman_corr(y_true, y_pred):
    a = pd.Series(np.asarray(y_true, dtype=float))
    b = pd.Series(np.asarray(y_pred, dtype=float))
    if a.nunique() < 2 or b.nunique() < 2:
        return None
    value = a.corr(b, method="spearman")
    return float(value) if pd.notna(value) else None


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman_corr(y_true, y_pred),
    }


def build_model(args):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the first grouped-CV XGBoost Cost Model."
    )
    parser.add_argument(
        "--input",
        default="costModel_v2/features/normalized_training_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="costModel_v2/models/xgboost_baseline",
    )
    parser.add_argument("--target", default="log_speedup_vs_seed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=3.0)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise SystemExit(f"[train] ERROR: input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"operator_name", args.target}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[train] ERROR: missing required columns: {sorted(missing)}")

    # Use only numeric TTIR-derived columns; explicitly exclude every label,
    # ID, grouping key and execution metadata field.
    candidate_features = [
        col for col in df.columns
        if col not in EXCLUDE_COLUMNS and col != args.target
    ]
    numeric_features = []
    for col in candidate_features:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted.replace([np.inf, -np.inf], np.nan)
            numeric_features.append(col)

    if not numeric_features:
        raise SystemExit("[train] ERROR: no numeric feature columns found.")

    # Global constant filtering is unsupervised (does not inspect target).
    features = [
        col for col in numeric_features
        if df[col].notna().any() and df[col].nunique(dropna=True) > 1
    ]

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[args.target, "operator_name"]).reset_index(drop=True)

    if len(df) < 10:
        raise SystemExit(f"[train] ERROR: too few valid rows: {len(df)}")

    groups = df["operator_name"].astype(str).to_numpy()
    y = df[args.target].astype(float).to_numpy()
    X = df[features].astype(float)

    group_count = len(np.unique(groups))
    n_splits = min(args.folds, group_count)
    if n_splits < 2:
        raise SystemExit("[train] ERROR: at least two distinct operators are required.")

    print("=" * 72)
    print("CostModel V2 - XGBoost baseline")
    print("=" * 72)
    print(f"Input             : {input_path}")
    print(f"Rows              : {len(df)}")
    print(f"Operators         : {group_count}")
    print(f"Numeric features   : {len(numeric_features)}")
    print(f"Non-constant feats : {len(features)}")
    print(f"Target             : {args.target}")
    print(f"Grouped CV folds   : {n_splits}")
    print(f"XGBoost version    : {xgb.__version__}")
    print("Leakage protection : labels/IDs/operator names excluded")
    print("=" * 72)

    splitter = GroupKFold(n_splits=n_splits)
    oof_xgb = np.full(len(df), np.nan, dtype=float)
    oof_dummy = np.full(len(df), np.nan, dtype=float)
    fold_ids = np.full(len(df), -1, dtype=int)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])
        overlap = train_groups.intersection(valid_groups)
        if overlap:
            raise RuntimeError(
                f"Group leakage detected in fold {fold}: {sorted(overlap)}"
            )

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # Remove features constant in this fold's training portion only.
        fold_features = [
            col for col in features
            if X_train[col].notna().any()
            and X_train[col].nunique(dropna=True) > 1
        ]
        if not fold_features:
            raise RuntimeError(f"No usable features in fold {fold}.")

        model = build_model(args)
        model.fit(X_train[fold_features], y_train)
        pred = model.predict(X_valid[fold_features])

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(np.zeros((len(train_idx), 1)), y_train)
        dummy_pred = dummy.predict(np.zeros((len(valid_idx), 1)))

        oof_xgb[valid_idx] = pred
        oof_dummy[valid_idx] = dummy_pred
        fold_ids[valid_idx] = fold

        xgb_metrics = regression_metrics(y_valid, pred)
        dummy_metrics = regression_metrics(y_valid, dummy_pred)

        fold_rows.append({
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "train_operators": int(len(train_groups)),
            "valid_operators": int(len(valid_groups)),
            "feature_count": int(len(fold_features)),
            **{f"xgb_{k}": v for k, v in xgb_metrics.items()},
            **{f"dummy_{k}": v for k, v in dummy_metrics.items()},
        })

        print(
            f"[fold {fold}/{n_splits}] "
            f"train={len(train_idx)} rows/{len(train_groups)} ops, "
            f"valid={len(valid_idx)} rows/{len(valid_groups)} ops, "
            f"features={len(fold_features)} | "
            f"XGB RMSE={xgb_metrics['rmse']:.5f}, "
            f"MAE={xgb_metrics['mae']:.5f}, "
            f"R2={xgb_metrics['r2']:.5f}, "
            f"Spearman={xgb_metrics['spearman']}"
        )

    if np.isnan(oof_xgb).any() or np.isnan(oof_dummy).any():
        raise RuntimeError("Some rows did not receive out-of-fold predictions.")

    overall_xgb = regression_metrics(y, oof_xgb)
    overall_dummy = regression_metrics(y, oof_dummy)

    # Final model is fitted on all labeled candidates after CV evaluation.
    final_features = [
        col for col in features
        if X[col].notna().any() and X[col].nunique(dropna=True) > 1
    ]
    final_model = build_model(args)
    final_model.fit(X[final_features], y)

    model_path = output_dir / "xgboost_baseline.json"
    final_model.get_booster().save_model(str(model_path))

    booster = final_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")

    importance_rows = [{
        "feature": feature,
        "gain": float(gain.get(feature, 0.0)),
        "weight": int(weight.get(feature, 0)),
        "cover": float(cover.get(feature, 0.0)),
    } for feature in final_features]
    importance_rows.sort(key=lambda row: row["gain"], reverse=True)
    pd.DataFrame(importance_rows).to_csv(
        output_dir / "feature_importance.csv", index=False
    )

    pred_df = pd.DataFrame({
        "candidate_id": df["candidate_id"] if "candidate_id" in df else "",
        "operator_name": groups,
        "selected_role": df["selected_role"] if "selected_role" in df else "",
        "fold": fold_ids,
        "y_true": y,
        "xgb_oof_prediction": oof_xgb,
        "dummy_oof_prediction": oof_dummy,
        "xgb_absolute_error": np.abs(y - oof_xgb),
        "dummy_absolute_error": np.abs(y - oof_dummy),
    })
    pred_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)

    report = {
        "experiment": "CostModel V2 first XGBoost baseline",
        "input": str(input_path),
        "rows": int(len(df)),
        "operators": int(group_count),
        "target": args.target,
        "target_definition": (
            "log(seed_latency_us / candidate_label_latency_us); "
            "higher values indicate faster candidates"
        ),
        "validation": {
            "method": "GroupKFold",
            "folds": int(n_splits),
            "group_column": "operator_name",
            "group_overlap": False,
            "xgboost_oof_metrics": overall_xgb,
            "dummy_mean_oof_metrics": overall_dummy,
            "fold_metrics_file": "fold_metrics.csv",
            "oof_predictions_file": "oof_predictions.csv",
        },
        "features": {
            "numeric_before_constant_filter": int(len(numeric_features)),
            "used_for_final_model": int(len(final_features)),
            "names": final_features,
            "selection_note": (
                "All numeric non-constant TTIR-derived columns were used. "
                "The globally label-ranked top-30 list was not used for CV, "
                "to avoid feature-selection leakage."
            ),
        },
        "xgboost": {
            "version": xgb.__version__,
            "parameters": {
                "objective": "reg:squarederror",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "reg_lambda": args.reg_lambda,
                "reg_alpha": args.reg_alpha,
                "tree_method": "hist",
                "n_jobs": args.n_jobs,
                "random_state": args.seed,
            },
        },
        "final_model": model_path.name,
        "feature_importance": "feature_importance.csv",
        "caveats": [
            "This is a small-data baseline, not a final performance claim.",
            "The final model is fitted on all labeled rows after grouped CV.",
            "Feature importance is descriptive, not causal.",
            "Hyperparameters have not been tuned.",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }

    with open(output_dir / "baseline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(output_dir / "model_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(final_features) + "\n")

    print("\n" + "=" * 72)
    print("OVERALL OUT-OF-FOLD RESULTS")
    print("=" * 72)
    for name, metrics in (
        ("XGBoost", overall_xgb),
        ("Dummy mean", overall_dummy),
    ):
        print(
            f"{name:12s} | "
            f"MAE={metrics['mae']:.6f} | "
            f"RMSE={metrics['rmse']:.6f} | "
            f"R2={metrics['r2']:.6f} | "
            f"Spearman={metrics['spearman']}"
        )

    print("\nSaved:")
    print(f"  Model       : {model_path}")
    print(f"  OOF preds   : {output_dir / 'oof_predictions.csv'}")
    print(f"  Fold metrics: {output_dir / 'fold_metrics.csv'}")
    print(f"  Importance  : {output_dir / 'feature_importance.csv'}")
    print(f"  Report      : {output_dir / 'baseline_report.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
