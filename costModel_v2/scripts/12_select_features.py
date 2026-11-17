#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First-pass, operator-aware feature filtering for CostModel V2."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

META_COLUMNS = [
    "candidate_id", "operator_name", "selected_role",
    "label_latency_us", "log_latency_us", "seed_latency_us",
    "relative_latency_vs_seed", "speedup_vs_seed",
    "log_speedup_vs_seed", "measurement_count", "device_id",
    "compile_status", "source_sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    vx = sum(x * x for x in dx)
    vy = sum(y * y for y in dy)
    if vx <= 1e-20 or vy <= 1e-20:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(vx * vy)


def center_by_operator(
    values: list[float | None], operators: list[str]
) -> list[float | None]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (operator, value) in enumerate(zip(operators, values)):
        if value is not None:
            groups[operator].append(i)

    centered: list[float | None] = [None] * len(values)
    for indices in groups.values():
        mean = sum(values[i] for i in indices) / len(indices)  # type: ignore[arg-type]
        for i in indices:
            centered[i] = values[i] - mean  # type: ignore[operator]
    return centered


def aligned_pearson(
    a: list[float | None], b: list[float | None]
) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    return pearson([x for x, _ in pairs], [y for _, y in pairs])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--input",
        default="costModel_v2/features/normalized_training_dataset.csv",
    )
    parser.add_argument(
        "--ranking",
        default="costModel_v2/features/feature_ranking.csv",
    )
    parser.add_argument("--output-dir", default="costModel_v2/features")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--corr-threshold", type=float, default=0.98)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if not 0 < args.corr_threshold <= 1:
        parser.error("--corr-threshold must be in (0, 1]")
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be in (0, 1]")

    root = Path(args.root).expanduser().resolve()

    def resolve(path_text: str) -> Path:
        path = Path(path_text).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    input_path = resolve(args.input)
    ranking_path = resolve(args.ranking)
    output_dir = resolve(args.output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Normalized dataset not found: {input_path}")
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Feature ranking not found: {ranking_path}")

    rows = read_csv(input_path)
    ranking_rows = read_csv(ranking_path)
    if not rows:
        raise RuntimeError(f"Normalized dataset is empty: {input_path}")
    if not ranking_rows:
        raise RuntimeError(f"Feature ranking is empty: {ranking_path}")

    columns = list(rows[0].keys())
    if "operator_name" not in columns:
        raise RuntimeError("Input CSV is missing required column: operator_name")

    feature_columns = [c for c in columns if c not in META_COLUMNS]
    if not feature_columns:
        raise RuntimeError("No feature columns found after excluding metadata/labels")

    operators = [r.get("operator_name", "") for r in rows]
    ranking_map = {r.get("feature", ""): r for r in ranking_rows}

    values: dict[str, list[float | None]] = {
        feature: [to_float(row.get(feature)) for row in rows]
        for feature in feature_columns
    }
    centered = {
        feature: center_by_operator(values[feature], operators)
        for feature in feature_columns
    }

    stats: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []

    # Quality checks and scores inherited from script 10's operator-aware ranking.
    for feature in feature_columns:
        valid = [x for x in values[feature] if x is not None]
        unique_count = len(set(valid))
        coverage = len(valid) / len(rows)
        centered_valid = [x for x in centered[feature] if x is not None]
        within_variance = (
            sum(x * x for x in centered_valid) / len(centered_valid)
            if centered_valid else 0.0
        )

        rank_row = ranking_map.get(feature, {})
        spearman = to_float(rank_row.get("spearman_within_operator"))
        pearson_within = to_float(rank_row.get("pearson_within_operator"))
        rank_score = abs(spearman) if spearman is not None else 0.0
        tie_score = abs(pearson_within) if pearson_within is not None else 0.0

        status = "eligible"
        reason = ""
        if coverage < args.min_coverage:
            status, reason = "excluded", "low_coverage"
        elif unique_count < 2 or within_variance <= 1e-12:
            status, reason = "excluded", "constant_within_operator"
        else:
            eligible.append(feature)

        stats[feature] = {
            "feature": feature,
            "status": status,
            "reason": reason,
            "representative_feature": "",
            "selected_rank": "",
            "valid_count": len(valid),
            "coverage": coverage,
            "unique_values": unique_count,
            "within_operator_variance": within_variance,
            "spearman_within_operator": spearman,
            "pearson_within_operator": pearson_within,
            "rank_score": rank_score,
            "tie_score": tie_score,
        }

    # Rank by the same absolute within-operator Spearman statistic as script 10.
    eligible.sort(
        key=lambda f: (
            stats[f]["rank_score"],
            stats[f]["tie_score"],
            stats[f]["within_operator_variance"],
            f,
        ),
        reverse=True,
    )

    # Record all high-correlation pairs among eligible features.
    redundant_pairs: list[dict[str, Any]] = []
    for i, feature_a in enumerate(eligible):
        for feature_b in eligible[i + 1:]:
            corr = aligned_pearson(centered[feature_a], centered[feature_b])
            if corr is not None and abs(corr) >= args.corr_threshold:
                redundant_pairs.append({
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "within_operator_pearson": corr,
                    "abs_correlation": abs(corr),
                })

    redundant_pairs.sort(key=lambda r: r["abs_correlation"], reverse=True)

    # Greedy de-duplication: preserve the higher-ranked feature as representative.
    kept: list[str] = []
    for feature in eligible:
        matches: list[tuple[float, str]] = []
        for selected_feature in kept:
            corr = aligned_pearson(centered[feature], centered[selected_feature])
            if corr is not None and abs(corr) >= args.corr_threshold:
                matches.append((abs(corr), selected_feature))

        if matches:
            _, representative = max(matches)
            stats[feature]["status"] = "excluded_redundant"
            stats[feature]["reason"] = "correlated_with_" + representative
            stats[feature]["representative_feature"] = representative
        else:
            kept.append(feature)
            stats[feature]["status"] = "non_redundant_candidate"
            stats[feature]["reason"] = "passed_correlation_filter"

    selected = kept[:args.top_k]
    for rank, feature in enumerate(selected, start=1):
        stats[feature]["status"] = "selected"
        stats[feature]["reason"] = "top_ranked_non_redundant"
        stats[feature]["selected_rank"] = rank

    for feature in kept[args.top_k:]:
        stats[feature]["status"] = "not_in_top_k"
        stats[feature]["reason"] = "ranked_below_top_k"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Include every emitted dictionary key in fieldnames (fixes the prior tie_score crash).
    ranking_fields = [
        "feature", "status", "reason", "representative_feature",
        "selected_rank", "valid_count", "coverage", "unique_values",
        "within_operator_variance", "spearman_within_operator",
        "pearson_within_operator", "rank_score", "tie_score",
    ]
    ranking_output = sorted(
        stats.values(),
        key=lambda s: (
            0 if s["status"] == "selected" else 1,
            s["selected_rank"] if isinstance(s["selected_rank"], int) else 10**9,
            -s["rank_score"],
            s["feature"],
        ),
    )
    write_csv(output_dir / "feature_selection_ranking.csv", ranking_output, ranking_fields)

    write_csv(
        output_dir / "feature_redundancy_pairs.csv",
        redundant_pairs,
        ["feature_a", "feature_b", "within_operator_pearson", "abs_correlation"],
    )

    meta_columns = [c for c in META_COLUMNS if c in columns]
    dataset_sizes = sorted(set([10, 20, 30, 40, args.top_k]))
    for requested_size in dataset_sizes:
        subset = selected[:requested_size]
        if not subset:
            continue
        actual_size = len(subset)
        output_rows = [
            {c: row.get(c, "") for c in meta_columns + subset}
            for row in rows
        ]
        write_csv(
            output_dir / f"selected_feature_dataset_top{actual_size}.csv",
            output_rows,
            meta_columns + subset,
        )

    selected_name = f"selected_features_top{len(selected)}.txt"
    (output_dir / selected_name).write_text(
        "\n".join(selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )

    report = {
        "rows": len(rows),
        "operators": len(set(operators)),
        "original_feature_count": len(feature_columns),
        "eligible_feature_count": len(eligible),
        "non_redundant_feature_count": len(kept),
        "selected_feature_count": len(selected),
        "parameters": {
            "top_k": args.top_k,
            "correlation_threshold": args.corr_threshold,
            "minimum_coverage": args.min_coverage,
        },
        "redundant_pair_count": len(redundant_pairs),
        "selected_features": selected,
        "selection_score": "absolute within-operator Spearman correlation from feature_ranking.csv",
        "redundancy_score": "absolute Pearson correlation after within-operator centering",
        "caveat": (
            "Exploratory filtering only. Feature-target association is not causal importance. "
            "Final model evaluation must split by operator to avoid candidate leakage."
        ),
    }
    (output_dir / "feature_selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 68)
    print("CostModel V2 Feature Selection")
    print("=" * 68)
    print("Rows                   :", len(rows))
    print("Operators              :", len(set(operators)))
    print("Original features      :", len(feature_columns))
    print("Eligible features      :", len(eligible))
    print("Redundant pairs        :", len(redundant_pairs))
    print("Non-redundant features :", len(kept))
    print("Selected features      :", len(selected))
    print("\nTop selected features:")
    for i, feature in enumerate(selected[:15], 1):
        print(
            f"{i:2d}. {feature:42s} "
            f"within_spearman={stats[feature]['spearman_within_operator']}"
        )
    print("\nOutput directory:", output_dir)
    print("Ranking           : feature_selection_ranking.csv")
    print("Redundancy pairs  : feature_redundancy_pairs.csv")
    print("Report            : feature_selection_report.json")
    print("Selected list     :", selected_name)
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
