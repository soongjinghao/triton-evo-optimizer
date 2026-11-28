#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and analyze the MSProf-feature / kernel-latency dataset.

Expected layout:
  <root>/costModel/datasets/<operator>/data/<role>/metadata.json
  <root>/costModel/datasets/<operator>/data/<role>/profiling/profile_summary.json
  <root>/costModel/datasets/<operator>/data/<role>/ir/{ttir,ttadapter,hivm_opt,...}/

The script is read-only with respect to the collected dataset. All outputs are
written under costModel/analysis/results/ by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_VERSION = "0.1.0"

# These are intentionally excluded from X. They are time labels, launch/task
# identity fields, or collection metadata and would cause leakage / noise.
KERNEL_NON_FEATURE_KEYS = {
    "device_id",
    "task_id",
    "stream_id",
    "op_name",
    "op_type",
    "op_state",
    "task_type",
    "task_start_time_us",
    "task_duration_us",
    "hf32_eligible",
    "input_shapes",
    "input_data_types",
    "output_shapes",
    "output_data_types",
}

LABEL_CANDIDATES = (
    ("metadata.profiling.timing.label_execution_time_us", "metadata"),
    ("profile_summary.timing.label_execution_time_us", "profile_summary"),
    ("metadata.profiling.timing.execution_time_us", "metadata"),
    ("profile_summary.timing.execution_time_us", "profile_summary"),
    ("metadata.evaluation.execution_time_us", "metadata_evaluation_fallback"),
)


def read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def get_path(obj: Any, dotted_path: str, default: Any = None) -> Any:
    cur = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_positive_float(value: Any) -> Optional[float]:
    number = as_float(value)
    return number if number is not None and number > 0 else None


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "success", "passed"}:
            return True
        if normalized in {"false", "0", "no", "n", "failed", "invalid"}:
            return False
    return None


def clean_key(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("%", "pct")
    text = re.sub(r"\([^)]*\)", lambda m: "_" + m.group(0)[1:-1].replace("/", "_per_").replace(" ", "_"), text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unnamed"


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def find_profile_summary(role_dir: Path, metadata: Optional[dict]) -> Tuple[Optional[dict], Optional[Path], str]:
    candidates: List[Path] = [role_dir / "profiling" / "profile_summary.json"]
    if metadata:
        rel = get_path(metadata, "profiling.summary_path")
        if isinstance(rel, str) and rel.strip():
            p = Path(rel)
            candidates.append(p if p.is_absolute() else role_dir / p)
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            value = read_json(path)
            if value is not None:
                return value, path, "profile_summary_file"
    # The collector mirrors profile_summary sections into metadata.profiling.
    if metadata and isinstance(metadata.get("profiling"), dict):
        inline = metadata["profiling"]
        if any(k in inline for k in ("kernel", "core_utilization", "memory", "timing")):
            return inline, None, "metadata_inline_profiling"
    return None, None, "missing"


def resolve_label(metadata: Optional[dict], profile: Optional[dict]) -> Tuple[Optional[float], str, Optional[bool], str]:
    """Return runtime_us, source, label_valid, validation_state."""
    label_valid = get_path(metadata or {}, "profiling.timing.label_valid")
    if label_valid is None:
        label_valid = get_path(profile or {}, "timing.label_valid")
    label_valid = as_bool(label_valid)

    sources = [
        (get_path(metadata or {}, "profiling.timing.label_execution_time_us"), "metadata.profiling.timing.label_execution_time_us"),
        (get_path(profile or {}, "timing.label_execution_time_us"), "profile_summary.timing.label_execution_time_us"),
        (get_path(metadata or {}, "profiling.timing.execution_time_us"), "metadata.profiling.timing.execution_time_us"),
        (get_path(profile or {}, "timing.execution_time_us"), "profile_summary.timing.execution_time_us"),
        (get_path(metadata or {}, "evaluation.execution_time_us"), "metadata.evaluation.execution_time_us"),
    ]
    for value, source in sources:
        number = as_positive_float(value)
        if number is not None:
            state = "valid" if label_valid is True else "invalid" if label_valid is False else "unknown"
            return number, source, label_valid, state

    state = "invalid" if label_valid is False else "unknown" if label_valid is None else "missing"
    return None, "unavailable", label_valid, state


def workload_fields(profile: Optional[dict], metadata: Optional[dict]) -> Dict[str, str]:
    kernel = (profile or {}).get("kernel") or get_path(metadata or {}, "profiling.kernel", {}) or {}
    fields = {
        "input_shapes": str(kernel.get("Input Shapes", "") or ""),
        "input_data_types": str(kernel.get("Input Data Types", "") or ""),
        "output_shapes": str(kernel.get("Output Shapes", "") or ""),
        "output_data_types": str(kernel.get("Output Data Types", "") or ""),
        "task_type": str(kernel.get("Task Type", "") or ""),
        "op_type": str(kernel.get("OP Type", "") or ""),
    }
    shape_keys = ("input_shapes", "input_data_types", "output_shapes", "output_data_types")
    if any(fields[k] for k in shape_keys):
        signature_obj = {k: fields[k] for k in shape_keys}
        signature = json.dumps(signature_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        signature_source = "profile_kernel_shapes_and_types"
    else:
        test_hash = get_path(metadata or {}, "source.test_sha256")
        if test_hash:
            signature = f"test_sha256:{test_hash}"
            signature_source = "metadata.source.test_sha256"
        else:
            candidate_id = (metadata or {}).get("candidate_id") or "unknown_candidate"
            signature = f"unknown:{candidate_id}"
            signature_source = "candidate_specific_unknown"
    fields["workload_signature"] = signature
    fields["workload_signature_source"] = signature_source
    return fields


def ir_counts(role_dir: Path) -> Dict[str, int]:
    ir_dir = role_dir / "ir"
    patterns = {
        "ttir_count": ("ttir", (".ttir.mlir", ".mlir")),
        "ttadapter_count": ("ttadapter", (".ttadapter.mlir", ".mlir")),
        "hivm_opt_count": ("hivm_opt", (".mlir",)),
        "hivm_count": ("hivm", (".mlir",)),
        "npubin_count": ("npubin", (".npubin",)),
        "npuir_count": ("npuir", (".mlir", ".ir", ".txt")),
    }
    result: Dict[str, int] = {}
    for output_key, (folder, suffixes) in patterns.items():
        directory = ir_dir / folder
        if not directory.exists():
            result[output_key] = 0
            continue
        files = [p for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json" and p.suffix.lower() in suffixes]
        # TTIR / TTAdapter / NPUBIN folders have stage-specific suffixes; this
        # avoids counting unrelated manifest or capture files.
        result[output_key] = len(files)
    return result


def flatten_numeric(obj: Any, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}__{clean_key(key)}" if prefix else clean_key(key)
            if isinstance(value, dict):
                out.update(flatten_numeric(value, child))
            else:
                number = as_float(value)
                if number is not None:
                    out[child] = number
    return out


def extract_profile_features(profile: Optional[dict], metadata: Optional[dict]) -> Dict[str, float]:
    if not profile:
        return {}
    kernel = profile.get("kernel") or {}
    core = profile.get("core_utilization") or {}
    memory = profile.get("memory") or {}

    task_type = str(kernel.get("Task Type", "") or "").upper()
    ai_core_info = core.get("ai_core") if isinstance(core, dict) else None
    ai_vector_info = core.get("ai_vector_core") if isinstance(core, dict) else None

    aic_app = None
    aiv_app = None
    if isinstance(ai_core_info, dict):
        app = str(ai_core_info.get("applicability", "")).lower()
        if app:
            aic_app = "not_applicable" not in app and app not in {"false", "no", "unavailable"}
    if isinstance(ai_vector_info, dict):
        app = str(ai_vector_info.get("applicability", "")).lower()
        if app:
            aiv_app = "not_applicable" not in app and app not in {"false", "no", "unavailable"}

    if aic_app is None and task_type:
        if "VECTOR" in task_type:
            aic_app = False
        elif "AI_CORE" in task_type or "CUBE" in task_type:
            aic_app = True
    if aiv_app is None and task_type:
        if "VECTOR" in task_type:
            aiv_app = True
        elif "AI_CORE" in task_type or "CUBE" in task_type:
            aiv_app = False

    features: Dict[str, float] = {}

    # Kernel-level hardware counters. Task Duration is deliberately excluded:
    # the independently collected timing label is the target variable.
    for key, value in kernel.items():
        normalized = clean_key(key)
        if normalized in KERNEL_NON_FEATURE_KEYS:
            continue
        if normalized.startswith("aic_") or normalized.startswith("aicore_") or normalized.startswith("cube_utilization"):
            if aic_app is False:
                continue
        if normalized.startswith("aiv_"):
            if aiv_app is False:
                continue
        number = as_float(value)
        if number is not None:
            features[f"kernel__{normalized}"] = number

    # Per-core utilization. Do not turn explicit "not applicable" zeros into
    # numeric observations; those zeros mean "not measured for this task type".
    for core_name, core_info in (core.items() if isinstance(core, dict) else []):
        if not isinstance(core_info, dict):
            continue
        applicability = str(core_info.get("applicability", "")).lower()
        if "not_applicable" in applicability or applicability in {"false", "no", "unavailable"}:
            continue
        if core_name == "ai_core" and aic_app is False:
            continue
        if core_name == "ai_vector_core" and aiv_app is False:
            continue
        for key, value in core_info.items():
            if key in {"source", "task_type", "applicability"}:
                continue
            number = as_float(value)
            if number is not None:
                features[f"core_utilization__{clean_key(core_name)}__{clean_key(key)}"] = number

    # Memory hierarchy / system-memory counters.
    if isinstance(memory, dict):
        features.update(flatten_numeric(memory, "memory"))

    return features


def collect_candidate(role_dir: Path, operator: str, root: Path, require_valid_label: bool) -> Tuple[Dict[str, Any], List[str]]:
    metadata_path = role_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else None
    profile, profile_path, profile_source = find_profile_summary(role_dir, metadata)
    runtime_us, label_source, label_valid, label_state = resolve_label(metadata, profile)
    ir = ir_counts(role_dir)
    workload = workload_fields(profile, metadata)
    features = extract_profile_features(profile, metadata)

    candidate_id = (metadata or {}).get("candidate_id") or f"{operator}__{role_dir.name}"
    metadata_status = (metadata or {}).get("status") or "unknown"
    evaluation_success = as_bool(get_path(metadata or {}, "evaluation.success"))
    correctness_success = as_bool(get_path(metadata or {}, "correctness.success"))
    hardware_profile_success = as_bool(get_path(metadata or {}, "evaluation.hardware_profile_success"))
    if hardware_profile_success is None:
        hardware_profile_success = as_bool(get_path(profile or {}, "collection.hardware_profile_success"))
    profile_mode = (metadata or {}).get("evaluation", {}).get("profile_mode") if metadata else None
    if profile_mode is None and profile:
        profile_mode = profile.get("profile_mode")

    issues: List[str] = []
    if metadata is None:
        issues.append("metadata_missing_or_invalid")
    elif metadata_status not in {"success", "unknown"}:
        issues.append(f"metadata_status_{metadata_status}")
    if evaluation_success is False:
        issues.append("evaluation_failed")
    if correctness_success is False:
        issues.append("correctness_failed")
    if label_state == "invalid":
        issues.append("latency_label_marked_invalid")
    elif runtime_us is None:
        issues.append("latency_label_missing_or_nonpositive")
    elif label_state == "unknown":
        issues.append("latency_label_validity_unknown")
    if require_valid_label and label_valid is not True:
        issues.append("strict_valid_label_required")
    if profile is None:
        issues.append("profile_summary_missing")
    if hardware_profile_success is False:
        issues.append("hardware_profile_failed")
    if not features:
        issues.append("no_numeric_profile_features")
    if ir["ttir_count"] == 0:
        issues.append("ttir_missing")

    # A failed/invalid label or failed collection must never enter correlation.
    fatal = {
        "metadata_missing_or_invalid",
        "evaluation_failed",
        "correctness_failed",
        "latency_label_marked_invalid",
        "latency_label_missing_or_nonpositive",
        "strict_valid_label_required",
        "profile_summary_missing",
        "hardware_profile_failed",
        "no_numeric_profile_features",
    }
    usable = runtime_us is not None and not any(x in fatal for x in issues)

    row: Dict[str, Any] = {
        "operator": operator,
        "candidate_role": role_dir.name,
        "candidate_id": candidate_id,
        "candidate_dir": safe_relative(role_dir, root),
        "metadata_status": metadata_status,
        "evaluation_success": evaluation_success,
        "correctness_success": correctness_success,
        "hardware_profile_success": hardware_profile_success,
        "profile_mode": profile_mode,
        "profile_summary_source": profile_source,
        "profile_summary_path": safe_relative(profile_path, root) if profile_path else "",
        "runtime_us": runtime_us,
        "label_source": label_source,
        "label_valid": label_valid,
        "label_valid_state": label_state,
        "workload_signature": workload["workload_signature"],
        "workload_signature_source": workload["workload_signature_source"],
        "input_shapes": workload["input_shapes"],
        "input_data_types": workload["input_data_types"],
        "output_shapes": workload["output_shapes"],
        "output_data_types": workload["output_data_types"],
        "task_type": workload["task_type"],
        "op_type": workload["op_type"],
        **ir,
        "has_ttir": ir["ttir_count"] > 0,
        "has_ttadapter": ir["ttadapter_count"] > 0,
        "has_hivm_opt": ir["hivm_opt_count"] > 0,
        "has_npubin": ir["npubin_count"] > 0,
        "profile_feature_count": len(features),
        "usable_for_correlation": usable,
        "quality_issues": ";".join(issues),
    }
    row.update(features)
    return row, issues


def discover_candidates(dataset_root: Path) -> List[Tuple[str, Path]]:
    found: List[Tuple[str, Path]] = []
    if not dataset_root.is_dir():
        return found
    for operator_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        data_dir = operator_dir / "data"
        if not data_dir.is_dir():
            continue
        for role_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            # Only direct children of data/ are candidates. Raw MSProf nested
            # profile_summary files are therefore never mistaken for samples.
            if (role_dir / "metadata.json").is_file() or (role_dir / "profiling").exists() or (role_dir / "ir").exists():
                found.append((operator_dir.name, role_dir))
    return found


def safe_corr(x: Sequence[float], y: Sequence[float], method: str) -> Tuple[Optional[float], Optional[float], int, str]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    n = len(pairs)
    if n < 3:
        return None, None, n, "too_few_samples"
    xa = np.asarray([p[0] for p in pairs], dtype=float)
    ya = np.asarray([p[1] for p in pairs], dtype=float)
    if np.unique(xa).size < 2 or np.unique(ya).size < 2:
        return None, None, n, "constant_variable"
    try:
        if method == "spearman":
            result = stats.spearmanr(xa, ya)
        else:
            result = stats.pearsonr(xa, ya)
        return float(result.statistic), float(result.pvalue), n, "ok"
    except Exception as exc:
        return None, None, n, f"error:{type(exc).__name__}"


def analyze_correlations(df: pd.DataFrame, feature_cols: List[str], min_group_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable = df[df["usable_for_correlation"] == True].copy()  # noqa: E712
    operator_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    group_columns = ["operator", "workload_signature"]
    groups = list(usable.groupby(group_columns, dropna=False, sort=True)) if not usable.empty else []

    for feature in feature_cols:
        valid = usable[["runtime_us", feature]].dropna() if not usable.empty else pd.DataFrame()
        pooled_p, pooled_pp, pooled_n, pooled_ps = safe_corr(valid[feature], valid["runtime_us"], "pearson") if not valid.empty else (None, None, 0, "no_data")
        pooled_s, pooled_sp, _, pooled_ss = safe_corr(valid[feature], valid["runtime_us"], "spearman") if not valid.empty else (None, None, 0, "no_data")

        group_spearman: List[float] = []
        group_pearson: List[float] = []
        valid_groups = 0
        for (operator, signature), group in groups:
            pair = group[[feature, "runtime_us"]].dropna()
            pearson_r, pearson_p, n, pearson_status = safe_corr(pair[feature], pair["runtime_us"], "pearson")
            spearman_r, spearman_p, _, spearman_status = safe_corr(pair[feature], pair["runtime_us"], "spearman")
            status = "ok" if n >= min_group_size and spearman_status == "ok" else (
                "too_few_samples" if n < min_group_size else spearman_status
            )
            if status == "ok":
                group_spearman.append(float(spearman_r))
                if pearson_r is not None:
                    group_pearson.append(float(pearson_r))
                valid_groups += 1
            operator_rows.append({
                "operator": operator,
                "workload_signature": signature,
                "feature": feature,
                "n": n,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
                "status": status,
            })

        summary_rows.append({
            "feature": feature,
            "pooled_n": pooled_n,
            "pooled_pearson_r": pooled_p,
            "pooled_pearson_p": pooled_pp,
            "pooled_pearson_status": pooled_ps,
            "pooled_spearman_rho": pooled_s,
            "pooled_spearman_p": pooled_sp,
            "pooled_spearman_status": pooled_ss,
            "workload_group_count": len(groups),
            "valid_workload_group_count": valid_groups,
            "median_within_group_spearman": float(np.median(group_spearman)) if group_spearman else None,
            "mean_within_group_spearman": float(np.mean(group_spearman)) if group_spearman else None,
            "std_within_group_spearman": float(np.std(group_spearman, ddof=1)) if len(group_spearman) > 1 else None,
            "positive_group_fraction": float(np.mean(np.asarray(group_spearman) > 0)) if group_spearman else None,
            "negative_group_fraction": float(np.mean(np.asarray(group_spearman) < 0)) if group_spearman else None,
            "median_within_group_pearson": float(np.median(group_pearson)) if group_pearson else None,
            "primary_scope": "within_operator_workload_spearman",
        })

    summary_df = pd.DataFrame(summary_rows)
    operator_df = pd.DataFrame(operator_rows)
    if not summary_df.empty:
        summary_df["abs_median_within_group_spearman"] = summary_df["median_within_group_spearman"].abs()
        summary_df = summary_df.sort_values(
            ["abs_median_within_group_spearman", "valid_workload_group_count"],
            ascending=[False, False], na_position="last",
        )
    return summary_df, operator_df


def build_feature_coverage(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    usable = df[df["usable_for_correlation"] == True]  # noqa: E712
    rows = []
    denominator = len(usable)
    for feature in feature_cols:
        values = pd.to_numeric(usable[feature], errors="coerce") if feature in usable else pd.Series(dtype=float)
        non_null = int(values.notna().sum())
        unique = int(values.dropna().nunique())
        rows.append({
            "feature": feature,
            "usable_candidate_count": denominator,
            "non_null_count": non_null,
            "coverage_ratio": non_null / denominator if denominator else 0.0,
            "unique_values": unique,
            "is_constant": unique <= 1,
            "eligible_for_correlation": non_null >= 3 and unique > 1,
        })
    return pd.DataFrame(rows).sort_values(["coverage_ratio", "unique_values"], ascending=[False, False]) if rows else pd.DataFrame()


def make_heatmap(df: pd.DataFrame, feature_cols: List[str], coverage_df: pd.DataFrame, output_path: Path, max_features: int = 24) -> Dict[str, Any]:
    usable = df[df["usable_for_correlation"] == True].copy()  # noqa: E712
    if usable.empty:
        return {"created": False, "reason": "no_usable_candidates"}
    eligible = coverage_df[coverage_df["eligible_for_correlation"] == True]  # noqa: E712
    eligible_features = [f for f in eligible["feature"].tolist() if f in feature_cols]
    if not eligible_features:
        return {"created": False, "reason": "no_features_with_variation"}

    # Select a compact, interpretable subset by coverage first, then absolute
    # pooled Spearman association with latency. The chart is explicitly pooled
    # and descriptive; operator-level conclusions come from grouped analysis.
    corr_scores = []
    for feature in eligible_features:
        pair = usable[[feature, "runtime_us"]].dropna()
        rho, _, n, status = safe_corr(pair[feature], pair["runtime_us"], "spearman")
        if status == "ok":
            corr_scores.append((feature, abs(rho), n))
    corr_scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
    chosen = [x[0] for x in corr_scores[:max_features]]
    if not chosen:
        chosen = eligible_features[:max_features]
    columns = chosen + ["runtime_us"]
    matrix = usable[columns].apply(pd.to_numeric, errors="coerce").corr(method="spearman", min_periods=3)
    if matrix.empty:
        return {"created": False, "reason": "correlation_matrix_empty"}

    width = max(9, min(18, 0.42 * len(columns) + 5))
    height = max(7, min(16, 0.42 * len(columns) + 4))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(columns, fontsize=7)
    ax.set_title("Pooled Spearman correlation (descriptive; between-operator effects included)", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"created": True, "features": chosen, "matrix_columns": columns}


def build_quality_report(
    rows: List[Dict[str, Any]],
    issue_rows: List[Dict[str, Any]],
    expected_operators: int,
    expected_candidates: int,
    dataset_root: Path,
    root: Path,
) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    operator_count = int(df["operator"].nunique()) if not df.empty else 0
    candidate_count = len(rows)
    metadata_status_counts = dict(Counter(str(r.get("metadata_status", "unknown")) for r in rows))
    label_state_counts = dict(Counter(str(r.get("label_valid_state", "unknown")) for r in rows))
    hardware_counts = dict(Counter(
        "true" if r.get("hardware_profile_success") is True else
        "false" if r.get("hardware_profile_success") is False else "unknown"
        for r in rows
    ))
    usable_count = sum(bool(r.get("usable_for_correlation")) for r in rows)
    usable_operators = len({r["operator"] for r in rows if r.get("usable_for_correlation")})
    candidate_ids = [r.get("candidate_id") for r in rows if r.get("candidate_id")]
    duplicates = [k for k, v in Counter(candidate_ids).items() if v > 1]

    return {
        "schema_version": "profile_latency_quality_v1",
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root),
        "dataset_root": str(dataset_root),
        "expected": {"operators": expected_operators, "candidates": expected_candidates},
        "observed": {
            "operators": operator_count,
            "candidate_directories": candidate_count,
            "usable_candidates_for_correlation": usable_count,
            "usable_operators": usable_operators,
        },
        "count_check": {
            "operator_count_matches_expected": operator_count == expected_operators,
            "candidate_count_matches_expected": candidate_count == expected_candidates,
        },
        "metadata_status_counts": metadata_status_counts,
        "label_validity_state_counts": label_state_counts,
        "hardware_profile_success_counts": hardware_counts,
        "ir_availability": {
            stage: sum(1 for r in rows if int(r.get(key, 0) or 0) > 0)
            for stage, key in (
                ("ttir", "ttir_count"),
                ("ttadapter", "ttadapter_count"),
                ("hivm_opt", "hivm_opt_count"),
                ("hivm", "hivm_count"),
                ("npubin", "npubin_count"),
                ("npuir", "npuir_count"),
            )
        },
        "missing_or_failed_counts": {
            "metadata_missing_or_invalid": sum("metadata_missing_or_invalid" in r.get("quality_issues", "") for r in rows),
            "profile_summary_missing": sum("profile_summary_missing" in r.get("quality_issues", "") for r in rows),
            "latency_label_missing": sum("latency_label_missing_or_nonpositive" in r.get("quality_issues", "") for r in rows),
            "latency_label_invalid": sum("latency_label_marked_invalid" in r.get("quality_issues", "") for r in rows),
            "hardware_profile_failed": sum("hardware_profile_failed" in r.get("quality_issues", "") for r in rows),
            "no_numeric_profile_features": sum("no_numeric_profile_features" in r.get("quality_issues", "") for r in rows),
            "ttir_missing": sum("ttir_missing" in r.get("quality_issues", "") for r in rows),
        },
        "duplicate_candidate_ids": duplicates,
        "candidate_issues": issue_rows,
        "notes": [
            "The candidate-level CSV includes discovered candidate directories, including failed/incomplete records.",
            "Correlation uses only rows marked usable_for_correlation=true.",
            "Task Duration(us), timing.execution_time_us and label_execution_time_us are not Profiling input features.",
            "Explicitly not-applicable AI Core / AI Vector Core metrics are omitted rather than treated as measured zeros.",
            "Pooled correlations mix operators and are descriptive only; within-operator/workload correlations are the primary comparison.",
            "With about five candidates per operator, per-operator correlation estimates are exploratory and statistically unstable.",
        ],
    }


def write_summary_md(path: Path, quality: dict, corr_df: pd.DataFrame, coverage_df: pd.DataFrame) -> None:
    observed = quality["observed"]
    lines = [
        "# Profiling–Kernel Runtime Analysis Summary",
        "",
        f"Generated: {quality['generated_at']}",
        "",
        "## Dataset overview",
        f"- Operators: {observed['operators']} (expected {quality['expected']['operators']})",
        f"- Candidate directories: {observed['candidate_directories']} (expected {quality['expected']['candidates']})",
        f"- Usable candidates for correlation: {observed['usable_candidates_for_correlation']}",
        f"- Operators with usable candidates: {observed['usable_operators']}",
        "",
        "## Label and profile quality",
        f"- Label validity states: `{json.dumps(quality['label_validity_state_counts'], ensure_ascii=False)}`",
        f"- Hardware profile status: `{json.dumps(quality['hardware_profile_success_counts'], ensure_ascii=False)}`",
        f"- IR availability: `{json.dumps(quality['ir_availability'], ensure_ascii=False)}`",
        "",
        "## Strongest within-workload associations",
    ]
    if corr_df.empty:
        lines.append("No correlation results were available. Check data-quality output and candidate coverage.")
    else:
        top = corr_df.dropna(subset=["median_within_group_spearman"]).head(12)
        if top.empty:
            lines.append("No feature had enough within-workload variation and samples for correlation.")
        else:
            lines.append("| Feature | Median group Spearman | Valid groups | Positive groups |")
            lines.append("|---|---:|---:|---:|")
            for _, row in top.iterrows():
                rho = row["median_within_group_spearman"]
                pos = row["positive_group_fraction"]
                lines.append(f"| `{row['feature']}` | {rho:.3f} | {int(row['valid_workload_group_count'])} | {pos:.1%} |")
    lines.extend([
        "",
        "## Highest-coverage Profiling features",
    ])
    if coverage_df.empty:
        lines.append("No numeric Profiling features were found.")
    else:
        lines.append("| Feature | Coverage | Unique values | Eligible |")
        lines.append("|---|---:|---:|---|")
        for _, row in coverage_df.head(15).iterrows():
            lines.append(f"| `{row['feature']}` | {row['coverage_ratio']:.1%} | {int(row['unique_values'])} | {bool(row['eligible_for_correlation'])} |")
    lines.extend([
        "",
        "## Interpretation cautions",
        "- Treat `median_within_group_spearman` as the primary exploratory statistic; pooled correlation can be dominated by differences between operator types and workload sizes.",
        "- A positive coefficient means the feature tends to increase with runtime; a negative coefficient means it tends to decrease. Correlation does not establish causation.",
        "- Ratios/utilization are not automatically better when larger. Interpret them together with cycles, memory traffic, task type and operator semantics.",
        "- The current dataset has roughly five candidates per operator. Do not use individual per-operator p-values as strong evidence.",
        "- This module analyzes measured Profiling data. A deployable two-stage Cost Model must later predict these Profiling targets from IR without running MSProf at inference time.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    script_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Analyze MSProf features against measured Triton Kernel runtime.")
    parser.add_argument("--root", type=Path, default=script_root, help="Project root (default: inferred from this script location).")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root; default: <root>/costModel/datasets.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; default: costModel/analysis/results/profile_latency_<timestamp>.")
    parser.add_argument("--expected-operators", type=int, default=50)
    parser.add_argument("--expected-candidates", type=int, default=250)
    parser.add_argument("--min-group-size", type=int, default=3, help="Minimum paired candidates per operator/workload group for grouped correlation.")
    parser.add_argument("--require-valid-label", action="store_true", help="Only include labels with label_valid=true; default also allows unknown validation state but flags it.")
    args = parser.parse_args()

    root = args.root.resolve()
    dataset_root = (args.dataset_dir or (root / "costModel" / "datasets")).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (root / "costModel" / "analysis" / "results" / f"profile_latency_{stamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("Profiling / Kernel Runtime Analysis")
    print("=" * 76)
    print(f"Project root : {root}")
    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {output_dir}")

    candidates = discover_candidates(dataset_root)
    print(f"Discovered candidate directories: {len(candidates)}")
    if not candidates:
        print("ERROR: no candidate directories found. Expected costModel/datasets/<operator>/data/<role>/")
        return 2

    rows: List[Dict[str, Any]] = []
    issue_rows: List[Dict[str, Any]] = []
    for index, (operator, role_dir) in enumerate(candidates, start=1):
        row, issues = collect_candidate(role_dir, operator, root, args.require_valid_label)
        rows.append(row)
        if issues:
            issue_rows.append({
                "operator": operator,
                "candidate_role": role_dir.name,
                "candidate_id": row["candidate_id"],
                "candidate_dir": row["candidate_dir"],
                "issues": issues,
            })
        if index % 25 == 0 or index == len(candidates):
            print(f"Processed {index}/{len(candidates)} candidates")

    df = pd.DataFrame(rows)
    # Build a stable candidate-level table: identifiers first, feature columns last.
    fixed_columns = [
        "operator", "candidate_role", "candidate_id", "candidate_dir", "metadata_status",
        "evaluation_success", "correctness_success", "hardware_profile_success", "profile_mode",
        "profile_summary_source", "profile_summary_path", "runtime_us", "label_source", "label_valid",
        "label_valid_state", "workload_signature", "workload_signature_source", "input_shapes",
        "input_data_types", "output_shapes", "output_data_types", "task_type", "op_type",
        "ttir_count", "ttadapter_count", "hivm_opt_count", "hivm_count", "npubin_count", "npuir_count",
        "has_ttir", "has_ttadapter", "has_hivm_opt", "has_npubin", "profile_feature_count",
        "usable_for_correlation", "quality_issues",
    ]
    feature_cols = sorted([c for c in df.columns if c not in fixed_columns])
    for col in fixed_columns:
        if col not in df.columns:
            df[col] = None
    df = df[fixed_columns + feature_cols]
    df.to_csv(output_dir / "profile_latency_dataset.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    quality = build_quality_report(rows, issue_rows, args.expected_operators, args.expected_candidates, dataset_root, root)
    (output_dir / "data_quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    usable_df = df[df["usable_for_correlation"] == True]  # noqa: E712
    feature_cols = [
        c for c in feature_cols
        if pd.to_numeric(usable_df[c], errors="coerce").notna().sum() >= 3
        and pd.to_numeric(usable_df[c], errors="coerce").nunique(dropna=True) > 1
    ] if not usable_df.empty else []

    coverage_df = build_feature_coverage(df, feature_cols)
    coverage_df.to_csv(output_dir / "feature_coverage.csv", index=False, encoding="utf-8-sig")

    correlation_df, operator_df = analyze_correlations(df, feature_cols, args.min_group_size)
    correlation_df.to_csv(output_dir / "correlation_results.csv", index=False, encoding="utf-8-sig")
    operator_df.to_csv(output_dir / "operator_correlation.csv", index=False, encoding="utf-8-sig")

    heatmap_info = make_heatmap(df, feature_cols, coverage_df, output_dir / "correlation_heatmap.png")
    quality["analysis_outputs"] = {
        "candidate_dataset": "profile_latency_dataset.csv",
        "feature_coverage": "feature_coverage.csv",
        "correlation_results": "correlation_results.csv",
        "operator_correlation": "operator_correlation.csv",
        "correlation_heatmap": "correlation_heatmap.png" if heatmap_info.get("created") else None,
        "heatmap_status": heatmap_info,
    }
    (output_dir / "data_quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_md(output_dir / "analysis_summary.md", quality, correlation_df, coverage_df)

    print("\n========== RESULT ==========")
    print(f"Operators discovered : {quality['observed']['operators']}")
    print(f"Candidates discovered: {quality['observed']['candidate_directories']}")
    print(f"Usable candidates    : {quality['observed']['usable_candidates_for_correlation']}")
    print(f"Numeric X features   : {len(feature_cols)}")
    print(f"Workload groups      : {int(usable_df.groupby(['operator', 'workload_signature']).ngroups) if not usable_df.empty else 0}")
    print(f"Output directory     : {output_dir}")
    print("Files:")
    for name in (
        "profile_latency_dataset.csv",
        "data_quality_report.json",
        "feature_coverage.csv",
        "correlation_results.csv",
        "operator_correlation.csv",
        "correlation_heatmap.png",
        "analysis_summary.md",
    ):
        p = output_dir / name
        if p.exists():
            print(f"  - {name}")
    if quality["observed"]["candidate_directories"] != args.expected_candidates:
        print(f"WARNING: expected {args.expected_candidates} candidates, found {quality['observed']['candidate_directories']}.")
    if quality["observed"]["operators"] != args.expected_operators:
        print(f"WARNING: expected {args.expected_operators} operators, found {quality['observed']['operators']}.")
    print("===========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
