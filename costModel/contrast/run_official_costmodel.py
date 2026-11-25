#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton-Ascend Official Cost Model ranking baseline (v2).

Target runtime: Triton-Ascend 3.2.0.
Dataset: costModel/datasets/<operator>/data/<role>/metadata.json

Design:
- One candidate per costmodel_bench call (3.2.0 adapter wraps it in a one-item list).
- Load per-operator/per-candidate arg bindings from JSON, or metadata if present.
- Auto-bind pid_{x,y,z}=0 for axes referenced by tt.get_program_id.
- Keep official estimates separate from full-kernel msprof latency labels.
- Main evaluation is within-workload ranking (Spearman, Kendall-tau-b,
  Top-1, Top-5, selection regret, pairwise ranking accuracy).
- MAE/MAPE/RMSE are intentionally not used as primary metrics because
  the official documentation describes pid bindings as static estimates
  and does not establish that the returned estimate equals full-grid
  end-to-end kernel latency.

No compilation, test execution, hardware profiling, or dataset mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ARG_BINDING_KEYS = {"arg_bindings", "costmodel_arg_bindings", "runtime_bindings"}
HARDWARE_CONFIG_KEYS = {"hardware_config", "costmodel_hardware_config"}
PROGRAM_ID_RE = re.compile(r"\btt\.get_program_id\s+([xyz])\b")
NUM_PROGRAMS_RE = re.compile(r"\btt\.get_num_programs\s+([xyz])\b")


def read_json(path: Path) -> Optional[dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def first_nested_value(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys and v not in (None, ""):
                return v
        for v in obj.values():
            found = first_nested_value(v, keys)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = first_nested_value(v, keys)
            if found not in (None, ""):
                return found
    return None


def positive_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError):
        return None


def get_real_latency(metadata: dict, profile: Optional[dict]) -> Tuple[Optional[float], str, bool]:
    timing = ((metadata.get("profiling") or {}).get("timing") or {})
    valid = timing.get("label_valid")
    if profile and valid is None:
        valid = ((profile.get("timing") or {}).get("label_valid"))

    candidates = [
        (timing.get("label_execution_time_us"), "metadata.profiling.timing.label_execution_time_us"),
        (timing.get("execution_time_us"), "metadata.profiling.timing.execution_time_us"),
        ((metadata.get("evaluation") or {}).get("execution_time_us"), "metadata.evaluation.execution_time_us"),
    ]
    if profile:
        pt = profile.get("timing") or {}
        candidates.extend([
            (pt.get("label_execution_time_us"), "profile_summary.timing.label_execution_time_us"),
            (pt.get("execution_time_us"), "profile_summary.timing.execution_time_us"),
        ])

    for value, source in candidates:
        x = positive_float(value)
        if x is not None:
            return x, source, valid is not False
    return None, "", valid is not False


def get_profile(candidate_dir: Path, metadata: dict) -> Tuple[Optional[dict], Optional[Path]]:
    paths = [candidate_dir / "profiling" / "profile_summary.json"]
    rel = ((metadata.get("profiling") or {}).get("summary_path"))
    if rel:
        paths.append(candidate_dir / str(rel))
    for p in paths:
        if p.is_file():
            obj = read_json(p)
            if obj:
                return obj, p
    return None, None


def get_kernel_name(profile: Optional[dict], metadata: dict) -> str:
    if profile:
        kernel = profile.get("kernel") or {}
        for key in ("Op Name", "OP Type", "Kernel Name", "kernel_name"):
            if kernel.get(key):
                return str(kernel[key]).strip().strip('"')
        if profile.get("kernel_name"):
            return str(profile["kernel_name"]).strip().strip('"')
    return str(metadata.get("operator_name") or "").strip()


FUNC_RE = re.compile(r'\btt\.func(?:\s+public)?\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))')


def ttir_symbols(text: str) -> List[str]:
    return [a or b for a, b in FUNC_RE.findall(text)]


def choose_ttir(candidate_dir: Path, kernel_name: str) -> Tuple[Optional[Path], str, str]:
    d = candidate_dir / "ir" / "ttir"
    files = sorted(d.glob("*.ttir.mlir"))
    if not files:
        files = sorted(p for p in d.glob("*ttir*") if p.is_file())
    if not files:
        return None, "", "no_ttir_file"

    matches = []
    for p in files:
        try:
            syms = ttir_symbols(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        exact = [s for s in syms if s == kernel_name]
        suffix = [s for s in syms if kernel_name and (s.endswith(kernel_name) or kernel_name.endswith(s))]
        if exact:
            matches.append((p, exact[0]))
        elif suffix:
            matches.append((p, suffix[0]))

    if len(matches) == 1:
        p, symbol = matches[0]
        return p, symbol, "matched_profile_kernel_name"
    if len(matches) > 1:
        return None, "", f"ambiguous_ttir_match:{len(matches)}"
    if len(files) == 1:
        syms = ttir_symbols(files[0].read_text(encoding="utf-8", errors="replace"))
        return files[0], syms[0] if syms else "", "single_ttir_fallback"
    return None, "", f"no_unique_kernel_match:{len(files)}"


def workload_signature(profile: Optional[dict], candidate_id: str) -> str:
    k = (profile or {}).get("kernel") or {}
    fields = [
        k.get("Input Shapes", ""),
        k.get("Input Data Types", ""),
        k.get("Output Shapes", ""),
        k.get("Output Data Types", ""),
    ]
    if all(x in (None, "") for x in fields):
        return f"unknown:{candidate_id}"
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def parse_binding_string(value: Any) -> Dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v is not None}
    if not isinstance(value, str):
        return {}
    out = {}
    for item in value.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def merge_bindings(*sources: Any) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for source in sources:
        merged.update(parse_binding_string(source))
    return merged


def axes_in_ttir(ttir_text: str) -> Tuple[List[str], List[str]]:
    pid_axes = sorted(set(PROGRAM_ID_RE.findall(ttir_text)))
    nprog_axes = sorted(set(NUM_PROGRAMS_RE.findall(ttir_text)))
    return pid_axes, nprog_axes


def load_binding_config(path: Path) -> dict:
    if not path.exists():
        template = {
            "_comment": (
                "Bindings are runtime scalar values for TTIR %argN. "
                "Use operator-level values only when all candidate roles share the same workload."
            ),
            "default": {},
            "operators": {
                "_chunk_state_fwd_kernel": {
                    "arg5": 16,
                    "arg6": 8,
                    "arg7": 8,
                    "arg8": 2,
                    "arg9": 4096,
                    "arg10": 2
                }
            },
            "candidates": {}
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return template
    obj = read_json(path)
    if not obj:
        raise ValueError(f"Invalid bindings JSON: {path}")
    return obj


def build_bindings(
    metadata: dict,
    operator: str,
    candidate_id: str,
    ttir_text: str,
    binding_config: dict,
) -> Tuple[str, List[str], List[str]]:
    metadata_value = first_nested_value(metadata, ARG_BINDING_KEYS)
    default_values = binding_config.get("default", {})
    operator_values = (binding_config.get("operators") or {}).get(operator, {})
    candidate_values = (binding_config.get("candidates") or {}).get(candidate_id, {})

    # Precedence: defaults < operator override < candidate override < metadata.
    bindings = merge_bindings(
        default_values,
        operator_values,
        candidate_values,
        metadata_value,
    )

    pid_axes, nprog_axes = axes_in_ttir(ttir_text)
    auto_added = []
    for axis in pid_axes:
        key = f"pid_{axis}"
        if key not in bindings:
            bindings[key] = "0"  # representative program ID, as in official example
            auto_added.append(key)

    missing_num_programs = []
    for axis in nprog_axes:
        key = f"num_programs_{axis}"
        if key not in bindings:
            missing_num_programs.append(key)

    text = ",".join(f"{k}={bindings[k]}" for k in sorted(bindings))
    return text, auto_added, missing_num_programs


def discover(dataset_dir: Path, operators: Sequence[str], roles: Sequence[str]) -> Tuple[List[dict], List[dict]]:
    records, skipped = [], []
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    for op_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        op = op_dir.name
        if operators and op not in operators:
            continue
        data_dir = op_dir / "data"
        if not data_dir.is_dir():
            continue

        for candidate_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            role = candidate_dir.name
            if roles and role not in roles:
                continue
            meta_path = candidate_dir / "metadata.json"
            meta = read_json(meta_path)
            if not meta:
                skipped.append({"operator": op, "candidate_role": role, "reason": "missing_or_invalid_metadata"})
                continue

            cid = str(meta.get("candidate_id") or f"{op}__{role}")
            status = meta.get("status")
            if status not in (None, "success"):
                skipped.append({"operator": op, "candidate_id": cid, "candidate_role": role,
                                "reason": f"metadata_status_{status}"})
                continue
            if (meta.get("evaluation") or {}).get("success") is False:
                skipped.append({"operator": op, "candidate_id": cid, "candidate_role": role,
                                "reason": "evaluation_not_successful"})
                continue
            if (meta.get("correctness") or {}).get("success") is False:
                skipped.append({"operator": op, "candidate_id": cid, "candidate_role": role,
                                "reason": "correctness_not_successful"})
                continue

            profile, profile_path = get_profile(candidate_dir, meta)
            real, label_source, label_valid = get_real_latency(meta, profile)
            if real is None or not label_valid:
                skipped.append({"operator": op, "candidate_id": cid, "candidate_role": role,
                                "reason": "missing_or_invalid_real_latency"})
                continue

            kernel_name = get_kernel_name(profile, meta)
            ttir_path, symbol, match_status = choose_ttir(candidate_dir, kernel_name)
            if ttir_path is None:
                skipped.append({"operator": op, "candidate_id": cid, "candidate_role": role,
                                "kernel_name": kernel_name, "reason": match_status})
                continue

            ttir_text = ttir_path.read_text(encoding="utf-8", errors="replace")
            records.append({
                "operator": op,
                "candidate_id": cid,
                "candidate_role": role,
                "kernel_name": kernel_name,
                "ttir_symbol": symbol,
                "ttir_path": str(ttir_path),
                "profile_summary_path": str(profile_path or ""),
                "ttir_match_status": match_status,
                "real_latency_us": real,
                "real_latency_source": label_source,
                "workload_signature": workload_signature(profile, cid),
                "ttir_text": ttir_text,
                "metadata": meta,
            })
    return records, skipped


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 2 or len(x) != len(y):
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else None


def spearman(real: Sequence[float], pred: Sequence[float]) -> Optional[float]:
    if len(real) < 3:
        return None
    return pearson(average_ranks(real), average_ranks(pred))


def kendall_tau_b(real: Sequence[float], pred: Sequence[float]) -> Optional[float]:
    if len(real) < 2:
        return None
    concordant = discordant = tie_real = tie_pred = 0
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            dr = real[i] - real[j]
            dp = pred[i] - pred[j]
            if dr == 0 and dp == 0:
                continue
            if dr == 0:
                tie_real += 1
            elif dp == 0:
                tie_pred += 1
            elif dr * dp > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(
        (concordant + discordant + tie_real) *
        (concordant + discordant + tie_pred)
    )
    return (concordant - discordant) / denom if denom else None


def group_ranking_metrics(rows: List[dict]) -> dict:
    valid = [
        r for r in rows
        if r.get("prediction_status") == "ok"
        and positive_float(r.get("costmodel_estimate_us")) is not None
        and positive_float(r.get("real_latency_us")) is not None
    ]
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in valid:
        groups[(r["operator"], r["workload_signature"])].append(r)

    spearmans, kendalls, pair_accs, regrets = [], [], [], []
    top1_hits = top1_groups = top5_hits = top5_groups = 0
    group_rows = []

    for (op, sig), group in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        if len(group) < 2:
            continue
        real = [float(r["real_latency_us"]) for r in group]
        pred = [float(r["costmodel_estimate_us"]) for r in group]

        sp = spearman(real, pred)
        kt = kendall_tau_b(real, pred)
        if sp is not None:
            spearmans.append(sp)
        if kt is not None:
            kendalls.append(kt)

        best_pred = min(pred)
        predicted_best = [i for i, v in enumerate(pred) if math.isclose(v, best_pred, rel_tol=1e-12, abs_tol=1e-15)]
        true_best = min(real)
        # Deterministic Top-1: candidate order is stable by candidate_id.
        selected_idx = min(predicted_best, key=lambda i: group[i]["candidate_id"])
        selected_real = real[selected_idx]
        top1_groups += 1
        top1_hits += int(selected_real <= true_best * (1.0 + 1e-9))
        regret = (selected_real / true_best - 1.0) * 100.0
        regrets.append(regret)

        if len(group) >= 5:
            top5_groups += 1
            top5_idx = sorted(range(len(group)), key=lambda i: (pred[i], group[i]["candidate_id"]))[:5]
            top5_hits += int(any(real[i] <= true_best * (1.0 + 1e-9) for i in top5_idx))

        concordant = discordant = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                dr, dp = real[i] - real[j], pred[i] - pred[j]
                if dr == 0 or dp == 0:
                    continue
                if dr * dp > 0:
                    concordant += 1
                else:
                    discordant += 1
        pair_total = concordant + discordant
        if pair_total:
            pair_accs.append(concordant / pair_total)

        group_rows.append({
            "operator": op,
            "candidate_count": len(group),
            "spearman": sp,
            "kendall_tau_b": kt,
            "selected_candidate_id": group[selected_idx]["candidate_id"],
            "selected_real_latency_us": selected_real,
            "best_real_latency_us": true_best,
            "selection_regret_pct": regret,
            "top5_hit": (any(real[i] <= true_best * (1.0 + 1e-9) for i in top5_idx)
                         if len(group) >= 5 else None),
        })

    return {
        "valid_prediction_count": len(valid),
        "rankable_group_count": len(group_rows),
        "mean_group_spearman": sum(spearmans) / len(spearmans) if spearmans else None,
        "spearman_group_count": len(spearmans),
        "mean_group_kendall_tau_b": sum(kendalls) / len(kendalls) if kendalls else None,
        "kendall_group_count": len(kendalls),
        "pairwise_ranking_accuracy": sum(pair_accs) / len(pair_accs) if pair_accs else None,
        "pairwise_group_count": len(pair_accs),
        "top1_accuracy": top1_hits / top1_groups if top1_groups else None,
        "top1_group_count": top1_groups,
        "top5_hit_rate": top5_hits / top5_groups if top5_groups else None,
        "top5_group_count": top5_groups,
        "mean_selection_regret_pct": sum(regrets) / len(regrets) if regrets else None,
        "regret_group_count": len(regrets),
        "per_group": group_rows,
    }


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def call_costmodel_one(costmodel_bench, version: str, item: dict) -> Tuple[Optional[float], str, Any]:
    """Adapt installed 3.2.0 iterable->dict API and newer item->tuple API."""
    key = item["config"]
    if version.startswith("3.2."):
        result = costmodel_bench([item])
    else:
        try:
            result = costmodel_bench(item)
        except (TypeError, AttributeError):
            result = costmodel_bench([item])

    if isinstance(result, dict):
        value = result.get(key)
    elif isinstance(result, tuple) and len(result) == 2:
        returned_key, value = result
        if returned_key != key:
            return None, f"unexpected_return_key:{returned_key}", result
    else:
        return None, f"unexpected_return_type:{type(result).__name__}", result

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None, "missing_or_non_numeric_latency", result

    if not math.isfinite(value) or value <= 0:
        return None, "non_finite_or_non_positive_latency", result
    return value, "ok", result


def main() -> int:
    parser = argparse.ArgumentParser(description="Official Triton-Ascend Cost Model ranking baseline v2")
    parser.add_argument("--root", type=Path, default=Path("/workspace/Agent/costModel"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--roles", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bindings-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-auto-pid", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    dataset = (args.dataset_dir or root / "datasets").resolve()
    out = (args.output_dir or root / "contrast" / "results" /
           f"official_costmodel_rank_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    out.mkdir(parents=True, exist_ok=True)

    bindings_path = (args.bindings_json or root / "contrast" / "official_costmodel_bindings.json").resolve()
    binding_config = load_binding_config(bindings_path)
    role_filter = [x.strip() for x in args.roles.split(",") if x.strip()]

    print("========== OFFICIAL COSTMODEL RANKING BASELINE V2 ==========")
    print("Dataset:", dataset)
    print("Output :", out)
    print("Bindings:", bindings_path)

    records, skipped = discover(dataset, args.operator, role_filter)
    if args.limit > 0:
        records = records[:args.limit]

    prepared = []
    for r in records:
        bindings, auto_pid, missing_num_programs = build_bindings(
            r["metadata"], r["operator"], r["candidate_id"], r["ttir_text"], binding_config
        )
        if args.no_auto_pid:
            # Remove automatically supplied representative pid values only.
            b = parse_binding_string(bindings)
            for key in auto_pid:
                b.pop(key, None)
            bindings = ",".join(f"{k}={b[k]}" for k in sorted(b))

        r["arg_bindings"] = bindings
        r["auto_pid_bindings"] = ",".join(auto_pid)
        r["missing_num_programs_bindings"] = ",".join(missing_num_programs)
        r["pid_axes"] = ",".join(axes_in_ttir(r["ttir_text"])[0])
        r["num_programs_axes"] = ",".join(axes_in_ttir(r["ttir_text"])[1])
        prepared.append(r)

    print("Matched candidates:", len(prepared))
    print("Skipped candidates:", len(skipped))
    print("Candidates with non-empty bindings:", sum(bool(r["arg_bindings"]) for r in prepared))
    print("Candidates with unresolved tt.get_num_programs:", sum(bool(r["missing_num_programs_bindings"]) for r in prepared))

    inventory_fields = [
        "operator", "candidate_id", "candidate_role", "kernel_name", "ttir_symbol",
        "ttir_path", "profile_summary_path", "ttir_match_status", "arg_bindings",
        "auto_pid_bindings", "pid_axes", "num_programs_axes",
        "missing_num_programs_bindings", "real_latency_us", "real_latency_source",
        "workload_signature",
    ]
    write_csv(out / "candidate_inventory.csv", prepared, inventory_fields)
    write_csv(out / "skipped_candidates.csv", skipped,
              ["operator", "candidate_id", "candidate_role", "kernel_name", "reason"])

    if args.dry_run:
        print("DRY RUN complete. No Cost Model calls made.")
        print("Inventory:", out / "candidate_inventory.csv")
        print("Bindings template/config:", bindings_path)
        return 0

    if not prepared:
        print("No eligible candidates.", file=sys.stderr)
        return 2

    try:
        import triton
        from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench
    except Exception as exc:
        print("Cannot import costmodel_bench:", repr(exc), file=sys.stderr)
        return 3

    version = getattr(triton, "__version__", "unknown")
    print("Triton version:", version)
    print("Installed API:", costmodel_bench)
    print("Running one candidate per call...")

    prediction_rows = []
    for idx, r in enumerate(prepared, 1):
        key = f"{r['operator']}::{r['candidate_id']}"
        item = {"config": key, "ttir": r["ttir_text"], "arg_bindings": r["arg_bindings"]}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimate, status, raw = call_costmodel_one(costmodel_bench, version, item)
            warning_text = " | ".join(str(w.message) for w in caught)
            if estimate is not None:
                prediction_status = "ok"
            else:
                prediction_status = status
            row = {
                "operator": r["operator"],
                "candidate_id": r["candidate_id"],
                "candidate_role": r["candidate_role"],
                "kernel_name": r["kernel_name"],
                "ttir_path": r["ttir_path"],
                "arg_bindings": r["arg_bindings"],
                "auto_pid_bindings": r["auto_pid_bindings"],
                "real_kernel_latency_us": r["real_latency_us"],
                "costmodel_estimate_us": estimate if estimate is not None else "",
                "prediction_status": prediction_status,
                "warning": warning_text,
                "raw_result": repr(raw),
                "workload_signature": r["workload_signature"],
            }
        except Exception as exc:
            row = {
                "operator": r["operator"],
                "candidate_id": r["candidate_id"],
                "candidate_role": r["candidate_role"],
                "kernel_name": r["kernel_name"],
                "ttir_path": r["ttir_path"],
                "arg_bindings": r["arg_bindings"],
                "auto_pid_bindings": r["auto_pid_bindings"],
                "real_kernel_latency_us": r["real_latency_us"],
                "costmodel_estimate_us": "",
                "prediction_status": f"exception:{type(exc).__name__}",
                "warning": "",
                "raw_result": repr(exc),
                "workload_signature": r["workload_signature"],
            }
        prediction_rows.append(row)
        print(f"[{idx:03d}/{len(prepared):03d}] {r['operator']} / {r['candidate_role']}: "
              f"{row['costmodel_estimate_us'] if row['costmodel_estimate_us'] != '' else row['prediction_status']}")

    fields = [
        "operator", "candidate_id", "candidate_role", "kernel_name", "ttir_path",
        "arg_bindings", "auto_pid_bindings", "real_kernel_latency_us",
        "costmodel_estimate_us", "prediction_status", "warning", "raw_result",
        "workload_signature",
    ]
    write_csv(out / "predictions.csv", prediction_rows, fields)

    ranking = group_ranking_metrics(prediction_rows)
    ok_count = sum(r["prediction_status"] == "ok" for r in prediction_rows)
    summary = {
        "baseline": "Triton-Ascend built-in Cost Model",
        "triton_version": version,
        "dataset_dir": str(dataset),
        "output_dir": str(out),
        "discovered_candidate_count": len(prepared),
        "skipped_candidate_count": len(skipped),
        "successful_estimate_count": ok_count,
        "failed_estimate_count": len(prediction_rows) - ok_count,
        "binding_config": str(bindings_path),
        "prediction_semantics": (
            "Official static Cost Model estimate with representative program_id bindings. "
            "The official example binds pid_x=0 and uses the estimate to rank candidate configs. "
            "This output is NOT assumed to be full-grid end-to-end kernel latency."
        ),
        "primary_evaluation": "Within the same operator + input-shape/dtype workload, compare costmodel estimates against measured full-kernel latency ordering.",
        "metrics": {k: v for k, v in ranking.items() if k != "per_group"},
        "per_group_metrics": ranking["per_group"],
        "notes": [
            "Triton 3.2.0 costmodel_bench accepts an iterable of items and returns {config: latency_us}; this script calls it with a one-item list per candidate.",
            "pid_x/y/z are bound to 0 when those axes occur in TTIR, following the official static program-id estimation example.",
            "num_programs_* are not guessed. Candidates using tt.get_num_programs without explicit bindings are flagged in candidate_inventory.csv.",
            "Absolute MAE/MAPE/RMSE are intentionally not reported as primary metrics because full-grid equivalence of the official estimate has not been established.",
            "Candidates are ranked only against candidates with the same operator and identical profile input/output shape/type signature.",
            "If an operator has only one successfully estimated candidate, it contributes no ranking metric.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFiles:")
    for name in ("candidate_inventory.csv", "predictions.csv", "skipped_candidates.csv", "summary.json"):
        print(out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
