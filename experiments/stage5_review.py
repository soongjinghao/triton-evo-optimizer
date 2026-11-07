#!/usr/bin/env python3
"""Create reproducible, read-only PAPER experiment review artifacts."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ("paper_b1", "paper_b2", "paper_full")
SEEDS = (11, 12, 13)
RESULT_COMMIT = "559f2f539c59886f155df92952141f210fa87c71"
FREEZE_COMMIT = "68ab813757679dd1f3c9ca3c7122c1b78a5dd0df"
RAG_SHA256 = "3a652ca492025e02986ac43c3d1a6cfc08c26a4c0297ab0d16dbcaff7533f397"
RUNS = (
    ("_act_quant_kernel", 12),
    ("_convert_req_index_to_global_index_kernel", 11),
    ("_dequantize_k_cache_fast_kernel", 12),
    ("_log_softmax_kernel", 13),
    ("_pack_seq_kernel", 12),
    ("assign_extend_cache_locs", 11),
    ("compute_identity_kernel", 13),
    ("l2norm_fwd_kernel1", 12),
    ("mean_kernel", 13),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def load_result(config, kernel, seed):
    path = ROOT / "experiments/results/filter_online" / config / f"{kernel}__r{seed}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def verify_result(config, path, rec):
    audit = rec.get("audit") or {}
    provenance = rec.get("provenance") or {}
    expected = {
        "success": rec.get("success") is True,
        "acceptance": (rec.get("acceptance") or {}).get("passed") is True,
        "budget_ok": audit.get("budget_ok") is True,
        "npu_calls": isinstance(audit.get("npu_calls"), int)
        and 0 < audit["npu_calls"] <= audit.get("eval_budget", 0),
        "tokens": isinstance(audit.get("llm_tokens"), int)
        and audit["llm_tokens"] > 0,
        "model_id": bool(rec.get("llm_model_ids")),
        "final_remeasure": audit.get("remeasure_repeats") == 5,
        "RAG": rec.get("rag_index_sha256") == RAG_SHA256,
        "commit": provenance.get("git_commit") == RESULT_COMMIT,
    }
    failed = [name for name, passed in expected.items() if not passed]
    if failed:
        raise RuntimeError(f"{config}/{path.name}: failed checks {failed}")


def read_events(config, kernel, seed):
    path = ROOT / "experiments/logs" / (
        f"filter_online__{config}__{kernel}__r{900 + seed}.jsonl")
    if not path.is_file():
        raise FileNotFoundError(f"Missing event log: {path}")
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid JSONL") from exc
    return path, events


def task_log_seed_counts(config, kernel, seed):
    path = ROOT / "experiments/results/phase4_recovery_20261020" / (
        f"{config}__{kernel}.log")
    if not path.is_file():
        return path, None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    ordinal = SEEDS.index(seed) + 1
    header = re.compile(
        rf"^=== \[{re.escape(config)}\] seed {ordinal}/3 算子 1/1: "
        rf"{re.escape(kernel)} ===$")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if header.match(line.strip())), None)
    if start is None:
        return path, None, None
    end = next((i for i in range(start + 1, len(lines))
                if re.match(rf"^=== \[{re.escape(config)}\] seed \d/3 算子 1/1:", lines[i])),
               len(lines))
    segment = "\n".join(lines[start:end])
    if "[复用]" in segment:
        return path, None, None
    return path, segment.count("连接失败(OpenAIConnectionError)"), segment.count("❌ Failed")


def capture_environment():
    package_names = (
        "torch", "torch-npu", "triton", "triton-ascend", "langchain-openai",
        "transformers", "faiss-cpu", "sentence-transformers", "numpy",
        "scipy", "pandas", "pytest",
    )
    packages = {}
    distributions = {
        d.metadata.get("Name", "").lower().replace("_", "-")
        for d in importlib.metadata.distributions()
    }
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    def read_version(path):
        p = Path(path)
        return p.read_text(encoding="utf-8", errors="replace").strip() if p.is_file() else None

    npu = {"torch_npu_imported": False, "device_count": None, "device_names": []}
    try:
        import torch
        import torch_npu
        npu["torch_npu_imported"] = True
        npu["device_count"] = torch.npu.device_count()
        npu["device_names"] = [
            torch.npu.get_device_name(i) for i in range(npu["device_count"])
        ]
    except Exception as exc:
        npu["probe_error_type"] = type(exc).__name__

    smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                         check=False)
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_scope": "Current host at Stage 5 review time; not a time-stamped capture from the original runs.",
        "result_commit": RESULT_COMMIT,
        "frozen_tag": "paper-full-v1",
        "frozen_commit": FREEZE_COMMIT,
        "current_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "packages": packages,
        "triton_distributions_detected": sorted(
            name for name in distributions if "triton" in name),
        "npu_devices": npu,
        "npu_smi_exit_code": smi.returncode,
        "npu_smi_info": smi.stdout.strip(),
        "npu_smi_stderr": smi.stderr.strip() or None,
        "cann_opp_version_info": read_version(
            "/usr/local/Ascend/cann-9.0.0/opp/version.info"),
        "driver_version_info": read_version(
            "/usr/local/Ascend/driver/version.info"),
        "historical_experiment_settings": {
            "configured_npu_devices": 1,
            "configured_llm_workers": 5,
            "llm_model_id": "140aced515cc4236a6141f2b8a77107f",
        },
        "rag_index_sha256": RAG_SHA256,
    }


def build_artifacts(output_dir, bootstrap_samples):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    input_entries = {}
    manifest_entries = {}
    source_entries = {}
    per_kernel = {}
    low_budget = []
    event_audit = []
    for config in CONFIGS:
        per_kernel[config] = {}
        paths = sorted((ROOT / "experiments/results/filter_online" / config)
                       .glob("*__r[0-9]*.json"))
        paths = [p for p in paths if p.stem.rsplit("__r", 1)[-1] in
                 {str(seed) for seed in SEEDS}]
        if len(paths) != 150:
            raise RuntimeError(f"{config}: expected 150 result JSONs, found {len(paths)}")
        for result_path in paths:
            rec = json.loads(result_path.read_text(encoding="utf-8"))
            kernel = rec["kernel"]
            seed = int(rec["seed"])
            verify_result(config, result_path, rec)
            if rec.get("config") != config or seed not in SEEDS:
                raise RuntimeError(f"Unexpected config/seed in {result_path}")
            candidate_path = result_path.with_suffix(".py")
            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            audit = rec["audit"]
            provenance = rec["provenance"]
            manifest_rel = Path("experiments/manifest") / f"{kernel}.json"
            manifest_path = ROOT / manifest_rel
            manifest_hash = sha256(manifest_path.read_bytes())
            if manifest_hash != rec.get("reference_manifest_sha256"):
                raise RuntimeError(f"Manifest fingerprint mismatch: {manifest_rel}")
            manifest_entries[str(manifest_rel)] = {
                "sha256": manifest_hash,
                "t_base_us": rec["reference_baseline_us"],
                "source": rec["reference_baseline_source"],
            }
            for name, digest in provenance.get("source_sha256", {}).items():
                old = source_entries.setdefault(name, digest)
                if old != digest:
                    raise RuntimeError(f"Source hash conflict: {name}")
            for absolute, digest in provenance.get("experiment_input_sha256", {}).items():
                input_path = Path(absolute)
                try:
                    rel = input_path.relative_to(ROOT)
                except ValueError as exc:
                    raise RuntimeError(f"Input outside project root: {absolute}") from exc
                actual = sha256(input_path.read_bytes())
                if actual != digest:
                    raise RuntimeError(f"Input fingerprint mismatch: {rel}")
                input_entries[str(rel)] = {
                    "sha256": actual,
                    "bytes": input_path.stat().st_size,
                    "kernel": kernel,
                }
            source_code = provenance.get("source_sha256", {})
            for name, expected in source_code.items():
                actual = sha256(subprocess.check_output(
                    ["git", "show", f"{RESULT_COMMIT}:{name}"], cwd=ROOT))
                if actual != expected:
                    raise RuntimeError(f"Committed source fingerprint mismatch: {name}")

            event_path, events = read_events(config, kernel, seed)
            eval_events = [e for e in events if e.get("event") == "eval"]
            if len(eval_events) != audit["npu_calls"]:
                raise RuntimeError(
                    f"{config}/{kernel}/seed{seed}: event NPU count "
                    f"{len(eval_events)} != result {audit['npu_calls']}")
            rejects = [e for e in events if e.get("event") == "reject"]
            eval_errors = Counter(e.get("error") or "none" for e in eval_events)
            reject_reasons = Counter(
                e.get("error") or e.get("operation") or "unspecified" for e in rejects)
            task_log_path, connection_retries, terminal_llm_failures = (
                task_log_seed_counts(config, kernel, seed))
            event_audit.append({
                "config": config, "kernel": kernel, "seed": seed,
                "npu_calls_result": audit["npu_calls"],
                "npu_calls_event_log": len(eval_events),
                "reject_count": len(rejects),
                "reject_reasons": dict(reject_reasons),
                "eval_error_counts": dict(eval_errors),
                "llm_connection_retries_seed_log": connection_retries,
                "llm_terminal_failures_seed_log": terminal_llm_failures,
                "event_log": str(event_path.relative_to(ROOT)),
                "task_log": str(task_log_path.relative_to(ROOT)),
            })
            if audit["npu_calls"] < audit["eval_budget"]:
                low_budget.append({
                    "config": config, "kernel": kernel, "seed": seed,
                    "npu_calls": audit["npu_calls"],
                    "budget": audit["eval_budget"],
                    "preflight_rejects": len(rejects),
                    "reject_reasons": dict(reject_reasons),
                    "eval_errors": dict(eval_errors),
                    "connection_retries": connection_retries,
                    "terminal_llm_failures": terminal_llm_failures,
                    "generation_evaluations": dict(Counter(
                        str(e.get("gen")) for e in eval_events)),
                    "event_log": str(event_path.relative_to(ROOT)),
                    "task_log": str(task_log_path.relative_to(ROOT)),
                    "assessment": "Budget cap respected; causes are itemized from event/log evidence. "
                                  "Do not infer a completed candidate-generation slot from an NPU-call count alone.",
                })
            row = {
                "config": config,
                "kernel": kernel,
                "seed": seed,
                "run_id": rec["run_id"],
                "S": float(rec["S"]),
                "t_best_us": float(rec["t_best_us"]),
                "reference_baseline_us": float(rec["reference_baseline_us"]),
                "npu_calls": audit["npu_calls"],
                "eval_budget": audit["eval_budget"],
                "budget_ok": audit["budget_ok"],
                "remeasure_repeats": audit["remeasure_repeats"],
                "wallclock_s": audit["wallclock_s"],
                "llm_tokens": audit["llm_tokens"],
                "llm_model_ids": ";".join(rec["llm_model_ids"]),
                "manifest_sha256": manifest_hash,
                "rag_index_sha256": rec["rag_index_sha256"],
                "git_commit": provenance["git_commit"],
                "success": rec["success"],
                "acceptance_passed": rec["acceptance"]["passed"],
            }
            records.append(row)
            per_kernel[config].setdefault(kernel, {})[seed] = float(rec["S"])

    if len(manifest_entries) != 50 or any(len(per_kernel[c]) != 50 for c in CONFIGS):
        raise RuntimeError("Expected 50 kernels and 50 manifest fingerprints")
    rag_hasher = hashlib.sha256()
    for name in ("index.faiss", "index.pkl"):
        rag_hasher.update(name.encode("utf-8"))
        rag_hasher.update((ROOT / "RAG/knowledge" / name).read_bytes())
    if rag_hasher.hexdigest() != RAG_SHA256:
        raise RuntimeError("RAG fingerprint mismatch")

    snapshot = {
        "schema": "tritonevo-paper-input-snapshot-v1",
        "result_commit": RESULT_COMMIT,
        "frozen_tag": "paper-full-v1",
        "frozen_commit": FREEZE_COMMIT,
        "rag_index_sha256": RAG_SHA256,
        "manifests": dict(sorted(manifest_entries.items())),
        "inputs": dict(sorted(input_entries.items())),
        "source_fingerprints": dict(sorted(source_entries.items())),
        "note": "This external snapshot manifest supplies dataset-level provenance missing from historical JSON; historical result JSON files were not modified.",
    }
    snapshot_bytes = canonical_json(snapshot) + b"\n"
    (output_dir / "dataset_snapshot.json").write_bytes(snapshot_bytes)
    snapshot_hash = sha256(snapshot_bytes)
    (output_dir / "dataset_snapshot.sha256").write_text(
        f"{snapshot_hash}  dataset_snapshot.json\n", encoding="utf-8")
    for row in records:
        row["dataset_snapshot_sha256"] = snapshot_hash
    (output_dir / "environment_snapshot.json").write_text(
        json.dumps(capture_environment(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")

    per_kernel_rows = []
    for kernel in sorted(per_kernel["paper_full"]):
        row = {"kernel": kernel}
        for config in CONFIGS:
            vals = [per_kernel[config][kernel][s] for s in SEEDS]
            row[f"{config}_seed11"] = vals[0]
            row[f"{config}_seed12"] = vals[1]
            row[f"{config}_seed13"] = vals[2]
            row[f"{config}_median"] = sorted(vals)[1]
        per_kernel_rows.append(row)
    with (output_dir / "per_kernel_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_kernel_rows[0]))
        writer.writeheader()
        writer.writerows(per_kernel_rows)

    seed_gms = {}
    for config in CONFIGS:
        seed_gms[config] = {
            str(seed): math.exp(sum(math.log(per_kernel[config][k][seed])
                                    for k in per_kernel[config]) / 50)
            for seed in SEEDS
        }
    median_gm = {
        config: math.exp(sum(math.log(row[f"{config}_median"])
                             for row in per_kernel_rows) / 50)
        for config in CONFIGS
    }
    wins = {
        config: sum(row[f"{config}_median"] > 1.0 for row in per_kernel_rows)
        for config in CONFIGS
    }
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise RuntimeError("SciPy is needed for the paired Wilcoxon test") from exc
    import numpy as np

    rng = np.random.default_rng(20261026)
    bootstrap = {}
    for comparator in ("paper_b1", "paper_b2"):
        paired_log_diffs = np.array([
            math.log(sorted(per_kernel["paper_full"][k].values())[1])
            - math.log(sorted(per_kernel[comparator][k].values())[1])
            for k in sorted(per_kernel["paper_full"])
        ])
        ratio = math.exp(float(paired_log_diffs.mean()))
        samples = np.empty(bootstrap_samples)
        for start in range(0, bootstrap_samples, 5000):
            stop = min(start + 5000, bootstrap_samples)
            indices = rng.integers(0, len(paired_log_diffs), size=(stop-start, len(paired_log_diffs)))
            samples[start:stop] = np.exp(paired_log_diffs[indices].mean(axis=1))
        ci = np.quantile(samples, [0.025, 0.975])
        full_values = [sorted(per_kernel["paper_full"][k].values())[1]
                       for k in sorted(per_kernel["paper_full"])]
        comp_values = [sorted(per_kernel[comparator][k].values())[1]
                       for k in sorted(per_kernel[comparator])]
        test = wilcoxon(full_values, comp_values, alternative="two-sided",
                        zero_method="wilcox", method="auto")
        bootstrap[comparator] = {
            "paired_geometric_mean_ratio": ratio,
            "relative_increase_percent": (ratio - 1) * 100,
            "paired_kernel_bootstrap_95ci_ratio": [float(ci[0]), float(ci[1])],
            "paired_kernel_bootstrap_95ci_relative_increase_percent": [
                (float(ci[0]) - 1) * 100, (float(ci[1]) - 1) * 100],
            "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_p_two_sided": float(test.pvalue),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": 20261026,
            "statistical_unit": "kernel; three independent search seeds reduced by median within kernel",
        }

    summary = {
        "result_commit": RESULT_COMMIT,
        "freeze_commit": FREEZE_COMMIT,
        "snapshot_sha256": snapshot_hash,
        "records": len(records),
        "kernels_per_config": {c: len(per_kernel[c]) for c in CONFIGS},
        "primary_statistic": "per-kernel median across seeds 11,12,13, followed by geometric mean across 50 kernels",
        "geometric_mean_of_per_kernel_medians": median_gm,
        "geometric_mean_by_seed": seed_gms,
        "kernels_with_median_speedup_above_one": wins,
        "paired_comparisons": bootstrap,
        "cost": {},
        "npu_call_distribution": {},
        "low_budget_record_count": len(low_budget),
        "low_budget_review": low_budget,
        "event_logs_matched_npu_calls": len(event_audit),
        "dataset_sha256_matches_meta_historical_json": False,
        "dataset_metadata_note": "Historical records have dataset_snapshot/dataset_sha256 null. The new external snapshot SHA binds the fixed inputs and manifests without rewriting historical JSON.",
    }
    for config in CONFIGS:
        rows = [r for r in records if r["config"] == config]
        summary["cost"][config] = {
            "mean_wallclock_s": sum(r["wallclock_s"] for r in rows) / len(rows),
            "mean_llm_tokens": sum(r["llm_tokens"] for r in rows) / len(rows),
            "total_npu_calls": sum(r["npu_calls"] for r in rows),
        }
        summary["npu_call_distribution"][config] = dict(sorted(
            Counter(r["npu_calls"] for r in rows).items()))

    with (output_dir / "run_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r:(r["config"],r["kernel"],r["seed"])))
    with (output_dir / "event_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(event_audit, key=lambda r:(r["config"],r["kernel"],r["seed"])):
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "snapshot_sha256": snapshot_hash,
        "records": len(records),
        "event_logs_matched_npu_calls": len(event_audit),
        "low_budget_records": len(low_budget),
        "geometric_mean_of_per_kernel_medians": median_gm,
        "kernels_with_median_speedup_above_one": wins,
        "paired_comparisons": bootstrap,
        "npu_call_distribution": summary["npu_call_distribution"],
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments/results/stage5_review_20261026"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    build_artifacts(Path(args.output_dir), args.bootstrap_samples)


if __name__ == "__main__":
    main()
