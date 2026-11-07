#!/usr/bin/env python3
"""Run paired PAPER-FULL single-factor ablations on the preselected 15 kernels."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import filter_online_exp as fe

KERNELS = (
    "_act_quant_kernel",
    "_dequantize_k_cache_fast_kernel",
    "_fwd_kernel_ep_gather",
    "_pack_seq_kernel",
    "_selective_scan_update_kernel",
    "_set_k_and_s_triton_kernel",
    "assign_extend_cache_locs",
    "chunk_local_cumsum_scalar_kernel",
    "compute_identity_kernel",
    "eye_kernel",
    "fused_gdn_gating_kernel",
    "l2norm_fwd_kernel2",
    "moe_align_block_size_stage",
    "recompute_w_u_fwd_kernel",
    "reshape_and_cache_kernel_flash",
)
SEEDS = (11, 12, 13)
RESULT_COMMIT = "559f2f539c59886f155df92952141f210fa87c71"
FREEZE_COMMIT = "68ab813757679dd1f3c9ca3c7122c1b78a5dd0df"
VARIANTS = {
    "abl_profile_on_rag_off": {"enable_profiling": True, "enable_rag": False},
    "abl_profile_off_rag_on": {"enable_profiling": False, "enable_rag": True},
    "abl_tournament_selection": {"selection": "tournament"},
    "abl_unprotected_crossover": {"crossover_mode": "unconstrained"},
    "abl_uniform_mutation": {"mutation_mode": "uniform"},
}
OUT_DIR = ROOT / "experiments" / "results" / "formal_ablations_20261026"
STATE_PATH = OUT_DIR / "run_state.json"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_protocol():
    manifest_hashes = {}
    for kernel in KERNELS:
        manifest = ROOT / "experiments" / "manifest" / f"{kernel}.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing frozen reference manifest: {manifest}")
        manifest_hashes[kernel] = sha256_file(manifest)
        for config in ("paper_b1", "paper_full"):
            for seed in SEEDS:
                result = (
                    ROOT / "experiments" / "results" / "filter_online" / config
                    / f"{kernel}__r{seed}.json"
                )
                if not result.is_file():
                    raise FileNotFoundError(f"Missing matched main result: {result}")
                record = json.loads(result.read_text(encoding="utf-8"))
                if record.get("success") is not True or record.get("seed") != seed:
                    raise RuntimeError(f"Invalid matched main result: {result}")
                if record.get("config") != config:
                    raise RuntimeError(f"Unexpected config in matched result: {result}")
                if record.get("provenance", {}).get("git_commit") != RESULT_COMMIT:
                    raise RuntimeError(f"Unexpected main-result commit: {result}")
                audit = record.get("audit") or {}
                if (audit.get("budget_ok") is not True
                        or audit.get("remeasure_repeats") != 5):
                    raise RuntimeError(f"Main result failed protocol checks: {result}")

    base = dict(fe.PAPER_FULL_CONFIG)
    configs = {}
    for name, changes in VARIANTS.items():
        cfg = {**base, **changes}
        if cfg.get("method") != "paper_full":
            raise RuntimeError(f"{name}: method must remain paper_full")
        if cfg.get("eval_budget") != 13:
            raise RuntimeError(f"{name}: budget or measurement protocol changed")
        if cfg.get("enable_candidate_filter") or cfg.get("enable_candidate_dedup"):
            raise RuntimeError(f"{name}: unrelated candidate filtering/dedup enabled")
        configs[name] = cfg

    return {
        "schema": "tritonevo-formal-paired-ablations-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": fe._git_commit(),
        "main_result_commit": RESULT_COMMIT,
        "frozen_commit": FREEZE_COMMIT,
        "runner_sha256": sha256_file(__file__),
        "kernels": list(KERNELS),
        "seeds": list(SEEDS),
        "configs": configs,
        "fixed_eval_budget_per_search": 13,
        "final_remeasure_repeats": 5,
        "matched_main_configs": ["paper_b1", "paper_full"],
        "manifest_sha256": manifest_hashes,
        "analysis_unit": "kernel; compare seed-matched outcomes and summarize seeds by per-kernel median",
        "historical_ablation_results": (
            "Older seed-0 component ablations are exploratory only and are not "
            "reused as formal paired results."
        ),
    }


def persist_state(protocol, state):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_sha256": hashlib.sha256(canonical_json(protocol)).hexdigest(),
               **state}
    temp = STATE_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(STATE_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="execute all five variants; default only validates/prints")
    args = parser.parse_args()

    protocol = build_protocol()
    protocol_path = OUT_DIR / "protocol.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old.get("runner_sha256") != protocol["runner_sha256"]:
            raise RuntimeError(
                f"Existing protocol runner hash differs: {protocol_path}; "
                "refusing to mix runs from different runner versions.")
        protocol = old
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    plan = [
        {"config": config, "kernel": kernel, "seed": seed}
        for seed in SEEDS
        for kernel in KERNELS
        for config in VARIANTS
    ]
    print(f"Validated: {len(VARIANTS)} variants × {len(KERNELS)} kernels × "
          f"{len(SEEDS)} seeds = {len(plan)} searches")
    print(f"Existing matched baseline: {len(KERNELS) * len(SEEDS)} kernels/seeds "
          "each for paper_b1 and paper_full")
    print(f"Protocol: {protocol_path}")
    if not args.run:
        print("Dry run only; pass --run to start NPU searches.")
        return 0

    state = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "status": "running",
        "planned": len(plan),
        "completed": [],
        "failed": [],
    }
    persist_state(protocol, state)
    for index, item in enumerate(plan, 1):
        config = item["config"]
        kernel = item["kernel"]
        seed = item["seed"]
        print(f"\n[{index}/{len(plan)}] {config} | {kernel} | seed={seed}",
              flush=True)
        fe.CONFIGS[config] = dict(protocol["configs"][config])
        try:
            record = fe.run_one(kernel, config, seed)
            state["completed"].append({
                **item,
                "S": record.get("S"),
                "npu_calls": (record.get("audit") or {}).get("npu_calls"),
                "result": str(
                    (fe.OUT_DIR / config / f"{kernel}__r{seed}.json").relative_to(ROOT)),
            })
        except Exception as exc:
            state["failed"].append({
                **item,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"[FAILED] {config}/{kernel}/seed{seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
        persist_state(protocol, state)
        print(f"[PROGRESS] {len(state['completed'])} completed, "
              f"{len(state['failed'])} failed, {len(plan) - index} remaining",
              flush=True)
        time.sleep(1)

    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = "completed" if not state["failed"] else "completed_with_failures"
    persist_state(protocol, state)
    print(f"\nFinal: {len(state['completed'])} successful/reused, "
          f"{len(state['failed'])} failed; state={STATE_PATH}")
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
