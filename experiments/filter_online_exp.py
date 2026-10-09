#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.6 候选预筛选的**在线（真机）**实验。

对照设计
--------
三种配置，NPU 预算是唯一的变量控制点：

  baseline  现行流程，每代生成 4 个 → 全评 4 个      NPU 评测 13 次
  filtered  每代生成 20 个 → 模型挑 4 个送评测        NPU 评测 13 次（完全相同）
  fulleval  每代生成 20 个 → 全部送评测（不过滤）      NPU 评测 45 次
            ↑ 子实验专用：让被模型排到后面的候选也有真值，
              事后即可算出在线精确率与召回率（表11 缺的那两格）

其中 baseline 与 filtered 的**评测次数严格相同**，多出来的只是廉价的 LLM 生成，
因此二者的差异只能来自"预算分配给哪些候选"。

指标（按"一代的变化"来量，不以端到端加速比为主）
-------------------------------------------
  - 每代送评测的候选中 G_mut > 1 的比例（代级改进率）
  - 每代最优候选的增益
  - 单位评测带来的有效改进次数（预算效率）
  - 最终 GM(S)（次要指标，明确其可能不显著）

数据安全
--------
  * 使用 run_id=900+ 与独立的日志/结果目录，绝不覆盖 results/full（3.1 主实验）
  * 并发通过环境变量 EA_MAX_WORKERS 控制，不超过 30

用法
----
    source set_env/set_api_local.sh
    EA_MAX_WORKERS=24 python experiments/filter_online_exp.py --config filtered --kernels eye_kernel
    python experiments/filter_online_exp.py --config baseline
    python experiments/filter_online_exp.py --config fulleval --limit 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.harness import build_ea                       # noqa: E402
from experiments import common                                  # noqa: E402
from experiments.analyze_logs import K15                        # noqa: E402

OUT_DIR = ROOT / "experiments" / "results" / "filter_online"
LOGS_DIR = ROOT / "experiments" / "logs"

POOL = 20          # 每代生成的候选池规模
TOPK = 4           # 每代送 NPU 评测的个数
RUN_BASE = 900     # 独立 run_id，避免与既有实验冲突

CONFIGS = {
    "baseline": dict(eval_budget=13),
    "filtered": dict(eval_budget=13, enable_candidate_filter=True,
                     candidate_pool_size=POOL),
    "fulleval": dict(eval_budget=13 + 2 * (POOL - TOPK),
                     enable_candidate_filter=True,
                     candidate_pool_size=POOL, filter_evaluate_all=True),
}


def run_one(kernel, cfg_name, seed):
    cfg = CONFIGS[cfg_name]
    run_id = RUN_BASE + seed
    # 续跑：结果文件已存在则直接复用。
    # 中断后重跑、或把 seed 从 2 个减为 1 个时，已完成的算子不必重做。
    out_dir = OUT_DIR / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{kernel}__r{seed}.json"
    if out_json.exists():
        print(f"  [复用] 已有结果 {out_json.name}")
        return json.loads(out_json.read_text(encoding="utf-8"))
    log_path = LOGS_DIR / f"filter_online__{cfg_name}__{kernel}__r{run_id}.jsonl"
    score_path = LOGS_DIR / f"filter_scores__{cfg_name}__{kernel}__r{run_id}.jsonl"
    for p in (log_path, score_path):
        if p.exists():
            p.unlink()

    ea, config = build_ea(
        kernel, "full", run_id=run_id, seed=seed, log=True,
        log_path=str(log_path), filter_log_path=str(score_path), **cfg)

    seed_codes = [common.read_code(p) for p in common.seed_code_paths(kernel)]
    t0 = time.time()
    best = ea.run(seed_codes)
    elapsed = time.time() - t0

    # 最终最优候选用与 3.1 一致的重复测量口径复测
    rec = {"kernel": kernel, "config": cfg_name, "seed": seed,
           "run_id": run_id, "seconds": round(elapsed, 1)}
    if best and best.code:
        res = ea.executor.measure_repeat(best.code, repeats=5, device_id=0)
        (out_dir / f"{kernel}__r{seed}.py").write_text(best.code, encoding="utf-8")
        rec.update({"success": bool(res.success), "t_best_us": res.execution_time,
                    "error": (res.error or "")[:200]})
        m = common.MANIFEST_DIR / f"{kernel}.json"
        if m.exists():
            man = json.loads(m.read_text(encoding="utf-8"))
            if man.get("t_base_us") and res.execution_time:
                rec["S"] = man["t_base_us"] / res.execution_time
    out_json.write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=("baseline", "filtered", "fulleval"),
                    required=True)
    ap.add_argument("--kernels", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    kernels = a.kernels or sorted(K15)
    if a.limit:
        kernels = kernels[:a.limit]

    workers = int(os.getenv("EA_MAX_WORKERS", "8"))
    print(f"[配置] {a.config}  候选池 {POOL if a.config != 'baseline' else 4}  "
          f"并发上限 {workers}  算子 {len(kernels)}  seed {list(a.seeds)}")
    if workers > 30:
        print("⚠ EA_MAX_WORKERS > 30，请调低")

    t0 = time.time()
    for si, seed in enumerate(a.seeds, 1):
        for ki, kernel in enumerate(kernels, 1):
            print(f"\n=== [{a.config}] seed {si}/{len(a.seeds)} "
                  f"算子 {ki}/{len(kernels)}: {kernel} ===")
            try:
                rec = run_one(kernel, a.config, seed)
                print(f"  完成: {rec.get('t_best_us')} us  "
                      f"S={rec.get('S')}  用时 {rec['seconds']}s")
            except Exception as e:
                print(f"  [错误] {kernel}: {type(e).__name__}: {e}")
    print(f"\n全部完成，用时 {(time.time()-t0)/3600:.2f} 小时")


if __name__ == "__main__":
    main()
