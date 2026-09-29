"""3.1 主实验流水线。

阶段 A（B0 + 种子重测）：
    python experiments/pipeline.py manifest --kernel mean_kernel

阶段 B（搜索）：
    python experiments/pipeline.py search --kernel mean_kernel --method full

阶段 C（串行复测 T_best）：
    python experiments/pipeline.py remeasure --kernel mean_kernel --method full

三个阶段的测量均使用完全相同的 msprof 命令与计时口径；
阶段 A/C 额外串行单设备重复 5 次取中位数。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime

from experiments import common
from experiments.harness import build_ea, build_executor, make_config


# ============================ 阶段 A：Manifest ============================
def stage_manifest(kernel_name: str, repeats: int = 5, device_id: int = 0):
    """重测参考实现 T_base 与各初始种子，产出 T_seed。"""
    common.ensure_dirs()

    # 已有相同 repeats 的结果则直接复用。T_base/T_seed 是纯测量量，
    # 重跑不改变结果；若每次重启都重测，50 个算子将额外消耗约 4 小时。
    mf = common.MANIFEST_DIR / f"{kernel_name}.json"
    if mf.exists():
        try:
            prev = json.loads(mf.read_text(encoding="utf-8"))
            if prev.get("repeats") == repeats and prev.get("t_base_us"):
                print(f"[Manifest] {kernel_name}: 已完成 (repeats={repeats})，跳过重测")
                return prev
        except Exception:
            pass

    config = make_config("b0", kernel_name, log=False)
    executor = build_executor(kernel_name, config)

    seeds = {}
    for p in common.seed_code_paths(kernel_name):
        code = common.read_code(p)
        res = executor.measure_repeat(code, repeats=repeats, device_id=device_id)
        seeds[p.name] = {
            "path": str(p),
            "latency_us": res.execution_time if res.success else None,
            "success": res.success,
            "error": res.error,
        }
        print(f"[Manifest] {kernel_name}/{p.name}: "
              f"{res.execution_time if res.success else 'FAILED'} us")

    ref = seeds.get(f"{kernel_name}.py", {})
    t_base = ref.get("latency_us")
    valid = [v["latency_us"] for v in seeds.values() if v["success"] and v["latency_us"]]
    t_seed = min(valid) if valid else None

    # 种子源码相似度（供 3.2 复用）
    sim = None
    paths = [p for p in common.seed_code_paths(kernel_name)]
    if len(paths) >= 2:
        from difflib import SequenceMatcher
        sim = SequenceMatcher(None, common.read_code(paths[0]),
                              common.read_code(paths[1])).ratio()

    record = {
        "kernel": kernel_name,
        "t_base_us": t_base,
        "t_seed_us": t_seed,
        "seed_similarity": sim,
        "repeats": repeats,
        "device_id": device_id,
        "seeds": seeds,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = common.MANIFEST_DIR / f"{kernel_name}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Manifest] 已保存 {out} | T_base={t_base} T_seed={t_seed} sim={sim}")
    return record


# ============================ 阶段 B：搜索 ============================
def stage_search(kernel_name: str, method: str, run_id: int = 0, seed: int = 0,
                 budget: int = None, tau: float = None, save_gen0: bool = False,
                 overrides: dict = None):
    """搜索阶段。tau: 种子相似度阈值；0 或负数表示不筛选（阈值置为 1.01）。
    overrides: 任意 EAConfig 字段覆盖，供 3.3 / 3.4 的消融配置使用。"""
    common.ensure_dirs()
    override = {}
    if budget:
        override["eval_budget"] = budget
    if tau is not None:
        override["seed_diversity_threshold"] = tau if tau > 0 else 1.01
    if overrides:
        override.update(overrides)

    ea, config = build_ea(kernel_name, method, run_id=run_id, seed=seed, **override)

    # 每次搜索重置事件日志：ExpLogger 以追加模式写入，若沿用上次的日志文件，
    # 历史事件会被重复计入 n_valid / n_ast_reject，导致有效候选率 > 100%。
    if getattr(config, "log_path", None):
        _lp = Path(config.log_path)
        if _lp.exists():
            _lp.unlink()

    # 同一 (method, kernel, run_id) 重跑时重置日志，避免事件重复计数
    if config.log_path and Path(config.log_path).exists():
        Path(config.log_path).unlink()

    seed_codes = [common.read_code(p) for p in common.seed_code_paths(kernel_name)]
    t0 = time.time()
    best = ea.run(seed_codes)
    elapsed = time.time() - t0

    # 3.2：不同 tau 的结果分目录存放，避免相互覆盖
    if tau is not None:
        tag = "none" if tau <= 0 else str(tau)
        out_dir = common.RESULTS_DIR / f"tau_{tag}"
    else:
        out_dir = common.RESULTS_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = out_dir / f"{kernel_name}__r{run_id}.py"
    code_path.write_text(best.code or "", encoding="utf-8")

    # 3.2：第 0 代种群快照与 D_G0
    gen0_codes = [ind.code for ind in getattr(ea, "gen0_snapshot", []) if ind.code]
    d_g0 = common.diversity_g0(gen0_codes)
    if save_gen0 and gen0_codes:
        gen0_dir = out_dir / f"{kernel_name}__r{run_id}_gen0"
        gen0_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(gen0_codes):
            (gen0_dir / f"{i}.py").write_text(c, encoding="utf-8")

    # 从事件日志统计 AST 拒绝数与真实评测数
    n_reject = n_eval_done = n_seeds_after = n_valid = 0
    if config.log_path and Path(config.log_path).exists():
        for line in Path(config.log_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "reject":
                n_reject += 1
            elif rec.get("event") == "eval":
                n_eval_done += 1
                if rec.get("latency_us"):
                    n_valid += 1
            elif rec.get("event") == "seed_filter":
                n_seeds_after = rec.get("seeds_after", 0)

    n_seeds_total = len(common.seed_code_paths(kernel_name))
    # LLM 成功率统计：LLM 全部失败时种群只剩种子，fitness 恒为 1.0，
    # 这是"假成功"，必须显式标记，避免污染最终统计。
    _stats = ea.genetic_ops.llm.get_stats()
    _pb = _stats.get("purpose_breakdown") or {}
    _llm_total = sum(p.get("count", 0) for p in _pb.values())
    _llm_ok = sum(p.get("success", 0) for p in _pb.values())
    _llm_rate = (_llm_ok / _llm_total) if _llm_total else 0.0
    _search_ok = bool(_llm_total and _llm_rate >= 0.5)

    record = {
        "kernel": kernel_name,
        "method": method,
        "run_id": run_id,
        "seed": seed,
        "tau": None if tau is None else (None if tau <= 0 else tau),
        "best_id": best.id,
        "best_fitness": best.fitness,
        "best_latency_us": best.metadata.get("execution_time"),
        "n_eval": ea.eval_count,
        "n_eval_done": n_eval_done,
        "n_valid": n_valid,
        "n_ast_reject": n_reject,
        "n_seeds_total": n_seeds_total,
        "n_seeds_retained": n_seeds_after or n_seeds_total,
        "gen0_count": len(gen0_codes),
        "d_g0": round(d_g0, 4),
        "elapsed_seconds": round(elapsed, 1),
        "llm_calls": _stats["call_count"],
        "llm_tokens": _stats["total_tokens"],
        "llm_success": _llm_ok,
        "llm_success_rate": round(_llm_rate, 3),
        "search_ok": _search_ok,
        "code_path": str(code_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_dir / f"{kernel_name}__r{run_id}.json"
    meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Search] {method}/{kernel_name}/r{run_id}: fitness={best.fitness:.4f} "
          f"n_eval={ea.eval_count} elapsed={elapsed:.0f}s")
    return record


# ============================ 阶段 C：复测 ============================
def stage_remeasure(kernel_name: str, method: str, run_id: int = 0,
                    repeats: int = 5, device_id: int = 0, tau: float = None):
    """对搜索得到的最优候选，在与 T_base/T_seed 相同的串行单设备协议下复测。"""
    common.ensure_dirs()
    if tau is not None:
        tag = "none" if tau <= 0 else str(tau)
        code_path = common.RESULTS_DIR / f"tau_{tag}" / f"{kernel_name}__r{run_id}.py"
    else:
        code_path = common.RESULTS_DIR / method / f"{kernel_name}__r{run_id}.py"
    # 上游 search 失败时不会产出代码文件（多因 LLM 不可达）。
    # 此时应跳过复测，而不是抛异常中断整个批次。
    if not code_path.exists():
        out_dir = code_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "kernel": kernel_name,
            "method": method,
            "run_id": run_id,
            "tau": None if tau is None else (None if tau <= 0 else tau),
            "t_best_us": None,
            "success": False,
            "error": "search_no_output",
            "repeats": repeats,
            "device_id": device_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _mp = out_dir / f"{kernel_name}__r{run_id}.remeasure.json"
        _mp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[Remeasure] {method}/{kernel_name}/r{run_id}: "
              f"跳过（搜索未产出代码）")
        return record

    # 搜索阶段 LLM 大面积失败（假成功）时，不再浪费 NPU 复测，直接标记无效
    out_dir = code_path.parent
    _srec_path = out_dir / f"{kernel_name}__r{run_id}.json"
    _search_ok = True
    if _srec_path.exists():
        try:
            _srec = json.loads(_srec_path.read_text(encoding="utf-8"))
            _search_ok = _srec.get("search_ok", True)
        except Exception:
            _search_ok = True

    if not _search_ok:
        record = {
            "kernel": kernel_name,
            "method": method,
            "run_id": run_id,
            "tau": None if tau is None else (None if tau <= 0 else tau),
            "t_best_us": None,
            "success": False,
            "error": "search_llm_failed",
            "repeats": repeats,
            "device_id": device_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _mp = out_dir / f"{kernel_name}__r{run_id}.remeasure.json"
        _mp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[Remeasure] {method}/{kernel_name}/r{run_id}: "
              f"跳过（搜索阶段 LLM 失败，判定为无效）")
        return record

    config = make_config(method, kernel_name, run_id=run_id, log=False)
    executor = build_executor(kernel_name, config)
    res = executor.measure_repeat(common.read_code(code_path), repeats=repeats,
                                  device_id=device_id)

    record = {
        "kernel": kernel_name,
        "method": method,
        "run_id": run_id,
        "tau": None if tau is None else (None if tau <= 0 else tau),
        "t_best_us": res.execution_time if res.success else None,
        "success": res.success,
        "error": res.error,
        "repeats": repeats,
        "device_id": device_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_dir / f"{kernel_name}__r{run_id}.remeasure.json"
    meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Remeasure] {method}/{kernel_name}/r{run_id}: "
          f"T_best={record['t_best_us']} us")
    return record


def main():
    ap = argparse.ArgumentParser(description="3.1 主实验流水线")
    ap.add_argument("stage", choices=["manifest", "search", "remeasure"])
    ap.add_argument("--kernel", "-k", required=True)
    ap.add_argument("--method", "-m", default="full", choices=["b1", "b2", "full"])
    ap.add_argument("--run-id", "-r", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--budget", type=int, default=None,
                    help="覆盖 NPU 评测预算（冒烟调试用，正式实验留空）")
    ap.add_argument("--tau", type=float, default=None,
                    help="种子相似度阈值 tau（3.2 敏感性实验）；0 表示不筛选")
    ap.add_argument("--save-gen0", action="store_true",
                    help="保存第 0 代候选源码，便于复核 D_G0")
    args = ap.parse_args()

    if args.stage == "manifest":
        stage_manifest(args.kernel, repeats=args.repeats, device_id=args.device)
    elif args.stage == "search":
        stage_search(args.kernel, args.method, run_id=args.run_id, seed=args.seed,
                     budget=args.budget, tau=args.tau, save_gen0=args.save_gen0)
    else:
        stage_remeasure(args.kernel, args.method, run_id=args.run_id,
                        repeats=args.repeats, device_id=args.device, tau=args.tau)


if __name__ == "__main__":
    main()
