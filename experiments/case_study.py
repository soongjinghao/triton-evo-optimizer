"""3.5 代表性算子案例：提取可测量数据并采集优化前后的 Profiling 指标。

对每个案例算子：
  T_base   参考实现延迟（取自 manifest，串行 5 次中位数）
  T_before 最快有效初始种子延迟（取自 manifest）
  T_after  该方法最终候选延迟（取自 remeasure）
  S        = T_base / T_after
  并分别对种子代码与最终候选各跑一次评测，从 Profiling 中取主导管线及其前后数值。

用法：
  python experiments/case_study.py -k mean_kernel _rms_norm_kernel
  python experiments/case_study.py -k mean_kernel -m full --run-id 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json

from experiments import common
from experiments.harness import build_executor, make_config


def latest_opprof(kernel: str, device: int = 0):
    base = common.DATASETS_DIR / kernel / "performance" / f"{kernel}_dev_{device}" / kernel
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("OPPROF_")]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def pipe_metrics(opprof_dir) -> dict:
    """从 PipeUtilization.csv 求各管线平均占用率（百分数）。"""
    csv_path = opprof_dir / "PipeUtilization.csv"
    if not csv_path.exists():
        return {}
    cols = {"aiv_vec_ratio": "Vector", "aiv_scalar_ratio": "Scalar",
            "aiv_mte2_ratio": "MTE2(读)", "aiv_mte3_ratio": "MTE3(写)"}
    acc = {v: [] for v in cols.values()}
    for row in csv.DictReader(csv_path.open(encoding="utf-8")):
        for key, name in cols.items():
            raw = row.get(key)
            if raw and raw != "NA":
                try:
                    acc[name].append(float(raw) * 100)
                except ValueError:
                    pass
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}


def measure_and_profile(kernel: str, code: str, device: int = 0):
    """跑一次评测（同时生成 Profiling），返回延迟与管线指标。"""
    config = make_config("full", kernel, log=False)
    executor = build_executor(kernel, config)
    res = executor.evaluate(code, timeout=600, device_id=device)
    prof = latest_opprof(kernel, device)
    metrics = pipe_metrics(prof) if prof else {}
    return (res.execution_time if res.success else None), metrics


def main():
    ap = argparse.ArgumentParser(description="3.5 代表性算子案例")
    ap.add_argument("--kernels", "-k", nargs="+", required=True)
    ap.add_argument("--method", "-m", default="full")
    ap.add_argument("--run-id", "-r", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    common.ensure_dirs()
    rows = []

    for kernel in args.kernels:
        man_path = common.MANIFEST_DIR / f"{kernel}.json"
        rem_path = (common.RESULTS_DIR / args.method
                    / f"{kernel}__r{args.run_id}.remeasure.json")
        best_path = common.RESULTS_DIR / args.method / f"{kernel}__r{args.run_id}.py"

        if not man_path.exists():
            print(f"[case] 跳过 {kernel}：无 manifest")
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))

        t_after = None
        if rem_path.exists():
            rem = json.loads(rem_path.read_text(encoding="utf-8"))
            if rem.get("success"):
                t_after = rem.get("t_best_us")
        if t_after is None and best_path.exists():
            print(f"[case] {kernel}：无复测结果，现测一次")
            t_after, _ = measure_and_profile(kernel, common.read_code(best_path), args.device)

        # 优化前：对最快种子跑一次，取 Profiling
        seed_path = common.seed_code_paths(kernel)[0]
        _, prof_before = measure_and_profile(kernel, common.read_code(seed_path), args.device)

        # 优化后：对最终候选跑一次
        prof_after = {}
        if best_path.exists():
            _, prof_after = measure_and_profile(kernel, common.read_code(best_path), args.device)

        # 取优化前占比最高的管线作为该案例的主要指标
        main_metric = max(prof_before, key=prof_before.get) if prof_before else "—"

        rows.append({
            "kernel": kernel,
            "method": args.method,
            "t_base": man.get("t_base_us"),
            "t_before": man.get("t_seed_us"),
            "t_after": t_after,
            "S": (man["t_base_us"] / t_after) if (man.get("t_base_us") and t_after) else None,
            "main_metric": main_metric,
            "metric_before": prof_before.get(main_metric),
            "metric_after": prof_after.get(main_metric),
            "profile_before": prof_before,
            "profile_after": prof_after,
        })
        print(f"[case] {kernel}: T_base={man.get('t_base_us')} "
              f"T_before={man.get('t_seed_us')} T_after={t_after} "
              f"主指标={main_metric}")

    print("\n表10 代表性算子案例的可测量数据")
    print(f"{'算子':<26}{'T_base':>10}{'T_before':>10}{'T_after':>10}"
          f"{'S':>8}{'主要指标':>12}{'前':>9}{'后':>9}")
    print("-" * 96)
    for r in rows:
        def f(v, d=2):
            return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
        print(f"{r['kernel']:<26}{f(r['t_base']):>10}{f(r['t_before']):>10}"
              f"{f(r['t_after']):>10}{f(r['S'], 4):>8}{r['main_metric']:>12}"
              f"{f(r['metric_before'], 1):>9}{f(r['metric_after'], 1):>9}")

    out = common.RESULTS_DIR / "table10_cases.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[case] 已写出 {out}")


if __name__ == "__main__":
    main()
