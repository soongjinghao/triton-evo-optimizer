"""3.3 / 3.4 消融实验驱动脚本。

所有配置都是 FULL 的变体，只覆盖被研究的那个组件，其余组件与预算保持一致。
配置名同时作为方法名使用，因此结果落在 results/{配置名}/ 下，
日志落在 logs/{配置名}__{kernel}__r{run}.jsonl，互不干扰。

用法：
  python experiments/ablation_components.py list
  python experiments/ablation_components.py run --group 3.3
  python experiments/ablation_components.py run --config a6 --kernels eye_kernel
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

from experiments import common

# ---------------- 配置定义 ----------------
# 3.3 Profiling 证据与知识库检索策略消融
CONFIG_33 = {
    "a6": {"enable_profiling": False},                        # 关闭 Profiling，保留受约束 RAG
    "b3": {"enable_rag": False},                              # 启用 Profiling，无 RAG
    "b4": {"rag_mode": "semantic"},                           # 启用 Profiling，普通语义 Top-k
    "full": {},                                               # 混合重排 + Guard（基线）
}

# 3.4.1 父代选择
CONFIG_341 = {
    "sel_roulette": {"selection": "roulette"},
    "sel_tournament": {"selection": "tournament"},
}

# 3.4.2 交叉策略
CONFIG_342 = {
    "cross_unconstrained": {"crossover_mode": "unconstrained"},
    "cross_protected": {"crossover_mode": "protected"},
}

# 3.4.3 变异策略
CONFIG_343 = {
    "mut_adaptive": {"mutation_mode": "adaptive"},
    "mut_uniform": {"mutation_mode": "uniform"},
    "mut_aggressive": {"mutation_mode": "aggressive"},
}

GROUPS = {
    "3.3": CONFIG_33,
    "3.4.1": CONFIG_341,
    "3.4.2": CONFIG_342,
    "3.4.3": CONFIG_343,
    "3.4": {**CONFIG_341, **CONFIG_342, **CONFIG_343},
}


def list_configs():
    print("可用配置：")
    for g, cfgs in GROUPS.items():
        if g == "3.4":
            continue
        print(f"\n  [{g}]")
        for name, ov in cfgs.items():
            print(f"    {name:<22} {ov if ov else '(FULL 基线)'}")
    print("\n  [3.4] 为 3.4.1 + 3.4.2 + 3.4.3 的全部 7 个配置")


def run(group=None, config=None, kernels=None, run_id=0, seed=0,
        repeats=5, device=0, skip_done=True):
    from experiments.pipeline import stage_search, stage_remeasure

    if config:
        cfgs = {config: _find_config(config)}
    else:
        cfgs = GROUPS[group]

    kernels = kernels or common.PILOT_KERNELS
    common.ensure_dirs()

    for cfg_name, overrides in cfgs.items():
        for k in kernels:
            out_py = common.RESULTS_DIR / cfg_name / f"{k}__r{run_id}.py"
            if skip_done and out_py.exists():
                print(f"[skip] {cfg_name}/{k}/r{run_id} 已完成")
                continue

            print(f"\n===== [{time.strftime('%H:%M:%S')}] {cfg_name} / {k} =====")
            try:
                stage_search(k, cfg_name, run_id=run_id, seed=seed,
                             overrides=overrides, save_gen0=True)
            except Exception as e:
                print(f"[run] {cfg_name}/{k} 搜索失败: {type(e).__name__}: {e}")
                continue
            try:
                stage_remeasure(k, cfg_name, run_id=run_id,
                                repeats=repeats, device_id=device)
            except Exception as e:
                print(f"[run] {cfg_name}/{k} 复测失败: {type(e).__name__}: {e}")


def _find_config(name):
    for cfgs in GROUPS.values():
        if name in cfgs:
            return cfgs[name]
    raise ValueError(f"未知配置: {name}")


def main():
    ap = argparse.ArgumentParser(description="3.3 / 3.4 消融实验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出所有可用配置")

    rp = sub.add_parser("run", help="运行消融配置")
    rp.add_argument("--group", choices=list(GROUPS.keys()),
                    help="按实验分组运行（3.3 / 3.4.1 / 3.4.2 / 3.4.3 / 3.4）")
    rp.add_argument("--config", help="只跑单个配置名")
    rp.add_argument("--kernels", "-k", nargs="+", default=None)
    rp.add_argument("--run-id", "-r", type=int, default=0)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--repeats", type=int, default=5)
    rp.add_argument("--device", type=int, default=0)
    rp.add_argument("--no-skip", action="store_true", help="已完成的也重跑")

    args = ap.parse_args()

    if args.cmd == "list":
        list_configs()
    else:
        if not args.group and not args.config:
            ap.error("需指定 --group 或 --config")
        run(group=args.group, config=args.config, kernels=args.kernels,
            run_id=args.run_id, seed=args.seed, repeats=args.repeats,
            device=args.device, skip_done=not args.no_skip)


if __name__ == "__main__":
    main()
