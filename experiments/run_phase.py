#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三阶段实验统一入口：每个阶段只跑这一个脚本。

子命令
------
  freeze     冻结训练数据快照（samples_frozen_YYYYMMDD.jsonl），后续训练/评估锚定它
  setup      冻结实验配置 + 记录 Git 提交 + 训练「排除 15 开发算子」的预筛选模型
  smoke      3 个算子 × {baseline,predictor,random} × 1 seed（流程自检，不计结论）
  phase1     15 开发算子 × 3 配置 × 3 seed（正式小规模验证）
  phase2     50 算子 5 折外层验证：--fold N 跑第 N 折（每折 10 测试 × 3 配置 × 3 seed）
             baseline / random 在 5 折的测试集上拼起来即覆盖全部 50 算子一次；
             predictor 每折用该折训练的模型，写入 predictor_foldN 独立目录防覆盖。
  diagnostic 对 3~5 个算子做候选池全评（fulleval），用于真实 P@4 / Recall@4

用法
----
  python experiments/run_phase.py freeze
  python experiments/run_phase.py setup
  EA_MAX_WORKERS=24 python experiments/run_phase.py smoke
  EA_MAX_WORKERS=24 python experiments/run_phase.py phase1
  EA_MAX_WORKERS=24 python experiments/run_phase.py phase2 --fold 0
  python experiments/run_phase.py diagnostic

依赖环境变量：EA_MAX_WORKERS（LLM 生成并发，建议 8，与芯片数无关）、
EA_NUM_DEVICES（NPU 卡数，默认 2，须 <= 实际芯片数）、set_env/set_api_local.sh
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import common
from experiments.splits import load_split  # noqa: E402
from experiments.candidate_filter_model_final import (  # noqa: E402
    train_and_save, load_df, add_pc_features, _sha256_file)
from experiments import filter_online_exp as fe  # noqa: E402
from experiments.filter_online_exp import run_one  # noqa: E402

EXP = ROOT / "experiments"
CONFIG_JSON = EXP / "experiment_config.json"
MODEL_DIR = EXP / "dataset"
# 冻结的训练数据快照文件名。训练数据会随实验推进持续增长（collect_data=True），
# 若不锚定快照，前后两次训练/评估的样本集就不可比，离线表11 与在线模型也会对不上。
DATASET_SNAPSHOT = os.getenv("EA_DATASET_SNAPSHOT", "samples_frozen_20261016.jsonl")

# 【修改】开发/留出集改从固定划分文件读取。
# 旧实现 `DEVS = list(K15)`，而 K15 是 analyze_logs 从 results/b3/ 目录 glob 出来的
# **set**：顺序不确定（影响"前 3 个算子"这类切片）、且目录一旦变动就为空。
# 现在所有训练/在线/汇总统一引用 experiments/splits/dev15.json。
SPLIT = load_split("dev15")
DEVS = sorted(SPLIT["test_kernels"])  # 15 个留出算子（已参与模型选型，作测试集）


def k50():
    return sorted(p.stem for p in common.MANIFEST_DIR.glob("*.json"))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def train_predictor(exclude, tag):
    # 【修改】强制使用冻结快照，不能读动态增长的 samples.jsonl。
    # 否则"这次训练用的数据"与"下次评估用的数据"不是同一批，
    # 离线表11 与在线模型对不上，且在线实验新产生的样本可能回流进训练集。
    snapshot = MODEL_DIR / DATASET_SNAPSHOT
    if not snapshot.is_file():
        raise FileNotFoundError(
            f"冻结数据快照不存在：{snapshot}\n"
            f"请先运行：python experiments/run_phase.py freeze")
    df = load_df(snapshot)
    ek = set(exclude)
    df = df[~df["kernel"].isin(ek)].copy()
    df = add_pc_features(df)
    out = MODEL_DIR / f"candidate_filter_model__{tag}.joblib"
    rep = EXP / "results" / f"candidate_filter_{tag}.json"
    train_and_save(df, out, rep, exclude_kernels=sorted(ek),
                   dataset_path=snapshot)
    # 把真实训练算子名单写回 meta（防泄漏校验依赖它）
    _write_train_kernels(out, df, sorted(ek), tag)
    return out


def _write_train_kernels(model_path, df, excluded, tag):
    """把真实训练算子名单写入 *.meta.json，供在线实验做防泄漏校验。"""
    meta_path = Path(model_path).with_name(Path(model_path).stem + ".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) \
            if meta_path.exists() else {}
    except Exception:
        meta = {}
    train = sorted(set(df["kernel"].astype(str)) - set(excluded))
    meta.update({
        "tag": tag,
        "train_kernels": train,          # 显式名单，在线实验据此校验
        "exclude_kernels": sorted(excluded),
        "n_train_kernels": len(train),
        "n_samples": int(len(df)),
        # 【修改】写成绝对路径。旧实现这里写的是相对文件名，
        # 导致 filter_online_exp 的 _sha256_file(ds) 找不到文件、数据哈希恒为 None。
        "dataset_snapshot": str(MODEL_DIR / DATASET_SNAPSHOT),
        "dataset_sha256": _sha256_file(MODEL_DIR / DATASET_SNAPSHOT),
        "split": SPLIT["name"],
        "split_version": SPLIT.get("version"),
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[train] 训练名单已写入 {meta_path.name} "
          f"（{len(train)} 算子 / {len(df)} 样本 / 快照 {DATASET_SNAPSHOT}）")


def model_for(tag):
    return str(MODEL_DIR / f"candidate_filter_model__{tag}.joblib")


def _run_config(kernels, config, seeds, model_path, tag=None):
    """跑一组 (配置 × 算子 × seed)。返回失败列表，绝不静默吞掉错误。

    【修改】旧实现捕获异常后只打印 [错误] 就继续，阶段末尾看起来"跑完了"，
    实际可能有一半组合失败。现在收集失败项、落盘到 JSON，并由 main 汇总后
    以非零退出码结束，让上层脚本能感知。
    """
    cfg_name = tag or config
    failures = []
    for seed in seeds:
        for k in kernels:
            try:
                rec = run_one(k, cfg_name, seed, model_path=model_path)
                s = rec.get("S")
                npu = (rec.get("audit") or {}).get("npu_calls")
                print(f"   {cfg_name:16s} {k:36s} seed{seed} S={s} npu={npu}")
            except Exception as e:
                failures.append({"config": cfg_name, "kernel": k, "seed": seed,
                                 "error": f"{type(e).__name__}: {e}"})
                print(f"   [错误] {cfg_name} {k} seed{seed}: "
                      f"{type(e).__name__}: {e}")
    return failures


def _dump_failures(failures, stage):
    if not failures:
        return
    p = EXP / "results" / f"filter_online_failures__{stage}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(failures, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"\n[失败记录] {len(failures)} 项 → {p}")


def setup():
    cfg = {
        "population_size": 6, "gen0_candidates": 5, "max_generations": 2,
        "children_per_generation": 4, "eval_budget": 13,
        "seed_diversity_threshold": 0.90, "selection": "ucb", "ucb_c": 0.5,
        "crossover_mode": "protected", "mutation_mode": "adaptive",
        "enable_profiling": True, "enable_rag": True,
        "enable_strategy_init": True, "remeasure_repeats": 5,
        "candidate_pool_size": 20, "topk": 4,
        # 【修改】第 0 代不启用预筛选，三组 Gen0 完全一致（见 evolutionary_algorithm 注释）
        "enable_candidate_filter_gen0": False,
        "split": SPLIT["name"],
        "split_version": SPLIT.get("version"),
        "dataset_snapshot": DATASET_SNAPSHOT,
        "dev_kernels(15)": DEVS,
        "git_commit": git_commit(),
        "note": "PAPER-FULL 父代选择采用 UCB(c=0.5)；预测模型按排除开发集训练，防泄漏；"
                "第 0 代不启用候选预筛选",
    }
    CONFIG_JSON.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[setup] 配置已冻结 → {CONFIG_JSON}")
    if not (MODEL_DIR / DATASET_SNAPSHOT).exists():
        print(f"[setup] ⚠️ 未找到数据快照 {DATASET_SNAPSHOT}，"
              f"请先执行 python experiments/run_phase.py freeze")
    train_predictor(exclude=DEVS, tag="dev15")
    print("\n[setup] 完成。下一步：python experiments/run_phase.py smoke")


def freeze():
    """冻结训练数据快照：之后的训练/评估一律锚定该文件，保证前后可比。

    背景：collect_data=True 会在每次真机评测后往 samples.jsonl 追加样本，
    数据集随实验推进持续增长。若不冻结，"表11 的离线评估"与"在线实验用的模型"
    就可能基于不同的样本集，论文自洽性受损；在线实验新产生的样本还可能回流。
    """
    src = MODEL_DIR / "samples.jsonl"
    dst = MODEL_DIR / DATASET_SNAPSHOT
    if dst.exists():
        print(f"[freeze] 快照已存在：{dst}（{dst.stat().st_size} 字节），不覆盖")
        return dst
    if not src.exists():
        raise SystemExit(f"[freeze] 源数据不存在：{src}")
    import shutil
    shutil.copy2(src, dst)
    n = sum(1 for _ in dst.open(encoding="utf-8"))
    print(f"[freeze] 已冻结 {dst.name}（{n} 条样本，源 {src.name}）")
    print(f"[freeze] 后续在线实验建议设 collect_data=False，避免新样本回流训练集")
    return dst


def smoke():
    ks = DEVS[:3]
    mp = model_for("dev15")
    print(f"[smoke] 3 算子 × 3 配置 × 1 seed；模型={mp}")
    fails = []
    for c in ("baseline", "predictor", "random"):
        fails += _run_config(ks, c, [0], (mp if c != "baseline" else None))
    _dump_failures(fails, "smoke")
    return fails


def phase1():
    ks = DEVS
    mp = model_for("dev15")
    print(f"[phase1] 15 开发算子 × 3 配置 × 3 seed；模型={mp}")
    fails = []
    for c in ("baseline", "predictor", "random"):
        fails += _run_config(ks, c, [0, 1, 2], (mp if c != "baseline" else None))
    _dump_failures(fails, "phase1")
    return fails


def phase2(fold):
    allk = k50()
    n = len(allk)
    fold_size = (n + 4) // 5
    folds = [allk[i * fold_size:(i + 1) * fold_size] for i in range(5)]
    test = folds[fold]
    out = train_predictor(exclude=test, tag=f"fold{fold}")
    mp = str(out)
    # 注入该折 predictor 配置，结果写入独立目录 predictor_foldN，避免不同折互相覆盖
    fe.CONFIGS[f"predictor_fold{fold}"] = dict(
        eval_budget=13, enable_candidate_filter=True,
        candidate_pool_size=fe.POOL, candidate_filter_mode="model")
    print(f"[phase2] fold{fold} 测试 {len(test)} 算子，训练 {len(allk)-len(test)} 算子；模型={mp}")
    # 【修改】random 也必须拿到该折模型：它虽不用于筛选，但要记录模型分做 Oracle 对照，
    # 且新版的防泄漏强校验要求显式模型路径（传 None 会被拒绝运行）。
    fails = []
    fails += _run_config(test, "baseline", [0, 1, 2], None)
    fails += _run_config(test, "random", [0, 1, 2], mp)
    fails += _run_config(test, "predictor", [0, 1, 2], mp,
                         tag=f"predictor_fold{fold}")
    _dump_failures(fails, f"phase2_fold{fold}")
    return fails


def diagnostic():
    ks = DEVS[:5]
    mp = model_for("dev15")
    # 【修改】旧实现传 model_path=None，fulleval 于是回落到默认模型
    # dataset/candidate_filter_model.joblib —— 该模型用全量数据训练（含这 15 个算子），
    # 用它跑在线评估属于数据泄漏。这里显式指定排除 dev15 的模型。
    print(f"[diagnostic] 全评诊断（fulleval）：{len(ks)} 算子；模型={mp}")
    fails = _run_config(ks, "fulleval", [0], mp)
    _dump_failures(fails, "diagnostic")
    return fails


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "freeze", "smoke", "phase1", "diagnostic"):
        sub.add_parser(name)
    p2 = sub.add_parser("phase2")
    p2.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    fails = []
    if a.cmd == "setup":
        setup()
    elif a.cmd == "freeze":
        freeze()
    elif a.cmd == "smoke":
        fails = smoke()
    elif a.cmd == "phase1":
        fails = phase1()
    elif a.cmd == "phase2":
        fails = phase2(a.fold)
    elif a.cmd == "diagnostic":
        fails = diagnostic()

    # 【修改】阶段结束必须报告失败数，并以非零退出码结束存在失败组合的阶段，
    # 不能只打印 [错误] 就让整阶段看起来成功。
    if fails:
        print(f"\n{'=' * 60}")
        print(f" 阶段 {a.cmd} 完成，但有 {len(fails)} 个组合失败")
        for f in fails[:10]:
            print(f"   - {f['config']} / {f['kernel']} / seed{f['seed']}: {f['error']}")
        if len(fails) > 10:
            print(f"   ... 另有 {len(fails) - 10} 项")
        print(f"{'=' * 60}")
        sys.exit(1)
    print(f"\n[完成] 阶段 {a.cmd} 全部成功")


if __name__ == "__main__":
    main()
