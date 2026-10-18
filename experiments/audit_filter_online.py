#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线候选预筛选实验的**自动审计**（纯读文件，不占用 NPU）。

在把结果写进论文之前跑一遍，提前发现会让整批数据失效的问题。
检查项（任一 FAIL 即不应进入统计）：

  1. 结果 JSON 完整      非空、可解析、必需字段齐全、audit.npu_calls 存在
  2. 模型隔离            必须有显式 train_kernels 名单，且测试算子不在其中
  3. 预算受控            npu_calls <= eval_budget（fulleval 为 45，其余 13）
  4. 分代日志正确        筛选日志的 gen 覆盖预期的代，且无重复计数
  5. 复测成功            t_best_us > 0
  6. 覆盖完整            期望的 (配置 × 算子 × seed) 组合是否齐全

用法
----
    cd /workspace/Agent
    python experiments/audit_filter_online.py                  # 审计 15 算子 × 3 配置 × seed0
    python experiments/audit_filter_online.py --seeds 0 1 2    # 审计 3 次重复
    python experiments/audit_filter_online.py --config predictor
    python experiments/audit_filter_online.py --strict         # 警告也视为失败

退出码：0 = 通过；1 = 存在 FAIL（或 --strict 下存在 WARN）。
"""

import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.splits import load_split                       # noqa: E402

OUT_DIR = ROOT / "experiments" / "results" / "filter_online"
LOGS_DIR = ROOT / "experiments" / "logs"

DEFAULT_CONFIGS = ["baseline", "predictor", "random"]

REQUIRED_FIELDS = ("kernel", "config", "seed", "run_id", "success",
                   "t_best_us", "audit")

# 兜底预算表：仅当配置不在 filter_online_exp.CONFIGS 中时使用
BUDGET_FALLBACK = {"baseline": 13, "predictor": 13, "random": 13,
                   "filtered": 13, "fulleval": 45}


def _cfg_dict(cfg):
    """取该配置的参数字典（含 phase2 运行时注入的 predictor_foldN）。"""
    try:
        from experiments import filter_online_exp as fe
        return dict(fe.CONFIGS.get(cfg, {})) or {}
    except Exception:
        return {}


def _needs_model(cfg):
    """【修改】按配置**字段**判断是否需要模型，而不是硬编码配置名。

    旧实现用固定集合 {"predictor","random","filtered","fulleval"}，
    会漏掉 phase2 动态注入的 predictor_fold0..4，导致五折实验跳过
    模型隔离的元数据记录与校验。
    """
    d = _cfg_dict(cfg)
    if d:
        return bool(d.get("enable_candidate_filter", False))
    # 配置名不在 CONFIGS（例如只看旧结果目录）时，退回名称前缀判断
    return cfg.startswith(("predictor", "random", "filtered", "fulleval"))


def _budget_of(cfg):
    """预算优先取 CONFIGS，兜底用静态表。"""
    d = _cfg_dict(cfg)
    if d.get("eval_budget"):
        return d["eval_budget"]
    return BUDGET_FALLBACK.get(cfg)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, scope, msg):
        self.rows.append((level, scope, msg))

    ok = lambda self, s, m: self.add("OK", s, m)          # noqa: E731
    warn = lambda self, s, m: self.add("WARN", s, m)      # noqa: E731
    fail = lambda self, s, m: self.add("FAIL", s, m)      # noqa: E731

    def count(self, level):
        return sum(1 for r in self.rows if r[0] == level)


def audit_one(rep, cfg, kernel, seed, idx_cache):
    d = OUT_DIR / cfg
    p = d / f"{kernel}__r{seed}.json"
    scope = f"{cfg}/{kernel}/r{seed}"

    if not p.exists():
        rep.fail(scope, "结果文件缺失")
        return
    if p.stat().st_size == 0:
        rep.fail(scope, "结果文件为空（上次运行被中断）")
        return
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        rep.fail(scope, f"结果文件无法解析: {type(e).__name__}")
        return

    missing = [k for k in REQUIRED_FIELDS if k not in rec]
    if missing:
        rep.fail(scope, f"缺字段 {missing}")
        return

    audit = rec.get("audit") or {}
    npu = audit.get("npu_calls")
    budget = _budget_of(cfg) or audit.get("eval_budget")

    # --- 1. 复测 ---
    if not rec.get("success") or not rec.get("t_best_us"):
        rep.fail(scope, f"复测失败: {rec.get('error') or 't_best_us 为空'}")
    else:
        rep.ok(scope, f"T_best={rec['t_best_us']:.4f}us  S={rec.get('S')}")

    # --- 2. 预算 ---
    if npu is None:
        rep.fail(scope, "audit 缺 npu_calls（无法证明预算受控）")
    elif budget and npu > budget:
        rep.fail(scope, f"超预算: npu_calls={npu} > {budget}")
    elif budget and npu < budget:
        rep.warn(scope, f"未用满预算: npu_calls={npu} < {budget}"
                        f"（论文应区分预算上限与实际评测次数）")
    else:
        rep.ok(scope, f"预算 OK: npu_calls={npu}/{budget}")

    # --- 3. 模型隔离（按配置字段判断，覆盖 phase2 的 predictor_foldN） ---
    if _needs_model(cfg):
        model = rec.get("filter_model")
        train = rec.get("train_kernels")
        if not model:
            rep.fail(scope, "未记录 filter_model（无法验证泄漏）")
        elif train is None:
            rep.fail(scope, "模型 meta 未写入显式 train_kernels 名单")
        elif kernel in set(train):
            rep.fail(scope, f"数据泄漏：{kernel} 出现在训练名单")
        else:
            rep.ok(scope, f"模型隔离 OK（训练 {len(train)} 算子，不含本算子）")

    # --- 3b. 版本指纹 ---
    # 【修改】分组键拆成两层：
    #   * 代码层（fix_tag + git_commit）：三组对照必须完全一致 → 全局唯一
    #   * 模型层（model_path + model_sha256 + dataset_*）：**按配置分别检查**
    # 旧实现把 model_sha256 也放进全局分组键，而 baseline 根本不加载模型（哈希为 None），
    # predictor/random 才有哈希，于是正常的三组对照会被误报为"版本不一致"。
    prov = rec.get("provenance")
    if not prov:
        rep.fail(scope, "缺 provenance（无法确认是修复后版本的结果）")
    else:
        idx_cache["code"].setdefault(
            (prov.get("fix_tag"), prov.get("git_commit")), []).append(scope)
        idx_cache["model"].setdefault(
            (cfg, prov.get("model_path"), prov.get("model_sha256"),
             prov.get("dataset_snapshot"), prov.get("dataset_sha256")),
            []).append(scope)
        # 两条检查都只对"需要模型"的配置生效：baseline 不加载模型，
        # 哈希为 None 是正常现象，不能当问题报（与评审指出的误报同源）。
        if _needs_model(cfg):
            if not prov.get("model_sha256"):
                rep.fail(scope, "启用预筛选却未记录模型哈希（无法验证模型版本）")
            if prov.get("dataset_sha256") is None:
                rep.fail(scope, "未记录数据快照哈希（快照路径可能未解析到实际文件）")
            elif not prov.get("dataset_sha256_matches_meta"):
                rep.warn(scope, "数据快照哈希与模型 meta 记录不一致")

    # --- 4. Token ---
    if audit.get("llm_tokens_missing"):
        rep.warn(scope, "llm_tokens 缺失（成本列需标为“—”，不可当 0）")

    # --- 5. 分代日志 ---
    log = LOGS_DIR / f"filter_online__{cfg}__{kernel}__r{900 + seed}.jsonl"
    flog = LOGS_DIR / f"filter_scores__{cfg}__{kernel}__r{900 + seed}.jsonl"
    if _needs_model(cfg) and not flog.exists():
        rep.warn(scope, f"缺筛选打分日志 {flog.name}")

    if flog.exists():
        gens = defaultdict(int)
        sel = defaultdict(int)
        for line in flog.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in r:
                continue
            gens[r.get("gen")] += 1
            if r.get("selected"):
                sel[r.get("gen")] += 1
        # 第 0 代默认不筛选 → 打分日志应从 gen=1 开始
        if 0 in gens:
            rep.warn(scope, f"筛选日志含 gen=0（第 0 代被纳入筛选，与默认口径不符）")
        if cfg != "fulleval" and any(v != 4 for v in sel.values()):
            rep.warn(scope, f"某代送评数不等于 4: {dict(sel)}")
        if cfg == "fulleval" and gens and min(gens.values()) < 20:
            rep.warn(scope, f"fulleval 某代候选池不足 20: {dict(gens)}")
        idx_cache[(cfg, seed)][kernel] = dict(gens)


def main():
    ap = argparse.ArgumentParser(description="在线候选预筛选实验审计")
    ap.add_argument("--config", "-c", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--kernels", "-k", nargs="+", default=None,
                    help="默认取 splits/dev15.json 的 15 个留出算子")
    ap.add_argument("--seeds", "-s", type=int, nargs="+", default=(0,))
    ap.add_argument("--strict", action="store_true",
                    help="警告也视为失败")
    args = ap.parse_args()

    split = load_split("dev15")
    kernels = args.kernels or split["test_kernels"]

    rep = Report()
    idx_cache = defaultdict(dict)

    print("=" * 74)
    print(" 在线候选预筛选实验审计")
    print(f" 配置: {args.config}")
    print(f" 算子: {len(kernels)} 个（划分 {split['name']} v{split.get('version')}）")
    print(f" seed : {list(args.seeds)}")
    print("=" * 74)

    total = 0
    for cfg in args.config:
        for seed in args.seeds:
            for k in kernels:
                total += 1
                audit_one(rep, cfg, k, seed, idx_cache)

    # 只打印非 OK 项 + 汇总
    for level, scope, msg in rep.rows:
        if level != "OK":
            print(f"  [{level}] {scope}: {msg}")

    # ---- 版本指纹一致性（分两层检查）----
    # 代码层：所有配置必须同版本（混入修复前后的结果 = 致命）
    code_groups = idx_cache.get("code", {})
    if len(code_groups) > 1:
        print(f"  [FAIL] 代码版本不一致：{len(code_groups)} 种版本混在一批结果里")
        for key, scopes in sorted(code_groups.items(), key=lambda x: -len(x[1])):
            print(f"          fix_tag={key[0]} commit={str(key[1])[:8]} "
                  f"× {len(scopes)} 例：{scopes[:2]}")
        rep.fail("provenance", "代码版本不一致（可能混入了修复前的旧结果）")

    # 模型层：**按配置**检查。同一配置内出现不同模型/不同数据快照才判异常。
    # （baseline 不加载模型 → 哈希为 None，属正常，不应与 predictor/random 比较）
    model_groups = idx_cache.get("model", {})
    by_cfg = defaultdict(dict)
    for key, scopes in model_groups.items():
        by_cfg[key[0]][key] = scopes
    for cfg, groups in sorted(by_cfg.items()):
        if len(groups) > 1:
            print(f"  [FAIL] 配置 {cfg} 内部模型/数据版本不一致：{len(groups)} 种")
            for key, scopes in sorted(groups.items(), key=lambda x: -len(x[1])):
                print(f"          model={str(key[2])[:8] if key[2] else '—'} "
                      f"data={str(key[4])[:8] if key[4] else '—'} "
                      f"× {len(scopes)} 例：{scopes[:2]}")
            rep.fail("provenance", f"配置 {cfg} 内部模型/数据版本不一致")

    n_fail = rep.count("FAIL")
    n_warn = rep.count("WARN")
    n_ok = rep.count("OK")

    print("-" * 74)
    print(f" 组合数 {total} | 检查通过 {n_ok} | 警告 {n_warn} | 失败 {n_fail}")
    def _short(v):
        return (str(v)[:8] if v else "—").ljust(8)

    for key, scopes in sorted(code_groups.items(), key=lambda x: -len(x[1])):
        print(f"  代码版本 fix_tag={key[0]} commit={str(key[1])[:8]} × {len(scopes)}")
    for cfg, groups in sorted(by_cfg.items()):
        for key, scopes in groups.items():
            print(f"  {cfg:<12} model={_short(key[2])} "
                  f"data={_short(key[4])} × {len(scopes)}")

    # 覆盖度
    missing = [(c, s, k) for c in args.config for s in args.seeds for k in kernels
               if not (OUT_DIR / c / f"{k}__r{s}.json").exists()]
    if missing:
        print(f" ⚠ 缺失 {len(missing)} 个组合，例如: {missing[:3]}")

    print("=" * 74)
    if n_fail == 0 and n_warn == 0:
        print(" 结论：✅ 全部通过，可进入统计")
        return 0
    if n_fail == 0:
        print(" 结论：⚠️  无致命问题，但有警告需确认")
        return 1 if args.strict else 0
    print(" 结论：❌ 存在致命问题，不要进入统计")
    return 1


if __name__ == "__main__":
    sys.exit(main())
