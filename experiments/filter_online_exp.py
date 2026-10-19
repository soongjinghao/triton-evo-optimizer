#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.5 候选预筛选的**在线（真机）**实验。

对照设计
--------
NPU 预算是唯一的变量控制点。第 0 代**不启用**预筛选，三组完全一致：

                 Gen0            Gen1            Gen2        NPU 评测
  baseline     5 → 全评 5      4 → 全评 4      4 → 全评 4      13 次
  predictor    5 → 全评 5     20 → 模型挑 4   20 → 模型挑 4     13 次
  random       5 → 全评 5     20 → 随机挑 4   20 → 随机挑 4     13 次
  fulleval     5 → 全评 5     20 → 全评 20    20 → 全评 20     45 次
                ↑ 子实验专用：让被模型排到后面的候选也有真值，
                  事后即可算出在线精确率与召回率

其中 baseline / predictor / random 的**评测次数严格相同**（均为 13），
多出来的只是廉价的 LLM 生成，因此三者的差异只能来自"预算分配给哪些候选"。
random 用于隔离"候选池从 4 扩到 20"本身的贡献——若 predictor 与 random 无差异，
说明收益来自扩池而非模型排序。

第 0 代为何不筛选
-----------------
第 0 代候选没有 parent_latency，模型在该子集上判别力接近随机（离线 AUC≈0.5，
见论文 3.5.5："仅对变异与交叉阶段启用预筛选"）。且第 0 代承担策略引导与种群
多样化职责。三组共用完全相同的第 0 代，差异才只能归因于进化阶段的排序决策。

防泄漏
------
predictor / filtered / random / fulleval **必须**显式传 --model-path。
启动前强制读取模型的 *.meta.json，要求 train_kernels 是显式算子名单，
且当前测试算子不在名单内；否则直接拒绝运行（旧实现对该场景是静默放行的）。

指标（按"一代的变化"来量，不以端到端加速比为主）
-------------------------------------------
  - 每代送评测的候选中 G_mut > 1 的比例（代级改进率）
  - 每代最优候选的增益
  - 单位评测带来的有效改进次数（预算效率）
  - 最终 GM(S)（次要指标，明确其可能不显著）

数据安全
--------
  * 使用 run_id=900+ 与独立的日志/结果目录，绝不覆盖 results/full（3.1 主实验）
  * LLM 生成并发由 EA_MAX_WORKERS 控制（默认 8，与芯片数无关，只是 API 并发）
  * 真机评测并发由 EA_NUM_DEVICES 控制（默认 2，须 <= 实际芯片数）

用法
----
    source set_env/set_api_local.sh
    EA_MAX_WORKERS=24 python experiments/filter_online_exp.py --config filtered --kernels eye_kernel
    python experiments/filter_online_exp.py --config baseline
    python experiments/filter_online_exp.py --config fulleval --limit 3
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.harness import build_ea, reference_measurement  # noqa: E402
from experiments import common                                  # noqa: E402
from experiments.analyze_logs import K15                        # noqa: E402

OUT_DIR = ROOT / "experiments" / "results" / "filter_online"
LOGS_DIR = ROOT / "experiments" / "logs"

POOL = 20          # 每代生成的候选池规模
TOPK = 4           # 每代送 NPU 评测的个数
RUN_BASE = 900     # 独立 run_id，避免与既有实验冲突

PAPER_FULL_CONFIG = {
    "method": "paper_full",
    "population_size": 6,
    "elite_ratio": 2 / 6,
    "max_generations": 2,
    "gen0_candidates": 5,
    "children_per_generation": 4,
    "eval_budget": 13,
    "seed_diversity_threshold": 0.90,
    "selection": "ucb",
    "ucb_c": 0.5,
    "crossover_mode": "protected",
    "mutation_mode": "adaptive",
    "enable_profiling": True,
    "enable_rag": True,
    "enable_strategy_init": True,
    "enable_crossover": True,
    "enable_candidate_filter": False,
    "enable_candidate_dedup": False,
    "genetic_v2": False,
    "require_rag_index": True,
    "pre_compile_check": False,
    "compile_gate_pool": False,
    "tiered_remeasure": False,
    "collect_data": False,
    "require_manifest_baseline": True,
}

CONFIGS = {
    "baseline": dict(eval_budget=13),
    "predictor": dict(eval_budget=13, enable_candidate_filter=True,
                      candidate_pool_size=POOL, candidate_filter_mode="model"),
    "random": dict(eval_budget=13, enable_candidate_filter=True,
                   candidate_pool_size=POOL, candidate_filter_mode="random"),
    # 【预算说明】第 0 代已改为不启用预筛选（generate_five → 全评 5），
    # 因此 fulleval 的实际送评量 = Gen0 5 + Gen1 POOL(20) + Gen2 POOL(20) = 45，
    # 与下式恰好一致：13 + 2 × (POOL − TOPK) = 13 + 2 × 16 = 45。
    # 若将来把第 0 代也纳入扩池，此式需改为 3 × POOL，否则会被预算硬闸截断。
    "fulleval": dict(eval_budget=13 + 2 * (POOL - TOPK),
                     enable_candidate_filter=True,
                     candidate_pool_size=POOL, filter_evaluate_all=True),
    # V2 与 V1 的**同预算**对照（评审 v8.2 阶段 5）：NPU 预算完全相同，
    # 只改生成机制（指纹去重 + 有限重试补位 + 目标明确的变异/结构化交叉）。
    # 去重失败的重复候选仍保留在池里（不缩小池规模），只打 duplicate_in_pool 标记，
    # 因此两组的候选池规模与 NPU 调用数严格可比。
    "fulleval_v2": dict(eval_budget=13 + 2 * (POOL - TOPK),
                        enable_candidate_filter=True,
                        candidate_pool_size=POOL, filter_evaluate_all=True,
                        enable_candidate_dedup=True, genetic_v2=True),

    # ================= 论文正式版本定义（评审 v8.2 第五节）=================
    # 三个版本必须显式区分，不能都叫 FULL：
    #   FULL-V1    ：原始交叉/变异，去重关，Predictor 关  → 冻结为论文基线
    #   FULL-V2    ：V2 生成机制，去重开，Predictor 关    → 继续验证候选质量
    #   FULL-V2-P  ：V2 + Predictor                    → 预测模型证明有效后再验证
    # 编译门保持独立开关（pre_compile_check），未经验证不默认打开。
    "full_v1": dict(eval_budget=13, selection="tournament"),
    "full_v2": dict(eval_budget=13,
                    enable_candidate_dedup=True, genetic_v2=True,
                    selection="tournament"),
    "full_v2_p": dict(eval_budget=13,
                      enable_candidate_dedup=True, genetic_v2=True,
                      enable_candidate_filter=True,
                      candidate_pool_size=POOL,
                      candidate_filter_mode="model",
                      selection="tournament"),
    # New paper protocol; legacy full_v1/full_v2 records remain explicitly
    # tournament-based and are not reused as PAPER-FULL results.
    "paper_full": dict(PAPER_FULL_CONFIG),
    "paper_tournament": {
        **PAPER_FULL_CONFIG,
        "method": "paper_tournament",
        "selection": "tournament",
        "ucb_c": 0.5,
    },
    "paper_full_v2": {
        **PAPER_FULL_CONFIG,
        "method": "paper_full",
        "enable_candidate_dedup": True,
        "genetic_v2": True,
    },
    "paper_full_v2_tournament": {
        **PAPER_FULL_CONFIG,
        "method": "paper_tournament",
        "selection": "tournament",
        "enable_candidate_dedup": True,
        "genetic_v2": True,
    },
    "paper_b1": {
        **PAPER_FULL_CONFIG,
        "method": "paper_b1",
        "enable_profiling": False,
        "enable_rag": False,
        "enable_strategy_init": False,
        "require_rag_index": False,
    },
    "paper_b2": {
        **PAPER_FULL_CONFIG,
        "method": "b2",
        "selection": "tournament",
        "enable_profiling": False,
        "enable_rag": False,
        "enable_strategy_init": False,
        "enable_crossover": False,
        "mutation_mode": "uniform",
        "require_rag_index": False,
    },
    # 兼容旧命名
    "filtered": dict(eval_budget=13, enable_candidate_filter=True,
                     candidate_pool_size=POOL, candidate_filter_mode="model"),
}


def _read_model_meta(model_path):
    if not model_path:
        return None
    meta = Path(model_path).parent / (Path(model_path).stem + ".meta.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# 结果 JSON 必须齐备的字段；缺任一即视为"未完成"，恢复运行时重跑而不是直接复用。
REQUIRED_FIELDS = ("kernel", "config", "seed", "run_id", "success",
                   "t_best_us", "S", "audit")


def _load_existing(out_json, force=False, provenance=None):
    """恢复已有结果：必须非空、可解析、字段齐全**且版本指纹匹配**，否则重跑。

    【修改】旧实现只判断 `out_json.exists()`。若上一次运行被中断（kill / 超时 / OOM），
    留下的空文件或半截 JSON 会被当成"已完成"而跳过，导致该算子永久缺失，
    且表面上看不出问题。这里把校验收紧，并追加版本指纹比对：
    代码版本（fix_tag + git commit）、模型哈希、数据快照哈希、筛选配置
    任一不一致就判定为"旧版本结果"，备份后重跑。
    """
    if force:
        return None
    if not out_json.exists():
        return None
    try:
        if out_json.stat().st_size == 0:
            print(f"  [恢复] {out_json.name} 为空文件（上次运行被中断），重跑")
            return None
        rec = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [恢复] {out_json.name} 无法解析({type(e).__name__})，重跑")
        return None
    if not isinstance(rec, dict):
        print(f"  [恢复] {out_json.name} 不是 JSON 对象，重跑")
        return None
    missing = [k for k in REQUIRED_FIELDS if k not in rec]
    if missing:
        print(f"  [恢复] {out_json.name} 缺字段 {missing}，重跑")
        return None
    # 【修复】评审指出：旧实现只检查 success 字段**存在**，值是 False 也会被复用。
    # 必须核查其值确实为 True，并且硬性验收也要通过。
    if rec.get("success") is not True:
        print(f"  [恢复] {out_json.name} 记录的 success={rec.get('success')!r}"
              f"（非 True），重跑")
        return None
    _acc = rec.get("acceptance")
    if not isinstance(_acc, dict) or _acc.get("passed") is not True:
        print(f"  [恢复] {out_json.name} 缺少通过的硬性验收记录，重跑")
        return None
    try:
        score = float(rec["S"])
        if not math.isfinite(score) or score <= 0:
            raise ValueError
    except (TypeError, ValueError):
        print(f"  [恢复] {out_json.name} 的 S 不是有限正数，重跑")
        return None
    audit = rec.get("audit")
    if (not isinstance(audit, dict)
            or audit.get("budget_ok") is not True
            or "npu_calls" not in audit):
        print(f"  [恢复] {out_json.name} audit 不完整或预算未通过，重跑")
        return None
    if provenance:
        bad = _provenance_mismatch(rec.get("provenance"), provenance)
        if bad:
            print(f"  [恢复] {out_json.name} 版本不匹配，重跑：")
            for b in bad[:6]:
                print(f"           - {b}")
            try:
                bak = out_json.with_name(out_json.name + ".old")
                if bak.exists():
                    bak.unlink()
                out_json.rename(bak)
                print(f"  [恢复] 旧结果已备份为 {bak.name}")
            except OSError:
                pass
            return None
    return rec


def _write_json_atomic(path, rec):
    """先写临时文件再原子替换，避免中断时留下空/半截结果文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _check_model_isolation(model_path, kernel):
    """【修改】强制校验：测试算子不得出现在训练名单中。

    旧实现的漏洞：`if meta and meta.get("train_kernels") not in (None, "ALL")` 才检查，
    而训练脚本一直把 train_kernels 写成 "ALL"（另用 exclude_kernels 表示排除集），
    于是防泄漏检查**从未真正生效**——这次正确排除了测试算子，靠的是训练脚本本身，
    不是这道检查。这里改为：
      * 必须显式提供模型路径；
      * 必须有可读的 *.meta.json；
      * train_kernels 必须是非空显式名单（不接受 "ALL"）；
      * 测试算子不得在名单内；
    任一不满足直接抛错拒绝运行，而不是静默继续。
    """
    if not model_path:
        raise RuntimeError(
            f"❌ 未提供预筛选模型路径，拒绝运行 {kernel}："
            f"无法验证模型是否见过该算子的数据（请传 --model-path）")
    if not Path(model_path).is_file():
        raise RuntimeError(
            f"❌ 模型文件不存在：{model_path}，拒绝运行 {kernel}")
    meta_path = Path(model_path).with_name(Path(model_path).stem + ".meta.json")
    if not meta_path.exists():
        raise RuntimeError(
            f"❌ 缺少模型元信息 {meta_path}，拒绝运行 {kernel}："
            f"无法验证训练/测试划分")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"❌ 模型元信息无法解析 {meta_path}: {e}")

    train = meta.get("train_kernels")
    if train is None or train == "ALL" or not isinstance(train, list) or not train:
        raise RuntimeError(
            f"❌ {meta_path.name} 的 train_kernels 不是显式名单（={train!r}），"
            f"拒绝运行 {kernel}。请用新版训练脚本重训以写入真实训练算子名单。")
    if kernel in set(train):
        raise RuntimeError(
            f"❌ 完整性自检失败：测试算子 {kernel} 出现在训练集（数据泄漏！）")
    return meta


# 本次修复代的标识。任何会改变结果语义的代码改动都应递增它，
# 使"修复前"的旧结果在恢复时自动失效，不会被当成新实验复用。
FIX_TAG = "20261019-paper-freeze-v1"


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha256_file(p):
    # 路径为空 / 文件不存在时返回 None，不能抛异常——
    # _build_provenance 会在快照未解析出来时以 None 调用它。
    if not p:
        return None
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_environment():
    packages = {}
    for package in (
            "torch", "torch-npu", "triton", "langchain-openai",
            "transformers", "faiss-cpu", "sentence-transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "npu_devices_configured": os.getenv("EA_NUM_DEVICES"),
        "llm_workers_configured": os.getenv("EA_MAX_WORKERS"),
    }


DATASET_DIR = ROOT / "experiments" / "dataset"


def _resolve_dataset_path(ds):
    """把 meta 里的 dataset_snapshot 解析成实际文件路径。

    【修改】meta 里存的可能是相对文件名（如 "samples_frozen_20261016.jsonl"），
    直接 `_sha256_file(ds)` 会因找不到文件而返回 None，指纹里的数据哈希就永远缺失，
    版本核对形同虚设。这里按优先级尝试：
        原样 → DATASET_DIR/<basename> → DATASET_DIR/<原路径>
    """
    if not ds:
        return None
    p = Path(ds)
    for cand in (p, DATASET_DIR / p.name, DATASET_DIR / p):
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _rag_index_sha256(required=False):
    index_dir = ROOT / "RAG" / "knowledge"
    files = [index_dir / "index.faiss", index_dir / "index.pkl"]
    missing = [p.name for p in files if not p.is_file()]
    if required and missing:
        raise RuntimeError(
            f"PAPER-FULL 要求固定 RAG 索引；{index_dir} 缺少 {', '.join(missing)}。"
            "请先构建/安装知识库索引，再启动真机验收。")
    if missing:
        return None
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


_FINGERPRINT_FILES = (
    "config.py",
    "main.py",
    "evolutionary_algorithm.py",
    "genetic_operators.py",
    "executor.py",
    "RAG/knowledge_node.py",
    "experiments/common.py",
    "experiments/harness.py",
    "experiments/filter_online_exp.py",
    "experiments/paper_full_config.json",
    "experiments/run_paper_acceptance.sh",
    "experiments/tests/test_parent_latency.py",
    "experiments/tests/test_filter_budget.py",
    "experiments/tests/test_paper_protocol.py",
    "baseline/baseline.json",
)


def _build_provenance(cfg_name, cfg, model_path, kernel=None):
    """本次运行的版本指纹：代码版本 / 模型版本 / 数据版本 / 筛选配置。

    【修改】恢复实验时逐项比对，全部匹配才允许复用。否则会把"修复前"的
    旧结果当成新实验的结果，是比跑失败更危险的静默错误。
    """
    meta_path = Path(model_path).with_name(
        Path(model_path).stem + ".meta.json") if model_path else None
    meta = {}
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    ds_path = _resolve_dataset_path(meta.get("dataset_snapshot"))
    ds_hash = _sha256_file(ds_path)
    # 实际哈希与 meta 记录是否一致：不一致说明模型不是用这份数据训的
    ds_declared = meta.get("dataset_sha256")
    ds_ok = bool(ds_hash and ds_declared and ds_hash == ds_declared)
    rag_hash = _rag_index_sha256(required=cfg.get("require_rag_index", False))
    source_hashes = {
        rel: _sha256_file(ROOT / rel)
        for rel in _FINGERPRINT_FILES
    }
    reference = None
    input_hashes = {}
    if kernel:
        reference = reference_measurement(
            kernel,
            "./baseline/baseline.json",
            require_manifest=cfg.get("require_manifest_baseline", False))
        input_paths = [*common.seed_code_paths(kernel), common.find_test_file(kernel)]
        input_hashes = {
            str(path.resolve()): _sha256_file(path)
            for path in input_paths
        }
    if ds_path and ds_declared and not ds_ok:
        print(f"  ⚠️ 数据快照哈希与模型 meta 记录不一致：{ds_path.name} "
              f"实际 {str(ds_hash)[:12]} vs 记录 {str(ds_declared)[:12]}")
    return {
        "fix_tag": FIX_TAG,
        "git_commit": _git_commit(),
        "config": cfg_name,
        # 关键实验参数（影响结果语义的部分）
        "params": {k: cfg.get(k) for k in sorted(cfg)},
        "model_path": str(model_path) if model_path else None,
        "model_sha256": _sha256_file(model_path) if model_path else None,
        # 存解析后的绝对路径 + 实际算出的哈希，便于恢复时直接核对
        "dataset_snapshot": str(ds_path) if ds_path else None,
        "dataset_sha256": ds_hash,
        "dataset_sha256_matches_meta": ds_ok,
        "rag_index_sha256": rag_hash,
        "source_sha256": source_hashes,
        "reference_baseline_us": reference["baseline_us"] if reference else None,
        "reference_baseline_source": reference["source"] if reference else None,
        "reference_manifest_path": reference["manifest_path"] if reference else None,
        "reference_manifest_sha256": reference["manifest_sha256"] if reference else None,
        "baseline_json_sha256": reference.get("baseline_json_sha256") if reference else None,
        "experiment_input_sha256": input_hashes,
        "runtime_environment": _runtime_environment(),
    }


def _provenance_mismatch(old, new):
    """返回不一致的字段列表；空列表表示可复用。"""
    if not isinstance(old, dict):
        return ["provenance（旧结果没有版本信息）"]
    bad = []
    for k in ("fix_tag", "git_commit", "config", "model_path",
              "model_sha256", "dataset_snapshot", "dataset_sha256",
              "dataset_sha256_matches_meta", "rag_index_sha256",
              "source_sha256", "reference_baseline_us",
              "reference_baseline_source", "reference_manifest_path",
              "reference_manifest_sha256", "baseline_json_sha256",
              "experiment_input_sha256", "runtime_environment"):
        if old.get(k) != new.get(k):
            bad.append(f"{k}: {old.get(k)!r} → {new.get(k)!r}")
    op = old.get("params") or {}
    np_ = new.get("params") or {}
    for k in sorted(set(op) | set(np_)):
        if op.get(k) != np_.get(k):
            bad.append(f"params.{k}: {op.get(k)!r} → {np_.get(k)!r}")
    return bad


def _sum_llm_tokens(log_path):
    """从运行日志 JSONL 累计 LLM token 消耗（字段缺失时返回 None）。"""
    p = Path(log_path)
    if not p.exists():
        return None
    pt = ct = 0
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            pt += int(o.get("prompt_tokens", 0) or 0)
            ct += int(o.get("completion_tokens", 0) or 0)
    except Exception:
        return None
    return (pt + ct) if (pt or ct) else None


def run_one(kernel, cfg_name, seed, model_path=None, force=False):
    cfg = dict(CONFIGS[cfg_name])
    ea_method = cfg.pop("method", "full")
    run_id = RUN_BASE + seed
    out_dir = OUT_DIR / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{kernel}__r{seed}.json"

    # ---- 步骤 0（【修改】前移到最前面）：模型隔离校验 ----
    # 旧顺序是「先复用旧结果 / 先跑完整次 NPU 搜索 → 最后才检查模型是否泄漏」，
    # 泄漏的模型可能已经污染了一整轮真机评测。改为按配置字段判断是否需要模型，
    # 而不是硬编码配置名（否则 phase2 的 predictor_foldN 会被漏掉）。
    meta = None
    if cfg.get("enable_candidate_filter", False):
        meta = _check_model_isolation(model_path, kernel)

    # ---- 步骤 1：构造版本指纹，恢复旧结果时逐项比对 ----
    prov = _build_provenance(cfg_name, cfg, model_path, kernel=kernel)
    reused = _load_existing(out_json, force=force, provenance=prov)
    if reused is not None:
        print(f"  [复用] 已有结果 {out_json.name}")
        return reused

    log_path = LOGS_DIR / f"filter_online__{cfg_name}__{kernel}__r{run_id}.jsonl"
    score_path = LOGS_DIR / f"filter_scores__{cfg_name}__{kernel}__r{run_id}.jsonl"
    for p in (log_path, score_path):
        if p.exists():
            p.unlink()

    # 显式传入模型路径与筛选模式，避免误用默认模型
    overrides = dict(cfg)
    if model_path:
        overrides["filter_model_path"] = model_path
    overrides.setdefault("candidate_filter_mode", "model")

    ea, config = build_ea(
        kernel, ea_method, run_id=run_id, seed=seed, log=True,
        log_path=str(log_path), filter_log_path=str(score_path), **overrides)
    if (prov.get("reference_baseline_us") is not None
            and config.reference_baseline_us != prov["reference_baseline_us"]):
        raise RuntimeError(
            "搜索执行器参考基线与结果 provenance 不一致："
            f"{config.reference_baseline_us} != {prov['reference_baseline_us']}")

    seed_codes = [common.read_code(p) for p in common.seed_code_paths(kernel)]
    t0 = time.time()
    best = ea.run(seed_codes)
    elapsed = time.time() - t0

    rec = {"kernel": kernel, "config": cfg_name, "seed": seed,
           "run_id": run_id, "seconds": round(elapsed, 1)}

    # ---- 记录模型版本与训练名单（隔离校验已在步骤 0 完成） ----
    if meta is not None:
        rec["train_kernels"] = sorted(set(meta["train_kernels"]))
        rec["exclude_kernels"] = meta.get("exclude_kernels", [])
        rec["train_dataset_snapshot"] = meta.get("dataset_snapshot")
        rec["train_n_samples"] = meta.get("n_samples")
    rec["filter_model"] = (str(model_path) if model_path
                           else ea.stats.get("filter_model_path"))
    rec["candidate_filter_mode"] = cfg.get("candidate_filter_mode", "model")
    # 版本指纹：恢复实验时逐项比对，防止复用修复前的旧结果
    rec["provenance"] = prov
    rec["rag_index_sha256"] = prov.get("rag_index_sha256")
    rec["reference_baseline_us"] = prov.get("reference_baseline_us")
    rec["reference_baseline_source"] = prov.get("reference_baseline_source")
    rec["reference_manifest_sha256"] = prov.get("reference_manifest_sha256")
    llm_history = getattr(ea.genetic_ops.llm, "call_history", [])
    rec["llm_model_ids"] = sorted({
        str(call["model"]) for call in llm_history
        if isinstance(call, dict) and call.get("model")
    })

    # 最终最优候选用与 3.1 一致的重复测量口径复测（5 次取中位数，不计入搜索预算）
    if best and best.code:
        res = ea.executor.measure_repeat(best.code, repeats=5, device_id=0)
        (out_dir / f"{kernel}__r{seed}.py").write_text(best.code, encoding="utf-8")
        rec.update({"success": bool(res.success), "t_best_us": res.execution_time,
                    "error": (res.error or "")[:200]})
        reference_baseline = prov.get("reference_baseline_us")
        if reference_baseline and res.execution_time:
            rec["S"] = reference_baseline / res.execution_time
    else:
        rec.update({"success": False, "t_best_us": None,
                    "error": "no_best_candidate（搜索没有产出可用候选）"})

    # ---- 审计：预算 / 拒绝 / 失败 / Token / 墙钟 ----
    _tok = _sum_llm_tokens(log_path)
    _npu = ea.stats.get("npu_calls")
    _budget = dict(CONFIGS[cfg_name]).get("eval_budget")
    rec["audit"] = {
        "npu_calls": _npu,
        "eval_budget": _budget,
        "budget_ok": bool(_npu is not None and _budget is not None
                          and _npu <= _budget),
        "static_rejected": ea.stats.get("static_rejected"),
        "ast_rejected": ea.stats.get("ast_rejected"),
        "compile_failed": ea.stats.get("compile_failed"),
        "llm_tokens": _tok,
        # 【修改】取不到 Token 时显式标记为缺失，而不是让它静默变成 0。
        # 比较"扩大候选池"的额外生成成本需要真实 Token；缺失必须可见、可审计。
        "llm_tokens_missing": (_tok is None),
        "remeasure_repeats": 5,
        "wallclock_s": round(elapsed, 1),
    }
    # ---- 硬性验收（评审 v8.2）：失败结果不得被当成"一次成功完成的搜索" ----
    # 旧实现只在抛异常时记失败，而最终复测失败仍会正常保存并返回，
    # 于是失败的搜索会被计入统计。这里做四项硬校验：
    problems = []
    if not rec.get("success") is True:
        problems.append(f"最终复测未成功（success={rec.get('success')!r}）"
                        + (f"，error={rec.get('error')!r}" if rec.get("error") else ""))
    _tb = rec.get("t_best_us")
    if not (isinstance(_tb, (int, float)) and math.isfinite(_tb) and _tb > 0):
        problems.append(f"t_best_us 不是有限正数（{_tb!r}）")
    _score = rec.get("S")
    if not (isinstance(_score, (int, float)) and math.isfinite(_score)
            and _score > 0):
        problems.append(f"S 不是有限正数（{_score!r}）")
    _miss = [k for k in REQUIRED_FIELDS if rec.get(k) is None and k != "t_best_us"]
    if _miss:
        problems.append(f"结果字段缺失/为空 {_miss}")
    if not (rec.get("audit") or {}).get("budget_ok", False):
        problems.append(f"预算超支或无法确认：npu_calls="
                        f"{(rec.get('audit') or {}).get('npu_calls')} / "
                        f"budget={(rec.get('audit') or {}).get('eval_budget')}")
    rec["acceptance"] = {"passed": not problems, "problems": problems}
    if problems:
        rec["success"] = False
        print(f"  ❌ 硬性验收不通过：")
        for p in problems:
            print(f"       - {p}")

    # 【修改】原子写：先写 .tmp 再 os.replace，避免中断留下空/半截结果文件。
    _write_json_atomic(out_json, rec)
    if problems:
        # 仍然把失败记录写盘（便于事后查），但让调用方知道这次搜索失败
        raise RuntimeError("acceptance_failed: " + " | ".join(problems))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="配置名，含 baseline / filtered / predictor / random / fulleval、"
                         "full_v1/full_v2 与 paper_full/paper_tournament/paper_full_v2/paper_b1/paper_b2")
    ap.add_argument("--kernels", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-path", default=None,
                    help="预筛选模型路径。predictor / random / fulleval 必需"
                         "（启动前做防泄漏强校验，缺失或名单不完整直接拒绝运行）；"
                         "random 用它记录模型分做 Oracle 对照")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的结果文件重新运行")
    a = ap.parse_args()
    if a.config not in CONFIGS:
        raise SystemExit(f"❌ 未知配置 {a.config}；可用：{sorted(CONFIGS)}")
    if CONFIGS[a.config].get("enable_candidate_filter", False) and not a.model_path:
        raise SystemExit(f"❌ 配置 {a.config} 启用了候选预筛选，必须传 --model-path"
                         f"（启动前要做防泄漏校验）")

    kernels = a.kernels or sorted(K15)
    if a.limit:
        kernels = kernels[:a.limit]

    workers = int(os.getenv("EA_MAX_WORKERS", "8"))
    ndev = int(os.getenv("EA_NUM_DEVICES", "2"))
    print(f"[配置] {a.config}  候选池 {POOL if a.config != 'baseline' else 4}  "
          f"LLM并发 {workers}  NPU卡数 {ndev}  算子 {len(kernels)}  seed {list(a.seeds)}")
    if workers > 30:
        print("⚠ EA_MAX_WORKERS(LLM生成并发) > 30，请调低以免触发 API 限流")

    t0 = time.time()
    failures = []          # [(config, seed, kernel, 错误)]
    for si, seed in enumerate(a.seeds, 1):
        for ki, kernel in enumerate(kernels, 1):
            print(f"\n=== [{a.config}] seed {si}/{len(a.seeds)} "
                  f"算子 {ki}/{len(kernels)}: {kernel} ===")
            try:
                rec = run_one(kernel, a.config, seed,
                              model_path=a.model_path, force=a.force)
                print(f"  完成: {rec.get('t_best_us')} us  "
                      f"S={rec.get('S')}  用时 {rec['seconds']}s")
            except Exception as e:
                failures.append((a.config, seed, kernel, f"{type(e).__name__}: {e}"))
                print(f"  [错误] {kernel}: {type(e).__name__}: {e}")

    # 【修复】评审指出：单个算子失败只打印错误就继续，最后仍显示"全部完成"，
    # 正式运行可能悄悄少跑几个算子。这里汇总失败任务并以**非零状态**退出。
    print(f"\n{'=' * 60}")
    print(f"任务汇总：成功 {len(kernels) * len(a.seeds) - len(failures)} / "
          f"{len(kernels) * len(a.seeds)}，失败 {len(failures)}")
    for cfg, seed, kernel, err in failures:
        print(f"  ❌ {cfg} seed={seed} {kernel}: {err}")
    print(f"用时 {(time.time()-t0)/3600:.2f} 小时")
    print("=" * 60)
    if failures:
        print("❌ 存在失败任务，以非零状态退出（正式实验不允许静默跳过）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
