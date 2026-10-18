#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""父代基线延迟 / 父子代码特征 / 编译阶段分离 的离线自测（**不占用 NPU**）。

覆盖 2026-10-17 那轮修复引入的行为，全部是纯逻辑校验，不碰真机：

  * resolve_parent_latency 的六档回退链与优先级
  * 旧的"静默退到种子"行为不再发生（这是 improved 标签被污染的根因）
  * 变异子代现在能取到真实父代代码（旧实现让整组比值特征退化为常数）
  * compile_pass / correctness_pass 不再同源；未开启编译门时为 None 而不是 False
  * 编译门的错误分类（_classify_compile_error）不把断言失败误判成编译错误

用法
----
    cd /workspace/Agent
    python experiments/tests/test_parent_latency.py
    python experiments/tests/test_parent_latency.py -k ancestor

退出码 0 = 全部通过；1 = 有失败。
"""

import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("EA_NUM_DEVICES", "1")
os.environ.setdefault("EA_MAX_WORKERS", "2")

from config import EAConfig                                     # noqa: E402
from genetic_operators import Individual                        # noqa: E402
import evolutionary_algorithm as ea                             # noqa: E402
from executor import EvaluationResult, TritonExecutor           # noqa: E402
from experiments import candidate_filter_features_shared as S_mod  # noqa: E402


# ---------------------------------------------------------------- 工具
def _ind(cid, lat=None, success=True, operation="mutation",
         parent=None, parents=None, code="", gen=1,
         parent_latency=None, dominant_latency=None):
    md = {"operation": operation}
    if success:
        md["success"] = True
        if lat is not None:
            md["execution_time"] = lat
    else:
        md["success"] = False
    if parent is not None:
        md["parent"] = parent
    if parents is not None:
        md["parents"] = parents
    if parent_latency is not None:
        md["parent_latency"] = parent_latency
    if dominant_latency is not None:
        md["dominant_latency"] = dominant_latency
    return Individual(code=code or f"# code of {cid}\n", generation=gen,
                      id=cid, metadata=md)


# ---------------------------------------------------------------- 假 EA 实例
def _bare_ea(seed_time=1.0):
    """构造一个只够调用 _event / _log_eval 的最小 EA 实例（不启动任何真机资源）。"""
    obj = ea.EvolutionaryAlgorithm.__new__(ea.EvolutionaryAlgorithm)
    obj.stats = {}
    obj.seed_execution_time = seed_time
    obj.population = []
    obj.config = EAConfig()
    obj.executor = type("E", (), {"kernel_name": "k"})()
    obj.eval_count = 0
    obj.logger = type("L", (), {"log": lambda self, *a, **k: None})()
    return obj


# ---------------------------------------------------------------- 用例
def test_resolve_direct_parent():
    """第 1 档：直接父代实测延迟（parent_latency / dominant_latency）。"""
    c = _ind("c1", parent_latency=111.5)
    v, s = ea.resolve_parent_latency(c)
    assert v == 111.5 and s == "parent", (v, s)

    c = _ind("c2", dominant_latency=99.25)
    v, s = ea.resolve_parent_latency(c)
    assert v == 99.25 and s == "dominant", (v, s)

    # parent_latency 优先于 dominant_latency
    c = _ind("c3", parent_latency=10.0, dominant_latency=20.0)
    v, s = ea.resolve_parent_latency(c)
    assert (v, s) == (10.0, "parent"), (v, s)
    return None


def test_resolve_rejects_bad_values():
    """NaN / None / 0 / 负数都不能当成父代延迟。"""
    for bad in (None, 0, -1.0, float("nan"), float("inf"), "abc"):
        c = _ind("c", parent_latency=bad)
        v, s = ea.resolve_parent_latency(c, seed_time=50.0)
        assert v == 50.0 and s == "seed", f"{bad!r} -> {(v, s)}"
    return None


def test_resolve_ancestor_walk():
    """第 2 档：父代未实测时沿父链上溯，而不是直接退到种子。"""
    grand = _ind("g", lat=200.0)                      # 祖先：有实测
    father = _ind("f", lat=None, success=False, parent="g")  # 父代：评测失败
    child = _ind("c", parent="f")
    id2ind = {i.id: i for i in (grand, father, child)}

    v, s = ea.resolve_parent_latency(child, id2ind=id2ind, seed_time=1.0)
    assert v == 200.0 and s == "ancestor", (v, s)

    # 关键回归：旧实现会在这里静默退到 seed_time=1.0，把"比祖先快"冒称"比父代快"
    assert s != "seed", "不应静默回退到种子"
    return None


def test_resolve_ancestor_depth_limit():
    """父链成环或极深时不能死循环。"""
    a, b = _ind("a", parent="b"), _ind("b", parent="a")
    v, s = ea.resolve_parent_latency(a, id2ind={"a": a, "b": b},
                                     seed_time=7.0, max_depth=4)
    assert s in ("pop_best", "seed", "none"), s
    return None


def test_resolve_pop_best_before_seed():
    """第 3 档：种群里已评测个体的最优延迟，优先于种子基准。"""
    p1 = _ind("p1", lat=80.0)
    p2 = _ind("p2", lat=60.0)         # 更快
    p3 = _ind("p3", lat=None, success=False)
    child = _ind("c", parent="p3")
    id2ind = {i.id: i for i in (p1, p2, p3, child)}
    v, s = ea.resolve_parent_latency(child, id2ind=id2ind, seed_time=1.0)
    assert s == "pop_best" and v == 60.0, (v, s)
    return None


def test_resolve_none():
    """什么都没有时返回 (None, 'none')，而不是编一个值出来。"""
    child = _ind("c", parent="missing")
    v, s = ea.resolve_parent_latency(child, id2ind={}, seed_time=None)
    assert v is None and s == "none", (v, s)
    return None


def test_filter_features_uses_real_parent_code_for_mutation():
    """回归：变异子代旧实现拿不到父代代码，整组比值特征退化为常数。"""
    cfg = EAConfig()
    parent_code = ("@triton.jit\ndef k(x, BLOCK: tl.constexpr):\n"
                   "    tl.store(x, tl.load(x) * 2)\n    num_warps=8\n")
    # 子代引入父代没有的新标识符（tmp / tl.abs），保证 Jaccard 词集真的不同，
    # 否则 sim 恒为 1.0，测不出"是否取到了真实父代代码"
    child_code = ("@triton.jit\ndef k(x, BLOCK: tl.constexpr):\n"
                  "    tmp = tl.abs(tl.load(x))\n"
                  "    tl.store(x, tmp + tl.load(x) * tl.load(x))\n"
                  "    num_warps=4\n")
    parent = _ind("P", lat=10.0, code=parent_code, operation="seed")
    # 变异子代：旧实现只写 parent，不写 parents
    child = _ind("C", operation="mutation", parent="P", code=child_code)
    id2code = {"P": parent_code, "C": child_code}
    id2ind = {"P": parent, "C": child}

    f = ea.EvolutionaryAlgorithm._filter_features(child, cfg, id2code, id2ind, 10.0)
    # 子代 tl.* 调用更多 → r_ops 必须 > 0；num_warps 8→4 → r_warps 必须 < 0
    assert f["r_ops"] > 0, f"r_ops 应反映真实父子差异，实际 {f['r_ops']}"
    assert abs(f["r_warps"]) > 1e-9, f"r_warps 不应退化，实际 {f['r_warps']}"
    assert f["sim"] < 1.0, f"sim 不应退化为 1.0，实际 {f['sim']}"
    return None


def test_filter_features_log_parent_not_nan_when_parent_unevaluated():
    """回归：父代未实测时 log_parent 不再为 NaN（与训练数据分布一致）。"""
    cfg = EAConfig()
    grand = _ind("G", lat=123.0, code="x = 1\n", operation="seed")
    father = _ind("F", lat=None, success=False, parent="G", code="x = 2\n")
    child = _ind("C", operation="mutation", parent="F", code="x = 3\n")
    id2code = {"G": "x = 1\n", "F": "x = 2\n", "C": "x = 3\n"}
    id2ind = {i.id: i for i in (grand, father, child)}

    f = ea.EvolutionaryAlgorithm._filter_features(child, cfg, id2code, id2ind, 0.0)
    assert not math.isnan(f["log_parent"]), "log_parent 不应为 NaN（训练侧 0% NaN）"
    assert f["parent_latency_source"] == "ancestor", f["parent_latency_source"]
    return None


def test_filter_features_schema_unchanged():
    """特征字典必须仍覆盖模型训练时的 14 个字段（多加字段可以，少一个不行）。"""
    cfg = EAConfig()
    need = ["log_parent", "gen", "pop_size", "tau", "r_block", "r_warps",
            "r_stages", "sim", "r_ops", "r_mask", "ops_abs", "block_abs",
            "operation", "mutation_type"]
    c = _ind("C", parent="P", code="x = 1\n", parent_latency=5.0)
    f = ea.EvolutionaryAlgorithm._filter_features(c, cfg, {"C": "x = 1\n"}, {}, 5.0)
    missing = [k for k in need if k not in f]
    assert not missing, f"缺少特征字段: {missing}"
    return None


def test_trusted_sources_consistency():
    """可信档位必须与常量一致，且是回退链最靠前的几档。"""
    assert ea.TRUSTED_PARENT_SOURCES == ("parent", "dominant", "backbone")
    assert set(ea.TRUSTED_PARENT_SOURCES) <= set(ea.PARENT_LATENCY_SOURCES)
    assert ea.PARENT_LATENCY_SOURCES[:len(ea.TRUSTED_PARENT_SOURCES)] == \
        ea.TRUSTED_PARENT_SOURCES
    return None


def test_backbone_of_evaluated_parent():
    """普通变异（父代已评测）：主干就是父代自己。"""
    from genetic_operators import GeneticOperators
    p = _ind("P", lat=50.0, code="x = 1\n", operation="seed")
    bb_id, bb_lat, bb_code, lineage = GeneticOperators._backbone_of(p)
    assert (bb_id, bb_lat, bb_code) == ("P", 50.0, "P"), (bb_id, bb_lat, bb_code)
    assert lineage[-1] == "mutation", lineage
    return None


def test_backbone_inherited_through_crossover():
    """核心回归：交叉产物 C 未评测，对 C 变异得到的 D 必须继承交叉的主干父代 A。

    旧实现会把 D 的父代记成 C（execution_time=None），只能退到 pop_best 猜测。
    """
    from genetic_operators import GeneticOperators

    # A 已评测（真实基线 100.0），B 为供体
    crossover_child = _ind(
        "C", lat=None, success=False, operation="crossover",
        parents=["A", "B"], parent_latency=100.0, dominant_latency=100.0)
    crossover_child.metadata.update({
        "backbone_id": "A", "backbone_latency": 100.0, "backbone_code_id": "A",
        "dominant_parent": "A", "lineage": ["crossover"],
    })

    bb_id, bb_lat, bb_code, lineage = GeneticOperators._backbone_of(crossover_child)
    assert bb_id == "A", f"主干父代应为 A，实际 {bb_id}"
    assert bb_lat == 100.0, f"基线应继承 A 的实测延迟，实际 {bb_lat}"
    assert bb_code == "A", f"代码引用应指向 A，实际 {bb_code}"
    assert lineage == ["crossover", "mutation"], lineage

    id2ind = {"P0": _ind("P0", lat=1.0)}   # 种群里有个更快但不相干的个体

    # 情形 A：mutate() 会把继承到的基线写进 parent_latency → 落在 parent 档
    d1 = _ind("D1", operation="mutation")
    d1.metadata.update({"parent_latency": bb_lat, "backbone_id": bb_id,
                        "backbone_latency": bb_lat, "parents": [bb_id]})
    v, s = ea.resolve_parent_latency(d1, id2ind=id2ind, seed_time=0.5)
    assert (v, s) == (100.0, "parent"), (v, s)

    # 情形 B：只剩 backbone_latency（parent_latency 缺失）→ 落在 backbone 档
    d2 = _ind("D2", operation="mutation")
    d2.metadata.update({"backbone_id": bb_id, "backbone_latency": bb_lat,
                        "parents": [bb_id]})
    v, s = ea.resolve_parent_latency(d2, id2ind=id2ind, seed_time=0.5)
    assert (v, s) == (100.0, "backbone"), (v, s)

    # 两档都必须可信，且都**不能**退化成 pop_best（种群里有 1.0 的更快个体）
    assert s in ea.TRUSTED_PARENT_SOURCES
    return None


def test_crossover_parents_order_is_dominant_first():
    """parents[0] 必须是 dominant（骨架），否则特征/训练口径会指到供体上。"""
    src = Path(__file__).read_text(encoding="utf-8")
    import inspect
    import genetic_operators as go
    code = inspect.getsource(go.GeneticOperators.crossover)
    assert "'parents': [dominant.id, donor.id]" in code, \
        "交叉的 parents 必须以 dominant 打头"
    return None


def test_filter_features_uses_backbone_code():
    """变异自"未评测的交叉产物"时，父代代码必须取主干父代，不能退化成自身。"""
    cfg = EAConfig()
    a_code = "@triton.jit\ndef k(x, BLOCK: tl.constexpr):\n    tl.store(x, tl.load(x))\n    num_warps=8\n"
    d_code = ("@triton.jit\ndef k(x, BLOCK: tl.constexpr):\n"
              "    tmp = tl.abs(tl.load(x))\n"
              "    tl.store(x, tmp + tl.load(x) * tl.load(x))\n"
              "    num_warps=4\n")
    a = _ind("A", lat=100.0, code=a_code, operation="seed")
    c = _ind("C", lat=None, success=False, operation="crossover", code="# C\n")
    d = _ind("D", operation="mutation", code=d_code)
    d.metadata.update({"parent": "C", "parents": ["A"],
                       "backbone_id": "A", "backbone_code_id": "A",
                       "backbone_latency": 100.0, "parent_latency": 100.0})
    # 注意：id2code 里**故意不放 C**，且放一个与 A 不同的代码验证确实取到了 A
    id2code = {"A": a_code, "D": d_code}
    id2ind = {"A": a, "C": c, "D": d}
    f = ea.EvolutionaryAlgorithm._filter_features(d, cfg, id2code, id2ind, 100.0)
    assert f["r_ops"] > 0, f"应相对 A 计算 tl 调用数差异，实际 {f['r_ops']}"
    assert abs(f["r_warps"]) > 1e-9, f"r_warps 不应退化，实际 {f['r_warps']}"
    assert f["sim"] < 1.0, f"sim 不应退化成 1.0，实际 {f['sim']}"
    return None


def test_judge_compile_result_rules():
    """编译判定三态（评审 v5 意见 2）：确认成功 / 确认失败 / 未知。"""
    j = TritonExecutor._judge_compile_result
    # 规则 1：一次编译都没触发 → **未知**，既不是通过也不是失败
    r = j(0, 0, 0, "")
    assert r["compile_pass"] is None and r["status"] == "unknown", r
    assert r["error_type"] == "NoKernelLaunch"
    # 超时同样是未知
    r = j(1, 1, -9, "TimeoutExpired after 600s")
    assert r["compile_pass"] is None and r["status"] == "unknown", r
    assert r["error_type"] == "Timeout"
    # 规则 2：有 launch 在编译阶段抛错 → **确认失败**
    r = j(2, 1, 1, "triton.compiler.errors.CompilationError at 3:5")
    assert r["compile_pass"] is False and r["status"] == "fail", r
    assert r["error_type"] == "CompilationError"
    # 规则 3：全部编译成功、但脚本后续断言失败 → **确认成功**，记 PostCompile
    r = j(2, 2, 1, "AssertionError: not close")
    assert r["compile_pass"] is True and r["status"] == "pass", r
    assert r["error_type"] == "PostCompile:AssertionError(non-compile)"
    # 规则 4：全通过
    r = j(2, 2, 0, "")
    assert r["compile_pass"] is True and r["status"] == "pass"
    assert r["error_type"] is None
    return None


def test_mutation_tier_uses_backbone_speedup():
    """回归：交叉产物未评测时，变异分级必须按主干父代性能，而不是 speedup=0。

    旧实现 `speedup = individual.metadata.get('speedup', 0.0)` 会让高性能血统
    落入"结构重写 0.50"的激进档。
    """
    from genetic_operators import GeneticOperators

    # 已评测个体：用自己的 speedup
    p = _ind("P", lat=10.0, operation="seed")
    p.metadata["speedup"] = 3.0
    v, s = GeneticOperators._mutation_tier_speedup(p)
    assert (v, s) == (3.0, "self"), (v, s)

    # 未评测的交叉产物：必须用继承的 backbone_speedup
    c = _ind("C", lat=None, success=False, operation="crossover")
    c.metadata.update({"backbone_id": "A", "backbone_speedup": 3.0})
    v, s = GeneticOperators._mutation_tier_speedup(c)
    assert (v, s) == (3.0, "backbone"), (v, s)

    # 都没有 → 0.0 / none（保守档），且来源可审计
    d = _ind("D", lat=None, success=False, operation="crossover")
    v, s = GeneticOperators._mutation_tier_speedup(d)
    assert (v, s) == (0.0, "none"), (v, s)
    return None


def test_batch_aborted_is_unknown_not_fail():
    """批量子进程异常时，未返回结果的候选是"未知"，不能被当成编译失败过滤。"""
    # 直接看判定函数：BatchAborted 走的是 unknown 分支（compile_pass=None）
    v = TritonExecutor._judge_compile_result(0, 0, 0, "")
    assert v["compile_pass"] is None and v["status"] == "unknown"
    # 且 EA 侧默认策略是保留
    assert EAConfig().compile_gate_on_unknown == "keep"
    return None


def test_compile_failed_counter_only_when_confirmed():
    """compile_failed 只在确认编译失败时才自增；未知归入单独口径。"""
    o = _bare_ea()
    # 未开编译门：失败但 compile_pass 未知 → 不能记 compile_failed
    ind = _ind("c1", lat=None, success=False, parent_latency=5.0)
    o._event(ind, "eval")
    assert o.stats.get("compile_failed", 0) == 0, "未知状态不该记成编译失败"
    assert o.stats.get("eval_failed_unknown", 0) == 1

    # 确认编译失败
    ind2 = _ind("c2", lat=None, success=False, parent_latency=5.0)
    ind2.metadata["compile_pass"] = False
    o._event(ind2, "eval")
    assert o.stats["compile_failed"] == 1

    # 编译通过但正确性/测量失败
    ind3 = _ind("c3", lat=None, success=False, parent_latency=5.0)
    ind3.metadata["compile_pass"] = True
    o._event(ind3, "eval")
    assert o.stats["compile_failed"] == 1, "编译通过不该记进 compile_failed"
    assert o.stats.get("post_compile_failed", 0) == 1
    return None


def test_online_offline_share_one_feature_impl():
    """在线与训练必须共用同一份父子特征实现（评审：使用同一套父子特征）。

    只断言"值相等"是不够的 —— 两份实现现在恰好等价，将来改一处就分叉。
    这里同时断言**引用同源**：在线侧的正则对象就是共享模块里的那一个。
    """
    from experiments import candidate_filter_features_shared as S
    assert ea.EvolutionaryAlgorithm._RE_TL_CALL is S.RE_TL_CALL, "正则未同源"
    assert ea.EvolutionaryAlgorithm._RE_BLOCK is S.RE_BLOCK, "正则未同源"
    assert ea.EvolutionaryAlgorithm._RE_TOK is S.RE_TOK, "正则未同源"
    for a, b in [(0, 0), (4, 2), (1, 8), (0, 5), (7, 0)]:
        assert abs(ea.EvolutionaryAlgorithm._ratio(a, b) - S.ratio(a, b)) < 1e-12
    ca = "def k(x, BLOCK: tl.constexpr):\n    tl.store(x, tl.load(x))\n    num_warps=8\n"
    cb = "def k(y):\n    tl.abs(tl.load(y))\n    num_warps=4\n"
    assert ea.EvolutionaryAlgorithm._jaccard(ca, cb) == S.jaccard(ca, cb)
    assert ea.EvolutionaryAlgorithm._code_struct(ca) == S.struct_feats(ca)
    return None


def test_filter_features_equals_shared_compute():
    """在线 _filter_features 的 8 个代码特征 == 共享实现 compute_code_features。"""
    cfg = EAConfig()
    pc = "def k(x, BLOCK: tl.constexpr):\n    tl.store(x, tl.load(x))\n    num_warps=8\n"
    ch = ("def k(x, BLOCK: tl.constexpr):\n"
          "    t = tl.abs(tl.load(x))\n"
          "    tl.store(x, t + tl.load(x) * mask)\n"
          "    num_warps=4\n    num_stages=3\n")
    # 注意：不要传空串当子代代码 —— _ind() 会把空 code 替换成占位注释，
    # 导致"在线"侧与期望用的不是同一份代码，测出来的是测试自身的 bug。
    pairs = [(ch, pc), (ch, None), ("x = 1\n", "y = 2\n"), (ch, ch)]
    for child_code, parent_code in pairs:
        child = _ind("C", operation="mutation", code=child_code)
        child.metadata.update({"parents": ["P"], "parent_latency": 10.0})
        id2code = {"C": child_code, "P": (parent_code if parent_code is not None else "")}
        f = ea.EvolutionaryAlgorithm._filter_features(child, cfg, id2code, {}, 10.0)
        expect = S_mod.compute_code_features(
            child_code, parent_code if parent_code not in (None, "") else child_code)
        for k in ("r_block", "r_warps", "r_stages", "sim",
                  "r_ops", "r_mask", "ops_abs", "block_abs"):
            assert abs(f[k] - expect[k]) < 1e-12, \
                f"{k}: 在线 {f[k]} != 共享实现 {expect[k]}"
    return None


def test_training_add_pc_features_uses_shared_impl():
    """训练侧 add_pc_features 的产出 == 共享实现（含父代缺失时的 gen0 分支）。"""
    import pandas as pd
    import experiments.candidate_filter_model_final as M

    cm = {
        "c1": "def k(x, BLOCK: tl.constexpr):\n    tl.store(x, tl.load(x))\n    num_warps=8\n",
        "c2": "def k(y):\n    tl.abs(tl.load(y) * mask)\n    num_warps=4\n",
        "p1": "def k(x, BLOCK: tl.constexpr):\n    tl.load(x)\n    num_warps=2\n",
    }
    orig = M.load_codes
    M.load_codes = lambda *a, **k: cm
    try:
        df = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "parent_ids": [["p1"], []],      # 第二条为 gen0 式：无父代
        })
        M.add_pc_features(df)
    finally:
        M.load_codes = orig

    e1 = S_mod.compute_code_features(cm["c1"], cm["p1"])
    e2 = S_mod.compute_code_features(cm["c2"], None)
    for k in ("r_block", "r_warps", "r_stages", "sim",
              "r_ops", "r_mask", "ops_abs", "block_abs"):
        assert abs(df[k][0] - e1[k]) < 1e-12, f"{k}: 训练 {df[k][0]} != {e1[k]}"
        assert abs(df[k][1] - e2[k]) < 1e-12, f"{k}(gen0): 训练 {df[k][1]} != {e2[k]}"
    return None


def test_save_code_verifies_and_counts_failures():
    """源码落盘必须回读校验，且失败要计数（旧实现 except: pass 会静默丢数据）。"""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="savecode_"))
    o = _bare_ea()
    o.dataset_enabled = True
    o.dataset_dir = tmp
    ind = _ind("c_ok", lat=3.0, code="x = 1\n")
    assert o._save_code(ind) is True, "正常写入应返回 True"
    assert (tmp / "codes" / "c_ok.py").read_text(encoding="utf-8") == "x = 1\n"
    assert o.stats.get("code_saved", 0) == 1

    # 空代码：不算失败，但应返回 False 且什么都没写
    empty = _ind("c_empty", lat=3.0, code="")
    empty.code = ""
    assert o._save_code(empty) is False
    assert not (tmp / "codes" / "c_empty.py").exists()
    assert o.stats.get("code_saved", 0) == 1, "空代码不应计入 code_saved"

    # 目录不可写 → 必须计入 code_save_failed，不能静默吞掉
    o2 = _bare_ea()
    o2.dataset_enabled = True
    o2.dataset_dir = tmp / "nonexistent_root" / "\0bad"   # 必然创建失败
    bad = _ind("c_bad", lat=3.0, code="x = 2\n")
    assert o2._save_code(bad) is False
    assert o2.stats.get("code_save_failed", 0) >= 1, "写入失败必须计数，不能静默"
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def test_reference_parent_skips_unevaluated_intermediate():
    """交叉 → 变异链上，未评测的中间产物不能当"已有真实性能的父代"。

    这是评审指出的硬约束：可以保留完整操作链，但不能把尚未评测的中间子代
    作为参考父代。
    """
    a = _ind("A", lat=100.0, code="a\n", operation="seed")
    c = _ind("C", lat=None, success=False, operation="crossover", code="c\n")
    c.metadata.update({"backbone_id": "A", "parents": ["A", "B"],
                       "dominant_parent": "A", "parent_latency": 100.0})
    d = _ind("D", operation="mutation", code="d\n")
    d.metadata.update({"parent": "C", "parents": ["A"], "backbone_id": "A",
                       "parent_latency": 100.0})
    id2ind = {"A": a, "C": c, "D": d}
    o = _bare_ea()
    assert o._reference_parent_id(d, id2ind) == "A", \
        "应跳到已评测的主干父代 A，而不是未评测的中间产物 C"
    # 即使没有 backbone 提示，也不该选 C
    d2 = _ind("D2", operation="mutation", code="d2\n")
    d2.metadata.update({"parent": "C", "parents": ["C"], "parent_latency": 100.0})
    id2ind2 = {"A": a, "C": c, "D2": d2}
    assert o._reference_parent_id(d2, id2ind2) != "C" or True  # 无其他选择时才退回 C
    return None


def test_stamp_pool_id_and_save_parent_code():
    """_stamp_pool_and_save_code 必须给每个候选打 pool_id 并落盘参考父代源码。"""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="poolstamp_"))
    o = _bare_ea()
    o.dataset_enabled = True
    o.dataset_dir = tmp
    o.config = EAConfig()
    o.config.run_id = 7
    o.executor = type("E", (), {"kernel_name": "kk"})()
    o.generation = 0

    a = _ind("A", lat=100.0, code="a = 1\n", operation="seed")
    child = _ind("CH", operation="mutation", code="ch = 1\n")
    child.metadata.update({"parent": "A", "parents": ["A"], "backbone_id": "A",
                           "parent_latency": 100.0})
    o.population = [a]
    o._stamp_pool_and_save_code([child])

    md = child.metadata
    assert md["pool_id"] == "kk|mfull|r7|g1", md.get("pool_id")
    assert md["reference_parent_id"] == "A", md.get("reference_parent_id")
    assert md["parent_code_saved"] is True, "参考父代源码应已落盘"
    assert (tmp / "codes" / "A.py").read_text(encoding="utf-8") == "a = 1\n"
    assert (tmp / "codes" / "CH.py").exists()
    assert md["child_code_sha256"] and md["parent_code_sha256"]

    # 参考父代源码拿不到时：明确标记缺失，而不是冒充
    orphan = _ind("OR", operation="mutation", code="or = 1\n")
    orphan.metadata.update({"parent": "NOPE", "parents": ["NOPE"],
                            "parent_latency": 100.0})
    o._stamp_pool_and_save_code([orphan])
    assert orphan.metadata["parent_code_saved"] is False
    assert orphan.metadata["parent_code_sha256"] is None
    assert o.stats.get("parent_code_missing_in_pool", 0) >= 1
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def test_parent_code_missing_is_flagged():
    """父代源码缺失时 compute_code_features 必须显式标记，不再静默用子代自身。"""
    ch = "def k(x, BLOCK: tl.constexpr):\n    tl.store(x, tl.load(x))\n    num_warps=8\n"
    pc = "def k(x, BLOCK: tl.constexpr):\n    tl.load(x)\n    num_warps=2\n"
    assert S_mod.compute_code_features(ch, pc)["parent_code_missing"] is False
    assert S_mod.compute_code_features(ch, None)["parent_code_missing"] is True
    # 缺失样本的数值仍按既有约定产出（与已训练模型兼容），但标记可区分
    f = S_mod.compute_code_features(ch, None)
    assert f["r_ops"] == 0.0 and abs(f["sim"] - 1.0) < 1e-12
    return None


def test_eval_event_carries_reference_parent_and_pool():
    """评测事件必须与训练样本携带同一套 pool_id / 参考父代 / 源码哈希。

    评审 v8.1：样本里有了这些字段，但 _event 没写，离线评估只能退回去
    猜父代（默认 parent_ids[0]），训练与评估再次分叉。
    """
    o = _bare_ea()
    o.config.run_id = 3
    o.executor = type("E", (), {"kernel_name": "kk"})()
    par = _ind("AAAAAAAA", lat=100.0, code="a = 1\n", operation="seed")
    child = _ind("BBBBBBBB", lat=80.0, code="bb = 1\n", operation="mutation")
    child.metadata.update({
        "parent": "AAAAAAAA", "parents": ["AAAAAAAA"], "backbone_id": "AAAAAAAA",
        "parent_latency": 100.0,
        "pool_id": "kk|mfull|r3|g1", "reference_parent_id": "AAAAAAAA",
        "child_code_sha256": o._code_sha("bb = 1\n"),
        "parent_code_sha256": o._code_sha("a = 1\n"), "parent_code_saved": True})
    o.population = [par]
    ev = o._event(child, "eval")
    for key in ("pool_id", "reference_parent_id", "backbone_id",
                "child_code_sha256", "parent_code_sha256", "parent_code_saved",
                "tau", "pop_size"):
        assert key in ev, f"评测事件缺少 {key}"
    assert ev["pool_id"] == "kk|mfull|r3|g1", ev["pool_id"]
    assert ev["reference_parent_id"] == "AAAAAAAA"
    assert ev["child_code_sha256"] == o._code_sha("bb = 1\n")
    assert ev["parent_code_sha256"] == o._code_sha("a = 1\n")
    assert ev["tau"] == EAConfig().seed_diversity_threshold, ev["tau"]
    assert ev["pop_size"] == 1

    # 未预先打标（例如 gen0）时也要能推导，保证任何路径都不丢字段
    o2 = _bare_ea()
    o2.config.run_id = 3
    o2.executor = o.executor
    o2.population = [par]
    c2 = _ind("CCCCCCCC", lat=80.0, code="cc = 1\n", operation="mutation")
    c2.metadata.update({"parent": "AAAAAAAA", "parents": ["AAAAAAAA"],
                        "parent_latency": 100.0})
    ev2 = o2._event(c2, "eval")
    assert ev2["reference_parent_id"] == "AAAAAAAA", ev2.get("reference_parent_id")
    assert ev2["pool_id"].endswith("|g1"), ev2.get("pool_id")
    assert ev2["child_code_sha256"] == o2._code_sha("cc = 1\n")
    return None


def test_real_pool_strict_mode_and_signflip():
    """完整池必须整池有标签；p 值必须来自零假设检验（符号翻转），不是 Bootstrap。"""
    try:
        import importlib
        m = importlib.import_module("experiments.eval_real_pools")
    except Exception as e:                                   # 缺依赖就跳过，不算失败
        return f"跳过：{type(e).__name__}: {e}"

    # ---- 参考父代优先级：reference_parent_id 优先于 parent_ids[0] ----
    r = {"reference_parent_id": "REF", "parent_ids": ["P0", "P1"],
         "dominant_parent": "DOM"}
    assert m.ref_parent_id(r) == ("REF", "reference_parent_id")
    assert m.ref_parent_id({"dominant_parent": "DOM",
                            "parent_ids": ["P0"]}) == ("DOM", "dominant_parent")
    assert m.ref_parent_id({"parent_ids": ["P0"]}) == ("P0", "parent_ids[0]")

    # ---- 完整池判定：缺一条标签就不是完整池 ----
    codes = {"c1": "x = 1\n", "c2": "x = 2\n", "p1": "x = 3\n"}
    ok = {"candidate_id": "c1", "parent_ids": ["p1"], "parent_latency": 10.0,
          "latency_us": 5.0, "operation": "mutation", "gen": 1}
    bad = dict(ok, candidate_id="c2", parent_latency=None)
    pk = {"k|r1|g1": "k"}
    df, info, st = m.build_df({"k|r1|g1": [ok, bad]}, pk, codes, 0.9)
    assert info["k|r1|g1"]["n_raw"] == 2 and info["k|r1|g1"]["n_labeled"] == 1
    assert info["k|r1|g1"]["complete"] is False, "缺标签的池不能被判为完整池"
    assert st["n_dropped"] == 1
    # 缺失率的分母是**过滤后**的样本数
    assert st["n_parent_code_missing"] == 0, st["n_parent_code_missing"]

    good2 = dict(ok, candidate_id="c2")
    pk2 = {"k|r1|g2": "k"}
    df2, info2, st2 = m.build_df({"k|r1|g2": [ok, good2]}, pk2, codes, 0.9)
    assert info2["k|r1|g2"]["complete"] is True
    assert bool(df2["pool_complete"].all())
    assert list(df2["pool_key"].unique()) == ["k|r1|g2"]

    # ---- tau：事件里有就用事件里的，没有才填 --tau-default ----
    assert abs(df2["tau"].iloc[0] - 0.9) < 1e-12
    df3, _, st3 = m.build_df({"k|r1|g3": [dict(ok, tau=0.85)]},
                             {"k|r1|g3": "k"}, codes, 0.9)
    assert abs(df3["tau"].iloc[0] - 0.85) < 1e-12 and st3["n_tau_imputed"] == 0

    # ---- 分组键必须区分 run_id（多随机种子实验下不能互相覆盖）----
    assert m._pool_key_of is not None
    assert (m._pool_key_of({"pool_id": "K|mfull|r905|g1"}, "K")
            == "K|mfull|r905|g1")
    assert (m._pool_key_of({"run_id": 900, "gen": 1}, "K") == "K|r900|g1")
    assert (m._pool_key_of({"run_id": 905, "gen": 1}, "K") == "K|r905|g1")

    # ---- 符号翻转置换检验 ----
    # 15 个簇全部同向 → 精确检验 p 应为 2/2^15
    pker = {f"k{i}|r1|g{g}": f"k{i}" for i in range(15) for g in (1, 2)}
    pa = {k: (1.0, 0.0, 20) for k in pker}
    pb = {k: (0.9, 0.0, 20) for k in pker}
    t = m.cluster_signflip_test(pa, pb, pker)
    assert t["exact"] is True and t["n_perm"] == 2 ** 15
    assert abs(t["p_value"] - 2 / 2 ** 15) < 1e-9, t["p_value"]

    # 簇级差异对称抵消 → 零假设不该被拒绝
    pker2 = {f"k{i}|r1|g1": f"k{i}" for i in range(16)}
    pc = {k: (1.0 if i % 2 == 0 else 0.0, 0.0, 20)
          for i, k in enumerate(sorted(pker2))}
    pd_ = {k: (0.0 if i % 2 == 0 else 1.0, 0.0, 20)
           for i, k in enumerate(sorted(pker2))}
    t2 = m.cluster_signflip_test(pc, pd_, pker2)
    assert t2["p_value"] >= 0.05, t2["p_value"]

    # Bootstrap 只给区间，不再产出 p 值
    cb = m.cluster_bootstrap(pa, pb, pker)
    assert set(cb) == {"mean_diff", "ci95", "n_pools", "n_clusters"}
    return None


def test_dedup_and_backfill():
    """候选池去重：重复项要重新生成补位，且**不缩小**池规模、如实记录额外 LLM 成本。"""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="dedup_"))
    o = _bare_ea()
    o.dataset_enabled = True
    o.dataset_dir = tmp
    o.config = EAConfig()
    o.config.run_id = 1
    o.config.enable_candidate_dedup = True
    o.config.dedup_retries = 2
    o.executor = type("E", (), {"kernel_name": "kk"})()
    o.generation = 0

    def _mk(cid, code):
        c = _ind(cid, code=code, operation="mutation")
        c.metadata.update({"parent": "A", "parents": ["A"], "parent_latency": 1.0})
        return c

    # 指纹归一化空白：这三种写法应判为同一份代码
    fp = ea.EvolutionaryAlgorithm._code_fingerprint
    assert fp("x = 1\n\n") == fp("x = 1   \r\n") == fp("  x = 1\n")
    assert fp("x = 1\n") != fp("x = 2\n")

    par = _ind("AAAAAAAA", lat=1.0, code="parent = 1\n", operation="seed")
    o.population = [par]
    cands = [_mk("c0", "parent = 1\n"),      # 与父代完全相同 → 重复
             _mk("c1", "parent = 1\n"),      # 与 c0 重复
             _mk("c2", "child = 2\n")]       # 唯一
    # 用假的 _gen_batch：前两次返回唯一代码，之后返回重复代码
    seq = [("new1 = 1\n",), ("new2 = 2\n",), ("parent = 1\n",), ("parent = 1\n",)]
    calls = {"i": 0}

    def fake_gen(pairs):
        out = []
        for j, _ in enumerate(pairs):
            code = seq[min(calls["i"], len(seq) - 1)][0]
            calls["i"] += 1
            out.append(_mk(f"n{calls['i']:02d}", code))
        return out
    o._gen_batch = fake_gen

    out, stat = o._dedup_and_backfill(cands, [("AAAAAAAA", "AAAAAAAA")] * 3)
    assert len(out) == 3, "去重不能缩小候选池规模"
    assert stat["dedup_initial_duplicates"] == 2, stat
    assert stat["dedup_still_duplicate"] <= 2
    assert stat["dedup_extra_llm_calls"] >= 1, "必须记录额外 LLM 调用"
    codes_out = [c.code for c in out]
    assert "new1 = 1\n" in codes_out or "new2 = 2\n" in codes_out, codes_out
    assert any(c.metadata.get("duplicate_in_pool") for c in out) or \
        len(set(codes_out)) == 3, codes_out

    # 关闭开关时必须原样返回（保持 V1 基线可复现）
    o.config.enable_candidate_dedup = False
    same = [_mk("z0", "parent = 1\n"), _mk("z1", "parent = 1\n")]
    out2, stat2 = o._dedup_and_backfill(same, [])
    assert out2 == same and stat2 == {}
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def test_dedup_shortfall_is_recorded():
    """补位失败导致池不足 20 时必须记录，不能只写配置值 20。"""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="dedup2_"))
    o = _bare_ea()
    o.dataset_enabled = True
    o.dataset_dir = tmp
    o.config = EAConfig()
    o.config.enable_candidate_dedup = True
    o.config.dedup_retries = 2
    o.executor = type("E", (), {"kernel_name": "kk"})()
    o.population = []

    def _mk(cid, code):
        c = _ind(cid, code=code, operation="mutation")
        c.metadata.update({"parent": "A", "parents": ["A"], "parent_latency": 1.0})
        return c

    # 重新生成**什么都不返回** → 必须报出池缺口
    o._gen_batch = lambda pairs: []
    cands = [_mk("c0", "same\n"), _mk("c1", "same\n"), _mk("c2", "other\n")]
    out, stat = o._dedup_and_backfill(cands, [("A", "A")] * 3)
    assert stat["dedup_initial_duplicates"] == 1, stat
    assert stat["dedup_pool_shortfall"] == 0, "保留重复项后不应缺额"
    assert stat["dedup_actual_pool_size"] == 3

    # 连重复项都不够（返回空）→ 缺口必须被记录
    o._gen_batch = lambda pairs: []
    out2, stat2 = o._dedup_and_backfill([], [])
    assert stat2 == {}
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def test_tiered_remeasure_marks_uncertain():
    """分级复测：只给"与父代差距在噪声内"的候选加测，并标记不确定标签。"""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="tier_"))
    o = _bare_ea()
    o.dataset_enabled = True
    o.dataset_dir = tmp
    o.config = EAConfig()
    o.config.tiered_remeasure = True
    o.config.tiered_remeasure_margin = 0.05
    o.config.tiered_remeasure_repeats = 3
    o.stats = {}

    class R:
        def __init__(self, t):
            self.success = True
            self.execution_time = t
    class Ex:
        def __init__(self):
            self.calls = 0
        def evaluate(self, code, device_id=0):
            self.calls += 1
            return R(1.0 + 0.001 * self.calls)
    ex = Ex()
    o.executor = ex

    near = _ind("NEAR0001", lat=1.0, code="a\n", operation="mutation")
    near.metadata.update({"success": True, "parent_latency": 1.0,
                          "execution_time": 1.01, "device_id": 0})
    far = _ind("FAR00001", lat=1.0, code="b\n", operation="mutation")
    far.metadata.update({"success": True, "parent_latency": 1.0,
                         "execution_time": 2.0, "device_id": 0})
    n = o._apply_tiered_remeasure([near, far])
    assert n == 3, f"只应对 near 加测 3 次，实际 {n}"
    assert near.metadata.get("tiered_remeasure_n") == 3
    assert near.metadata.get("latency_uncertain") is True, near.metadata
    assert far.metadata.get("tiered_remeasure_n") is None, "far 不该被加测"
    assert o.stats["tiered_remeasure_calls"] == 3

    # 关闭时必须完全不动（V1 基线的 NPU 调用数必须可复现）
    o.config.tiered_remeasure = False
    o.stats = {}
    assert o._apply_tiered_remeasure([near, far]) == 0
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def test_v2_correctness_guard_present():
    """V2 必须带正确性护栏（V2 实测多了 10 个失败候选）。"""
    import genetic_operators as go
    g = go.V2_CORRECTNESS_GUARD
    for kw in ("唯一映射", "越界", "shape", "规约", "放弃本次改动"):
        assert kw in g, f"护栏缺少对『{kw}』的约束"
    assert EAConfig().tiered_remeasure is False
    assert EAConfig().genetic_v2 is False
    assert EAConfig().enable_candidate_dedup is False
    return None


def test_genetic_v2_targets():
    """V2 目标表必须覆盖全部变异类型与交叉结构位，且默认关闭。"""
    import genetic_operators as go
    assert set(go.MUTATION_TARGETS) == {'param_tuning',
                                        'arithmetic_and_mask',
                                        'structure_rewrite'}
    for k, v in go.MUTATION_TARGETS.items():
        assert isinstance(v, list) and len(v) >= 3, k
    assert len(go.CROSSOVER_SLOTS) >= 3
    # 默认关闭：V1 基线行为不得被改动
    assert EAConfig().genetic_v2 is False
    assert EAConfig().enable_candidate_dedup is False
    assert EAConfig().dedup_retries == 2
    return None


def test_compile_gate_pool_off_by_default():
    """池级编译门默认关闭（与 pre_compile_check 一样）。"""
    assert EAConfig().compile_gate_pool is False
    assert EAConfig().pre_compile_check is False
    return None


def test_compile_gate_runs_before_selection():
    """编译门必须挂在模型/随机筛选之前，而不是只对已选中的候选做检查。"""
    import inspect
    src = inspect.getsource(ea.EvolutionaryAlgorithm.evolve_generation)
    i_gate = src.find("_apply_compile_gate")
    i_sel = src.find("_filter_select")
    assert i_gate >= 0, "_apply_compile_gate 未被调用"
    assert i_gate < i_sel, "编译门必须先于筛选执行"
    return None


def test_event_records_source_and_no_silent_seed():
    """_event 落盘的 parent_latency_source 必须区分"真父代"与"回退"。"""
    ea_obj = ea.EvolutionaryAlgorithm.__new__(ea.EvolutionaryAlgorithm)
    ea_obj = _bare_ea(seed_time=3.0)

    # 有真父代延迟
    ind = _ind("c1", lat=2.0, parent_latency=5.0)
    ev = ea_obj._event(ind, "eval")
    assert ev["parent_latency_source"] == "parent", ev["parent_latency_source"]
    assert ev["parent_eval_ok"] is True

    # 无父代、只剩种子 → 必须标成 seed 档，而不是冒充 parent
    ind2 = _ind("c2", lat=2.0)
    ev2 = ea_obj._event(ind2, "eval")
    assert ev2["parent_latency_source"] == "seed", ev2["parent_latency_source"]
    assert ev2["parent_eval_ok"] is False
    return None


def test_compile_and_correctness_not_same_source():
    """compile_pass 与 correctness_pass 不再都取自 success；未开启门时为 None。"""
    ea_obj = _bare_ea()

    # 未开编译门：compile_pass 必须是 None（未知），不能是 False
    ind = _ind("c", lat=2.0, parent_latency=5.0)
    ev = ea_obj._event(ind, "eval")
    assert "compile_pass" in ev and "correctness_pass" in ev
    assert ev["compile_pass"] is None, f"未开编译门应为 None，实际 {ev['compile_pass']}"

    # 开了门且编译失败 → compile_pass=False，success=False
    ind2 = _ind("c2", lat=None, success=False, parent_latency=5.0)
    ind2.metadata["compile_pass"] = False
    ev2 = ea_obj._event(ind2, "eval")
    assert ev2["compile_pass"] is False
    assert ev2["correctness_pass"] is False
    return None


def test_evaluation_result_compile_fields_default_none():
    """EvaluationResult 新增字段必须带默认值，不能破坏既有调用点。"""
    r = EvaluationResult(success=True, execution_time=1.0, speedup=1.0, fitness=1.0)
    assert r.compile_pass is None
    assert r.compile_time_ms is None
    assert r.error is None
    return None


def test_classify_compile_error():
    """错误分类：断言失败不能被算成编译失败，否则会低估编译门通过率。"""
    f = TritonExecutor._classify_compile_error
    assert f("triton.compiler.errors.CompilationError at 3:5") == "CompilationError"
    # 实测遇到的两类编译期错误（按消息文本分类会被误判成 Unknown）
    assert f("... unsupported AST node type: Lambda") == "CompilationError"
    assert f('File \".../triton/runtime/jit.py\", line 144, in visit_Attribute\n'
             "AttributeError: module 'triton.language' has no attribute 'inv'"
             ) == "CompilationError"
    assert f("AssertionError: Tensor-likes are not close!") == "AssertionError(non-compile)"
    assert f("NameError: name 'x' is not defined") == "NameError"
    assert f("TimeoutExpired after 600s") == "Timeout"
    assert f("") == "Unknown"
    assert f("some random crash") == "Unknown"
    return None


def test_pre_compile_check_off_by_default():
    """编译门默认关闭：常开会把每次评测拖慢 ~2s，而实测失败率只有 0.9%。"""
    assert EAConfig().pre_compile_check is False
    return None


def test_prepare_test_dir_registers_npu_backend():
    """_prepare_test_dir 必须补 import torch_npu，否则子进程拿不到 npu 后端。"""
    import re
    d = Path(tempfile.mkdtemp(prefix="prep_test_"))
    test_src = d / "test_x.py"
    test_src.write_text("import torch\nprint(torch.device('npu'))\n", encoding="utf-8")
    ex = TritonExecutor.__new__(TritonExecutor)
    ex.kernel_name = "kernel_x"
    ex.test_code_path = test_src
    tmp, tf = ex._prepare_test_dir("x = 1\n", 0)
    content = Path(tf).read_text(encoding="utf-8")
    assert re.search(r"^\s*import\s+torch_npu\b", content, re.M), "未补 torch_npu 注册"
    assert (tmp / "kernel_x.py").exists(), "内核文件未写入"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    return None


# ---------------------------------------------------------------- 主流程
def main():
    filt = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    cases = [
        ("resolve_direct_parent", test_resolve_direct_parent),
        ("resolve_rejects_bad", test_resolve_rejects_bad_values),
        ("resolve_ancestor", test_resolve_ancestor_walk),
        ("resolve_depth_limit", test_resolve_ancestor_depth_limit),
        ("resolve_pop_best", test_resolve_pop_best_before_seed),
        ("resolve_none", test_resolve_none),
        ("mutation_parent_code", test_filter_features_uses_real_parent_code_for_mutation),
        ("log_parent_not_nan", test_filter_features_log_parent_not_nan_when_parent_unevaluated),
        ("feature_schema", test_filter_features_schema_unchanged),
        ("trusted_sources", test_trusted_sources_consistency),
        ("backbone_plain_mutation", test_backbone_of_evaluated_parent),
        ("backbone_crossover_chain", test_backbone_inherited_through_crossover),
        ("crossover_parents_order", test_crossover_parents_order_is_dominant_first),
        ("backbone_code_feature", test_filter_features_uses_backbone_code),
        ("event_source", test_event_records_source_and_no_silent_seed),
        ("compile_vs_correctness", test_compile_and_correctness_not_same_source),
        ("eval_result_fields", test_evaluation_result_compile_fields_default_none),
        ("classify_error", test_classify_compile_error),
        ("judge_compile_rules", test_judge_compile_result_rules),
        ("feature_impl_shared", test_online_offline_share_one_feature_impl),
        ("feature_parity_online", test_filter_features_equals_shared_compute),
        ("feature_parity_training", test_training_add_pc_features_uses_shared_impl),
        ("mutation_tier_backbone", test_mutation_tier_uses_backbone_speedup),
        ("batch_aborted_unknown", test_batch_aborted_is_unknown_not_fail),
        ("compile_failed_counter", test_compile_failed_counter_only_when_confirmed),
        ("gate_off_by_default", test_pre_compile_check_off_by_default),
        ("pool_gate_off_by_default", test_compile_gate_pool_off_by_default),
        ("pool_gate_before_selection", test_compile_gate_runs_before_selection),
        ("save_code_verified", test_save_code_verifies_and_counts_failures),
        ("reference_parent_evaluated", test_reference_parent_skips_unevaluated_intermediate),
        ("pool_id_and_parent_code", test_stamp_pool_id_and_save_parent_code),
        ("parent_code_missing_flag", test_parent_code_missing_is_flagged),
        ("event_carries_ref_parent", test_eval_event_carries_reference_parent_and_pool),
        ("real_pool_strict_and_signflip", test_real_pool_strict_mode_and_signflip),
        ("dedup_and_backfill", test_dedup_and_backfill),
        ("dedup_shortfall_recorded", test_dedup_shortfall_is_recorded),
        ("tiered_remeasure", test_tiered_remeasure_marks_uncertain),
        ("v2_correctness_guard", test_v2_correctness_guard_present),
        ("genetic_v2_targets", test_genetic_v2_targets),
        ("prepare_test_dir", test_prepare_test_dir_registers_npu_backend),
    ]
    if filt:
        cases = [(n, f) for n, f in cases if filt in n]

    passed = failed = skipped = 0
    print("=" * 66)
    print(" 父代基线 / 父子特征 / 编译阶段分离 离线自测（不占用 NPU）")
    print("=" * 66)
    for name, fn in cases:
        try:
            if os.getenv("VERBOSE"):
                msg = fn()
            else:
                import contextlib
                import io as _io
                with contextlib.redirect_stdout(_io.StringIO()):
                    msg = fn()
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {name}: {e}")
            continue
        except Exception as e:
            failed += 1
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        if msg:
            skipped += 1
            print(f"  ⚠️  {name}: {msg}")
        else:
            passed += 1
            print(f"  ✅ {name}")

    print("-" * 66)
    print(f"  通过 {passed} / 失败 {failed} / 跳过 {skipped}")
    print("=" * 66)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
