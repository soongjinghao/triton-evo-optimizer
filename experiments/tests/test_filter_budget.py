#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选预筛选：预算 / 分代 / 模型隔离 的离线自测（**不占用 NPU**）。

用假的生成器与假的执行器替换 LLM 与 msprof，只验证控制流与计数：
  * 第 0 代固定 5 生成 / 5 送评，且不进入预筛选
  * 进化代 20 生成 → 4 送评
  * 全流程 NPU 评测次数恒为 13（fulleval 为 45）
  * 预算硬闸：剩余额度不足时截断，绝不超评
  * 筛选日志的 gen 编号为 0/1/2，与真值日志一致
  * 随机筛选可复现，且跨算子互相独立
  * 模型隔离强校验：缺模型 / train_kernels="ALL" / 测试算子在训练名单 → 拒绝运行

用法
----
    cd /workspace/Agent
    python experiments/tests/test_filter_budget.py            # 全部用例
    python experiments/tests/test_filter_budget.py -k budget  # 只跑名字含 budget 的

退出码 0 = 全部通过；1 = 有失败。
"""

import os
import sys
import json
import shutil
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# 必须在导入 evolutionary_algorithm 之前，避免它去探测真实卡数/建目录
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="ea_budget_test_"))
os.environ.setdefault("EA_NUM_DEVICES", "1")
os.environ.setdefault("EA_MAX_WORKERS", "4")

from config import EAConfig                                    # noqa: E402
from executor import EvaluationResult                          # noqa: E402
from genetic_operators import Individual                        # noqa: E402
from evolutionary_algorithm import EvolutionaryAlgorithm        # noqa: E402
from experiments import common                                  # noqa: E402

MODEL = ROOT / "experiments" / "dataset" / "candidate_filter_model__dev15.joblib"

# 假候选源码：彼此有差异，保证 _filter_features 的父子比/Jaccard 有意义
_CODE_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel_{i}(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n_elements
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, x * {scale} + {bias}, mask=m)
"""


# ---------------------------------------------------------------- 假组件
class FakeExecutor:
    """不碰 NPU、不跑 msprof 的假执行器。"""

    def __init__(self, kernel_name="fake_kernel"):
        self.kernel_name = kernel_name
        self.calls = 0

    def evaluate(self, code, timeout=600, device_id=0):
        self.calls += 1
        # 用代码长度造一个稳定但各不相同的"延迟"，保证 fitness 有区分度
        latency = max(1.0, 100.0 - (len(code) % 97) * 0.9)
        return EvaluationResult(success=True, execution_time=latency,
                                speedup=1.0, fitness=1.0)


class FakeGenOps:
    def __init__(self):
        self.n_strategies = 0

    def analyze_kernel(self, codes, n, profiling_context=""):
        self.n_strategies = n
        return ([f"策略{i}|诊断|动作" for i in range(n)], None, "rag")

    def generate_initial_individual_with_strategy(self, baseline_code="",
                                                  strategy_str="",
                                                  rag_rules_text=""):
        raise AssertionError("本测试直接替换 _process_single_generation_task")

    def crossover(self, p1, p2):
        raise AssertionError("本测试直接替换 _process_single_generation_task")

    def mutate(self, ind):
        raise AssertionError("本测试直接替换 _process_single_generation_task")


def _make_config(**over):
    cfg = EAConfig()
    cfg.method = "full"
    cfg.population_size = 6
    cfg.max_generations = 2
    cfg.gen0_candidates = 5
    cfg.children_per_generation = 4
    cfg.eval_budget = 13
    cfg.enable_profiling = False          # 不解析 Profiling，避免读真实文件
    cfg.enable_rag = False
    cfg.enable_strategy_init = True
    cfg.save_code = False
    cfg.collect_data = False
    cfg.dataset_dir = str(_TMP_ROOT / "dataset")
    cfg.save_code_dir = str(_TMP_ROOT / "saved")
    cfg.random_seed = 0
    cfg.run_id = 0
    cfg.log_path = str(_TMP_ROOT / "logs" / "ea.jsonl")
    cfg.filter_log_path = str(_TMP_ROOT / "logs" / "filter.jsonl")
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _build_ea(cfg, kernel="fake_kernel"):
    ex = FakeExecutor(kernel)
    ea = EvolutionaryAlgorithm(FakeGenOps(), ex, cfg)
    n_gen = [0]

    def fake_task(self, task_type="", seed_code="", strategy_str="",
                  parent1=None, parent2=None, rag_rules_text=""):
        n_gen[0] += 1
        i = n_gen[0]
        gen = self.generation if task_type == "gen0_strategy" \
            else self.generation + 1
        code = _CODE_TMPL.format(i=i, scale=1.0 + i * 0.01, bias=i)
        md = {
            "operation": ("gen0_strategy" if task_type == "gen0_strategy"
                          else "mutation"),
            "mutation_type": ("none" if task_type == "gen0_strategy"
                              else "arithmetic_and_mask"),
            "parents": ([parent1.id] if parent1 is not None else []),
            "parent_latency": (getattr(parent1, "metadata", {}).get(
                "execution_time") if parent1 is not None else None),
        }
        return Individual(code=code, generation=gen, metadata=md)

    ea._process_single_generation_task = fake_task.__get__(ea, EvolutionaryAlgorithm)
    ea._n_gen = n_gen
    return ea, ex


def _read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


# ---------------------------------------------------------------- 用例
def _reset_logs(cfg):
    for p in (cfg.log_path, cfg.filter_log_path):
        if p and Path(p).exists():
            Path(p).unlink()


def test_gen0_not_filtered():
    """第 0 代固定 5 生成 / 5 送评，不进入预筛选。"""
    if not MODEL.exists():
        return "跳过（无 dev15 模型）"
    cfg = _make_config(enable_candidate_filter=True, candidate_pool_size=20,
                       filter_model_path=str(MODEL),
                       candidate_filter_mode="model")
    _reset_logs(cfg)
    ea, ex = _build_ea(cfg)
    ea.initialize_population([_CODE_TMPL.format(i=999, scale=1.0, bias=0)])
    assert ea._n_gen[0] == 5, f"Gen0 应生成 5 个，实际 {ea._n_gen[0]}"
    assert ea.eval_count == 5, f"Gen0 应评测 5 次，实际 {ea.eval_count}"
    # 未启用 Gen0 筛选 → 筛选日志不应有 gen=0 的记录
    gens = {r.get("gen") for r in _read_jsonl(cfg.filter_log_path)}
    assert 0 not in gens, f"Gen0 不应进入预筛选，但筛选日志出现 gen=0：{gens}"
    return None


def test_evolve_pool_then_select():
    """进化代 20 生成 → 4 送评，两代共 8 次，加上 Gen0 共 13 次。"""
    if not MODEL.exists():
        return "跳过（无 dev15 模型）"
    cfg = _make_config(enable_candidate_filter=True, candidate_pool_size=20,
                       filter_model_path=str(MODEL),
                       candidate_filter_mode="model")
    _reset_logs(cfg)
    ea, ex = _build_ea(cfg)
    ea.initialize_population([_CODE_TMPL.format(i=999, scale=1.0, bias=0)])
    before = ea._n_gen[0]
    ea.evolve_generation()
    mid_gen = ea._n_gen[0] - before
    assert mid_gen == 20, f"Gen1 应生成 20 个，实际 {mid_gen}"
    assert ea.eval_count == 9, f"Gen0+Gen1 应评测 9 次，实际 {ea.eval_count}"

    before = ea._n_gen[0]
    ea.evolve_generation()
    assert ea._n_gen[0] - before == 20, "Gen2 应生成 20 个"
    assert ea.eval_count == 13, f"全流程应评测 13 次，实际 {ea.eval_count}"
    return None


def test_fulleval_within_budget():
    """fulleval：Gen0 5 + Gen1 20 + Gen2 20 = 45，不得超过预算。"""
    if not MODEL.exists():
        return "跳过（无 dev15 模型）"
    cfg = _make_config(enable_candidate_filter=True, candidate_pool_size=20,
                       filter_model_path=str(MODEL), filter_evaluate_all=True,
                       eval_budget=13 + 2 * (20 - 4))
    _reset_logs(cfg)
    ea, ex = _build_ea(cfg)
    ea.initialize_population([_CODE_TMPL.format(i=999, scale=1.0, bias=0)])
    ea.evolve_generation()
    ea.evolve_generation()
    assert ea.eval_count <= cfg.eval_budget, \
        f"fulleval 超预算：{ea.eval_count} > {cfg.eval_budget}"
    assert ea.eval_count == 45, \
        f"fulleval 应为 45（5+20+20），实际 {ea.eval_count}"
    return None


def test_budget_hard_gate():
    """预算硬闸：剩余额度不足时截断，绝不超评。"""
    cfg = _make_config(eval_budget=13)
    _reset_logs(cfg)
    ea, ex = _build_ea(cfg)
    # 只剩 1 次额度。注意额度按 eval_attempts 计算（含异常的设备调用），
    # 所以两个计数都要置位；只置 eval_count 会掩盖"异常未计预算"的旧缺陷。
    ea.eval_count = 12
    ea.eval_attempts = 12
    batch = [Individual(code=_CODE_TMPL.format(i=i, scale=1.0, bias=i),
                        generation=1, metadata={"operation": "mutation"})
             for i in range(20)]
    res = ea._evaluate_individuals_batch(batch)
    assert ea.eval_attempts == 13, \
        f"预算硬闸失效：应停在 13，实际 {ea.eval_attempts}（超评 {ea.eval_attempts-13} 次）"
    assert ea.eval_count == 13, f"成功计数应同为 13，实际 {ea.eval_count}"
    assert ea.stats.get("npu_calls") == 13, "stats.npu_calls 应等于尝试次数"
    skipped = [r for r in res if r.metadata.get("error") == "eval_budget_exhausted"]
    assert len(skipped) == 19, f"应有 19 个被截断，实际 {len(skipped)}"
    return None


def test_gen_numbering():
    """开启 Gen0 筛选时，筛选日志的 gen 必须是 0/1/2（旧实现 Gen0/Gen1 都写 1）。"""
    if not MODEL.exists():
        return "跳过（无 dev15 模型）"
    cfg = _make_config(enable_candidate_filter=True, candidate_pool_size=20,
                       filter_model_path=str(MODEL),
                       candidate_filter_mode="model",
                       enable_candidate_filter_gen0=True)
    _reset_logs(cfg)
    ea, ex = _build_ea(cfg)
    ea.initialize_population([_CODE_TMPL.format(i=999, scale=1.0, bias=0)])
    ea.evolve_generation()
    ea.evolve_generation()
    gens = sorted({r.get("gen") for r in _read_jsonl(cfg.filter_log_path)})
    assert gens == [0, 1, 2], f"筛选日志 gen 应为 [0,1,2]，实际 {gens}"
    return None


def test_random_select_reproducible():
    """随机筛选：同配置可复现；换算子后随机流不同。"""
    if not MODEL.exists():
        return "跳过（无 dev15 模型）"
    # 注意：不能用 Individual.id 比对——它由 hashlib.md5(str(time.time())) 生成，
    # 两次运行必然不同。这里比对候选源码（构造方式固定），才是稳定的选择结果。
    def _pick(kernel):
        cfg = _make_config(enable_candidate_filter=True, candidate_pool_size=20,
                           filter_model_path=str(MODEL),
                           candidate_filter_mode="random")
        _reset_logs(cfg)
        ea, ex = _build_ea(cfg, kernel=kernel)
        ea.initialize_population([_CODE_TMPL.format(i=999, scale=1.0, bias=0)])
        cands = [Individual(code=_CODE_TMPL.format(i=i, scale=1.0, bias=i),
                            generation=1, metadata={"operation": "mutation"})
                 for i in range(20)]
        sel = ea._random_select(cands, 4)
        assert len(sel) == 4, f"应选出 4 个，实际 {len(sel)}"
        return [c.code for c in sel]

    a1 = _pick("kernel_a")
    b = _pick("kernel_b")
    a2 = _pick("kernel_a")
    assert a1 == a2, "同配置下随机筛选不可复现"
    assert a1 != b, "不同算子复用了同一条随机流（旧实现的缺陷）"
    return None


def test_model_isolation_check():
    """模型隔离强校验：缺模型 / "ALL" / 测试算子在训练名单 → 必须拒绝。"""
    from experiments import filter_online_exp as fe
    from experiments.splits import load_split

    split = load_split("dev15")
    k_test = split["test_kernels"][0]

    # 1) 未提供模型路径
    try:
        fe._check_model_isolation(None, k_test)
    except RuntimeError as e:
        assert "未提供预筛选模型路径" in str(e)
    else:
        return "缺模型路径时未拒绝运行"

    # 2) train_kernels = "ALL"（旧实现的漏洞场景）
    d = _TMP_ROOT / "iso"
    d.mkdir(parents=True, exist_ok=True)
    m = d / "bad.joblib"
    m.write_bytes(b"")
    (d / "bad.meta.json").write_text(
        json.dumps({"train_kernels": "ALL", "exclude_kernels": []}),
        encoding="utf-8")
    try:
        fe._check_model_isolation(str(m), k_test)
    except RuntimeError as e:
        assert "不是显式名单" in str(e), str(e)
    else:
        return "train_kernels='ALL' 时未拒绝运行（旧漏洞仍在）"

    # 3) 测试算子出现在训练名单
    m2 = d / "leak.joblib"
    m2.write_bytes(b"")
    (d / "leak.meta.json").write_text(
        json.dumps({"train_kernels": [k_test, "other_kernel"]}),
        encoding="utf-8")
    try:
        fe._check_model_isolation(str(m2), k_test)
    except RuntimeError as e:
        assert "泄漏" in str(e), str(e)
    else:
        return "测试算子在训练名单中却未拒绝运行"

    # 4) 真实 dev15 模型 → 必须通过
    if MODEL.exists():
        meta = fe._check_model_isolation(str(MODEL), k_test)
        assert isinstance(meta.get("train_kernels"), list)
        assert k_test not in meta["train_kernels"]
    return None


def test_eval_attempt_counts_exception():
    """设备评测抛异常也要占用预算（旧实现只在正常返回后自增）。"""
    cfg = _make_config(eval_budget=5)
    _reset_logs(cfg)
    ea, ex = _build_ea(cfg)

    class RaisingExecutor(FakeExecutor):
        def evaluate(self, code, timeout=600, device_id=0):
            self.calls += 1
            raise RuntimeError("device timeout")

    ea.executor = RaisingExecutor()
    batch = [Individual(code=_CODE_TMPL.format(i=i, scale=1.0, bias=i),
                        generation=1, metadata={"operation": "mutation"})
             for i in range(5)]
    ea._evaluate_individuals_batch(batch)
    assert ea.eval_attempts == 5, \
        f"异常评测应计入尝试次数，实际 eval_attempts={ea.eval_attempts}"
    assert ea.stats.get("npu_calls") == 5, \
        f"npu_calls 应等于尝试次数，实际 {ea.stats.get('npu_calls')}"
    assert ea.eval_count == 0, "eval_count 只统计成功返回，应为 0"

    # 关键：下一代据此算剩余额度，必须为 0（否则会安排超预算的额外评测）
    room = cfg.eval_budget - ea.eval_attempts
    assert room == 0, f"异常后剩余额度应为 0，实际 {room}"
    return None


def test_provenance():
    """版本指纹：代码/模型/数据/配置任一变化都必须让旧结果失效。"""
    from experiments import filter_online_exp as fe

    cfg_a = {"eval_budget": 13, "enable_candidate_filter": True,
             "candidate_pool_size": 20}
    prov_a = fe._build_provenance("predictor", cfg_a, str(MODEL))
    assert prov_a["fix_tag"] == fe.FIX_TAG
    assert prov_a["model_sha256"], "未记录模型哈希"

    # 完全一致 → 可复用
    assert fe._provenance_mismatch(prov_a, prov_a) == []

    # 换了模型 → 不可复用
    prov_b = dict(prov_a, model_sha256="deadbeef")
    bad = fe._provenance_mismatch(prov_a, prov_b)
    assert any("model_sha256" in b for b in bad), f"换模型未被识别: {bad}"

    # 改了筛选配置 → 不可复用
    prov_c = fe._build_provenance("predictor",
                                  dict(cfg_a, candidate_pool_size=30), str(MODEL))
    bad = fe._provenance_mismatch(prov_a, prov_c)
    assert any("candidate_pool_size" in b for b in bad), f"改配置未被识别: {bad}"

    # 旧结果没有 provenance → 不可复用
    assert fe._provenance_mismatch({}, prov_a), "无版本信息的旧结果被判为可复用"
    return None


def test_isolation_before_search():
    """模型隔离校验必须在启动任何搜索之前完成，不能等跑完再查。"""
    from experiments import filter_online_exp as fe
    from experiments.splits import load_split

    k = load_split("dev15")["test_kernels"][0]
    orig_build, orig_run = fe.build_ea, None
    started = []

    def boom(*a, **kw):
        started.append(1)
        raise AssertionError("模型校验通过前不应启动搜索")

    fe.build_ea = boom
    try:
        # 1) 不传模型 → 应在启动搜索前拒绝
        try:
            fe.run_one(k, "predictor", 0, model_path=None)
        except RuntimeError as e:
            assert "未提供预筛选模型路径" in str(e), str(e)
        else:
            return "缺模型时未拒绝运行"
        assert not started, "模型校验前就已启动搜索（旧缺陷：检查太晚）"

        # 2) 泄漏模型 → 同样应在启动搜索前拒绝
        d = _TMP_ROOT / "late"
        d.mkdir(parents=True, exist_ok=True)
        m = d / "leak.joblib"
        m.write_bytes(b"")
        (d / "leak.meta.json").write_text(
            json.dumps({"train_kernels": [k, "x"]}), encoding="utf-8")
        try:
            fe.run_one(k, "predictor", 0, model_path=str(m))
        except RuntimeError as e:
            assert "泄漏" in str(e), str(e)
        else:
            return "泄漏模型未拒绝运行"
        assert not started, "泄漏模型竟然先跑完了一整轮搜索才报错"
    finally:
        fe.build_ea = orig_build
    return None


def test_dataset_snapshot_resolution():
    """meta 里写相对文件名时，也必须能解析到实际文件并算出真实哈希。"""
    from experiments import filter_online_exp as fe

    rel = "samples_frozen_20261016.jsonl"
    p = fe._resolve_dataset_path(rel)
    assert p is not None and p.is_file(), f"相对快照名未解析到实际文件：{rel}"
    assert p == fe.DATASET_DIR / rel, f"解析结果不符合预期：{p}"
    # 绝对路径同样要能解析
    assert fe._resolve_dataset_path(str(p)) == p, "绝对路径解析失败"
    # 不存在 / 空值 → None，不能抛异常
    assert fe._resolve_dataset_path("no_such_snapshot.jsonl") is None
    assert fe._resolve_dataset_path(None) is None
    assert fe._sha256_file(None) is None

    # 关键：provenance 必须拿到真实哈希（旧实现这里是 None）
    if MODEL.exists():
        cfg = {"eval_budget": 13, "enable_candidate_filter": True,
               "candidate_pool_size": 20, "candidate_filter_mode": "model"}
        prov = fe._build_provenance("predictor", cfg, str(MODEL))
        assert prov["dataset_snapshot"], "快照路径为空"
        assert prov["dataset_sha256"], "数据快照哈希为空（路径未解析到实际文件）"
    return None


def test_result_recovery():
    """结果文件恢复：空文件 / 缺字段 / audit 不完整 都必须判定为未完成。"""
    from experiments import filter_online_exp as fe

    d = _TMP_ROOT / "rec"
    d.mkdir(parents=True, exist_ok=True)
    good = {"kernel": "k", "config": "predictor", "seed": 0, "run_id": 900,
            "success": True, "t_best_us": 1.0,
            "audit": {"npu_calls": 13}}

    p = d / "empty.json"
    p.write_text("", encoding="utf-8")
    assert fe._load_existing(p) is None, "空文件被误判为已完成"

    p = d / "partial.json"
    p.write_text(json.dumps({"kernel": "k"}), encoding="utf-8")
    assert fe._load_existing(p) is None, "缺字段被误判为已完成"

    p = d / "noaudit.json"
    bad = dict(good)
    bad["audit"] = {}
    p.write_text(json.dumps(bad), encoding="utf-8")
    assert fe._load_existing(p) is None, "audit 不完整被误判为已完成"

    p = d / "good.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    assert fe._load_existing(p) == good, "完整结果应被复用"

    # 原子写：写完后不存在 .tmp 残留
    p = d / "atomic.json"
    fe._write_json_atomic(p, good)
    assert p.exists() and json.loads(p.read_text(encoding="utf-8")) == good
    assert not list(d.glob("atomic.json.tmp*")), "原子写留下了 .tmp 残留"
    return None


def test_split_consistency():
    """划分文件：15/35 且训练与测试无交集。"""
    from experiments.splits import load_split
    sp = load_split("dev15")
    assert len(sp["test_kernels"]) == 15, f"测试集应 15 个，实际 {len(sp['test_kernels'])}"
    assert len(sp["train_kernels"]) == 35, f"训练集应 35 个，实际 {len(sp['train_kernels'])}"
    assert not (set(sp["test_kernels"]) & set(sp["train_kernels"])), "训练/测试有交集"
    return None


# ---------------------------------------------------------------- 主流程
def main():
    filt = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    cases = [
        ("split_consistency", test_split_consistency),
        ("gen0_not_filtered", test_gen0_not_filtered),
        ("evolve_pool", test_evolve_pool_then_select),
        ("fulleval_budget", test_fulleval_within_budget),
        ("budget_hard_gate", test_budget_hard_gate),
        ("gen_numbering", test_gen_numbering),
        ("random_reproducible", test_random_select_reproducible),
        ("model_isolation", test_model_isolation_check),
        ("isolation_before_search", test_isolation_before_search),
        ("eval_attempt", test_eval_attempt_counts_exception),
        ("provenance", test_provenance),
        ("dataset_snapshot", test_dataset_snapshot_resolution),
        ("result_recovery", test_result_recovery),
    ]
    if filt:
        cases = [(n, f) for n, f in cases if filt in n]

    passed = failed = skipped = 0
    print("=" * 66)
    print(" 候选预筛选离线自测（不占用 NPU）")
    print("=" * 66)
    for name, fn in cases:
        try:
            # EA 的打印很吵；需要看细节时设 VERBOSE=1
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
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
