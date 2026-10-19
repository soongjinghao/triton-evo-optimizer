#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PAPER-FULL wiring and UCB parent-selection invariants (offline only)."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EA_NUM_DEVICES", "1")
os.environ.setdefault("EA_MAX_WORKERS", "2")

from config import EAConfig
from evolutionary_algorithm import EvolutionaryAlgorithm
from experiments import common
from experiments.harness import make_config, reference_measurement
from experiments.filter_online_exp import CONFIGS
from genetic_operators import Individual
from RAG.knowledge_node import TritonKnowledgeNode


class CaptureLogger:
    def __init__(self):
        self.records = []

    def log(self, record):
        self.records.append(record)


def _bare_ucb_ea():
    obj = EvolutionaryAlgorithm.__new__(EvolutionaryAlgorithm)
    obj.population = [
        Individual(code=f"code-{i}", fitness=fitness, id=f"p{i}")
        for i, fitness in enumerate((1.0, 0.9, 0.8, 0.7, 0.6, 0.5))
    ]
    obj.config = SimpleNamespace(selection="ucb", ucb_c=0.5,
                                 method="paper_full", run_id=17)
    obj._ucb_counts = {}
    obj.logger = CaptureLogger()
    obj.executor = SimpleNamespace(kernel_name="test_kernel")
    obj.generation = 0
    return obj


def test_paper_config():
    cfg = make_config("paper_full", "test_kernel", log=False)
    assert cfg.selection == "ucb"
    assert cfg.ucb_c == 0.5
    assert cfg.population_size == 6
    assert int(cfg.elite_ratio * cfg.population_size) == 2
    assert cfg.max_generations == 2
    assert cfg.gen0_candidates == 5
    assert cfg.children_per_generation == 4
    assert cfg.eval_budget == 13
    assert cfg.seed_diversity_threshold == 0.90
    assert cfg.crossover_mode == "protected"
    assert cfg.mutation_mode == "adaptive"
    assert cfg.require_rag_index is True
    assert cfg.enable_candidate_filter is False
    assert cfg.pre_compile_check is False
    assert cfg.compile_gate_pool is False
    assert cfg.tiered_remeasure is False
    assert cfg.require_manifest_baseline is True


def test_paper_b1_b2_smoke_configurations():
    assert CONFIGS["paper_b1"]["method"] == "paper_b1"
    assert CONFIGS["paper_b1"]["selection"] == "ucb"
    assert CONFIGS["paper_b1"]["eval_budget"] == 13
    assert CONFIGS["paper_b2"]["method"] == "b2"
    assert CONFIGS["paper_b2"]["enable_crossover"] is False
    assert CONFIGS["paper_b2"]["mutation_mode"] == "uniform"
    assert CONFIGS["paper_b2"]["eval_budget"] == 13
    assert CONFIGS["paper_b2"]["require_manifest_baseline"] is True


def test_reference_manifest_validation():
    with tempfile.TemporaryDirectory(prefix="paper_manifest_") as td:
        original = common.MANIFEST_DIR
        common.MANIFEST_DIR = Path(td)
        try:
            try:
                reference_measurement("missing_kernel", "./baseline/baseline.json",
                                      require_manifest=True)
            except RuntimeError as exc:
                assert "manifest" in str(exc)
            else:
                raise AssertionError("invalid manifest must fail explicitly")

            manifest = Path(td) / "test_kernel.json"
            manifest.write_text(
                '{"t_base_us": 10.0, "t_seed_us": 8.0, "repeats": 5, "device_id": 0}',
                encoding="utf-8")
            record = common.read_reference_manifest("test_kernel")
            assert record["t_base_us"] == 10.0
            assert record["repeats"] == 5
            assert record["sha256"]

            manifest.write_text('{"t_base_us": 0}', encoding="utf-8")
            try:
                common.read_reference_manifest("test_kernel")
            except RuntimeError as exc:
                assert "有限正数" in str(exc)
            else:
                raise AssertionError("nonpositive reference must fail explicitly")
        finally:
            common.MANIFEST_DIR = original


def test_executor_uses_manifest_reference():
    import experiments.harness as harness

    with tempfile.TemporaryDirectory(prefix="paper_executor_baseline_") as td:
        root = Path(td)
        old_manifest_dir = common.MANIFEST_DIR
        old_find_test_file = common.find_test_file
        old_executor = harness.TritonExecutor
        common.MANIFEST_DIR = root / "manifest"
        common.MANIFEST_DIR.mkdir()
        test_file = root / "test_kernel.py"
        test_file.write_text("# test stub\n", encoding="utf-8")
        common.find_test_file = lambda _kernel: test_file

        captured = {}

        def executor_factory(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        harness.TritonExecutor = executor_factory
        (common.MANIFEST_DIR / "test_kernel.json").write_text(
            '{"t_base_us": 8.5, "t_seed_us": 7.0, "repeats": 5, "device_id": 0}',
            encoding="utf-8")
        try:
            cfg = make_config("paper_full", "test_kernel", log=False)
            executor = harness.build_executor("test_kernel", cfg)
            assert executor.baseline_time == 8.5
            assert cfg.reference_baseline_us == 8.5
            assert cfg.reference_baseline_source == "manifest"
            assert cfg.reference_manifest_sha256

            (common.MANIFEST_DIR / "test_kernel.json").unlink()
            try:
                harness.build_executor("test_kernel", cfg)
            except RuntimeError as exc:
                assert "manifest" in str(exc)
            else:
                raise AssertionError("PAPER executor must fail closed without manifest")
        finally:
            common.MANIFEST_DIR = old_manifest_dir
            common.find_test_file = old_find_test_file
            harness.TritonExecutor = old_executor


def test_ucb_selects_distinct_parents_and_updates_counts():
    ea = _bare_ucb_ea()
    for _ in range(20):
        parent1, parent2 = ea._select_parents_ucb()
        assert parent1.id != parent2.id
    assert sum(ea._ucb_counts.values()) == 40
    assert len(ea.logger.records) == 20
    assert all(r["selection_mode"] == "ucb" for r in ea.logger.records)
    assert all(len(set(r["parent_ids"])) == 2 for r in ea.logger.records)


def test_ucb_requires_two_distinct_individuals():
    ea = _bare_ucb_ea()
    ea.population = ea.population[:1]
    try:
        ea._select_parents_ucb()
    except RuntimeError as exc:
        assert "two distinct individuals" in str(exc)
    else:
        raise AssertionError("UCB must not silently select the same parent twice")


def test_required_rag_index_fails_closed():
    with tempfile.TemporaryDirectory(prefix="paper_rag_test_") as td:
        try:
            TritonKnowledgeNode(knowledge_dir=td, require_index=True)
        except RuntimeError as exc:
            assert "RAG 索引不完整" in str(exc)
        else:
            raise AssertionError("Required RAG index should fail closed when absent")


def main():
    tests = [
        test_paper_config,
        test_paper_b1_b2_smoke_configurations,
        test_reference_manifest_validation,
        test_executor_uses_manifest_reference,
        test_ucb_selects_distinct_parents_and_updates_counts,
        test_ucb_requires_two_distinct_individuals,
        test_required_rag_index_fails_closed,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"  ❌ {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n通过 {len(tests) - len(failures)}/{len(tests)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
