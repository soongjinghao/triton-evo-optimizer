"""3.1 主实验装配器：按方法构造 Config / Executor / EvolutionaryAlgorithm。

方法预设（严格对应 3.1 正文）：
  B1  = 进化搜索骨架 - Profiling - RAG - 策略初始化（保留 adaptive 变异与保护交叉）
  B2  = 单轨迹 LLM 迭代：无种群、无交叉，uniform 变异，仅接受更快候选
  FULL= 完整 TritonEvo-NPU
"""

import os
import json
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent

# 与 main.py 保持一致：必须在导入 langchain / transformers 之前设置
os.environ.setdefault("HF_HOME", "/workspace/user_data/Agent/RAG/.hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
os.chdir(AGENT_DIR)  # RAG/knowledge 与 baseline.json 使用相对路径

from config import EAConfig
from executor import TritonExecutor
from genetic_operators import GeneticOperators
from evolutionary_algorithm import EvolutionaryAlgorithm
from llm_interface import LLMInterface
from optimizer_agent import get_baseline_from_json

from experiments import common

METHOD_PRESETS = {
    "b1": dict(enable_profiling=False, enable_rag=False, enable_strategy_init=False,
               enable_crossover=True, mutation_mode="adaptive", selection="tournament"),
    "b2": dict(enable_profiling=False, enable_rag=False, enable_strategy_init=False,
               enable_crossover=False, mutation_mode="uniform", selection="tournament"),
    "full": dict(enable_profiling=True, enable_rag=True, enable_strategy_init=True,
                 enable_crossover=True, mutation_mode="adaptive", selection="tournament"),
}


def make_config(method: str, kernel_name: str, run_id: int = 0, seed: int = 0,
                log: bool = True, **override) -> EAConfig:
    cfg = EAConfig()
    cfg.method = method
    # b0 仅做重测，组件开关无意义，沿用 full 的默认值
    for k, v in METHOD_PRESETS.get(method, METHOD_PRESETS["full"]).items():
        setattr(cfg, k, v)

    # 统一搜索预算：N_eval = 5 + 4 + 4 = 13
    cfg.population_size = 6
    cfg.max_generations = 2
    cfg.gen0_candidates = 5
    cfg.children_per_generation = 4
    cfg.eval_budget = 13
    cfg.remeasure_repeats = 5

    cfg.random_seed = seed
    cfg.run_id = run_id
    cfg.save_code = False  # 实验期不落盘 saved_codes，避免污染历史存档

    if log:
        common.ensure_dirs()
        cfg.log_path = str(common.LOGS_DIR / f"{method}__{kernel_name}__r{run_id}.jsonl")

    for k, v in override.items():
        setattr(cfg, k, v)
    return cfg


def build_executor(kernel_name: str, config: EAConfig) -> TritonExecutor:
    baseline_time = get_baseline_from_json(config.baseline_json, kernel_name, 1)
    test_file = common.find_test_file(kernel_name)
    return TritonExecutor(
        baseline_time=baseline_time or 1.0,
        test_code_path=str(test_file),
        config=config,
        kernel_name=kernel_name,
        work_dir=common.DATASETS_DIR / kernel_name,
    )


def build_ea(kernel_name: str, method: str, run_id: int = 0, seed: int = 0,
             **override):
    config = make_config(method, kernel_name, run_id=run_id, seed=seed, **override)

    # 若该 Kernel 已完成 B0 重测，用统一的 T_seed 作为适应度锚点
    man = common.MANIFEST_DIR / f"{kernel_name}.json"
    if man.exists():
        try:
            t_seed = json.loads(man.read_text(encoding="utf-8")).get("t_seed_us")
            if t_seed:
                config.seed_anchor_time = float(t_seed)
                print(f"[harness] 适应度锚点采用 manifest T_seed = {t_seed:.4f} us")
        except Exception:
            pass

    executor = build_executor(kernel_name, config)
    llm = LLMInterface(config)
    gops = GeneticOperators(llm, config)
    ea = EvolutionaryAlgorithm(gops, executor, config)
    return ea, config
