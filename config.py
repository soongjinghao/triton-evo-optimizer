#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module - Evolutionary Algorithm Parameter Configuration

Students configure all parameters by modifying this file, including:
- Evolutionary algorithm parameters (population size, generations, etc.)
- Available large model list (supports model evolution)
- Baseline JSON file path
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EAConfig:
    """
    Evolutionary Algorithm Configuration Parameters

    Modification instructions:
    1. Directly modify the default values below
    2. Some parameters (population_size, max_generations, debug) can be overridden via command line
    3. Large model list llm_models can only be modified in this file, not supported via command line
    """

    # ==================== Evolutionary Algorithm Parameters ====================
    population_size: int = 6          # Population size
    max_generations: int = 2          # Max evolution generations

    # Genetic operation parameters
    crossover_rate: float = 0.8          # Crossover probability
    mutation_rate: float = 0.5          # Mutation probability
    elite_ratio: float = 0.34            # Elite preservation ratio

    # ==================== Large Model Configuration (Key) ====================
    # 仅配置两类模型：Flash 用于快速分析，Pro 用于复杂生成和代码修复
    flash_model: str = os.getenv("ENGINE_FLASH", "deepseek-v4-flash-ga-260731")
    pro_model: str = os.getenv("ENGINE_PRO", "deepseek-v4-pro-ga-260813")

    llm_models: List[str] = field(default_factory=lambda: [
        os.getenv("ENGINE_FLASH", "deepseek-v4-flash-ga-260731"),
        os.getenv("ENGINE_PRO", "deepseek-v4-pro-ga-260813")
    ])

    # LLM generation parameters
    llm_temperature: float = 0.2         # Generation temperature
    # 本地与云端统一取同一值；可用环境变量 MAX_LLM_TOKENS 覆盖
    max_llm_tokens: int = int(os.getenv("MAX_LLM_TOKENS", "16888"))

    # Model evolution parameters
    model_switch_prob: float = 0.2         # Probability of switching model during mutation

    # ==================== Evaluation Parameters ====================
    baseline_json: str = "./baseline/baseline.json"  # Baseline data JSON file path
    timeout_seconds: int = 300             # Compilation/execution timeout
    max_iterations: int = 5                # Max fix attempts per operator

    # ==================== 消融实验配置（3.1 主实验） ====================
    # method: b0(仅重测) | b1(无证据引导EA) | b2(单轨迹LLM迭代) | full(完整TritonEvo-NPU)
    method: str = "full"
    enable_profiling: bool = True      # Profiling 硬件证据是否注入生成提示
    enable_rag: bool = True            # 是否启用受约束 RAG 检索
    enable_strategy_init: bool = True  # 是否启用策略驱动的第 0 代初始化
    enable_crossover: bool = True      # 是否执行父代交叉（B2 关闭）
    mutation_mode: str = "adaptive"    # adaptive | uniform | aggressive
    selection: str = "tournament"      # tournament | roulette
    crossover_mode: str = "protected"  # protected | unconstrained
    rag_mode: str = "hybrid"        # hybrid(混合重排+Guard) | semantic(普通语义Top-k)

    # 统一搜索预算（3.1）
    gen0_candidates: int = 5           # 第 0 代生成的候选数
    children_per_generation: int = 4   # 每代生成的子代数
    eval_budget: int = 13              # NPU 评测预算 = 5 + 4 + 4
    remeasure_repeats: int = 5         # 阶段 A/C 串行复测次数，取中位数

    # 种子相似度去重阈值（3.2 敏感性实验；>1.0 表示不筛选）
    seed_diversity_threshold: float = 0.85

    # 适应度锚点：>0 时用它替代搜索内实测的种子延迟，使搜索内 fitness 与最终 F 口径一致
    seed_anchor_time: float = 0.0

    # 训练数据采集：记录每个进入 NPU 评测的候选及其父子延迟关系，
    # 用于后续训练候选预筛选模型（3.6 节）。只落盘，不改变任何实验行为。
    collect_data: bool = True
    dataset_dir: str = "./experiments/dataset"

    # 随机性与日志
    random_seed: int = 0
    run_id: int = 0
    log_path: Optional[str] = None     # 非空时写入 JSONL 事件日志

    # ==================== Debug Parameters ====================
    debug: bool = False                    # Whether to enable debug mode

    # ==================== Code Saving Parameters ====================
    save_code: bool = True               # Whether to save generated code (开关：是否保存生成的代码)
    save_code_dir: str = "./saved_codes"   # Directory to save generated code

    # ==================== API Configuration ====================
    # Injected by evaluation system, students do not need to modify
    api_url: Optional[str] = None
    api_key: Optional[str] = None

    def print_config(self):
        """Print current configuration"""
        print(f"[Config] Evolutionary Algorithm Configuration:")
        print(f"         Population size: {self.population_size}")
        print(f"         Max generations: {self.max_generations}")
        print(f"         Crossover rate: {self.crossover_rate}")
        print(f"         Mutation rate: {self.mutation_rate}")
        print(f"         Elite ratio: {self.elite_ratio}")
        print(f"[Config] Large Model Configuration:")
        print(f"         Flash model: {self.flash_model}")
        print(f"         Pro model: {self.pro_model}")
        print(f"         Available models: {', '.join(self.llm_models)}")
        print(f"         Model switch probability: {self.model_switch_prob}")
        print(f"[Config] Baseline Configuration:")
        print(f"         Baseline JSON: {self.baseline_json}")
        print(f"[Config] LLM Parameters:")
        print(f"         Temperature: {self.llm_temperature}")
        print(f"         Max tokens: {self.max_llm_tokens}")
        print(f"[Config] Code Saving Parameters:")
        print(f"         Save code: {self.save_code}")
        print(f"         Save directory: {self.save_code_dir}")