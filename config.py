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
    max_llm_tokens: int = 16888          # Max tokens per call

    # Model evolution parameters
    model_switch_prob: float = 0.2         # Probability of switching model during mutation

    # ==================== Evaluation Parameters ====================
    baseline_json: str = "./baseline/baseline.json"  # Baseline data JSON file path
    timeout_seconds: int = 300             # Compilation/execution timeout
    max_iterations: int = 5                # Max fix attempts per operator

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