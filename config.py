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
    # 3.2 敏感性实验在 17 个对阈值敏感的 Kernel 上标定为 0.90
    # （GM(S)=1.2510，为各档最优；0.85~0.95 为稳健区间）
    seed_diversity_threshold: float = 0.90

    # 适应度锚点：>0 时用它替代搜索内实测的种子延迟，使搜索内 fitness 与最终 F 口径一致
    seed_anchor_time: float = 0.0

    # 训练数据采集：记录每个进入 NPU 评测的候选及其父子延迟关系，
    # 用于后续训练候选预筛选模型（3.6 节）。只落盘，不改变任何实验行为。
    collect_data: bool = True
    dataset_dir: str = "./experiments/dataset"

    # ==================== 候选预筛选（3.5/3.6） ====================
    enable_candidate_filter: bool = False   # 是否启用模型预筛选
    # 第 0 代是否也启用预筛选。默认 False，且论文 3.5.5 已按此结论撰写：
    # 离线分析表明第 0 代候选缺少 parent_latency，模型判别力接近随机（AUC≈0.5），
    # 而第 0 代同时承担策略引导与种群多样化职责。
    # 因此正式对照中第 0 代固定为「生成 5 → 全评 5」，与无预筛选基线完全一致，
    # 使三组差异只能来自进化阶段（交叉/变异）的排序决策。
    # 若后续要研究第 0 代预测筛选，请设为 True 并作为**独立消融**单独报告。
    enable_candidate_filter_gen0: bool = False
    candidate_pool_size: int = 0            # 启用预筛选时每代候选池规模（须 > children_per_generation）
    filter_evaluate_all: bool = False       # 全评诊断：候选池全部送 NPU（不过滤），用于在线 P@4/Recall
    candidate_filter_mode: str = "model"    # model=模型排序 | random=随机筛选（隔离扩池贡献）
    filter_model_path: Optional[str] = None # 显式指定预筛选模型路径；None 用默认路径
    integrity_check: bool = True            # 运行后做预算/数据划分/覆盖一致性自检

    # ==================== 候选生成质量（评审 v8.2 阶段 3/5） ====================
    # 【背景】新采集的全池数据显示：每代 20 个候选里平均只有 9.5 种不同源码，
    # 20% 的子代源码与参考父代完全相同（重复率 52.5%）。
    # 在确认"同一源码 == 同一实现配置"之前默认关闭，保持 V1 基线可复现；
    # V1/V2 对照时显式打开。
    enable_candidate_dedup: bool = False    # 是否对候选池做指纹去重 + 重新生成补位
    dedup_retries: int = 2                  # 每个重复候选最多重新生成几次
    # V2 生成算子：目标明确的变异 + 结构化交叉（与去重一起构成 V2）
    genetic_v2: bool = False

    # ---- 分级复测（评审 v8.2 决策 1）----
    # 同一源码的计时 CV 实测 2.5%~7.7%，1%~3% 的改进无法可靠区分。
    # 全部候选都复测会把 NPU 成本翻几倍，因此只对"与父代差距 ≤ margin"的
    # 候选追加 repeats 次测量并取中位数；仍在噪声内则把标签标为不确定。
    # 默认关闭：V1 基线的 NPU 调用数与评测口径必须保持可复现。
    tiered_remeasure: bool = False
    tiered_remeasure_margin: float = 0.05
    tiered_remeasure_repeats: int = 5

    # 编译可行性门（任务③'）。默认关闭：
    # 实测一次完整评测 ~16.5s（端到端 22s）中编译只占 ~2.0s，冷启动占 ~7.8s，
    # 而现有日志里整体失败率仅 0.9% —— 常开编译门的净收益接近 0，只会拖慢搜索。
    # 打开它的用途是：(a) 把"编译失败"与"正确性失败"真正分离，量化编译门的
    # 拦截率与失败类型；(b) 批量预编译后为候选池补位（提高池位利用率）。
    # 关闭时 compile_pass 记 None（未知），不记 False，避免低估编译门价值。
    pre_compile_check: bool = False

    # 池级编译门（与 pre_compile_check 是两件事，默认同样关闭）：
    #   pre_compile_check —— 挂在 evaluate 里，只对**已选中**的候选做，用于诊断
    #                        （区分编译失败 / 正确性失败）。
    #   compile_gate_pool —— 挂在"模型选 4"**之前**对整池批量预编译，用于提高池位利用率。
    # 批量化后整池成本 ≈ 7.8s 冷启动 + N × 2.0s 净编译。是否划算取决于实测拦截率，
    # 在拦截率测出来之前保持关闭，避免把算力花在一个收益未证实的环节上。
    compile_gate_pool: bool = False
    # 编译结果可能是"未知"（没触发 JIT 编译 / 批量子进程异常 / 超时）。
    # 未知**不等于**编译失败，处理策略显式可配：
    #   keep  = 保留（默认）—— 不把"没测出来"当成"没过"
    #   drop  = 一并剔除（更保守，只在确认拦截率足够高时才用）
    #   abort = 直接抛错中止本轮，交由人工排查
    compile_gate_on_unknown: str = "keep"

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