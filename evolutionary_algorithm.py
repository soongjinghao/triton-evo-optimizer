#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolutionary Algorithm Main Logic Module - High-Throughput & Unified Pipeline
"""

import random
import json
import time
import os
import queue
import csv
import difflib
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# 候选生成的并发上限（本地 vLLM 显存/并发有限，可用环境变量调小）
MAX_WORKERS = int(os.getenv("EA_MAX_WORKERS", "8"))

from config import EAConfig
from genetic_operators import GeneticOperators, Individual
from executor import TritonExecutor, EvaluationResult
from experiments.common import ExpLogger, set_random_seed


class EvolutionaryAlgorithm:
    """进化算法主类，负责种群管理、世代迭代和评估流程"""
    
    def __init__(self, 
                 genetic_ops: GeneticOperators,
                 executor: TritonExecutor,
                 config: EAConfig):
        """初始化进化算法，绑定遗传算子、执行器和配置，初始化种群状态"""
        self.genetic_ops = genetic_ops
        self.executor = executor
        self.config = config

        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
        self.stagnant_generations = 0
        self.last_best_fitness = 0.0
        self.seed_execution_time = 0.0

        # 3.1 主实验：搜索预算计数与事件日志
        self.eval_count = 0
        self.gen0_snapshot: List[Individual] = []  # 3.2：第 0 代种群快照，用于计算 D_G0
        self.logger = ExpLogger(config.log_path)
        self._init_dataset()
        set_random_seed(getattr(config, 'random_seed', 0))

        self._init_save_dir()

    def _init_save_dir(self):
        """初始化代码保存目录"""
        if self.config.save_code:
            save_dir = Path(self.config.save_code_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[EA] 代码保存目录已初始化: {save_dir}")

    def _save_individual_code(self, individual: Individual, prefix: str = "") -> None:
        """将个体代码保存到文件，文件名包含代数、适应度和操作类型"""
        if not self.config.save_code or not individual.code:
            return
        
        save_dir = Path(self.config.save_code_dir)
        generation = individual.generation
        fitness = individual.fitness
        operation = individual.metadata.get('operation', 'unknown')
        
        filename = f"{prefix}gen{generation}_fit{fitness:.4f}_{operation}_{uuid.uuid4().hex[:8]}.py"
        filepath = save_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(individual.code)
        
        print(f"[EA] 💾 代码已保存: {filename}")

    def _extract_profiling_context(
        self, device_id: int = 0, ind: Optional[Individual] = None
    ) -> str:
        """解析 executor 导出的 CSV 文件"""
        diag_lines = ["【硬件 Profiling 深度诊断日志】:", "- 说明：以下结论是候选优化假设，不是强制命令；只有代码契约、边界、连续性和对齐条件可证明时才能采用。"]

        official_base = getattr(self.executor, "baseline_time", 0.0)
        curr_time = (
            ind.metadata.get("execution_time", 0.0)
            if ind
            else self.seed_execution_time
        )

        if official_base > 0 and curr_time > 0:
            rel_speedup = (
                (self.seed_execution_time / curr_time)
                if self.seed_execution_time > 0
                else 1.0
            )
            base_speedup = official_base / curr_time - 1.0
            diag_lines.append(
                f"- 性能坐标: 官方基线={official_base:.2f} us | 当前耗时={curr_time:.2f} us "
                f"(相对基线:{base_speedup * 100:+.2f}% | 相对种子:{rel_speedup:.2f}x)"
            )

        try:
            possible_dirs = [
                self.executor.performance_dir
                / f"{self.executor.kernel_name}_dev_{device_id}"
                / self.executor.kernel_name,
                self.executor.performance_dir / self.executor.kernel_name,
            ]

            target_dir = next((p for p in possible_dirs if p.exists()), None)

            if target_dir:
                opprof_dirs = [
                    d
                    for d in target_dir.iterdir()
                    if d.is_dir() and d.name.startswith("OPPROF_")
                ]
                if opprof_dirs:
                    latest_dir = max(opprof_dirs, key=lambda d: d.stat().st_mtime)

                    # 2.1 OpBasicInfo.csv
                    basic_csv = latest_dir / "OpBasicInfo.csv"
                    if basic_csv.exists():
                        with open(basic_csv, "r", encoding="utf-8") as f:
                            reader = list(csv.DictReader(f))
                            if reader:
                                row = reader[0]
                                bdim = row.get("Block Dim", "NA")
                                op_type = row.get("Op Type", "vector")
                                diag_lines.append(
                                    f"- 网格结构: 算子类型={op_type} | Block Dim={bdim}"
                                )

                    # 2.2 PipeUtilization.csv
                    pipe_csv = latest_dir / "PipeUtilization.csv"
                    if pipe_csv.exists():
                        with open(pipe_csv, "r", encoding="utf-8") as f:
                            reader = list(csv.DictReader(f))
                            if reader:
                                vec_r, scalar_r, mte2_r, mte3_r = [], [], [], []
                                for r in reader:
                                    if (
                                        r.get("aiv_vec_ratio")
                                        and r["aiv_vec_ratio"] != "NA"
                                    ):
                                        vec_r.append(float(r["aiv_vec_ratio"]))
                                    if (
                                        r.get("aiv_scalar_ratio")
                                        and r["aiv_scalar_ratio"] != "NA"
                                    ):
                                        scalar_r.append(
                                            float(r["aiv_scalar_ratio"])
                                        )
                                    if (
                                        r.get("aiv_mte2_ratio")
                                        and r["aiv_mte2_ratio"] != "NA"
                                    ):
                                        mte2_r.append(float(r["aiv_mte2_ratio"]))
                                    if (
                                        r.get("aiv_mte3_ratio")
                                        and r["aiv_mte3_ratio"] != "NA"
                                    ):
                                        mte3_r.append(float(r["aiv_mte3_ratio"]))

                                avg_v = (
                                    sum(vec_r) / len(vec_r) * 100
                                    if vec_r
                                    else 0
                                )
                                avg_s = (
                                    sum(scalar_r) / len(scalar_r) * 100
                                    if scalar_r
                                    else 0
                                )
                                avg_m2 = (
                                    sum(mte2_r) / len(mte2_r) * 100
                                    if mte2_r
                                    else 0
                                )
                                avg_m3 = (
                                    sum(mte3_r) / len(mte3_r) * 100
                                    if mte3_r
                                    else 0
                                )

                                diag_lines.append(
                                    f"- 管线利用率: Vector={avg_v:.1f}%, Scalar={avg_s:.1f}%, MTE2(读)={avg_m2:.1f}%, MTE3(写)={avg_m3:.1f}%"
                                )

                                util_map = {
                                    'Scalar 标量': avg_s,
                                    'Vector 向量': avg_v,
                                    'MTE2 读访存': avg_m2,
                                    'MTE3 写访存': avg_m3,
                                }
                                top_pipe = max(util_map, key=util_map.get)
                                top_val = util_map[top_pipe]

                                actions = []
                                if avg_s > 35.0 and avg_s > avg_v:
                                    actions.append("标量占比偏高（优先检查标量地址计算、不变量外提；仅在 stride 恒为 1 可证明时简化 Stride 乘法）")
                                if avg_m2 > 35.0 or avg_m3 > 35.0:
                                    actions.append("MTE 内存搬运受限（检查访存连续性与任务打包；仅在基址/行跨度对齐可证明时使用 tl.multiple_of，仅在尾界不可能出现时删除 Mask）")
                                if avg_v > 45.0:
                                    actions.append("Vector 算术受限（评估倒数代除与 exp/log、exp2/log2 等价实现；必须保持 dtype、特殊值和误差要求）")

                                if top_val > 0:
                                    if actions:
                                        diag_lines.append(f"⚠️ 诊断结论: 当前主瓶颈为 [{top_pipe} ({top_val:.1f}%)]。" + " | ".join(actions))
                                    else:
                                        diag_lines.append(f"💡 诊断结论: 各管线负载均衡 (主导: {top_pipe} {top_val:.1f}%)，可对比测试网格收拢、独立规约域内的安全打包与 BLOCK_SIZE 微调；不得改变完整计算域。")
                                else:
                                    diag_lines.append("💡 诊断结论: Profiling 硬件计数器未捕获（极短内核），优先采用低风险参数微调、公共子表达式消除和地址不变量外提；对 Stride 简化与对齐声明必须先证明。")

                    # 2.3 MemoryUB.csv
                    ub_csv = latest_dir / "MemoryUB.csv"
                    if ub_csv.exists():
                        with open(ub_csv, "r", encoding="utf-8") as f:
                            reader = list(csv.DictReader(f))
                            if reader:
                                rb, wb = [], []
                                for r in reader:
                                    if (
                                        r.get("aiv_ub_read_bw_vector(GB/s)")
                                        and r["aiv_ub_read_bw_vector(GB/s)"]
                                        != "NA"
                                    ):
                                        rb.append(
                                            float(
                                                r["aiv_ub_read_bw_vector(GB/s)"]
                                            )
                                        )
                                    if (
                                        r.get("aiv_ub_write_bw_vector(GB/s)")
                                        and r["aiv_ub_write_bw_vector(GB/s)"]
                                        != "NA"
                                    ):
                                        wb.append(
                                            float(
                                                r[
                                                    "aiv_ub_write_bw_vector(GB/s)"
                                                ]
                                            )
                                        )
                                avg_rb = sum(rb) / len(rb) if rb else 0
                                avg_wb = sum(wb) / len(wb) if wb else 0

                                if avg_rb > 0 or avg_wb > 0:
                                    diag_lines.append(
                                        f"- UB吞吐: 向量读={avg_rb:.2f} GB/s, 向量写={avg_wb:.2f} GB/s"
                                    )

        except Exception as e:
            diag_lines.append(f"⚠️ 解析 Profiling 告警: {e}")

        return "\n".join(diag_lines)

    def _process_single_generation_task(self, task_type: str, seed_code: str = "", strategy_str: str = "", 
                                         parent1: Individual = None, parent2: Individual = None, rag_rules_text: str = "") -> Individual:
        """处理单个生成任务，支持策略生成、交叉变异等多种任务类型，并处理 AST 门禁回退"""
        # 静态门禁拒绝后重新生成补位，保证各方法的真实 NPU 评测次数一致
        max_retries = 3
        new_ind = None
        for attempt in range(max_retries):
            if task_type == 'gen0_strategy':
                new_ind = self.genetic_ops.generate_initial_individual_with_strategy(
                    baseline_code=seed_code,
                    strategy_str=strategy_str,
                    rag_rules_text=rag_rules_text
                )
                fallback_code = seed_code
            elif task_type == 'crossover_mutate':
                if (parent1 and parent2 and getattr(self.config, 'enable_crossover', True)
                        and random.random() < self.config.crossover_rate):
                    child = self.genetic_ops.crossover(parent1, parent2)
                else:
                    child = parent1 if parent1 and parent1.fitness >= getattr(parent2, 'fitness', 0.0) else parent2

                if child and child.code and random.random() < self.config.mutation_rate:
                    child = self.genetic_ops.mutate(child)

                new_ind = child
                fallback_code = parent1.code if parent1 else seed_code
            else:
                new_ind = Individual(code=seed_code, generation=self.generation)
                fallback_code = seed_code

            if new_ind and new_ind.code:
                break
            print(f"[EA] retry {attempt + 1}/{max_retries}: candidate rejected by AST gate, regenerating")
            # 每次静态门禁拒绝都记录，用于统计 AST 通过率与有效候选率
            if new_ind is not None:
                self.logger.log(self._event(new_ind, 'reject'))

        if not new_ind.code:
            err = new_ind.metadata.get('syntax_error', 'Unknown AST/Contract/Preflight error')
            print(f"[EA] 🛑 候选代码触发静态门禁，跳过真机评估！原因: {err}")
            rejected_metadata = dict(new_ind.metadata)
            rejected_metadata.update({
                'evaluated': True,
                'success': False,
                'preflight_rejected': True,
                'error': err,
            })
            new_ind = Individual(
                code="",
                generation=self.generation,
                metadata=rejected_metadata,
                model_used=getattr(new_ind, 'model_used', 'unknown'),
            )
            self.logger.log(self._event(new_ind, 'reject'))

        return new_ind

    def initialize_population(self, seed_codes: List[str]) -> None:
        """初始化种群：Stage 1 校验去重 -> Stage 2 Profiling 诊断 -> Stage 3 RAG 导流 -> Stage 4 并发生成"""
        print(f"[EA] Initializing population, target size: {self.config.population_size}")
        print(f"[EA] 🔍 [Stage 1/4] 启动种子代码双卡并发校验与去重 (种子数量: {len(seed_codes)})...")
        valid_seed_individuals: List[Individual] = []

        def eval_seed_task(args):
            """子任务：在指定 NPU 设备上评估一个种子代码"""
            i, code = args
            dev_id = i % 2  # 双设备轮询分配
            res = self.executor.evaluate(code, timeout=600, device_id=dev_id)
            return code, res, dev_id

        with ThreadPoolExecutor(max_workers=min(len(seed_codes), 2)) as pool:
            futures = [pool.submit(eval_seed_task, (i, code)) for i, code in enumerate(seed_codes)]
            for future in as_completed(futures):
                try:
                    code, res, dev_id = future.result()
                    if res.success and res.execution_time > 0:
                        ind = Individual(code=code, generation=0, metadata={
                            'operation': 'seed',
                            'evaluated': True,
                            'success': True,
                            'execution_time': res.execution_time,
                            'speedup': res.speedup,
                            'device_id': dev_id
                        })
                        prof_ctx = self._extract_profiling_context(device_id=dev_id, ind=ind)
                        ind.metadata['profiling_context'] = prof_ctx
                        valid_seed_individuals.append(ind)
                    else:
                        print(f"[EA] ❌ 种子代码未能通过功能/性能测试，已过滤。")
                except Exception as e:
                    print(f"[EA] ❌ 种子测试异常: {e}")

        if not valid_seed_individuals:
            raise RuntimeError("【致命错误】：所有传入的种子代码均未能通过正确性/性能压测！")

        valid_seed_individuals.sort(key=lambda x: x.metadata['execution_time'])

        if len(valid_seed_individuals) >= 2:
            code1 = valid_seed_individuals[0].code
            code2 = valid_seed_individuals[1].code
            sim_ratio = difflib.SequenceMatcher(None, code1, code2).ratio()
            sim_threshold = getattr(self.config, 'seed_diversity_threshold', 0.85)
            print(f"[EA] 📊 2 个合法种子代码相似度为: {sim_ratio * 100:.2f}% (阈值 τ={sim_threshold})")

            retained_before = len(valid_seed_individuals)
            if sim_ratio >= sim_threshold:
                print(f"[EA] ✂️ 种子相似度达到阈值，仅保留运行速度更快者 "
                      f"(耗时: {valid_seed_individuals[0].metadata['execution_time']:.2f}us)，丢弃重复种子！")
                valid_seed_individuals = [valid_seed_individuals[0]]

            # 3.2：记录种子筛选决策，供种子保留率与决策变化统计使用
            self.logger.log({
                "event": "seed_filter",
                "method": getattr(self.config, 'method', 'full'),
                "kernel": self.executor.kernel_name,
                "run_id": getattr(self.config, 'run_id', 0),
                "sim_ratio": round(sim_ratio, 4),
                "threshold": sim_threshold,
                "seeds_before": retained_before,
                "seeds_after": len(valid_seed_individuals),
                "filtered": len(valid_seed_individuals) < retained_before,
            })

        anchor_seed = valid_seed_individuals[0]
        _a = float(getattr(self.config, 'seed_anchor_time', 0.0) or 0.0)
        self.seed_execution_time = (_a if _a > 0 else anchor_seed.metadata['execution_time'])
        anchor_seed.fitness = 1.0
        print(f"[EA] 🏆 黄金基准种子锁定成功！ID: {anchor_seed.id}, 基准耗时: {self.seed_execution_time:.2f}us")

        self.population = list(valid_seed_individuals)
        for ind in self.population:
            self._save_individual_code(ind, prefix="seed_")
            self._save_code(ind)  # 3.6：保存种子源码，供事后计算与子代的代码差异

        # Stage 2：Profiling 证据（B1 关闭该引导信息）
        if getattr(self.config, 'enable_profiling', True):
            print(f"[EA] 🔬 [Stage 2/4] 解析黄金种子的 Profiling 硬件诊断日志...")
            profiling_context = anchor_seed.metadata.get('profiling_context', '')
            if not profiling_context:
                profiling_context = self._extract_profiling_context(device_id=anchor_seed.metadata.get('device_id', 0), ind=anchor_seed)
            print(f"\n{profiling_context}\n")
        else:
            profiling_context = ""
            print(f"[EA] ⛔ [Stage 2/4] Profiling 引导已关闭（方法要求），不注入硬件证据")

        # 固定第 0 代候选数，保证各 Kernel 的 NPU 评测预算恒为 eval_budget
        required_variants = getattr(self.config, 'gen0_candidates', 5)
        untested_candidates: List[Individual] = []

        if required_variants > 0:
            if getattr(self.config, 'enable_strategy_init', True):
                print(f"[EA] 🧠 [Stage 3/4] 结合 Profiling 诊断向 RAG 检索知识库，生成 {required_variants} 条重构策略...")
                strategies, _, rag_rules = self.genetic_ops.analyze_kernel(
                    [anchor_seed.code], required_variants, profiling_context=profiling_context
                )
            else:
                print(f"[EA] ⛔ [Stage 3/4] 策略驱动初始化已关闭，改为无引导的普通初始化")
                strategies = [None] * required_variants
                rag_rules = ""

            print(f"\n[EA] 📋 RAG 导流与 LLM 脑暴出的 {len(strategies)} 套重构策略蓝图：")
            print("=" * 80)
            for idx, strat in enumerate(strategies, 1):
                parts = (strat or "无策略蓝图（无引导初始化）").split('|')
                st_name = parts[0].strip() if len(parts) > 0 else "未命名策略"
                st_diag = parts[1].strip() if len(parts) > 1 else "无诊断"
                st_act  = parts[2].strip() if len(parts) > 2 else "无具体动作"
                print(f"  策略 [{idx}]: {st_name}")
                print(f"   ├─ 瓶颈诊断: {st_diag}")
                print(f"   └─ 实施动作: {st_act}\n")
            print("=" * 80 + "\n")

            print(f"[EA] 🚀 [Stage 4/4] 并发生成 {len(strategies)} 个高多样性 Gen 0 重构个体...")
            with ThreadPoolExecutor(max_workers=min(len(strategies), MAX_WORKERS)) as thread_pool:
                futures = [
                    thread_pool.submit(
                        self._process_single_generation_task,
                        task_type='gen0_strategy',
                        seed_code=anchor_seed.code,
                        strategy_str=strategy_str,
                        rag_rules_text=rag_rules
                    )
                    for strategy_str in strategies
                ]
                for future in as_completed(futures):
                    try:
                        untested_candidates.append(future.result())
                    except Exception as e:
                        print(f"[EA] ❌ Gen 0 并发生成异常: {e}")

            print(f"\n[EA] 🧪 策略子代汇总完毕（共 {len(untested_candidates)} 个），启动压测评估...")
            untested_candidates = self._evaluate_individuals_batch(untested_candidates)

            for ind in untested_candidates:
                self.population.append(ind)
                self._save_individual_code(ind, prefix="init_")

        # 预算恒定：种群规模超出设定时按适应度截断（保证 N_eval 不随种子数漂移）
        if len(self.population) > self.config.population_size:
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.population = self.population[:self.config.population_size]

        self.best_individual = max(self.population, key=lambda x: x.fitness)
        self.gen0_snapshot = list(self.population)  # 3.2：冻结第 0 代种群
        print(f"\n[EA] 🎉 种群初始化全部完成！初始最佳 Fitness: {self.best_individual.fitness:.4f} "
              f"(第 0 代种群规模 {len(self.gen0_snapshot)})")

    def select_parents(self) -> Tuple[Individual, Individual]:
        """父代选择：tournament（默认 3 元锦标赛）或 roulette（轮盘赌，供 3.4.1 对比）"""
        if getattr(self.config, 'selection', 'tournament') == 'roulette':
            return self._select_parents_roulette()

        tournament_size = min(3, len(self.population))
        cand1 = random.sample(self.population, tournament_size)
        parent1 = max(cand1, key=lambda x: x.fitness)
        
        cand2 = random.sample(self.population, tournament_size)
        parent2 = max(cand2, key=lambda x: x.fitness)
        
        if parent1 == parent2 and len(self.population) > 1:
            remaining = [ind for ind in self.population if ind != parent1]
            parent2 = max(random.sample(remaining, min(tournament_size, len(remaining))), key=lambda x: x.fitness)
        self._log_selection(parent1, parent2, "tournament")
        return parent1, parent2

    def _log_selection(self, parent1, parent2, mode):
        """3.4.1：记录每次父代选择的被选父代适应度，用于衡量选择压力。"""
        self.logger.log({
            "event": "selection",
            "method": getattr(self.config, 'method', 'full'),
            "kernel": self.executor.kernel_name,
            "run_id": getattr(self.config, 'run_id', 0),
            "gen": self.generation,
            "selection_mode": mode,
            "parent_ids": [parent1.id, parent2.id],
            "parent_fitness": [round(parent1.fitness, 6), round(parent2.fitness, 6)],
        })

    def _select_parents_roulette(self) -> Tuple[Individual, Individual]:
        """轮盘赌选择：按适应度比例采样两个父代"""
        weights = [max(ind.fitness, 0.0) for ind in self.population]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(self.population)
            total = float(len(self.population))

        def _pick():
            r = random.random() * total
            acc = 0.0
            for ind, w in zip(self.population, weights):
                acc += w
                if acc >= r:
                    return ind
            return self.population[-1]

        parent1 = _pick()
        parent2 = _pick()
        if parent1 == parent2 and len(self.population) > 1:
            parent2 = _pick()
        self._log_selection(parent1, parent2, "roulette")
        return parent1, parent2

    def evolve_generation(self) -> None:
        """执行一代演化：精英保留 + 并发交叉变异生成子代 + 批量评估"""
        new_population = []
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        elite_count = int(self.config.elite_ratio * self.config.population_size)
        elites = sorted_pop[:elite_count]
        new_population.extend(elites)

        # 固定每代子代数，使 N_eval = gen0_candidates + max_generations * children_per_generation
        needed_children = getattr(self.config, 'children_per_generation', 4)
        if needed_children <= 0:
            self.population = new_population
            return

        tasks_args = [self.select_parents() for _ in range(needed_children)]

        print(f"[EA] 🚀 并发发射 {needed_children} 个子代变异/交叉 LLM 请求...")
        child_candidates: List[Individual] = []

        with ThreadPoolExecutor(max_workers=min(needed_children, MAX_WORKERS)) as thread_pool:
            futures = [
                thread_pool.submit(
                    self._process_single_generation_task,
                    task_type='crossover_mutate',
                    parent1=p1,
                    parent2=p2
                )
                for p1, p2 in tasks_args
            ]

            for future in as_completed(futures):
                try:
                    child_candidates.append(future.result())
                except Exception as e:
                    print(f"[EA] ❌ 变异线程异常: {e}")

        print(f"[EA] 🧪 批量生成完毕，启动压测评估...")
        
        for child in child_candidates:
            child.generation = self.generation + 1
        
        child_candidates = self._evaluate_individuals_batch(child_candidates)
        
        for child in child_candidates:
            if not child.code:
                child.fitness = 0.0
                child.metadata['success'] = False
            
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        current_best = max(self.population, key=lambda x: x.fitness)

        if current_best.fitness > self.last_best_fitness + 1e-6:
            self.stagnant_generations = 0
            self.last_best_fitness = current_best.fitness
        else:
            self.stagnant_generations += 1

        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
            print(f"[EA] 🎉 找到更优版本! New Best Fitness: {current_best.fitness:.4f}")
            self._save_individual_code(self.best_individual, prefix="best_")

        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        print(f"[EA] Gen {self.generation} Summary: Best={current_best.fitness:.4f}, Avg={avg_fitness:.4f}")

    # ---------------- 3.6：训练数据采集 ----------------
    def _init_dataset(self):
        """初始化数据集目录。仅落盘，不参与任何搜索决策。"""
        self.dataset_enabled = bool(getattr(self.config, 'collect_data', False))
        if not self.dataset_enabled:
            return
        self.dataset_dir = Path(getattr(self.config, 'dataset_dir', './experiments/dataset'))
        (self.dataset_dir / "codes").mkdir(parents=True, exist_ok=True)
        self.samples_path = self.dataset_dir / "samples.jsonl"

    def _save_code(self, ind: Individual):
        """保存候选源码，供事后计算代码差异特征（按 candidate_id 索引）。"""
        if not getattr(self, 'dataset_enabled', False) or not ind.code:
            return
        try:
            (self.dataset_dir / "codes" / f"{ind.id}.py").write_text(ind.code, encoding="utf-8")
        except Exception:
            pass

    def _profile_metrics(self, device_id):
        """从本次评测产生的 OPPROF 中提取管线利用率与 Block Dim。

        这些是硬件侧证据特征，事后无法从覆盖式的 Profiling 目录中恢复，
        因此必须在评测当时采集。全程容错，任何异常都只返回空字典。
        """
        try:
            base = (self.executor.performance_dir
                    / f"{self.executor.kernel_name}_dev_{device_id}"
                    / self.executor.kernel_name)
            if not base.exists():
                return {}
            dirs = [p for p in base.iterdir()
                    if p.is_dir() and p.name.startswith("OPPROF_")]
            if not dirs:
                return {}
            latest = max(dirs, key=lambda p: p.stat().st_mtime)

            m = {}
            pipe = latest / "PipeUtilization.csv"
            if pipe.exists():
                rows = list(csv.DictReader(pipe.open(encoding="utf-8")))
                for key, name in [("aiv_vec_ratio", "vec"),
                                  ("aiv_scalar_ratio", "scalar"),
                                  ("aiv_mte2_ratio", "mte2"),
                                  ("aiv_mte3_ratio", "mte3")]:
                    vals = []
                    for r in rows:
                        raw = r.get(key)
                        if raw and raw != "NA":
                            try:
                                vals.append(float(raw))
                            except ValueError:
                                pass
                    if vals:
                        m[name] = round(sum(vals) / len(vals) * 100, 2)

            for c in sorted(latest.rglob("OpBasicInfo*.csv")):
                rows = list(csv.DictReader(c.open(encoding="utf-8")))
                if rows:
                    m["block_dim"] = rows[0].get("Block Dim")
                    break
            return m
        except Exception:
            return {}

    def _collect_sample(self, ind: Individual):
        """采集一条训练样本：给定父子信息，标签为『子代是否优于父代』。

        只记录成功取得延迟的候选；评测失败的候选没有标签，不进入数据集。
        """
        if not getattr(self, 'dataset_enabled', False):
            return
        md = ind.metadata or {}
        latency = md.get('execution_time')
        if not latency or latency <= 0:
            return
        # Gen0 候选的父代为锚点种子；交叉/变异子代用生成时记录的父代延迟
        parent_latency = md.get('parent_latency') or getattr(self, 'seed_execution_time', 0)
        if not parent_latency or parent_latency <= 0:
            return

        g = parent_latency / latency
        parent = md.get('parent')
        sample = {
            "kernel": self.executor.kernel_name,
            "method": getattr(self.config, 'method', 'full'),
            "run_id": getattr(self.config, 'run_id', 0),
            "tau": getattr(self.config, 'seed_diversity_threshold', None),
            "gen": ind.generation,
            "candidate_id": ind.id,
            "operation": md.get('operation'),
            "mutation_type": md.get('mutation_type'),
            "parent_ids": md.get('parents') or ([parent] if parent else []),
            "parent_latency": round(parent_latency, 4),
            "child_latency": round(latency, 4),
            "g_mut": round(g, 6),
            "label": 1 if g > 1.0 else 0,
            "applied_strategy": (md.get('applied_strategy') or "")[:200],
            "seed_time": round(getattr(self, 'seed_execution_time', 0) or 0, 4),
            "pop_size": len(self.population),
            "child_profile": self._profile_metrics(md.get('device_id', 0)),
            "model": getattr(ind, 'model_used', 'unknown'),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with open(self.samples_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _event(self, ind: Individual, event: str) -> dict:
        """构造一条 JSONL 事件记录（3.1 主实验的原始数据）。"""
        md = ind.metadata or {}
        parent = md.get('parent')
        return {
            "event": event,
            "method": getattr(self.config, 'method', 'full'),
            "kernel": self.executor.kernel_name,
            "run_id": getattr(self.config, 'run_id', 0),
            "eval_index": self.eval_count,
            "gen": ind.generation,
            "candidate_id": ind.id,
            "operation": md.get('operation'),
            "mutation_type": md.get('mutation_type'),
            "parent_ids": md.get('parents') or ([parent] if parent else []),
            "dominant_parent": md.get('dominant_parent'),
            "parent_latency": md.get('parent_latency') or md.get('dominant_latency'),
            "ast_pass": not md.get('preflight_rejected', False),
            "ast_reason": md.get('syntax_error'),
            "compile_pass": bool(md.get('success', False)),
            "correctness_pass": bool(md.get('success', False)),
            "latency_us": md.get('execution_time') if md.get('success') else None,
            "device_id": md.get('device_id'),
            "model": getattr(ind, 'model_used', 'unknown'),
            "error": md.get('error'),
        }

    def _evaluate_individuals_batch(self, individuals: List[Individual]) -> List[Individual]:
        """批量评估个体：双 NPU 并发执行，计算相对种子的加速比作为适应度"""
        task_queue = queue.Queue()
        for idx, ind in enumerate(individuals):
            if not ind.metadata.get('evaluated', False):
                if not ind.id or ind.id.startswith('hash_') or len(ind.id) != 8:
                    ind.id = uuid.uuid4().hex[:8]
                # 预算耗尽：不再占用 NPU，直接判定为无效候选
                if self.config.eval_budget > 0 and self.eval_count >= self.config.eval_budget:
                    ind.metadata.update({'evaluated': True, 'success': False,
                                         'error': 'eval_budget_exhausted'})
                    ind.fitness = 0.0
                    self.logger.log(self._event(ind, 'skipped'))
                    continue
                task_queue.put((idx, ind))
        
        results = list(individuals)
        total_tasks = task_queue.qsize()
        completed_tasks = [0]
        
        if total_tasks == 0:
            return results

        print(f"[EA] 🚀 启动双 NPU 并发评估，共 {total_tasks} 个待评估个体...")
        
        def device_worker(device_id):
            """设备工作线程：从任务队列取个体，在指定 NPU 上评估"""
            while not task_queue.empty():
                try:
                    idx, ind = task_queue.get_nowait()
                except queue.Empty:
                    break
                
                try:
                    result = self.executor.evaluate(ind.code, timeout=600, device_id=device_id)
                    self.eval_count += 1

                    ind.metadata.update({
                        'evaluated': True,
                        'success': result.success,
                        'speedup': result.speedup,
                        'execution_time': result.execution_time,
                        'error': result.error,
                        'device_id': device_id
                    })

                    if getattr(self.config, 'enable_profiling', True):
                        prof_ctx = self._extract_profiling_context(device_id=device_id, ind=ind)
                        ind.metadata['profiling_context'] = prof_ctx
                    else:
                        ind.metadata['profiling_context'] = ""

                    ind.fitness = 0.0
                    results[idx] = ind
                    self.logger.log(self._event(ind, 'eval'))
                    self._save_code(ind)
                    self._collect_sample(ind)
                except Exception as e:
                    print(f"[EA] ❌ 个体 {ind.id} 评估异常: {e}")
                    ind.metadata.update({
                        'evaluated': True,
                        'success': False,
                        'error': str(e)
                    })
                    ind.fitness = 0.0
                    results[idx] = ind
                
                task_queue.task_done()
                completed_tasks[0] += 1

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(device_worker, 0), pool.submit(device_worker, 1)]
            for future in futures:
                future.result()
        
        task_queue.join()
        
        for ind in results:
            if ind.metadata.get('evaluated', False):
                if ind.metadata.get('success', False) and ind.metadata.get('execution_time', 0) > 0:
                    baseline_speedup = ind.metadata.get('speedup', 0.0)
                    if self.seed_execution_time > 0:
                        relative_speedup = self.seed_execution_time / ind.metadata['execution_time']
                        ind.fitness = min(relative_speedup, 50.0)
                        print(f"[EA] ✅ 个体 {ind.id} ({ind.metadata.get('operation')}): "
                              f"耗时={ind.metadata['execution_time']:.2f}us, "
                              f"相对基线加速比={baseline_speedup:.4f}x, "
                              f"相对Seed加速比={relative_speedup:.4f}x, "
                              f"Fitness={ind.fitness:.4f}")
                    else:
                        ind.fitness = 1.0 if ind.metadata.get('operation') == 'seed' else 0.0
                else:
                    ind.fitness = 0.0
                    print(f"[EA] ❌ 个体 {ind.id} ({ind.metadata.get('operation')}): 评估失败, Fitness=0")
        
        return results

    def run_hill_climb(self, seed_codes: List[str]) -> Individual:
        """B2：单轨迹 LLM 迭代优化。不维护多父代种群，不执行父代交叉；
        每次生成一个候选并 NPU 评测，仅当候选有效且更快时才替换当前候选。"""
        print("[EA] 🥾 [B2] 单轨迹迭代：先评测初始种子并锁定起点（种子评测不计入预算）...")
        valid = []
        for i, code in enumerate(seed_codes):
            res = self.executor.evaluate(code, timeout=600, device_id=i % 2)
            if res.success and res.execution_time > 0:
                ind = Individual(code=code, generation=0, metadata={
                    'operation': 'seed', 'evaluated': True, 'success': True,
                    'execution_time': res.execution_time, 'speedup': res.speedup,
                    'device_id': i % 2,
                })
                if not getattr(self.config, 'enable_profiling', True):
                    ind.metadata['profiling_context'] = ""
                valid.append(ind)
            else:
                print(f"[EA] ❌ [B2] 种子 {i} 未通过评测，已过滤")

        if not valid:
            raise RuntimeError("【致命错误】B2：所有初始种子均未通过评测")

        valid.sort(key=lambda x: x.metadata['execution_time'])
        current = valid[0]
        _a = float(getattr(self.config, 'seed_anchor_time', 0.0) or 0.0)
        self.seed_execution_time = (_a if _a > 0 else current.metadata['execution_time'])
        current.fitness = 1.0
        self.population = [current]
        self.best_individual = current
        self._save_code(current)
        print(f"[EA] 🏁 [B2] 起点锁定: {current.id}, 耗时={self.seed_execution_time:.2f}us")

        for step in range(self.config.eval_budget):
            if self.eval_count >= self.config.eval_budget:
                break

            child = None
            for attempt in range(3):  # 静态门禁拒绝后重新生成补位
                child = self.genetic_ops.mutate(current)
                child.generation = step + 1
                if child.code:
                    break
                print(f"[EA] \u267b\ufe0f [B2] \u5019\u9009\u672a\u901a\u8fc7\u9759\u6001\u95e8\u7981\uff0c\u91cd\u65b0\u751f\u6210 ({attempt + 1}/3)")
                self.logger.log(self._event(child, 'reject'))  # 每次拒绝都记录
            if not child.code:
                continue

            child = self._evaluate_individuals_batch([child])[0]
            if child.metadata.get('success') and child.metadata.get('execution_time', 0) > 0:
                t_child = child.metadata['execution_time']
                t_cur = current.metadata.get('execution_time', float('inf'))
                if t_child < t_cur:
                    print(f"[EA] ⬆️ [B2] step {step + 1}: 接受更快候选 {child.id} "
                          f"({t_cur:.2f}us -> {t_child:.2f}us)")
                    current = child
                else:
                    print(f"[EA] ➖ [B2] step {step + 1}: 候选 {child.id} 未更快，保留当前候选")

            if self.best_individual is None or current.fitness > self.best_individual.fitness:
                self.best_individual = current

        self.generation = self.config.eval_budget
        print(f"[EA] ✅ [B2] 迭代结束，Best Fitness: {self.best_individual.fitness:.4f}")
        return self.best_individual

    def run(self, seed_codes: List[str]) -> Individual:
        """运行完整的进化流程：初始化种群 -> 迭代演化 -> 返回最优个体"""
        if getattr(self.config, 'method', 'full') == 'b2':
            return self.run_hill_climb(seed_codes)

        self.initialize_population(seed_codes)
        for gen in range(self.config.max_generations):
            # 预算耗尽时停止后续世代，避免空转消耗 LLM 预算
            if (self.config.eval_budget > 0
                    and self.eval_count >= self.config.eval_budget):
                print(f"[EA] ⏹ NPU 评测预算已耗尽 (N_eval={self.eval_count})，停止后续世代")
                break
            print(f"\n[EA] ===== Generation {gen + 1}/{self.config.max_generations} =====")
            self.evolve_generation()

        print(f"\n[EA] 所有代演化完成。Ultimate Best Fitness: {self.best_individual.fitness:.4f}")
        return self.best_individual