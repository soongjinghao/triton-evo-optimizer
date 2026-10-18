#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolutionary Algorithm Main Logic Module - High-Throughput & Unified Pipeline
"""

import random
import json
import math
import time
import os
import queue
import csv
import hashlib
import threading
import difflib
import re
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# 候选生成的并发上限（本地 vLLM 显存/并发有限，可用环境变量调小）
MAX_WORKERS = int(os.getenv("EA_MAX_WORKERS", "8"))


# ============================================================================
# 父代基线延迟解析（候选预筛选的"任务 0"修复）
# ----------------------------------------------------------------------------
# 背景（2026-10-17 用 5 算子 fulleval 的 200 个 gen>=1 候选实测）：
#   * 37.5%（75/200）的候选拿不到父代实测延迟，原因是主干父代个体自身没有被
#     评测过（execution_time 为 None），或交叉只写了 dominant_latency 而没写
#     parent_latency。
#   * 旧实现在拿不到时静默回退到 seed_execution_time，于是 improved 标签的含义
#     从"比父代快"偷偷变成"比种子快"——对 gen2 候选这个标准宽松得多；同时
#     _filter_features 的 log_parent 特征被填成 NaN，而训练快照里 parent_latency
#     的 NaN 比例是 0%，形成训练/推理分布不一致。
#
# 修复策略不是"猜一个值"，而是给出一条**可审计的回退链**，每一步都留痕，
# 让下游分析可以只信最可信的两档：
#   parent / dominant —— 直接父代的实测延迟（唯一可用于 improved 标签的档位）
#   ancestor          —— 父代未实测，沿父链上溯到最近一个有实测延迟的祖先
#   pop_best          —— 父链全未实测，退到当前种群已评测个体的最优（最小）延迟
#   seed              —— 只剩种子基准
#   none              —— 完全不可用（improved 不可当作 False，应记为未知）
# ============================================================================
PARENT_LATENCY_SOURCES = ("parent", "dominant", "backbone", "ancestor",
                          "pop_best", "seed", "none")
# 这三档的父代基线都是"某个真实个体的实测延迟"，可用于 improved 标签与训练样本：
#   parent   —— 变异的直接父代实测值
#   dominant —— 交叉的主干（骨架）父代实测值
#   backbone —— "交叉 → 变异"链上继承下来的主干父代实测值
#               （中间产物未评测时，孙代仍能拿到被评测过的那个真实个体）
TRUSTED_PARENT_SOURCES = ("parent", "dominant", "backbone")


def _positive_float(v):
    """把任意值转成正的有限 float；失败返回 None（NaN/None/0 都视为不可用）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _measured_latency(ind):
    """取个体的实测延迟；未评测/评测失败返回 None（不拿种子值冒充）。"""
    md = getattr(ind, "metadata", None) or {}
    if not md.get("success", False):
        return None
    return _positive_float(md.get("execution_time"))


def resolve_parent_latency(child, id2ind=None, seed_time=None, max_depth=4):
    """解析候选的父代基线延迟，返回 (value, source)。

    Args:
        child: 候选 Individual（读其 metadata 的 parent/parents/parent_latency/
            dominant_latency）
        id2ind: {candidate_id: Individual}，用于沿父链上溯与种群最优回退；
            传 None 时跳过 ancestor / pop_best 两档。
        seed_time: 种子基准延迟，作为倒数第二档回退。
        max_depth: 父链上溯的最大层数，防止极端情况下的深递归。

    Returns:
        (value, source)：value 为 None 时 source == "none"。
    """
    md = getattr(child, "metadata", None) or {}

    # 第 1 档：直接父代实测延迟（交叉写 dominant_latency，变异写 parent_latency）
    for key, src in (("parent_latency", "parent"), ("dominant_latency", "dominant")):
        v = _positive_float(md.get(key))
        if v is not None:
            return v, src

    # 第 2 档：主干父代（"交叉 → 变异"链继承下来的被评测个体）。
    # 中间产物未评测时这是唯一的真实基线，比退到 pop_best 精确得多。
    v = _positive_float(md.get("backbone_latency"))
    if v is not None and md.get("backbone_id"):
        return v, "backbone"

    # 第 3 档：沿父链上溯，找最近一个被实测过的祖先
    id2ind = id2ind or {}
    if id2ind:
        frontier, seen, depth = [], set(), 0
        p = md.get("parent")
        if p:
            frontier.append(str(p))
        for q in (md.get("parents") or []):
            frontier.append(str(q))
        # 主干/主干父代也要进搜索队列，否则"交叉 → 变异"链断在这里
        for q in (md.get("backbone_id"), md.get("dominant_parent")):
            if q:
                frontier.append(str(q))
        while frontier and depth < max_depth:
            nxt = []
            for pid in frontier:
                if pid in seen:
                    continue
                seen.add(pid)
                ind = id2ind.get(pid)
                if ind is None:
                    continue
                v = _measured_latency(ind)
                if v is not None:
                    return v, "ancestor"
                imd = getattr(ind, "metadata", None) or {}
                ip = imd.get("parent")
                if ip:
                    nxt.append(str(ip))
                for q in (imd.get("parents") or []):
                    nxt.append(str(q))
                # 中间产物上的主干父代也要继续上溯，
                # 否则 A →(交叉) C →(变异) D 这条链在 C 处断掉
                for q in (imd.get("backbone_id"), imd.get("dominant_parent")):
                    if q:
                        nxt.append(str(q))
            frontier = nxt
            depth += 1

        # 第 3 档：当前种群里已评测个体的最优（最小）延迟。
        # 这是一个"偏严"的基线（要超过种群最优才算改进），比"退到种子"更保守，
        # 也避免把大量候选错误地标成 improved。
        best = None
        for ind in id2ind.values():
            v = _measured_latency(ind)
            if v is not None and (best is None or v < best):
                best = v
        if best is not None:
            return best, "pop_best"

    # 第 4 档：种子基准
    st = _positive_float(seed_time)
    if st is not None:
        return st, "seed"

    return None, "none"


def _resolve_num_devices(default: int = 2) -> int:
    """按 EA_NUM_DEVICES 与真实可用 NPU 数量取较小值。

    某些容器/物理机实际只有 1 张可用 NPU（如仅 /dev/davinci1 存在、被 CANN 映射为
    逻辑设备 0），此时仍按默认 2 把评测轮询到 device 1，会因 aclInit
    'Invalid device ID'(107001) 而整批失败，且种子/子代评测会被静默过滤。
    以 torch.npu.device_count() 为上界，从根本上避免把任务调度到不存在的卡上。
    """
    raw = int(os.getenv("EA_NUM_DEVICES", "0") or "0")
    requested = raw if raw > 0 else default
    try:
        import torch_npu  # noqa: F401  触发 NPU 后端注册
        import torch
        cnt = int(torch.npu.device_count())
    except Exception:
        cnt = 0
    n = min(requested, cnt) if cnt > 0 else requested
    n = max(1, n)  # 至少 1 张，否则整个流程无法做真机评测
    if cnt > 0 and n != requested:
        print(f"[EA] ⚠️ 检测到可用 NPU 仅 {cnt} 张，EA_NUM_DEVICES 由 {requested} 收敛为 {n}")
    return n


# 真机评测并发只受芯片数约束，与 LLM 生成并发(EA_MAX_WORKERS)无关。
# 默认 2；实际设备数由 _resolve_num_devices 按真实 NPU 数量收敛（<= 物理卡数）。
NUM_DEVICES = _resolve_num_devices(2)

from config import EAConfig
# 候选预筛选特征：训练侧与在线侧共用的**唯一**实现（详见该模块文档）
from experiments.candidate_filter_features_shared import (
    RE_BLOCK, RE_WARPS, RE_STAGES, RE_TL_CALL, RE_MASK, RE_TOK,
    struct_feats as shared_struct_feats,
    ratio as shared_ratio,
    jaccard as shared_jaccard,
    compute_code_features,
    normalize_pids,
)
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

        # 3.4.1：UCB 选择的“历史被选次数”计数（个体 id -> 次数），用于探索项
        self._ucb_counts = {}

        # 3.1 主实验：搜索预算计数与事件日志
        self.eval_count = 0
        # 【修改】已**发起**的设备评测尝试次数。与 eval_count 的区别：
        # 设备调用已发出但抛异常（编译/运行/超时）时，eval_attempts 计数而 eval_count 不计数。
        # 预算必须按 eval_attempts 控制——异常的那次同样占用了真机资源，
        # 否则下一代会算出偏大的剩余额度，安排超出预算的额外评测。
        self.eval_attempts = 0
        # ---- 3.5/3.6 审计与完整性检查 ----
        self.stats = {
            "npu_calls": 0,             # = eval_attempts：实际设备评测次数（含异常）
            "npu_calls_completed": 0,   # = eval_count：成功返回的次数
            "static_rejected": 0, "ast_rejected": 0,
            "compile_failed": 0, "llm_tokens": None,
            "filter_model_path": getattr(config, "filter_model_path", None),
            "candidate_filter_mode": getattr(config, "candidate_filter_mode", "model"),
        }
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
                performed = False
                if (parent1 and parent2 and getattr(self.config, 'enable_crossover', True)
                        and random.random() < self.config.crossover_rate):
                    child = self.genetic_ops.crossover(parent1, parent2)
                    performed = True
                else:
                    child = parent1 if parent1 and parent1.fitness >= getattr(parent2, 'fitness', 0.0) else parent2

                # 未交叉时必须变异：否则子代是父代的纯副本，会在去重时被剔除，
                # 白白占用一个预算名额，导致各方法实际 NPU 评测次数不一致。
                if child and child.code and (random.random() < self.config.mutation_rate or not performed):
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
        print(f"[EA] 🔍 [Stage 1/4] 启动种子代码 {NUM_DEVICES} 卡并发校验与去重 (种子数量: {len(seed_codes)})...")
        valid_seed_individuals: List[Individual] = []

        def eval_seed_task(args):
            """子任务：在指定 NPU 设备上评估一个种子代码"""
            i, code = args
            dev_id = i % NUM_DEVICES  # 按实际卡数轮询分配
            res = self.executor.evaluate(code, timeout=600, device_id=dev_id)
            return code, res, dev_id

        with ThreadPoolExecutor(max_workers=min(len(seed_codes), NUM_DEVICES)) as pool:
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

        # 固定第 0 代候选数，保证各 Kernel 的 NPU 评测预算恒为 eval_budget。
        #
        # 【修改】第 0 代默认**不启用**候选预筛选，固定「生成 required_variants 个 → 全部送评测」。
        # 旧实现是「生成 pool_size(20) 个 → 模型/随机挑 5 个送评测」，与论文 3.5.5 的结论冲突：
        # 论文明确写道「第 0 代接近随机，因此集成时仅对变异与交叉阶段启用预筛选」。
        # 且第 0 代候选没有 parent_latency（_filter_features 会填 NaN），模型在该子集上
        # 无判别力，用它筛掉候选只会引入额外方差，还会让三组对照在第 0 代就不一致。
        # 现在三组（baseline / predictor / random）的第 0 代完全相同，
        # 差异只能来自进化阶段的排序决策，归因才干净。
        required_variants = getattr(self.config, 'gen0_candidates', 5)
        pool_size = int(getattr(self.config, 'candidate_pool_size', 0) or 0)
        use_filter = (bool(getattr(self.config, 'enable_candidate_filter', False))
                      and bool(getattr(self.config, 'enable_candidate_filter_gen0', False))
                      and pool_size > required_variants)
        n_gen = pool_size if use_filter else required_variants
        if use_filter:
            print(f"[EA] ⚠️ 第 0 代已开启预筛选（非默认）：{n_gen} → {required_variants}")
        untested_candidates: List[Individual] = []

        if required_variants > 0:
            if getattr(self.config, 'enable_strategy_init', True):
                print(f"[EA] 🧠 [Stage 3/4] 结合 Profiling 诊断向 RAG 检索知识库，生成 {n_gen} 条重构策略...")
                strategies, _, rag_rules = self.genetic_ops.analyze_kernel(
                    [anchor_seed.code], n_gen, profiling_context=profiling_context
                )
            else:
                print(f"[EA] ⛔ [Stage 3/4] 策略驱动初始化已关闭，改为无引导的普通初始化")
                strategies = [None] * n_gen
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

            # 3.6：默认不生效（enable_candidate_filter_gen0=False）。
            # 仅当显式为第 0 代开启预筛选时才走这里：先生成大池，再由模型/随机挑出
            # required_variants 个送评测（fulleval 下 filter_evaluate_all=True 保留全部、不过滤）。
            if use_filter:
                do_select = not bool(getattr(self.config, 'filter_evaluate_all', False))
                mode = getattr(self.config, 'candidate_filter_mode', 'model')
                if mode == 'random':
                    untested_candidates = self._random_select(
                        untested_candidates, required_variants, do_select=do_select)
                else:
                    untested_candidates = self._filter_select(
                        untested_candidates, required_variants, do_select=do_select)

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
        """父代选择（3.4.1 消融，共四种策略）：

        - tournament：三元锦标赛，局部竞争取优（默认）
        - roulette  ：轮盘赌，按适应度比例加权采样（强选择压力）
        - uniform   ：均匀随机，完全不给选择压力（下界基线）
        - ucb       ：UCB 式探索-利用平衡选择（本文提出）
        """
        mode = str(getattr(self.config, 'selection', 'tournament') or 'tournament').lower()
        if mode == 'roulette':
            return self._select_parents_roulette()
        if mode == 'uniform':
            return self._select_parents_uniform()
        if mode == 'ucb':
            return self._select_parents_ucb()

        # 默认：三元锦标赛
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

    def _select_parents_uniform(self) -> Tuple[Individual, Individual]:
        """3.4.1：均匀随机选择——完全不施加选择压力，作为下界基线。

        作用是回答“父代选择本身是否带来收益”：若各策略均优于 uniform，
        则说明选择压力确实有效；策略之间的差异则可进一步归因于压力强度。
        """
        pop = self.population
        parent1 = random.choice(pop)
        parent2 = random.choice(pop)
        if parent1 == parent2 and len(pop) > 1:
            parent2 = random.choice([ind for ind in pop if ind != parent1])
        self._log_selection(parent1, parent2, "uniform")
        return parent1, parent2

    def _select_parents_ucb(self) -> Tuple[Individual, Individual]:
        """3.4.1：UCB 式探索-利用平衡选择（本文提出）。

        在选择权重中同时考虑归一化适应度（利用）与“尚未被充分采样”的奖励项（探索）：

            score_i = f_i / f_max + c * sqrt( ln(N + 2) / (1 + n_i) )

        其中 n_i 为该个体历史被选中的次数，N 为累计选择次数，c 为探索系数。
        轮盘赌只按适应度加权，容易把有限的评测预算过度集中在少数高适应度父代上；
        uniform 则完全丢弃适应度信息。UCB 通过 n_i 显式补偿“尚未被充分尝试”的个体，
        在两者之间取得平衡。采样仍按 score 加权随机进行，以保留种群随机性。
        """
        pop = self.population
        fmax = max((max(ind.fitness, 0.0) for ind in pop), default=0.0)
        if fmax <= 0:
            # 全部个体均无有效适应度，退化为均匀随机
            return self._select_parents_uniform()

        c = float(getattr(self.config, 'ucb_c', 0.5) or 0.5)
        counts = self._ucb_counts
        n_total = sum(counts.values())

        scores = []
        for ind in pop:
            f_norm = max(ind.fitness, 0.0) / fmax
            bonus = c * float(np.sqrt(np.log(n_total + 2.0) / (1.0 + counts.get(ind.id, 0))))
            scores.append(max(f_norm + bonus, 0.0))

        def _pick():
            total = sum(scores)
            if total <= 0:
                return random.choice(pop)
            r = random.random() * total
            acc = 0.0
            for ind, s in zip(pop, scores):
                acc += s
                if acc >= r:
                    return ind
            return pop[-1]

        parent1 = _pick()
        parent2 = _pick()
        if parent1 == parent2 and len(pop) > 1:
            parent2 = _pick()

        # 更新被选次数，供后续各代的探索项使用
        for p in (parent1, parent2):
            counts[p.id] = counts.get(p.id, 0) + 1

        self._log_selection(parent1, parent2, "ucb")
        return parent1, parent2

    # ---------------------------------------------------------------- 3.6 预筛选
    _FILTER_MODEL = None

    _FILTER_MODELS = {}  # path -> loaded model（按路径缓存，避免不同实验误用同一模型）

    @classmethod
    def _load_filter_model(cls, path=None):
        """惰性加载 3.6 的候选预筛选模型（仅数值特征的逻辑回归）。

        path 为 None 时使用默认路径；否则按显式路径加载并独立缓存，
        确保 random / predictor 等不同划分的模型不会相互覆盖。
        """
        if path is None:
            path = Path(__file__).resolve().parent / "experiments" / "dataset" / \
                "candidate_filter_model.joblib"
        path = str(path)
        if path not in cls._FILTER_MODELS:
            import joblib
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(
                    f"未找到预筛选模型 {path}；请先用 "
                    f"experiments/candidate_filter_model_final.py 训练并指定 --model-out")
            cls._FILTER_MODELS[path] = joblib.load(p)
            print(f"[EA] 🔧 已加载预筛选模型 {path}")
        return cls._FILTER_MODELS[path]

    # ---- 3.6 预筛选：父子代码相对特征 ----
    # 【统一】原先这里自带一份 _RE_* / _code_struct / _ratio / _jaccard，
    # 而训练脚本 candidate_filter_model_final.py 用的是 v6 模块的另一份实现。
    # 两份代码**当前**恰好等价，但属于人工保持的一致性 —— 任一侧改动都会静默造成
    # 训练/推理分布分叉，而这种分叉离线评估看不出来。
    # 现在都改为引用唯一实现 experiments.candidate_filter_features_shared，
    # 并由 experiments/tests/test_feature_parity.py 逐字段断言两者一致。
    _RE_BLOCK = RE_BLOCK
    _RE_WARPS = RE_WARPS
    _RE_STAGES = RE_STAGES
    _RE_TL_CALL = RE_TL_CALL
    _RE_MASK = RE_MASK
    _RE_TOK = RE_TOK

    @staticmethod
    def _code_struct(code):
        return shared_struct_feats(code)

    @staticmethod
    def _ratio(c, p):
        return shared_ratio(c, p)

    @staticmethod
    def _jaccard(a, b):
        return shared_jaccard(a, b)

    @staticmethod
    def _filter_features(child, config, id2code=None, id2ind=None, seed_time=None):
        """构造一条预筛选输入（全部评测前可得，与训练时口径一致）。

        【修复 1】父代基线延迟改用 resolve_parent_latency 的回退链解析。
        旧实现 `md.get("parent_latency") or md.get("dominant_latency")` 在父代未被
        评测时直接为 None，log_parent 被填成 NaN；而训练快照里 parent_latency 的
        NaN 比例是 0% —— 这是训练/推理分布不一致的一个直接来源。

        【修复 2】父代代码不再只认 md["parents"]。
        变异子代只在 metadata 里写 `parent`（字符串），不写 `parents`，于是旧实现的
        pids 恒为空，pc_code 回退成子代自身，r_block/r_warps/r_stages/r_ops/r_mask
        全部退化为 0.0、sim 退化成 1.0。而训练侧
        （candidate_filter_model_final.py:add_pc_features）读的是 JSONL 的
        parent_ids，变异样本在那里是 `[parent]`，拿到的是**真实**父子比值。
        即：训练时这对特征有信息，在线时整对特征恒为常数 —— 第二个训练/推理不一致。
        这里与训练口径对齐：parents 缺失时回退到 [md["parent"]]。
        """
        md = child.metadata or {}
        p_lat, p_src = resolve_parent_latency(child, id2ind=id2ind, seed_time=seed_time)
        ch_code = child.code or ""
        ch = EvolutionaryAlgorithm._code_struct(ch_code)
        # 与训练口径一致：parents 缺失时回退到 md["parent"]（变异子代）
        pids = md.get("parents") or ([md["parent"]] if md.get("parent") else [])
        # 【修复】"交叉 → 变异"链上，直接前驱 C 未评测、其代码也不在种群索引里，
        # 此时必须显式用 backbone_code_id（那次交叉的 dominant）取父代代码；
        # 否则 pc_code 会回退成子代自身，整组比值特征退化为常数、sim 退化为 1.0。
        pc_ids = ([str(md["backbone_code_id"])] if md.get("backbone_code_id") else []) \
            or [str(pids[0])] if pids else []
        pc_code = None
        for pid in pc_ids:
            if id2code is not None and id2code.get(pid):
                pc_code = id2code[pid]
                break
        if pc_code is None:
            pc_code = ch_code  # 无父代代码：比值=0、sim=1（与训练侧 gen0 分支一致）
        # 8 个父子代码特征由**唯一实现**产出，与训练侧逐字段一致
        cf = compute_code_features(ch_code, pc_code)
        return {
            "log_parent": float(np.log1p(p_lat)) if p_lat and p_lat > 0 else np.nan,
            "parent_latency": float(p_lat) if p_lat and p_lat > 0 else np.nan,
            "parent_latency_source": p_src,
            "gen": float(child.generation),
            "pop_size": float(getattr(config, "population_size", 6)),
            "tau": float(getattr(config, "seed_diversity_threshold", -1.0)),
            "operation": md.get("operation") or "none",
            "mutation_type": md.get("mutation_type") or "none",
            "r_block": cf["r_block"],
            "r_warps": cf["r_warps"],
            "r_stages": cf["r_stages"],
            "sim": cf["sim"],
            "r_ops": cf["r_ops"],
            "r_mask": cf["r_mask"],
            "ops_abs": cf["ops_abs"],
            "block_abs": cf["block_abs"],
        }

    def _apply_compile_gate(self, candidates):
        """在模型/随机筛选**之前**对整池做批量预编译，剔除编译失败的候选。

        与 `pre_compile_check`（挂在 evaluate 里、只对已选中候选做）是两件事：
          * pre_compile_check  → 目的：区分"编译失败"与"正确性失败"，用于诊断
          * compile_gate_pool  → 目的：提高池位利用率（把编译不过的候选踢掉，
                                 让筛选面对的是一个"全部可编译"的池）

        默认关闭：净编译虽只有 ~2s，但冷启动 ~7.8s；批量化后整池成本
        ≈ 7.8 + N × 2.0s，是否划算要看实测拦截率（现有历史数据只有 0.9%），
        因此在拦截率测出来之前不自动打开。
        """
        if not getattr(self.config, 'compile_gate_pool', False):
            return candidates
        if not candidates:
            return candidates
        if not hasattr(self.executor, 'compile_check_batch'):
            print("[EA] ⚠️ 执行器不支持批量编译，跳过编译门")
            return candidates

        device_id = 0
        on_unknown = str(getattr(self.config, 'compile_gate_on_unknown', 'keep')
                         or 'keep').lower()
        t0 = time.time()
        try:
            infos = self.executor.compile_check_batch(
                [c.code for c in candidates], device_id=device_id)
        except Exception as e:
            print(f"[EA] ⚠️ 批量编译异常，跳过编译门: {e}")
            return candidates
        wall = time.time() - t0
        if len(infos) != len(candidates):
            print("[EA] ⚠️ 批量编译结果数量不匹配，跳过编译门")
            return candidates

        keep, dropped, unknown = [], [], []
        for c, info in zip(candidates, infos):
            status = info.get('status') or (
                'pass' if info.get('compile_pass') is True else
                'unknown' if info.get('compile_pass') is None else 'fail')
            c.metadata['compile_pass'] = info.get('compile_pass')
            c.metadata['compile_time_ms'] = info.get('compile_time_ms')
            c.metadata['compile_error_type'] = info.get('error_type')
            if status == 'fail':
                dropped.append({
                    "candidate_id": c.id,
                    "operation": c.metadata.get('operation'),
                    "mutation_type": c.metadata.get('mutation_type'),
                    "error_type": info.get('error_type'),
                    "error_message": (info.get('error_message') or "")[:300],
                })
                continue
            if status == 'unknown':
                # 未确认不能编译 —— 按 compile_gate_on_unknown 处理，默认**保留**，
                # 绝不能把"没测出来"当成"编译失败"直接过滤掉。
                unknown.append({
                    "candidate_id": c.id,
                    "error_type": info.get('error_type'),
                })
                if on_unknown == 'drop':
                    continue
            keep.append(c)

        self.stats['compile_gate'] = {
            "enabled": True,
            "pool_in": len(candidates),
            "pool_out": len(keep),
            "confirmed_fail": len(dropped),
            "unknown": len(unknown),
            "on_unknown": on_unknown,
            "wall_s": round(wall, 1),
            "per_candidate_s": round(wall / max(1, len(candidates)), 2),
        }
        self.stats['compile_gate_dropped'] = \
            self.stats.get('compile_gate_dropped', 0) + len(dropped)
        self.stats['compile_gate_unknown'] = \
            self.stats.get('compile_gate_unknown', 0) + len(unknown)
        for d in dropped:
            self.logger.log(self._event(
                type("D", (), {"id": d["candidate_id"], "generation": self.generation + 1,
                               "metadata": {"operation": d["operation"],
                                            "mutation_type": d["mutation_type"],
                                            "compile_pass": False,
                                            "compile_error_type": d["error_type"],
                                            "preflight_rejected": False},
                               "code": ""})(), 'compile_reject'))

        print(f"[EA] 🛂 编译门：池 {len(candidates)} → {len(keep)}"
              f"（确认编译失败剔除 {len(dropped)}，未知保留 {len(unknown)}，"
              f"耗时 {wall:.1f}s，折合 {wall / max(1, len(candidates)):.2f}s/候选）")
        return keep

    def _filter_select(self, candidates, k, do_select=True):
        """用预筛选模型给候选池打分；默认返回得分最高的 k 个，同时记录全部打分。

        do_select=False 用于"全评子实验"：不过滤，把候选池全部送评测，
        这样被模型排在后面的候选也有真值，事后即可算出在线精确率与召回率。
        """
        import pandas as pd

        model = self._load_filter_model(getattr(self.config, "filter_model_path", None))
        self.stats["filter_model_path"] = str(
            getattr(self.config, "filter_model_path", None)
            or (Path(__file__).resolve().parent / "experiments" / "dataset"
                / "candidate_filter_model.joblib"))
        cols = ["log_parent", "gen", "pop_size", "tau", "r_block", "r_warps",
                "r_stages", "sim", "r_ops", "r_mask", "ops_abs", "block_abs",
                "operation", "mutation_type"]
        # 父子代码映射：父代代码取 parent_ids[0]，与训练口径一致
        id2code = {str(getattr(ind, "id", "")): (getattr(ind, "code", "") or "")
                   for ind in getattr(self, "population", [])}
        id2ind = {str(getattr(ind, "id", "")): ind
                  for ind in getattr(self, "population", [])}
        rows = [self._filter_features(c, self.config, id2code, id2ind,
                                      getattr(self, "seed_execution_time", 0))
                for c in candidates]
        X = pd.DataFrame(rows)[cols]
        scores = model.predict_proba(X)[:, 1]

        # 并列时按 candidate_id 兜底，保证同一次实验可复现（np.argsort 默认快排不稳定）
        order = sorted(range(len(candidates)),
                       key=lambda i: (-float(scores[i]), candidates[i].id))[:k]
        keep = set(int(i) for i in order)
        selected = [candidates[int(i)] for i in order]

        # 打分与选择记录：事后可算出"被选中的候选中真正改进的比例"，
        # 配合"全评子实验"还能得到被拒绝候选的真值，从而算在线精确率/召回率。
        # 【修改】gen 口径与 _log_eval 对齐：直接用候选自身的 generation（Gen0=0 / Gen1=1 / Gen2=2）。
        # 旧实现写 self.generation + 1，而 self.generation 在 evolve_generation 末尾才自增，
        # 于是 Gen0 与 Gen1 的筛选记录都被写成 gen=1；而同一批候选的真值日志（_log_eval）
        # 用 ind.generation 记录为 0 和 1。两者按 gen 分组时对不上，
        # 「代级改进率」这类按代聚合的指标会算错（Gen0 与 Gen1 混在一起）。
        self._log_filter([{
            "kernel": getattr(self.executor, "kernel_name", ""),
            "run_id": getattr(self.config, "run_id", 0),
            "gen": int(getattr(candidates[i], "generation", self.generation)),
            "candidate_id": candidates[i].id,
            "operation": rows[i]["operation"],
            "mutation_type": rows[i]["mutation_type"],
            "parent_latency": rows[i]["parent_latency"],
            "parent_latency_source": rows[i].get("parent_latency_source"),
            # 【新增】主干父代与变异分级的审计字段。
            # 评审要求"让离线训练和在线预测使用同一套父子特征"必须可验证 ——
            # 光在代码里改了不算数，得能在日志里看到实际用了哪一档。
            # 旧日志里没有这些字段，正是 v5 之前无法证明父代传递是否修好的原因。
            "backbone_id": candidates[i].metadata.get("backbone_id"),
            "parent_latency_source_chain": candidates[i].metadata.get("lineage"),
            "mutation_tier_source": candidates[i].metadata.get("mutation_tier_source"),
            "mutation_tier_speedup": candidates[i].metadata.get("mutation_tier_speedup"),
            "score": float(scores[i]),
            "selected": bool(i in keep),
        } for i in range(len(candidates))])

        # 记一档"父代基线可信度"：只有 parent/dominant 档的 improved 标签才等价于
        # "比父代快"，其余档位是回退值，事后分析必须能看出来（详见
        # experiments/analyze_candidate_distribution.py 的 --strict-source）。
        src_counter = {}
        for r in rows:
            s = r.get("parent_latency_source") or "none"
            src_counter[s] = src_counter.get(s, 0) + 1
        self.stats["parent_latency_sources"] = src_counter

        print(f"[EA] 🔍 预筛选：候选池 {len(candidates)} → 送评测 {len(selected)} "
              f"（最高分 {scores[order[0]]:.3f}，最低 {scores[order[-1]]:.3f}）")
        return selected if do_select else candidates

    def _random_select(self, candidates, k, do_select=True):
        """随机筛选控制组：与模型筛选同候选池、同预算，但随机选 k 个送 NPU。

        若配置了模型路径则仍记录模型打分（仅供事后 Oracle 对照），
        但选择完全随机，从而隔离『扩大生成池』与『预测排序』各自的贡献。
        """
        import pandas as pd
        cols = ["log_parent", "gen", "pop_size", "tau", "r_block", "r_warps",
                "r_stages", "sim", "r_ops", "r_mask", "ops_abs", "block_abs",
                "operation", "mutation_type"]
        id2code = {str(getattr(ind, "id", "")): (getattr(ind, "code", "") or "")
                   for ind in getattr(self, "population", [])}
        id2ind = {str(getattr(ind, "id", "")): ind
                  for ind in getattr(self, "population", [])}
        rows = [self._filter_features(c, self.config, id2code, id2ind,
                                      getattr(self, "seed_execution_time", 0))
                for c in candidates]
        scores = None
        try:
            model = self._load_filter_model(getattr(self.config, "filter_model_path", None))
            X = pd.DataFrame(rows)[cols]
            scores = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"[EA] 🎲 随机筛选：未加载模型（仅随机，不记录模型分）: {e}")

        # 【修改】旧种子只含 (random_seed, generation)，同一 seed 下不同算子、不同 run
        # 会复用同一条随机流，跨算子的"随机"并不独立。这里混入 run_id 与算子名，
        # 并用 sha256 而非内置 hash（后者受 PYTHONHASHSEED 影响，跨进程不稳定）。
        _gen = int(getattr(candidates[0], "generation", self.generation)) if candidates \
            else int(self.generation)
        _seed_material = "|".join([
            str(int(getattr(self.config, "random_seed", 0))),
            str(int(getattr(self.config, "run_id", 0))),
            str(getattr(self.executor, "kernel_name", "")),
            str(_gen),
        ])
        rng = random.Random(
            int(hashlib.sha256(_seed_material.encode("utf-8")).hexdigest()[:8], 16))
        idxs = list(range(len(candidates)))
        rng.shuffle(idxs)
        order = idxs[:k]
        keep = set(order)
        selected = [candidates[i] for i in order]

        self._log_filter([{
            "kernel": getattr(self.executor, "kernel_name", ""),
            "run_id": getattr(self.config, "run_id", 0),
            "gen": _gen,
            "candidate_id": candidates[i].id,
            "operation": rows[i]["operation"],
            "mutation_type": rows[i]["mutation_type"],
            "parent_latency": rows[i]["parent_latency"],
            "parent_latency_source": rows[i].get("parent_latency_source"),
            # 与 _filter_select 保持一致：主干父代与变异分级的审计字段
            "backbone_id": candidates[i].metadata.get("backbone_id"),
            "parent_latency_source_chain": candidates[i].metadata.get("lineage"),
            "mutation_tier_source": candidates[i].metadata.get("mutation_tier_source"),
            "mutation_tier_speedup": candidates[i].metadata.get("mutation_tier_speedup"),
            "score": (float(scores[i]) if scores is not None else None),
            "selected": bool(i in keep),
            "selection": "random",
        } for i in range(len(candidates))])

        print(f"[EA] 🎲 随机筛选：候选池 {len(candidates)} → 送评测 {len(selected)}（随机）")
        return selected if do_select else candidates

    def _log_filter(self, records):
        """记录每代全部候选的预筛选打分与是否被选中，供事后算在线精确率。"""
        path = getattr(self.config, "filter_log_path", None)
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for r in records:
                    r.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[EA] 预筛选日志写入失败: {e}")

    def _log_eval(self, ind: "Individual") -> None:
        """候选评测后，把真实延迟与是否改进追加写入预筛选日志。

        仅当开启候选预筛选（enable_candidate_filter）时记录。配合 fulleval
        （filter_evaluate_all）模式下同一次运行内既有打分记录又有全评真值，
        即可事后算出在线精确率与召回率，无需跨运行匹配候选。
        """
        if not getattr(self.config, "enable_candidate_filter", False):
            return
        path = getattr(self.config, "filter_log_path", None)
        if not path:
            return
        md = ind.metadata or {}
        latency = md.get("execution_time")
        if not latency or latency <= 0 or not md.get("success", False):
            return
        # 【修复】旧实现 `md.get("parent_latency") or seed_execution_time` 在拿不到父代
        # 实测延迟时静默退化成"比种子快"。这在 gen>=1 上是一个**宽松得多**的标准，
        # 于是 improved 把两种含义不同的比较混在一起，P@4 / AUC 的标签口径不一致
        # （实测：fulleval 5 算子里 37.5% 的 gen>=1 候选走的就是这条回退路径，
        #  且它们的 parent_latency 全部等于同一个种子常数）。
        # 现在改为走 resolve_parent_latency 回退链，并把来源一并落盘；
        # 下游分析用 --strict-source 只信 parent/dominant 两档。
        id2ind = {str(getattr(x, "id", "")): x for x in getattr(self, "population", [])}
        parent_latency, p_src = resolve_parent_latency(
            ind, id2ind=id2ind, seed_time=getattr(self, "seed_execution_time", 0))
        if not parent_latency or parent_latency <= 0:
            return
        rec = {
            "event": "eval",
            "kernel": getattr(self.executor, "kernel_name", ""),
            "run_id": getattr(self.config, "run_id", 0),
            "gen": ind.generation,
            "candidate_id": ind.id,
            "child_latency": round(latency, 4),
            "parent_latency": round(parent_latency, 4),
            "parent_latency_source": p_src,
            "improved": bool(parent_latency > latency),
            "label": 1 if parent_latency > latency else 0,
            # 只有 parent/dominant 档时 improved 才等价于"比父代快"；
            # 其余档位是回退基线，comparable=False 便于下游严格口径统计。
            "comparable": p_src in TRUSTED_PARENT_SOURCES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[EA] 预筛选真值日志写入失败: {e}")

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

        # 3.6：启用预筛选时先生成更大的候选池，再由模型挑出 needed_children 个送评测。
        # NPU 评测次数保持不变（仍是 needed_children 个），
        # 多出来的只是廉价的 LLM 生成，因此对照的是"预算如何分配"而非"预算多少"。
        pool_size = int(getattr(self.config, 'candidate_pool_size', 0) or 0)
        use_filter = (bool(getattr(self.config, 'enable_candidate_filter', False))
                      and pool_size > needed_children)
        n_gen = pool_size if use_filter else needed_children

        tasks_args = [self.select_parents() for _ in range(n_gen)]

        print(f"[EA] 🚀 并发发射 {n_gen} 个子代变异/交叉 LLM 请求...")
        child_candidates: List[Individual] = self._gen_batch(tasks_args)

        # ---- 候选池指纹去重 + 有限次重新生成补位（评审 v8.2 阶段 3）----
        child_candidates, _dedup_stat = self._dedup_and_backfill(
            child_candidates, tasks_args)
        for _k, _v in _dedup_stat.items():
            self.stats[_k] = self.stats.get(_k, 0) + _v

        for child in child_candidates:
            child.generation = self.generation + 1

        # ---- 候选池标识 + 源码可靠落盘（必须在筛选之前，未入选的也要保存）----
        self._stamp_pool_and_save_code(child_candidates)

        # ---- 3.6 预筛选流水线 ----
        # 正确顺序（评审意见 2）：
        #   生成 20 → **对整池做批量预编译** → 剔除编译失败 → 模型/随机选 4 → NPU 全评
        # 注意编译门必须在"选 4"**之前**，否则它只是给已选中的候选做检查，
        # 回答不了"预编译能否提高池位利用率"这个问题。
        if use_filter:
            child_candidates = self._apply_compile_gate(child_candidates)

        if use_filter:
            # filter_evaluate_all：全评子实验专用，记录打分但不过滤
            do_select = not bool(getattr(self.config, 'filter_evaluate_all', False))
            mode = getattr(self.config, 'candidate_filter_mode', 'model')
            if mode == 'random':
                child_candidates = self._random_select(
                    child_candidates, needed_children, do_select=do_select)
            else:
                child_candidates = self._filter_select(
                    child_candidates, needed_children, do_select=do_select)

        print(f"[EA] 🧪 批量生成完毕，启动压测评估...")

        child_candidates = self._evaluate_individuals_batch(child_candidates)
        # ---- 分级复测：只给"与父代差距落在噪声内"的候选加测 ----
        _n_rm = self._apply_tiered_remeasure(child_candidates)
        if _n_rm:
            print(f"[EA] 🔬 分级复测：额外 {_n_rm} 次 NPU 调用"
                  f"（仅用于拉近父代的候选，标签可信度优先于成本）")
        # 评测后追加真实延迟与是否改进，供事后算在线精确率/召回率
        for _c in child_candidates:
            self._log_eval(_c)

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

    @staticmethod
    def _code_sha(code):
        """源码 SHA256，用于把「样本里的哈希」与「codes/ 里的文件」一一对上。"""
        import hashlib
        return hashlib.sha256((code or "").encode("utf-8")).hexdigest()

    def _save_code(self, ind: Individual) -> bool:
        """保存候选源码，供事后计算代码差异特征（按 candidate_id 索引）。

        【修复】旧实现 `except Exception: pass` —— 写入失败被静默吞掉，
        事后表现为"父代代码不在库中"，而实际上可能是磁盘/权限问题。
        这类失败在离线审计里看起来和"父代从未被保存"一模一样，
        导致我们无法区分"没采到"和"采到了但丢了"。
        现在：写入后**回读校验**（长度 + sha256），失败计入 stats 并打印告警。
        """
        if not getattr(self, 'dataset_enabled', False) or not ind.code:
            return False
        target = self.dataset_dir / "codes" / f"{ind.id}.py"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(ind.code, encoding="utf-8")
            # 回读校验，确认字节真的落盘
            if target.read_text(encoding="utf-8") != ind.code:
                raise IOError(f"回读不一致: {target}")
        except Exception as e:
            self.stats['code_save_failed'] = self.stats.get('code_save_failed', 0) + 1
            print(f"[EA] ⚠️ 源码落盘失败 {ind.id}: {e}")
            return False
        self.stats['code_saved'] = self.stats.get('code_saved', 0) + 1
        return True

    def _save_code_by(self, cid, code) -> bool:
        """按 candidate_id 落盘一段源码（用于保存参考父代的源码）。"""
        if not getattr(self, 'dataset_enabled', False) or not cid or not code:
            return False
        target = self.dataset_dir / "codes" / f"{cid}.py"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            if target.read_text(encoding="utf-8") != code:
                raise IOError(f"回读不一致: {target}")
        except Exception as e:
            self.stats['code_save_failed'] = self.stats.get('code_save_failed', 0) + 1
            print(f"[EA] ⚠️ 父代源码落盘失败 {cid}: {e}")
            return False
        return True

    def _reference_parent_id(self, child, id2ind=None):
        """求"训练标签真正参照的那位已评测父代"的 ID。

        规则（与 resolve_parent_latency 的取值顺序保持一致，只是这里要的是 ID）：
          主干父代 backbone_id → parents → parent → dominant_parent，
          只接受**已被实测过**的个体；全都不可用时退回第一个已知 ID（并记为不可用）。
        关键约束：**不能把尚未评测的中间子代当作"已有真实性能的父代"** ——
        交叉 → 变异链上的中间产物必须跳过。
        """
        md = child.metadata or {}
        id2ind = id2ind or {}
        cand = []
        for key in ("backbone_id", "dominant_parent"):
            if md.get(key):
                cand.append(str(md[key]))
        for p in (md.get("parents") or []):
            cand.append(str(p))
        if md.get("parent"):
            cand.append(str(md["parent"]))
        for pid in cand:
            ind = id2ind.get(pid)
            if ind is not None and _measured_latency(ind) is not None:
                return pid
        return cand[0] if cand else None

    @staticmethod
    def _code_fingerprint(code):
        """候选源码指纹：归一化空白后的 SHA256。

        【为什么需要】新采集的全池数据显示候选重复率 52.5%（每池 20 个里平均
        只有 9.5 种源码），20% 的子代与参考父代源码完全相同 —— 这些重复候选
        既浪费 LLM 生成，也挤占候选池里本可以给新实现的位置。

        【为什么只归一化空白】只有确认"同一源码 == 同一实现配置"后才能按指纹
        去重。目前已知同一源码在池内的延迟极差可达 18%，在复测确认这是测量
        波动还是外部启动配置差异之前，指纹保守地只吃空白差异。
        """
        import hashlib
        import re as _re
        c = code or ""
        c = "\n".join(line.rstrip() for line in c.replace("\r\n", "\n").split("\n"))
        c = _re.sub(r"\n{3,}", "\n\n", c).strip()
        return hashlib.sha256(c.encode("utf-8")).hexdigest()

    def _gen_batch(self, pairs):
        """并发生成一批子代（供去重补位时重复调用）。"""
        out = []
        if not pairs:
            return out
        with ThreadPoolExecutor(max_workers=min(len(pairs), MAX_WORKERS)) as pool:
            futures = [pool.submit(self._process_single_generation_task,
                                   task_type='crossover_mutate',
                                   parent1=p1, parent2=p2) for p1, p2 in pairs]
            for f in as_completed(futures):
                try:
                    out.append(f.result())
                except Exception as e:
                    print(f"[EA] ❌ 变异线程异常: {e}")
        return out

    def _dedup_and_backfill(self, candidates, pairs):
        """候选池指纹去重 + 有限次重新生成补位。

        评审 v8.2 阶段 3：产生重复候选时允许有限次数重新生成，并记录额外 LLM 成本。
        设计取舍：
          * **不缩小候选池** —— 补不满就保留重复项（池规模是实验自变量），
            但会给它打 `duplicate_in_pool` 标记，事后可统计真正的重复率；
          * 去重参照集 = 当前种群（父代/精英）+ 本代已接受候选，
            所以"与参考父代完全相同"这类重复会被抓到；
          * 额外 LLM 调用次数如实计入 `dedup_extra_llm_calls`。
        """
        if not candidates:
            return candidates, {}
        max_retry = int(getattr(self.config, 'dedup_retries', 0) or 0)
        if not getattr(self.config, 'enable_candidate_dedup', False) or max_retry <= 0:
            return candidates, {}

        seen = set()
        for ind in getattr(self, 'population', []):
            fp = self._code_fingerprint(getattr(ind, 'code', ''))
            if fp:
                seen.add(fp)

        target = len(candidates)
        final, pending = [], []
        for c in candidates:
            fp = self._code_fingerprint(getattr(c, 'code', ''))
            (pending if (not fp or fp in seen) else final).append(c)
            if fp and fp not in seen:
                seen.add(fp)
        n_init_dup = len(pending)
        # 原始重复项留作**兜底**：重新生成若返回不足，宁可保留重复项，
        # 也不能让候选池悄悄小于配置值（评审要求：正式实验必须知道实际生成了多少）
        orig_dups = list(pending)

        extra_calls = 0
        for _ in range(max_retry):
            if not pending:
                break
            idx = [i % len(pairs) for i in range(len(pending))] if pairs else []
            fresh = self._gen_batch([pairs[i] for i in idx]) if pairs else []
            extra_calls += len(fresh)
            still = []
            for c in fresh:
                fp = self._code_fingerprint(getattr(c, 'code', ''))
                if fp and fp not in seen and len(final) < target:
                    seen.add(fp)
                    c.metadata['dedup_regenerated'] = True
                    final.append(c)
                else:
                    still.append(c)
            pending = still

        # 补不满就保留重复项（先用新生成的，再用原始兜底），并显式标记
        n_still_dup = 0
        for c in list(pending) + orig_dups:
            if len(final) >= target:
                break
            if any(c is x for x in final):
                continue
            c.metadata['duplicate_in_pool'] = True
            n_still_dup += 1
            final.append(c)
        stat = {"dedup_pool_size": target,
                "dedup_initial_duplicates": n_init_dup,
                "dedup_replaced": max(0, n_init_dup - n_still_dup),
                "dedup_still_duplicate": n_still_dup,
                "dedup_extra_llm_calls": extra_calls,
                # 【修复】评审指出：重新生成失败/返回不足时可能悄悄导致池不足 20。
                # 正式实验必须知道"实际生成了多少"，不能只写配置值 20。
                "dedup_pool_shortfall": max(0, target - len(final)),
                "dedup_actual_pool_size": len(final)}
        if stat["dedup_pool_shortfall"]:
            print(f"[EA] ⚠️ 去重补位后候选池不足：期望 {target}，实际 "
                  f"{len(final)}（缺 {stat['dedup_pool_shortfall']}）——"
                  f"LLM 生成失败或返回数量不足，请检查日志")
        print(f"[EA] 🧹 去重：初始重复 {n_init_dup}/{target}，"
              f"重新生成补位 {stat['dedup_replaced']}，"
              f"仍重复 {n_still_dup}，额外 LLM 调用 {extra_calls}，"
              f"实际池大小 {len(final)}/{target}")
        return final, stat

    def _apply_tiered_remeasure(self, candidates):
        """分级复测（评审 v8.2 决策 1）：只对"与父代差距落在噪声内"的候选加测。

        背景：同码复测显示计时 CV 达 2.5%~7.7%，1%~3% 的改进无法可靠区分。
        全部候选都测 5 次会把 NPU 成本翻几倍，因此分级：
          * 与父代差距 > margin（默认 5%）→ 单次评测即可，结论可信；
          * 差距 ≤ margin → 追加 `tiered_remeasure_repeats` 次，取中位数；
            重测后仍在噪声内 → 打 `latency_uncertain` 标记，标签视为不确定。
        额外 NPU 调用如实计入 `stats['tiered_remeasure_calls']`。
        """
        if not getattr(self.config, 'tiered_remeasure', False):
            return 0
        margin = float(getattr(self.config, 'tiered_remeasure_margin', 0.05) or 0.05)
        repeats = int(getattr(self.config, 'tiered_remeasure_repeats', 5) or 5)
        extra = 0
        for c in candidates:
            md = c.metadata or {}
            if not md.get('success'):
                continue
            pl = _positive_float(md.get('parent_latency'))
            cl = _positive_float(md.get('execution_time'))
            if pl is None or cl is None:
                continue
            if abs(cl - pl) / pl > margin:
                continue
            samples = []
            for _ in range(max(1, repeats)):
                extra += 1
                res = self.executor.evaluate(c.code, device_id=md.get('device_id', 0))
                v = _positive_float(getattr(res, 'execution_time', None)) if res else None
                if res and res.success and v is not None:
                    samples.append(float(v))
            if not samples:
                md['latency_uncertain'] = True
                md['tiered_remeasure_n'] = 0
                continue
            samples.sort()
            med = samples[len(samples) // 2]
            md['execution_time'] = med
            md['tiered_remeasure_n'] = len(samples)
            md['tiered_remeasure_samples'] = [round(v, 4) for v in samples]
            md['latency_uncertain'] = bool(abs(med - pl) / pl <= margin)
            c.fitness = (pl / med) if med > 0 else c.fitness
        self.stats['tiered_remeasure_calls'] = \
            self.stats.get('tiered_remeasure_calls', 0) + extra
        return extra

    def _stamp_pool_and_save_code(self, candidates):
        """给本代候选打唯一 pool_id，并可靠落盘子代 + 参考父代源码。

        【为什么需要 pool_id】
        事后算 P@k 时必须按"同一次运行里真实一起生成的那一批"分组。
        只按 (kernel, gen) 分组会把多次运行/多批生成的候选并成一个假池 ——
        实测历史数据里 (kernel, gen) 分组下每池约 70 条，而线上实际是 20 选 4，
        Top-4 的难度、随机基线、新旧模型对比全都会失真。

        【为什么生成时就要落盘】
        旧流程只在评测时保存源码，未入选的候选（以及被用作父代但自身未评测的个体）
        的代码就永远丢了，事后无法恢复父子代码特征 —— 这正是清洗时
        1805 条变异样本父代代码缺失的根因。
        """
        if not candidates:
            return
        run_id = getattr(self.config, 'run_id', 0)
        kernel = getattr(self.executor, 'kernel_name', '')
        method = getattr(self.config, 'method', 'full')
        gen = self.generation + 1
        pool_id = f"{kernel}|m{method}|r{run_id}|g{gen}"

        id2ind = {str(getattr(x, "id", "")): x for x in getattr(self, "population", [])}
        n_ref_missing = 0
        for c in candidates:
            md = c.metadata
            md['pool_id'] = pool_id
            ref = self._reference_parent_id(c, id2ind)
            md['reference_parent_id'] = ref
            if not md.get('backbone_id') and ref:
                md['backbone_id'] = ref

            self._save_code(c)
            md['child_code_sha256'] = self._code_sha(c.code)

            ref_ind = id2ind.get(ref) if ref else None
            pcode = getattr(ref_ind, 'code', None) if ref_ind is not None else None
            if ref and pcode:
                ok = self._save_code_by(ref, pcode)
                md['parent_code_sha256'] = self._code_sha(pcode)
                md['parent_code_saved'] = bool(ok)
            else:
                # 明确标记缺失，不再静默把子代自身当父代
                md['parent_code_sha256'] = None
                md['parent_code_saved'] = False
                n_ref_missing += 1
        self.stats['parent_code_missing_in_pool'] = \
            self.stats.get('parent_code_missing_in_pool', 0) + n_ref_missing

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
        # Gen0 候选的父代为锚点种子；交叉/变异子代用生成时记录的父代延迟。
        # 【修复】旧实现在父代未实测时静默回退到种子延迟，于是样本标签变成
        # "比种子快"而不是"比父代快"，两种语义的样本被混进同一份训练集。
        # 现在走统一回退链；且**只有 parent/dominant 档才落盘**，避免用回退基线
        # 污染训练集（训练快照要求 parent_latency 语义单一，实测现有 7431 行
        # 快照里 NaN 比例为 0%，正是不落盘不可信样本换来的）。
        id2ind = {str(getattr(x, "id", "")): x for x in getattr(self, "population", [])}
        parent_latency, p_src = resolve_parent_latency(
            ind, id2ind=id2ind, seed_time=getattr(self, 'seed_execution_time', 0))
        if not parent_latency or parent_latency <= 0:
            return
        if p_src not in TRUSTED_PARENT_SOURCES:
            self.stats['samples_dropped_untrusted_parent'] = \
                self.stats.get('samples_dropped_untrusted_parent', 0) + 1
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
            "parent_latency_source": p_src,
            # ---- 可追溯信息（评审 v8 要求）----
            # pool_id 唯一标识"同一次运行里真实一起生成的那一批"，
            # 事后算真实 20 选 4 必须按它分组，不能只按 (kernel, gen)。
            "pool_id": md.get("pool_id"),
            # reference_parent_id 必须对应**训练标签用到的那位真实已评测父代**；
            # 交叉 → 变异链上的未评测中间产物不能当父代。
            "reference_parent_id": md.get("reference_parent_id"),
            "backbone_id": md.get("backbone_id"),
            "lineage": md.get("lineage"),
            "mutation_tier_source": md.get("mutation_tier_source"),
            "mutation_tier_speedup": md.get("mutation_tier_speedup"),
            # 源码哈希：把样本里的哈希与 codes/ 目录里的文件一一对上，
            # 源码本体不重复写进 JSONL（避免文件膨胀）
            "child_code_sha256": md.get("child_code_sha256") or self._code_sha(ind.code),
            "parent_code_sha256": md.get("parent_code_sha256"),
            "parent_code_saved": md.get("parent_code_saved"),
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
        except Exception as e:
            # 同 _save_code：写出失败必须留痕，否则离线审计无法区分
            # 「没采到」和「采到了但丢了」
            self.stats['sample_write_failed'] = \
                self.stats.get('sample_write_failed', 0) + 1
            print(f"[EA] \u26a0\ufe0f \u6837\u672c\u843d\u76d8\u5931\u8d25 {ind.id}: {e}")

    def _event(self, ind: Individual, event: str) -> dict:
        """构造一条 JSONL 事件记录（3.1 主实验的原始数据）。"""
        md = ind.metadata or {}
        # ---- 3.5/3.6 审计计数（每次事件仅记一次） ----
        if event == 'reject':
            self.stats['static_rejected'] = self.stats.get('static_rejected', 0) + 1
            if md.get('preflight_rejected'):
                self.stats['ast_rejected'] = self.stats.get('ast_rejected', 0) + 1
        elif event == 'compile_reject':
            # 编译门在"选 k"之前剔掉的候选：它没有进 NPU，也不该混进 static_rejected
            self.stats['compile_rejected'] = self.stats.get('compile_rejected', 0) + 1
        elif event == 'eval':
            if not md.get('success', False):
                # 【修复】旧实现把所有评测失败一律记成 compile_failed，
                # 即使根本没开编译门 —— 于是"编译失败率"这个数一直是假的
                # （实测 225 次里只有 2 次整体失败，却被全记在编译头上）。
                # 现在只有**确认**编译失败（compile_pass is False）才计数；
                # 编译状态未知（None）或已确认编译通过但正确性/测量失败，
                # 分别归入 eval_failed_unknown / eval_failed_post_compile。
                cp = md.get('compile_pass')
                if cp is False:
                    self.stats['compile_failed'] = \
                        self.stats.get('compile_failed', 0) + 1
                elif cp is True:
                    self.stats['post_compile_failed'] = \
                        self.stats.get('post_compile_failed', 0) + 1
                else:
                    self.stats['eval_failed_unknown'] = \
                        self.stats.get('eval_failed_unknown', 0) + 1
        parent = md.get('parent')
        # 父代基线：走统一回退链 parent → dominant → ancestor → pop_best → seed
        _id2ind = {str(getattr(x, "id", "")): x for x in getattr(self, "population", [])}
        _p_lat, _p_src = resolve_parent_latency(
            ind, id2ind=_id2ind, seed_time=getattr(self, 'seed_execution_time', 0))
        # 【修复】评审 v8.1：训练样本里有 pool_id / reference_parent_id，
        # 但**评测事件里没有**，于是离线评估只能退回去猜父代（默认取
        # parent_ids[0]，而交叉场景下它未必是主干父代），训练和评估再次分叉。
        # 这里补齐：事件与样本必须携带同一套参考父代与候选池标识。
        _ref = md.get("reference_parent_id")
        if not _ref:
            _ref = self._reference_parent_id(ind, _id2ind)
        _pool_id = md.get("pool_id")
        if not _pool_id:
            _pool_id = (f"{self.executor.kernel_name}|"
                        f"m{getattr(self.config, 'method', 'full')}|"
                        f"r{getattr(self.config, 'run_id', 0)}|g{ind.generation}")
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
            # ---- 候选池 / 参考父代 / 源码哈希（与训练样本同一套口径）----
            "pool_id": _pool_id,
            "reference_parent_id": _ref,
            "backbone_id": md.get('backbone_id'),
            "child_code_sha256": md.get('child_code_sha256')
                                 or self._code_sha(getattr(ind, 'code', '') or ''),
            "parent_code_sha256": md.get('parent_code_sha256'),
            "parent_code_saved": md.get('parent_code_saved'),
            # tau 直接影响模型特征：离线评估若拿不到就只能猜，
            # 旧脚本硬编码 -1.0 与训练分布（主流 0.90）不一致。
            "tau": getattr(self.config, 'seed_diversity_threshold', None),
            "pop_size": len(getattr(self, 'population', []) or []),
            # 父代延迟：走统一回退链（parent → dominant → ancestor → pop_best → seed）。
            # 注意：父代评估失败时其 execution_time 为 None，此时不再直接冒充为种子耗时，
            # 而是先沿父链上溯找最近一个被实测过的祖先，仍找不到才退到种群最优/种子，
            # 并把实际来源一并落盘（parent_latency_source），保证"相对父代的增益"
            # 和"相对种子的增益"在事后分析中可区分。
            "parent_latency": _p_lat,
            "parent_eval_ok": _p_src in TRUSTED_PARENT_SOURCES,
            "parent_latency_source": _p_src,
            "ast_pass": not md.get('preflight_rejected', False),
            "ast_reason": md.get('syntax_error'),
            # 【修复】旧实现两行都读 md['success']，compile_pass 恒等于 correctness_pass，
            # 于是"编译失败率"这类指标无从计算（实测 225 次评测里只能看到 overall
            # 失败率 0.9%，无法归因到编译还是正确性）。
            # 现在 compile_pass 取自独立的编译门；未开启编译门时为 None（未知），
            # 不写成 False —— 未知与失败必须可区分，否则会低估编译门的价值。
            "compile_pass": md.get('compile_pass'),
            "compile_time_ms": md.get('compile_time_ms'),
            "correctness_pass": bool(md.get('success', False)),
            "latency_us": md.get('execution_time') if md.get('success') else None,
            "device_id": md.get('device_id'),
            "model": getattr(ind, 'model_used', 'unknown'),
            "error": md.get('error'),
            # 【修改】补记该候选生成时的真实 Token 消耗。
            # 来源：genetic_operators._track_tokens 会写入 ind.metadata['usage']。
            # 旧实现不落盘这两个字段，导致 filter_online_exp 的 _sum_llm_tokens 恒为 None，
            # 无法比较"扩大候选池"带来的额外生成成本。取不到时写 None（缺失），不写 0。
            "prompt_tokens": (md.get('usage') or {}).get('prompt_tokens'),
            "completion_tokens": (md.get('usage') or {}).get('completion_tokens'),
        }

    def _evaluate_individuals_batch(self, individuals: List[Individual]) -> List[Individual]:
        """批量评估个体：多 NPU 并发执行（卡数由 EA_NUM_DEVICES 决定），计算相对种子的加速比作为适应度"""
        task_queue = queue.Queue()

        # ---- 预算硬闸：入队前先算剩余额度，只把前 room 个放进队列 ----
        # 【修改】旧实现在入队循环里逐个判断 `self.eval_count >= eval_budget`，
        # 但 eval_count 真正的自增发生在 worker 内部（evaluate 之后），整批入队时它尚未变化，
        # 于是「本批第一个通过检查」之后的所有候选会被一起放行，单批即可超出预算。
        # 典型场景：fulleval 候选池 20、剩余额度 5 → 20 个全部入队 → 超评 15 次。
        # 改为先算额度再截断，保证任何情况下 npu_calls <= eval_budget，且截断顺序确定可复现。
        budget = int(getattr(self.config, 'eval_budget', 0) or 0)
        # 【修改】额度按"已发起的设备评测次数"计算，而不是"成功完成的次数"。
        # 若设备调用已发出但抛异常（编译/运行/超时），它同样占用了真机资源；
        # 旧实现只在 evaluate() 正常返回后才自增，异常时该次不计入，
        # 下一代据此算出偏大的剩余额度，就可能安排超出预算的额外评测。
        room = (budget - self.eval_attempts) if budget > 0 else len(individuals)
        room = max(0, room)

        for idx, ind in enumerate(individuals):
            if not ind.metadata.get('evaluated', False):
                if not ind.id or ind.id.startswith('hash_') or len(ind.id) != 8:
                    ind.id = uuid.uuid4().hex[:8]
                # 预算耗尽（含本批已把剩余额度用满）：不再占用 NPU，直接判定为无效候选
                if room <= 0:
                    ind.metadata.update({'evaluated': True, 'success': False,
                                         'error': 'eval_budget_exhausted'})
                    ind.fitness = 0.0
                    self.logger.log(self._event(ind, 'skipped'))
                    continue
                task_queue.put((idx, ind))
                room -= 1
        
        results = list(individuals)
        total_tasks = task_queue.qsize()
        completed_tasks = [0]
        # 多线程下 eval_attempts / eval_count 的自增必须加锁保护：
        # `x += 1` 不是原子操作，NUM_DEVICES>1 时并发自增可能丢计数。
        _cnt_lock = threading.Lock()
        
        if total_tasks == 0:
            return results

        print(f"[EA] 🚀 启动 {NUM_DEVICES} 卡 NPU 并发评估，共 {total_tasks} 个待评估个体...")
        
        def device_worker(device_id):
            """设备工作线程：从任务队列取个体，在指定 NPU 上评估"""
            while not task_queue.empty():
                try:
                    idx, ind = task_queue.get_nowait()
                except queue.Empty:
                    break

                # ---- 先占额度：设备调用一旦发出即占用预算，无论成功、编译失败还是抛异常 ----
                with _cnt_lock:
                    self.eval_attempts += 1
                    self.stats['npu_calls'] = self.eval_attempts

                try:
                    result = self.executor.evaluate(ind.code, timeout=600, device_id=device_id)
                    with _cnt_lock:
                        self.eval_count += 1
                        self.stats['npu_calls_completed'] = self.eval_count

                    ind.metadata.update({
                        'evaluated': True,
                        'success': result.success,
                        'speedup': result.speedup,
                        'execution_time': result.execution_time,
                        'error': result.error,
                        'device_id': device_id,
                        # 【新增】编译阶段单独落盘（任务③'）。
                        # 旧实现把 compile_pass 与 correctness_pass 都取自 success，
                        # 两个字段恒等，"编译失败率"根本算不出来。
                        # 开启 --pre-compile-check 后二者才真正分离；未开启时为 None，
                        # 明确表示"未知"，不用 False 冒充。
                        'compile_pass': getattr(result, 'compile_pass', None),
                        'compile_time_ms': getattr(result, 'compile_time_ms', None),
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

        with ThreadPoolExecutor(max_workers=NUM_DEVICES) as pool:
            futures = [pool.submit(device_worker, d) for d in range(NUM_DEVICES)]
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
            res = self.executor.evaluate(code, timeout=600, device_id=i % NUM_DEVICES)
            if res.success and res.execution_time > 0:
                ind = Individual(code=code, generation=0, metadata={
                    'operation': 'seed', 'evaluated': True, 'success': True,
                    'execution_time': res.execution_time, 'speedup': res.speedup,
                    'device_id': i % NUM_DEVICES,
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