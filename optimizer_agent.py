#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton Optimization Agent Main Class - Unified Interface

[Note] This file is provided by the organizers, students do not need to modify it
This class is responsible for coordinating components and providing a standard calling interface for the evaluation system
"""

import os
import time
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import EAConfig
from llm_interface import LLMInterface
from executor import TritonExecutor
from genetic_operators import GeneticOperators
from evolutionary_algorithm import EvolutionaryAlgorithm


def get_baseline_from_json(baseline_json_path: str, kernel_name: str, test_case_id: int = 1) -> Optional[float]:
    """
    Read baseline time for the specified kernel and test case from the baseline JSON file

    Args:
        baseline_json_path: Baseline JSON file path
        kernel_name: Kernel name
        test_case_id: Test case number (1, 2, 3)

    Returns:
        Baseline time (us), returns None if not found
    """
    try:
        with open(baseline_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        grouped_data = data.get('grouped_by_kernel', {})

        if kernel_name not in grouped_data:
            print(f"[Baseline] Warning: Kernel not found in JSON: {kernel_name}")
            return None

        kernel_baseline = grouped_data[kernel_name]

        # Find the baseline corresponding to test_case_id
        for item in kernel_baseline:
            test_file = item.get('test_file', '')
            # Extract test case number: test_kernel_name_1.py -> 1
            match = re.search(r'_([0-9]+)\.py$', test_file)
            if match:
                current_id = int(match.group(1))
                if current_id == test_case_id:
                    return item.get('task_duration_us')

        print(f"[Baseline] Warning: Cannot find baseline for kernel {kernel_name} test_case {test_case_id}")
        return None

    except Exception as e:
        print(f"[Baseline] Error: Failed to read JSON: {e}")
        return None


class TritonOptimizerAgent:
    """
    Triton Optimization Agent Main Class (Unified Interface)

    Students complete the optimization logic by implementing EvolutionaryAlgorithm and GeneticOperators
    This class is responsible for coordinating components and providing a standard calling interface for the evaluation system

    Evaluation interfaces:
    - setup(): Initialize execution environment
    - optimize(): Execute optimization (standard interface)
    - get_results(): Get optimization results (return up to 5 versions)
    """

    def __init__(self, config: Optional[EAConfig] = None):
        """
        Initialize Agent

        Args:
            config: Configuration object, if None use default configuration
        """
        self.config = config or EAConfig()
        self.llm = LLMInterface(self.config)
        self.executor: Optional[TritonExecutor] = None
        self.ea: Optional[EvolutionaryAlgorithm] = None

        # Result records
        self.optimization_history: List[Dict] = []

    def setup(self, 
              baseline_code: str, 
              test_code: str, 
              kernel_name: str = "kernel",
              work_dir: Optional[str] = None,
              test_case_id: int = 1):
        """
        Initialize execution environment

        Args:
            baseline_code: Baseline Triton code (used to get code structure, not for performance measurement)
            test_code: Test code (import statements will be modified)
            kernel_name: Operator name
            work_dir: Working directory path
            test_case_id: Which test case baseline to use (1, 2, 3)
        """
        print(f"[Agent] Initializing execution environment...")
        print(f"       └─ Operator: {kernel_name}")
        print(f"       └─ Test case: {test_case_id}")

        # Read baseline time from JSON
        baseline_time = get_baseline_from_json(
            self.config.baseline_json,
            kernel_name,
            test_case_id
        )

        if baseline_time is None:
            raise RuntimeError(f"Cannot read baseline time for kernel {kernel_name} from JSON")

        print(f"       └─ Baseline: {baseline_time:.2f}μs")

        # Find test file path (test_code is a code string, need to find the file path)
        # If test_code is a file path, use it directly
        if os.path.isfile(test_code):
            test_code_path = test_code
        else:
            # test_code is code content, need to write to a temporary file
            work_dir_path = Path(work_dir) if work_dir else Path(".")
            test_code_path = work_dir_path / f"test_{kernel_name}.py"
            with open(test_code_path, 'w', encoding='utf-8') as f:
                f.write(test_code)

        self.executor = TritonExecutor(
            baseline_time=baseline_time,
            test_code_path=str(test_code_path),
            config=self.config,
            kernel_name=kernel_name,
            work_dir=Path(work_dir) if work_dir else Path(".")
        )

        genetic_ops = GeneticOperators(self.llm, self.config)
        self.ea = EvolutionaryAlgorithm(genetic_ops, self.executor, self.config)

    def optimize(self, seed_codes: List[str], max_time: int = 600) -> Dict[str, Any]:
        """
        Execute optimization - [Core interface, called by evaluation system]

        Workflow:
        1. Run evolutionary algorithm
        2. Collect optimization results
        3. Return dictionary containing code, fitness, and statistics

        Args:
            seed_codes: Initial seed code list (must include baseline_code)
            max_time: Maximum running time (seconds), controlled by evaluation system

        Returns:
            result: Dictionary containing the following fields:
                - best_code: Best code
                - best_fitness: Best fitness
                - speedup: Speedup ratio
                - generations: Actual evolution generations
                - time_elapsed: Time consumed
                - llm_stats: LLM call statistics
                - top5_codes: Top 5 best codes (for final submission)
        """
        if self.ea is None:
            raise RuntimeError("Please call setup() first to initialize")

        print(f"\n[Agent] Starting optimization, max time: {max_time}s")
        start_time = time.time()

        # Run evolutionary algorithm
        best = self.ea.run(seed_codes)

        # Collect results
        elapsed = time.time() - start_time

        # Get top5 individuals (for final submission, at most 5)
        top5_individuals = self._get_top_k(5)
        top5_codes = [
            {
                'code': ind.code,
                'fitness': ind.fitness,
                'generation': ind.generation,
                'id': ind.id
            }
            for ind in top5_individuals
        ]

        result = {
            'best_code': best.code,
            'best_fitness': best.fitness,
            'speedup': best.metadata.get('speedup', 0),
            'generations': self.ea.generation,
            'time_elapsed': elapsed,
            'llm_stats': self.llm.get_stats(),
            'top5_codes': top5_codes
        }

        #相对于官方文件做出的修改：在优化完成后，打印llm调用信息
        self.optimization_history.append(result)

        print(f"\n[Agent] Optimization completed:")
        print(f"  - Best fitness: {result['best_fitness']:.4f}")
        print(f"  - Speedup: {result['speedup']:.4f}")
        print(f"  - Evolution generations: {result['generations']}")
        print(f"  - Time elapsed: {result['time_elapsed']:.2f}s")
        print(f"  - LLM call count: {result['llm_stats']['call_count']}")

        self.llm.print_call_summary()

        return result

    def _get_top_k(self, k: int) -> List[Any]:
        """
        Get top-k individuals

        Args:
            k: Number of individuals to return

        Returns:
            List[Individual]: Top k individuals sorted by fitness
        """
        sorted_pop = sorted(self.ea.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:k]

    def save_results(self, output_dir: str, kernel_name: str) -> None:
        """
        Save optimization results to file

        Args:
            output_dir: Output directory
            kernel_name: Operator name
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.optimization_history:
            print("[Agent] No optimization results to save")
            return

        latest_result = self.optimization_history[-1]

        # Save best code
        best_code_path = output_path / f"{kernel_name}_best.py"
        with open(best_code_path, 'w') as f:
            f.write(latest_result['best_code'])
        print(f"[Agent] Best code saved: {best_code_path}")

        # Save top5 codes
        for i, code_info in enumerate(latest_result['top5_codes']):
            code_path = output_path / f"{kernel_name}_v{i+1}.py"
            with open(code_path, 'w') as f:
                f.write(code_info['code'])

        # Save statistics
        stats_path = output_path / f"{kernel_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'best_fitness': latest_result['best_fitness'],
                'speedup': latest_result['speedup'],
                'generations': latest_result['generations'],
                'time_elapsed': latest_result['time_elapsed'],
                'llm_stats': latest_result['llm_stats'],
                'top5_summary': [
                    {
                        'id': c['id'],
                        'fitness': c['fitness'],
                        'generation': c['generation']
                    }
                    for c in latest_result['top5_codes']
                ]
            }, f, indent=2)

        print(f"[Agent] Statistics saved: {stats_path}")