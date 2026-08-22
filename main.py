#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entry - One-click run evolutionary algorithm optimization

Usage examples:
    python main.py --input-dir ./datasets/sglang0119 --output-dir ./output --kernel add_kernel

Parameter description:
    --input-dir: Input folder path, containing Triton operators to be optimized
    --output-dir: Output folder path
    --kernel: Specify the operator name to optimize (optional, default processes all operators)
    --population-size: Population size (optional, overrides config value)
    --max-generations: Max evolution generations (optional, overrides config value)
    --debug: Enable debug mode (optional)

All LLM configurations please modify the llm_models list in config.py
"""

#相对于官方文件做出的修改：重定向HF_HOME缓存路径并开启离线模式，避免网络问题
import os
os.environ["HF_HOME"] = "/workspace/user_data/Agent/RAG/.hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import argparse
import logging
from pathlib import Path
from typing import List

from config import EAConfig
from optimizer_agent import TritonOptimizerAgent


def parse_args():
    """Parse command-line arguments"""
    # First get default values from config for help information
    default_config = EAConfig()

    parser = argparse.ArgumentParser(
        description="Triton Auto-Optimization System Based on Evolutionary Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Optimize all operators in the specified folder
  python main.py --input-dir ./datasets/test --output-dir ./output

  # Optimize a specific operator
  python main.py --input-dir ./datasets/test --output-dir ./output --kernel add_kernel

  # Use custom evolution parameters (override config values)
  python main.py --input-dir ./datasets/test --output-dir ./output \
                 --population-size 15 --max-generations 30

  # Enable debug mode
  python main.py --input-dir ./datasets/test --output-dir ./output --debug

Configuration:
  LLM list please modify the llm_models field in config.py
  Current default model list: {', '.join(default_config.llm_models)}
        """
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Input folder path, containing Triton operators (.py files) to be optimized"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output folder path"
    )

    parser.add_argument(
        "--kernel", "-k",
        type=str,
        #相对于官方文件做出的修改：新增支持指定多个kernel
        nargs='+',
        default=None,
        help="Specify the operator name(s) to optimize (without .py suffix), can specify multiple, default processes all operators"
    )

    parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=None,
        help=f"Population size (default uses config value: {default_config.population_size})"
    )

    parser.add_argument(
        "--max-generations", "-g",
        type=int,
        default=None,
        help=f"Max evolution generations (default uses config value: {default_config.max_generations})"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, output detailed logs"
    )

    return parser.parse_args()


def generate_default_test(kernel_name: str) -> str:
    """
    Generate default test code

    Students can modify this function as needed to add custom test logic
    """
    return f'''# Default test code
import torch
import triton

def test_kernel():
    # TODO: Implement test logic according to specific operator
    print("Running default test for {kernel_name}")
    # Here should call the kernel and verify correctness
    return True

if __name__ == "__main__":
    test_kernel()
'''


def get_seed_codes(input_dir: Path, kernel_name: str) -> List[str]:
    """
    Get seed code list - supports multiple naming conventions

    Directory structure: dataset/kernel_name/kernel_name.py

    Supports:
    1. kernel_name/kernel_name.py (main file)
    2. kernel_name/kernel_name_1.py to kernel_name_10.py (multiple variants)
    3. kernel_name/variants/ subfolder
    """
    seed_codes = []

    # Kernel-specific directory
    kernel_dir = input_dir / kernel_name

    if not kernel_dir.exists():
        raise ValueError(f"Operator directory not found: {kernel_dir}")

    print(f"[Loader] Loading operator directory: {kernel_dir}")

    # 1. Main code: kernel_name.py
    main_path = kernel_dir / f"{kernel_name}.py"
    if main_path.exists():
        with open(main_path, 'r', encoding='utf-8') as f:
            seed_codes.append(f.read())
        print(f"[Loader] Loading main code: {main_path.name}")

    # 2. Multiple variant files (kernel_name_1.py to kernel_name_10.py)
    for i in range(1, 11):  # 1-10
        variant_path = kernel_dir / f"{kernel_name}_{i}.py"
        if variant_path.exists():
            with open(variant_path, 'r', encoding='utf-8') as f:
                seed_codes.append(f.read())
            print(f"[Loader] Loading variant {i}: {variant_path.name}")

    # 3. variants/ subfolder
    variants_dir = kernel_dir / "variants"
    if variants_dir.exists():
        variant_files = sorted(variants_dir.glob("*.py"))
        for variant_file in variant_files:
            # Avoid reloading already loaded files
            already_loaded = [f"{kernel_name}.py"] + [f"{kernel_name}_{i}.py" for i in range(1,11)]
            if variant_file.name not in already_loaded:
                with open(variant_file, 'r', encoding='utf-8') as f:
                    seed_codes.append(f.read())
                print(f"[Loader] Loading additional variant: {variant_file.name}")

    if not seed_codes:
        raise ValueError(f"No seed code found for operator {kernel_name}")

    print(f"[Loader] Loaded {len(seed_codes)} seed codes in total")
    return seed_codes


def find_test_file(kernel_path: Path, kernel_name: str) -> Path:
    """
    Find test file

    Priority:
    1. test_{kernel_name}_1.py (test case 1)
    2. test_{kernel_name}_2.py (test case 2)
    3. test_{kernel_name}_3.py (test case 3)
    4. test_{kernel_name}.py (without number)

    Returns:
        Test file path

    Raises:
        ValueError: No test file found
    """
    # Priority: find numbered test files (1, 2, 3)
    for test_case_id in range(1, 4):
        test_file = kernel_path / f"test_{kernel_name}_{test_case_id}.py"
        if test_file.exists():
            print(f"[Loader] Loading test code: {test_file.name}")
            return test_file

    # Fallback to unnumbered test file
    test_file = kernel_path / f"test_{kernel_name}.py"
    if test_file.exists():
        print(f"[Loader] Loading test code: {test_file.name}")
        return test_file

    raise ValueError(
        f"No test file found for operator {kernel_name}. "
        f"Please ensure test_{kernel_name}_1.py or test_{kernel_name}.py exists in directory {kernel_path}"
    )


def load_kernel_code(kernel_path: Path, kernel_name: str) -> tuple[str, Path]:
    """
    Load operator code and test file path

    Convention:
    - Each operator directory contains:
      - kernel_name.py: Main code (contains Triton kernel)
      - test_kernel_name_1.py: Test code (test case 1, priority)
      - test_kernel_name_2.py: Test code (test case 2)
      - test_kernel_name_3.py: Test code (test case 3)
      - test_kernel_name.py: Test code (without number, fallback)

    Args:
        kernel_path: Operator directory path (e.g. dataset/addcdiv/)
        kernel_name: Operator name

    Returns:
        (code, test_file_path): Main code and test file path
    """
    kernel_file = kernel_path / f"{kernel_name}.py"

    # Read main code
    if not kernel_file.exists():
        raise ValueError(f"Main code file not found: {kernel_file}")

    with open(kernel_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # Find test file (return path, do not read content)
    test_file = find_test_file(kernel_path, kernel_name)

    return code, test_file


def optimize_single_kernel(
    kernel_name: str,
    input_dir: Path,
    output_dir: Path,
    config: EAConfig
) -> bool:
    """
    Optimize a single operator

    Directory structure:
    input_dir/
    -- kernel_name/
        -- kernel_name.py          # Main code
        -- test_kernel_name_1.py   # Test code (test case 1)
        -- test_kernel_name_2.py   # Test code (test case 2, optional)
        -- test_kernel_name_3.py   # Test code (test case 3, optional)
        -- kernel_name_1.py        # Variant 1 (optional)
        -- variants/               # More variants (optional)
    """
    print(f"\n{'='*60}")
    print(f"Optimizing operator: {kernel_name}")
    print(f"{'='*60}")

    try:
        # Kernel-specific directory
        kernel_dir = input_dir / kernel_name

        # Load main code and test file path
        baseline_code, test_file = load_kernel_code(kernel_dir, kernel_name)

        # Get all seed codes (including variants)
        seed_codes = get_seed_codes(input_dir, kernel_name)

        # Initialize Agent, pass test file path (not code content)
        agent = TritonOptimizerAgent(config)
        agent.setup(
            baseline_code, 
            str(test_file),  # Pass file path! Avoid setup writing temp files to data directory
            kernel_name=kernel_name,
            work_dir=str(kernel_dir)  # Pass kernel-specific directory
        )

        # Run optimization
        result = agent.optimize(seed_codes, max_time=1200)

        # Save results to output directory
        kernel_output_dir = output_dir / kernel_name
        agent.save_results(str(kernel_output_dir), kernel_name)

        # Determine success
        success = result['best_fitness'] > 0

        if success:
            print(f"[Main] Optimization successful: {kernel_name} (fitness={result['best_fitness']:.4f})")
        else:
            print(f"[Main] Optimization failed: {kernel_name} (did not pass functional test)")

        return success

    except Exception as e:
        print(f"[Main] Optimization exception: {kernel_name}")
        print(f"  Error: {str(e)}")
        if config.debug:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry"""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s'
    )

    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input folder does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config - load from config.py, command-line parameters override some values
    config = EAConfig()

    # Override config values with command-line parameters (if provided)
    if args.population_size is not None:
        config.population_size = args.population_size
        print(f"[Main] Command-line override: population_size = {config.population_size}")
    if args.max_generations is not None:
        config.max_generations = args.max_generations
        print(f"[Main] Command-line override: max_generations = {config.max_generations}")
    if args.debug:
        config.debug = True

    # Print complete configuration
    print(f"\n{'='*60}")
    print("System Configuration Information")
    print(f"{'='*60}")
    config.print_config()
    print(f"{'='*60}")

    print(f"\n[Main] Path information:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")

    # Check model configuration
    if not config.llm_models:
        print(f"\nError: No LLM configured!")
        print(f"Please add at least one model to the llm_models list in config.py")
        print(f"Available models: deepseek-v3, deepseek-v3.1, deepseek-r1, qwen3, qwen3-coder, kimi-k2-instruct, glm-4.5, etc.")
        sys.exit(1)

    print(f"\n[Main] Model evolution configuration:")
    print(f"       Available models: {', '.join(config.llm_models)}")
    print(f"       Initial model: {config.llm_models[0]}")
    print(f"       Model switch probability: {config.model_switch_prob}")
    print(f"       ({config.model_switch_prob*100:.0f}% probability of switching model during mutation)")

    # Determine operators to process
    if args.kernel:
        kernels = args.kernel
    else:
        # Auto-discover all operators (find subfolders)
        kernels = []
        for item in input_dir.iterdir():
            if item.is_dir():
                # Check if corresponding .py file exists
                kernel_file = item / f"{item.name}.py"
                if kernel_file.exists():
                    kernels.append(item.name)

        kernels.sort()

    if not kernels:
        print("Error: No operator directory found")
        print(f"Please ensure there are subfolders under {input_dir}, e.g.: addcdiv/addcdiv.py")
        sys.exit(1)

    print(f"[Main] Found {len(kernels)} operators to optimize: {', '.join(kernels[:5])}" 
          + (f" ...total {len(kernels)}" if len(kernels) > 5 else ""))

    # Batch processing
    success_count = 0
    results = {}

    for kernel_name in kernels:
        success = optimize_single_kernel(
            kernel_name, input_dir, output_dir, config
        )
        results[kernel_name] = success
        if success:
            success_count += 1

    # Summary report
    print(f"\n{'='*60}")
    print("Optimization Completion Report")
    print(f"{'='*60}")
    print(f"Total: {len(kernels)} operators")
    print(f"Success: {success_count} operators")
    print(f"Failed: {len(kernels) - success_count} operators")
    print(f"Success rate: {success_count/len(kernels)*100:.1f}%")

    for kernel_name, success in results.items():
        status = "Success" if success else "Failed"
        print(f"  - {kernel_name}: {status}")

    # Return exit code
    sys.exit(0 if success_count > 0 else 1)


if __name__ == "__main__":
    main()