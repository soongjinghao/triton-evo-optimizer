"""3.1 主实验公共模块：路径、事件日志、Kernel 装载、执行器构造。"""

import json
import time
import random
from pathlib import Path
from typing import List, Optional

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = AGENT_DIR / "datasets"
EXPERIMENTS_DIR = AGENT_DIR / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
MANIFEST_DIR = EXPERIMENTS_DIR / "manifest"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# 3.1 试点算子：覆盖 1.3us ~ 415us 三个量级与多种算子类型
PILOT_KERNELS = [
    "eye_kernel",                  # 1.34us  超短 Kernel，测量噪声探针
    "matmul_kernel_simplified",    # 2.44us  Tile 类
    "_act_quant_kernel",           # 5.18us  量化类
    "_rms_norm_kernel",            # 256.50us 归一化类
    "mean_kernel",                 # 415.12us 归约类
]

METHODS = ["b1", "b2", "full"]


def ensure_dirs():
    for d in (LOGS_DIR, MANIFEST_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def list_kernels() -> List[str]:
    return sorted(d.name for d in DATASETS_DIR.iterdir()
                  if d.is_dir() and (d / f"{d.name}.py").exists())


def read_code(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def seed_code_paths(kernel_name: str) -> List[Path]:
    """返回该 Kernel 的全部初始种子源码路径（主实现 + 变体）。"""
    kdir = DATASETS_DIR / kernel_name
    paths = [kdir / f"{kernel_name}.py"]
    for i in range(1, 11):
        p = kdir / f"{kernel_name}_{i}.py"
        if p.exists():
            paths.append(p)
    return [p for p in paths if p.exists()]


def find_test_file(kernel_name: str) -> Path:
    kdir = DATASETS_DIR / kernel_name
    for i in range(1, 4):
        p = kdir / f"test_{kernel_name}_{i}.py"
        if p.exists():
            return p
    p = kdir / f"test_{kernel_name}.py"
    if p.exists():
        return p
    raise ValueError(f"未找到测试文件: {kernel_name}")


class ExpLogger:
    """JSONL 事件日志：每个候选一行，覆盖 AST 拒绝与 NPU 评测两类事件。"""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict):
        if not self.path:
            return
        record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def geometric_mean(values) -> float:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def diversity_g0(codes) -> float:
    """第 0 代多样性 D_G0 = 1 - 平均两两源码相似度。

    在第 0 代种群的全部不同候选个体上计算（相同源码先去重）。
    """
    from difflib import SequenceMatcher

    uniq = []
    for c in codes:
        if c and c not in uniq:
            uniq.append(c)
    n = len(uniq)
    if n < 2:
        return 0.0

    sims = [SequenceMatcher(None, uniq[i], uniq[j]).ratio()
            for i in range(n) for j in range(i + 1, n)]
    return 1.0 - sum(sims) / len(sims)


def seed_similarity(kernel_name: str):
    """返回该 Kernel 两个初始种子的源码相似度（不足两个种子时返回 None）。"""
    from difflib import SequenceMatcher

    paths = seed_code_paths(kernel_name)
    if len(paths) < 2:
        return None
    return SequenceMatcher(None, read_code(paths[0]), read_code(paths[1])).ratio()
