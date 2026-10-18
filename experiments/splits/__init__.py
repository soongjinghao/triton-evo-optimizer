#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定的算子划分（训练 / 留出），供训练、在线实验与汇总统一引用。

为什么需要它
------------
此前"15 个开发算子"是从 analyze_logs.K15 推导的，而 K15 是扫描
results/b3/ 目录 glob 出来的 **set**：

  * 顺序不确定 → `DEVS[:3]`（冒烟取哪 3 个算子）在每次运行都可能不同；
  * 硬依赖目录 → b3 结果被清理后 K15 为空，run_phase 会静默跑 0 个算子；
  * 无法记录版本 → 离线表11 与在线实验可能用了不同的划分，论文不自洽。

改为把划分固化成 JSON：训练脚本、run_phase、analyze_logs 全部从这里读，
并带 version 与 dataset_snapshot，保证"哪份数据 + 哪个划分 + 哪个模型"可追溯。

用法
----
    from experiments.splits import load_split
    sp = load_split("dev15")
    sp["test_kernels"]   # 15 个留出算子
    sp["train_kernels"]  # 其余 35 个训练算子
"""

import json
from pathlib import Path

SPLITS_DIR = Path(__file__).resolve().parent


def split_path(name: str) -> Path:
    return SPLITS_DIR / f"{name}.json"


def load_split(name: str = "dev15") -> dict:
    """读取划分文件；不存在时报错（不静默返回空集，避免跑到 0 个算子）。"""
    p = split_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"未找到划分文件 {p}。可用划分："
            f"{[f.stem for f in SPLITS_DIR.glob('*.json')]}")
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("test_kernels", "train_kernels"):
        data.setdefault(key, [])
        data[key] = sorted(set(data[key]))
    # 一致性自检：训练与测试不得相交
    overlap = set(data["test_kernels"]) & set(data["train_kernels"])
    if overlap:
        raise ValueError(f"划分 {name} 训练/测试存在交集（泄漏）：{sorted(overlap)}")
    return data


def write_split(name, test_kernels, train_kernels, version, dataset_snapshot,
                note="", overwrite=False):
    """写入（或更新）一份划分。"""
    p = split_path(name)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} 已存在，overwrite=True 才会覆盖")
    data = {
        "name": name,
        "version": version,
        "dataset_snapshot": dataset_snapshot,
        "n_test": len(test_kernels),
        "n_train": len(train_kernels),
        "test_kernels": sorted(set(test_kernels)),
        "train_kernels": sorted(set(train_kernels)),
        "note": note,
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
