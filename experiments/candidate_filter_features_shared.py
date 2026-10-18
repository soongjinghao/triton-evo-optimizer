#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选预筛选的**唯一**父子特征实现（训练侧与在线侧共用）。

为什么要有这个模块
------------------
评审意见："让离线训练和在线预测使用同一套父子特征。"

在此之前存在两份实现：

  * 在线：`evolutionary_algorithm.EvolutionaryAlgorithm._filter_features`
    用 `_code_struct` / `_ratio` / `_jaccard`
  * 训练：`experiments/candidate_filter_model_final.py::add_pc_features`
    用 `candidate_filter_features_v6.struct_feats` / `ratio` / `jaccard`

两处的正则与公式**当前**恰好一致，但这是靠人工保持的：任何一处改动正则、
改 `ratio` 的缺失处理、或改"父代代码取谁"，都会静默地让训练分布与在线分布分叉，
而这种分叉在离线评估上完全看不出来，只会表现为"在线 AUC 掉到 0.49"。

本模块是唯一实现，两边都 `from experiments.candidate_filter_features_shared import ...`，
从机制上杜绝漂移。新增 `test_feature_parity.py` 会直接断言两侧产出逐字段相等。

依赖
----
只用标准库（re / math / os / json / glob），不 import torch / sklearn / pandas，
这样在线主流程（evolutionary_algorithm）与离线训练脚本都能安全地引用它。
"""

import glob
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODES_DIR = ROOT / "experiments" / "dataset" / "codes"

# ---------------------------------------------------------------- 正则
RE_BLOCK = re.compile(r"BLOCK[A-Z_]*\s*[:=]\s*(\d+)")
RE_WARPS = re.compile(r"num_warps\s*=\s*(\d+)")
RE_STAGES = re.compile(r"num_stages\s*=\s*(\d+)")
RE_TL_CALL = re.compile(r"tl\.[a-z_]+")
RE_MASK = re.compile(r"mask")
RE_TOK = re.compile(r"[A-Za-z_]\w*")


# ---------------------------------------------------------------- 基础量
def _mx(rx, code, default=0.0):
    v = [int(x) for x in rx.findall(code)]
    return float(max(v)) if v else default


def struct_feats(code):
    """从一段 Triton 代码抽取静态特征；空代码返回全 0。"""
    code = code or ""
    if not code:
        return dict(block=0.0, warps=0.0, stages=0.0,
                    ops=0.0, mask=0.0)
    return dict(
        block=_mx(RE_BLOCK, code),
        warps=_mx(RE_WARPS, code),
        stages=_mx(RE_STAGES, code),
        ops=float(len(RE_TL_CALL.findall(code))),
        mask=float(len(RE_MASK.findall(code))),
    )


def ratio(c, p):
    """父子比值取对数。两边都缺失时返回 0（表示"无变化"），与训练/在线旧口径一致。"""
    c = c if c is not None else 0.0
    p = p if p is not None else 0.0
    if c <= 0 and p <= 0:
        return 0.0
    return math.log((c + 1.0) / (p + 1.0))


def jaccard(a, b):
    a, b = a or "", b or ""
    if not a or not b:
        return 0.0
    sa, sb = set(RE_TOK.findall(a)), set(RE_TOK.findall(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------- 组合特征
def compute_code_features(child_code, parent_code):
    """由子代代码与父代代码算出 8 个父子相对特征。

    Args:
        child_code: 子代代码文本
        parent_code: **参考父代**的代码文本。
            - 传 None 表示"父代源码缺失/无真实父代"。此时仍按既有约定
              （父代=子代自身）产出数值，以保证与已训练模型的口径兼容，
              但会同时把 `parent_code_missing` 置为 True。
            - 【修复】旧实现静默地把"缺失"当成"父代=子代自身"，
              于是这批样本的数值特征与真实 gen0 完全一样，事后**无法区分**：
              离线审计看着像正常样本，实际上 8 个特征全是常数。
              现在显式给出缺失标记，训练/评估可以据此剔除。

    Returns:
        dict，含 8 个模型数值特征 + `parent_code_missing` 标记。
    """
    ch = child_code or ""
    pc = parent_code if parent_code is not None else ch
    chf, pcf = struct_feats(ch), struct_feats(pc)
    return dict(
        r_block=ratio(chf["block"], pcf["block"]),
        r_warps=ratio(chf["warps"], pcf["warps"]),
        r_stages=ratio(chf["stages"], pcf["stages"]),
        sim=jaccard(ch, pc),
        r_ops=ratio(chf["ops"], pcf["ops"]),
        r_mask=ratio(chf["mask"], pcf["mask"]),
        ops_abs=math.log1p(chf["ops"]),
        block_abs=math.log1p(chf["block"]),
        # True = 该样本的 8 个代码特征是退化的，不应参与训练/评估
        parent_code_missing=parent_code is None,
    )


CODE_FEATURE_KEYS = ("r_block", "r_warps", "r_stages", "sim",
                     "r_ops", "r_mask", "ops_abs", "block_abs")


# ---------------------------------------------------------------- 代码库
def load_codes(codes_dir=None):
    """读候选代码库 {candidate_id: code}。"""
    d = Path(codes_dir) if codes_dir else CODES_DIR
    m = {}
    for p in glob.glob(str(d / "*.py")):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                m[os.path.splitext(os.path.basename(p))[0]] = f.read()
        except OSError:
            continue
    return m


def normalize_pids(pids):
    """把 parent_ids 规范化成字符串列表（兼容 JSON 字符串 / 单字符串 / None）。"""
    if pids is None:
        return []
    if isinstance(pids, str):
        s = pids.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                return [str(x) for x in json.loads(s.replace("'", '"'))]
            except Exception:
                return [s]
        return [s]
    return [str(x) for x in pids]
