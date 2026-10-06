#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 3.3 与 3.4 的开头各补一句：两套消融实验都在同一批 15 个算子的子集上进行。

背景
----
第三章的算子集合并不统一：3.1 主实验覆盖 50 个算子，3.2 阈值敏感性用 17 个
（与 K15 仅 7 个重叠），而 3.3 与 3.4 各小节共用同一批 15 个算子（K15）。
正文中此前只在 3.3 的第二段一带而过（"主表统一使用同一批 15 个代表性算子"），
3.4 开头完全没有交代，读者无法判断跨节数字是否可比，因此需要显式点明。

口径数据（供正文引用，取自 experiments/manifest/*/t_base_us）
------------------------------------------------------------
15 个算子的 T_base 覆盖 1.58 ~ 526.83 μs，跨三个数量级；
按 <5 / 5~100 / ≥100 μs 三档占比为 47% / 33% / 20%，
与 50 个算子全集的 52% / 30% / 18% 基本一致，可据此说明其代表性。

用法
----
    python experiments/update_ch3_kernel_subset.py
"""
import shutil
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.docx_util import find_paragraph, set_text   # noqa: E402

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"
BACKUP = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分_加算子集说明前备份.docx"

# ---- 锚点：两节正文的第一段 ----
A_33 = "本实验验证硬件证据和知识库检索"
A_34 = "本节分别验证父代选择、交叉和变异三个组件"

# ---- 要补的句子 ----
NOTE_33 = (
    "本节以及后续 3.4 各节的消融实验，均统一在 3.1 主实验所用 50 个算子中选取的 "
    "15 个代表性算子子集上进行；该子集的参考实现延迟覆盖 1.58 至 526.83 μs，"
    "跨三个数量级，且各延迟档的占比与 50 个算子的全集基本一致，因此可以保证跨小节的口径可比。"
)
NOTE_34 = "与 3.3 相同，本节的三个实验均统一在上述 15 个算子的子集上进行。"


def append_once(p, note):
    """句末补一句；已存在则不再重复。"""
    if note in p.text:
        return False
    set_text(p, p.text.rstrip() + note)
    return True


def main():
    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print(f"已备份: {BACKUP.name}")
    else:
        print(f"备份已存在，跳过: {BACKUP.name}")

    d = Document(DOC)
    _, p33 = find_paragraph(d, A_33)
    _, p34 = find_paragraph(d, A_34)

    c33 = append_once(p33, NOTE_33)
    c34 = append_once(p34, NOTE_34)

    d.save(DOC)
    print(f"\n已写回: {DOC.name}")
    print(f"  - 3.3 开头: {'已补入' if c33 else '已存在，跳过'}")
    print(f"  - 3.4 开头: {'已补入' if c34 else '已存在，跳过'}")
    print("  说明: 幂等，可重复运行；其余脚本按文字锚点定位，不受影响")


if __name__ == "__main__":
    main()
