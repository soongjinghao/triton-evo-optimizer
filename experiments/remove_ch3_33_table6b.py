#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""删除 3.3 中的「表6b 加速比配对比较（以 FULL 为基准）」及其说明段。

删除理由
--------
表6b 采用「FULL 胜 / FULL 负 / 持平」的胜负计数口径，与全文其余各表
（有效候选率、GM(S)、GM(G_cross)、GM(G_mut) 等比值型指标）的口径不一致，
放在正文里显得突兀，因此连同表题、表体与说明段一并删除。
3.3 的结论仍由表6（有效候选率 + GM(S)）承载，删表6b 不影响论证完整性。

删除范围
--------
  1. 表题段：「表6b 加速比配对比较（以 FULL 为基准）」
  2. 表体  ：紧随该表题之后的那张表（7 列，首格为「对照配置」）
  3. 说明段：「表6b 对每个算子逐一比较 FULL 与对照配置的加速比……」

用法
----
    python experiments/remove_ch3_33_table6b.py
"""
import shutil
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.docx_util import (                     # noqa: E402
    find_paragraph, remove_element, table_after,
)

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"
BACKUP = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分_删表6b前备份.docx"

A_CAPTION = "表6b 加速比配对比较"
A_DESC = "表6b 对每个算子逐一比较 FULL 与对照配置的加速比"
A_ANALYSIS = "表6显示，在统一的 15 个算子上"

# 分析段中围绕表6b 展开的那一整块：从引入句到收尾句（含）
CUT_START = "为得到稳健判断，表6b 补充逐算子配对比较。"
CUT_END = "未见任一消融配置能够稳定优于完整方法。"

# 原结论里的"未达统计显著"依赖表6b 的配对口径，删表后改为可由表6 直接支撑的表述
OLD_CLAIM = "在单轮搜索的方差水平下，各消融配置与完整方法的加速比差异未达统计显著"
NEW_CLAIM = ("在单轮搜索的方差水平下，各消融配置与完整方法的加速比处于同一量级"
             "（GM(S) 介于 1.9697 与 2.2715 之间）")


def fix_analysis(d):
    """删除分析段中围绕表6b 的整块论述，并修正依赖该表的结论措辞。"""
    from experiments.docx_util import set_text
    _, p = find_paragraph(d, A_ANALYSIS, required=False)
    if p is None:
        return False
    text = p.text.strip()
    if CUT_START not in text:
        return False
    i = text.index(CUT_START)
    j = text.index(CUT_END) + len(CUT_END)
    new = (text[:i] + text[j:]).replace(OLD_CLAIM, NEW_CLAIM)
    new = new.replace("。 。", "。").replace("  ", " ")
    set_text(p, new)
    return True


def main():
    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print(f"已备份: {BACKUP.name}")
    else:
        print(f"备份已存在，跳过: {BACKUP.name}")

    d = Document(DOC)
    _, cap = find_paragraph(d, A_CAPTION, required=False)
    if cap is None:
        # 表已删过，但仍要保证分析段里的表6b 论述被清理（否则重跑会漏改）
        print("表6b 已不存在，跳过删表")
    else:
        # 表体必须在删除表题之前定位，否则找不到相对位置
        tbl = table_after(d, cap)
        _, desc = find_paragraph(d, A_DESC, required=False)

        if tbl is not None:
            remove_element(tbl)
            print("  - 已删除表体")
        remove_element(cap)
        print("  - 已删除表题段")
        if desc is not None:
            remove_element(desc)
            print("  - 已删除说明段")

    # 分析段里围绕表6b 的整块论述一并删除
    if fix_analysis(d):
        print("  - 已删除分析段中围绕表6b 的论述，并修正其结论措辞")
    else:
        print("  - 分析段无需修改")

    d.save(DOC)

    # 检查全文是否还有对表6b 的残留引用
    d2 = Document(DOC)
    left = [(i, p.text.strip()[:90]) for i, p in enumerate(d2.paragraphs)
            if "表6b" in p.text]
    print(f"\n已写回: {DOC.name}  (段落 {len(d2.paragraphs)}，表格 {len(d2.tables)})")
    if left:
        print(f"⚠ 仍有 {len(left)} 处提到「表6b」，需人工确认：")
        for i, t in left:
            print(f"    P[{i}] {t}")
    else:
        print("  ✓ 全文已无「表6b」的残留引用")


if __name__ == "__main__":
    main()
