#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 3.3 的「第 0 代候选质量」表（表6c）与说明写入第三章 Word。

数据来源：experiments/results/table6c_gen0_quality.json
          （由 analyze_33_gen0.py 从现有日志算出，不需要 NPU）

放在哪
------
插在 3.3 的表6 分析段之后、3.4 之前：先用表6 / 表6b 讲端到端结果，
再把度量下沉到组件真正起作用的层次（第 0 代），自成一个完整块。

编号用「表6c」：文档已有「表4（a）/（b）」「表6b」的先例，用字母后缀
可避免给表6 之后的所有表重新编号、进而改动全文的交叉引用。

用法
----
    python experiments/update_ch3_33_gen0.py
"""
import copy
import json
import shutil
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.docx_util import (                     # noqa: E402
    find_paragraph, find_table, insert_paragraph_after,
    insert_table_after, set_cell, set_text, table_after,
)

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"
BACKUP = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分_加表6c前备份.docx"
RES = ROOT / "experiments" / "results" / "table6c_gen0_quality.json"

# ---- 锚点 ----
A_AFTER = "表6显示，在统一的 15 个算子上"      # 3.3 的端到端分析段
A_INTRO = "表6 与表6b 报告的是搜索结束后的端到端结果"
A_CAPTION = "表6c 第 0 代候选质量对照"
A_NOTE = "由表6c可见"
A_TABLE_TPL = "对照配置"                      # 表6b：7 列，作版式模板


def build_texts(res):
    rows = res["rows"]
    by = {r["label"]: r for r in rows}

    intro = (
        "表6 与表6b 报告的是搜索结束后的端到端结果，而 Profiling 证据与知识库检索这两个组件"
        "实际只作用于第 0 代候选的生成。为把度量下沉到组件真正起作用的层次，"
        "这里补充一组第 0 代候选的直接对照：对各配置在搜索过程中生成的全部第 0 代候选，"
        "以最快种子延迟 T_seed 与候选实测延迟计算 S0 = T_seed / T_candidate，"
        "统计候选有效率、S0 的几何平均与中位数、S0 大于 1 的比例，"
        "以及每个算子上第 0 代最优候选的加速比，结果如表6c所示。"
        "需要说明的是，第 0 代候选的延迟取自搜索过程中的单次实测，"
        "含约 ±15% 的量测抖动（该数值由 3.4.3 的无操作对照组实测得到），"
        "因此本表用于说明组件作用的层次，其数值差异本身不足以单独支撑显著性结论。"
    )

    pw = {q["label"]: q for q in res["pairwise"]}
    ps = "、".join(f"与 {q['label']} 为 {q['p']:.3f}" for q in res["pairwise"])
    note = (
        f"由表6c可见，五个配置的第 0 代候选有效率介于 "
        f"{min(r['valid_rate'] for r in rows)*100:.1f}% 与 "
        f"{max(r['valid_rate'] for r in rows)*100:.1f}% 之间，"
        f"S0 的中位数均落在 {min(r['median_s0'] for r in rows):.4f} 至 "
        f"{max(r['median_s0'] for r in rows):.4f} 附近，"
        "即多数第 0 代候选与最快种子基本持平而略有不及，"
        "这与第 0 代以策略引导的多样性探索为主、收益主要由后续进化取得的设计预期一致。"
        "在更能反映第 0 代质量的“每算子最优候选”指标上，"
        f"B3 为 {by['B3']['gm_best_s0']:.4f}、A7 为 {by['A7']['gm_best_s0']:.4f}、"
        f"B4 为 {by['B4']['gm_best_s0']:.4f}、A6 为 {by['A6']['gm_best_s0']:.4f}、"
        f"FULL 为 {by['FULL']['gm_best_s0']:.4f}；"
        "各配置互有高低，未呈现一致排序，且 FULL 在有效率与每算子最优候选两项上均处于末位或接近末位。"
        f"以 {pw['A6']['n_pairs']} 个算子的配对结果做 Wilcoxon 符号秩检验，"
        f"FULL 与各对照的差异均不显著（p 值分别为 {ps}）。"
        "结合表6 的端到端结果同样未呈现显著差异可以认为："
        "在本文的搜索预算与算子规模下，Profiling 证据与受约束知识库检索对第 0 代候选质量的影响，"
        "无法在单次实测的噪声水平之上被分离出来；"
        "3.1 主实验中 FULL 优于两个对照方法的整体结论，来自多组件的联合作用，"
        "而非其中任一组件可单独度量的贡献。"
    )
    return intro, note


def build_table(res):
    head = ["配置", "候选数", "有效率", "GM(S0)", "中位 S0", "P(S0>1)", "每算子最优 S0(GM)"]
    data = [head]
    for r in res["rows"]:
        data.append([
            r["label"], str(r["n_candidates"]),
            f"{r['valid_rate']*100:.1f}%",
            f"{r['gm_s0']:.4f}", f"{r['median_s0']:.4f}",
            f"{r['p_better']*100:.1f}%", f"{r['gm_best_s0']:.4f}",
        ])
    return data


def main():
    res = json.loads(RES.read_text(encoding="utf-8"))
    intro, note = build_texts(res)
    data = build_table(res)

    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print(f"已备份: {BACKUP.name}")
    else:
        print(f"备份已存在，跳过: {BACKUP.name}")

    d = Document(DOC)
    _, cap = find_paragraph(d, A_CAPTION, required=False)
    if cap is not None:
        # 已存在：只更新表体与两段文字，不重复插入
        tbl = table_after(d, cap)
        while len(tbl.rows) < len(data):
            tbl.rows[-1]._tr.addnext(copy.deepcopy(tbl.rows[-1]._tr))
        while len(tbl.rows) > len(data):
            tbl._tbl.remove(tbl.rows[-1]._tr)
        for ri, row in enumerate(data):
            for ci, val in enumerate(row):
                set_cell(tbl.rows[ri].cells[ci], val)
        _, p_intro = find_paragraph(d, A_INTRO)
        _, p_note = find_paragraph(d, A_NOTE)
        set_text(p_intro, intro)
        set_text(p_note, note)
        print("  - 表6c 已存在，就地更新")
    else:
        _, anchor = find_paragraph(d, A_AFTER)
        _, tpl = find_table(d, A_TABLE_TPL)
        p_intro = insert_paragraph_after(anchor, intro, anchor)
        p_cap = insert_paragraph_after(p_intro, A_CAPTION + "（相对最快种子 T_seed）", anchor)
        insert_paragraph_after(p_cap, note, anchor)
        insert_table_after(p_cap, data, d, template=tpl)   # 落在标题与解读之间
        print("  - 表6c 已插入（引言 → 表题 → 表 → 解读）")

    d.save(DOC)
    print(f"\n已写回: {DOC.name}")
    print("  - 位置: 3.3 表6 分析段之后、3.4 之前")
    print("  - 幂等，可重复运行；其余脚本按文字锚点定位，不受影响")


if __name__ == "__main__":
    main()
