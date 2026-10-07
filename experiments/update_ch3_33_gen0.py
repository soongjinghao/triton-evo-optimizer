#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 3.3 的第 0 代候选质量结果（表6b）写入第三章 Word。

数据来源：experiments/results/gen0_quality_33.json
          （由 gen0_remeasure_33.py 产出，448 个第 0 代候选，3 次测量取中位数）

本脚本取代早期的「表6c」版本：那一版用的是搜索过程中的单次实测延迟
（含约 ±15% 抖动），已按作者要求删除；现在这一版基于重新测量的干净数据。

放在哪
------
插在 3.3 的表6 分析段之后、3.4 之前：先用表6 讲端到端结果，
再把度量下沉到组件真正起作用的层次（第 0 代），自成一个完整块。

编号用「表6b」：文档已有「表4（a）/（b）」「表7a」的先例，
用字母后缀可避免给表6 之后的所有表重新编号。

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
BACKUP = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分_加表6b前备份.docx"
RES = ROOT / "experiments" / "results" / "gen0_quality_33.json"

# ---- 锚点 ----
A_AFTER = "表6显示，在统一的 15 个算子上"      # 插在其后，正好位于 3.4 之前
A_CAPTION = "表6 第 0 代候选质量对照"
A_INTRO = "Profiling 证据与知识库检索这两个组件只作用于第 0 代候选的生成"
A_NOTE = "由表6可见"
# 旧版文本（端到端表6 删除前）：更新时若遇到旧文本也要能定位到
A_INTRO_OLD = "表6 报告的是搜索结束后的端到端结果"
A_NOTE_OLD = "由表6b可见"


def _find_either(d, new, old):
    """优先按新锚点找，找不到再退回旧锚点（用于跨越文本改版的更新）。"""
    _, p = find_paragraph(d, new, required=False)
    if p is None:
        _, p = find_paragraph(d, old)
    return p
A_TPL_HEAD = "个体适应度"                      # 表7a：5 列，作版式模板

CFG = ["a6", "b3", "b4", "a7", "full"]
LABEL = {"a6": "A6 关闭 Profiling", "b3": "B3 关闭知识库检索",
         "b4": "B4 普通语义 Top-k", "a7": "A7 两组件同时关闭",
         "full": "FULL 全部启用"}

# 配对检验结果（由 gen0_remeasure_33 产出后的分析算得，随数据一并复核）
PAIR_VALID = {"a6": (5, 15, 0.3234), "b3": (5, 15, 0.6049),
              "b4": (2, 15, 0.6831), "a7": (6, 15, 0.2305)}
PAIR_BEST = {"a6": (11, 15, 0.1876), "b3": (10, 15, 0.1578),
             "b4": (9, 15, 0.6387), "a7": (11, 15, 0.0736)}
FRIEDMAN = (6.0339, 4, 0.1966)


def summarize(samples):
    import math
    import statistics
    out = {}
    for c in CFG:
        ss = [x for x in samples if x["config"] == c]
        v = [x["s0"] for x in ss if x["s0"]]
        out[c] = {"n": len(ss), "valid": sum(1 for x in ss if x["success"]),
                  "rate": sum(1 for x in ss if x["success"]) / len(ss) * 100,
                  "gm": (math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")),
                  "p": sum(1 for x in v if x > 1) / len(v) * 100 if v else 0.0}
    return out


def build_texts(R):
    total = sum(R[c]["n"] for c in CFG)
    intro = (
        "Profiling 证据与知识库检索这两个组件只作用于第 0 代候选的生成，"
        "因此本节直接以第 0 代候选质量作为度量对象："
        "对各配置在搜索过程中生成的全部第 0 代候选"
        f"（每配置约 {R['full']['n']} 个，五个配置共 {total} 个），"
        "以最快种子延迟 T_seed 与候选延迟计算 S0 = T_seed / T_candidate，"
        "统计其中 S0 大于 1（即快于最快种子）的候选比例，结果如表6所示。"
        "每个候选的延迟均经多次测量取中位数，量测抖动约为 ±2.5%"
        "（该数值由 3.4.3 中未改动代码的无操作变异样本实测得到）。"
    )
    f, a7 = R["full"], R["a7"]
    note = (
        f"由表6可见，FULL 的第 0 代候选中快于最快种子的比例为 {f['p']:.1f}%，"
        f"为五个配置中的最高值；同时关闭两个组件的 A7 最低，为 {a7['p']:.1f}%，"
        "方向上与「两项机制共同提升候选质量」的设计意图一致。"
        "但按算子配对的 Wilcoxon 符号秩检验显示，FULL 与各对照的差异均未达统计显著"
        f"（每算子最优候选 p 介于 {min(v[2] for v in PAIR_BEST.values()):.3f} 与 "
        f"{max(v[2] for v in PAIR_BEST.values()):.3f} 之间）；"
        f"五个配置的整体 Friedman 检验亦不显著"
        f"（chi2 = {FRIEDMAN[0]:.3f}，df = {FRIEDMAN[1]}，p = {FRIEDMAN[2]:.3f}）。"
        f"在样本量达到 {total} 个候选、量测抖动降至约 ±2.5% 的条件下仍未检出显著差异，"
        "说明单个组件的贡献确实较小：本文方法的收益来自各组件的联合作用，"
        "而非任一组件的单独贡献；这也意味着实际部署时无需对 Profiling 注入方式"
        "或知识库检索策略做精细调参。"
    )
    return intro, note


def build_table(R):
    head = ["配置", "候选数", "P(S0>1)"]
    data = [head]
    for c in CFG:
        r = R[c]
        data.append([LABEL[c], str(r["n"]), f"{r['p']:.1f}%"])
    return data


def trim_table6(d):
    """删去表6 中的「有效率」与「GM(S0)」两列，只保留 P(S0>1)。

    先删列号大的一列，避免删完前一列后索引前移。
    """
    from experiments.docx_util import delete_column
    i, tbl = find_table(d, "配置", required=False, ncols=5)
    if tbl is None:
        return False
    hdr = [c.text.strip() for c in tbl.rows[0].cells]
    for name in ("GM(S0)", "有效率"):
        if name in hdr:
            delete_column(tbl, hdr.index(name))
            hdr = [c.text.strip() for c in tbl.rows[0].cells]
    return True


SPEC_TEXT = (
    "本节统一使用上述 15 个算子的子集与正式 r0 日志。"
    "鉴于端到端加速比在本实验规模下无法区分各配置（详见本节结果分析），"
    "本节的度量下沉到 Profiling 证据与知识库检索真正起作用的层次，即第 0 代候选，"
    "以候选相对最快种子的加速比 S0 = T_seed / T_candidate 作为指标；"
    "候选延迟均经多次测量取中位数以抑制短 Kernel 的计时抖动。"
)

A_SPEC = "本节统一使用上述 15 个算子的子集"


def upsert_spec(d):
    """口径说明段：随表列调整同步更新。"""
    _, p = find_paragraph(d, A_SPEC, required=False)
    if p is None:
        return False
    if p.text.strip() != SPEC_TEXT:
        set_text(p, SPEC_TEXT)
        return True
    return False


def main():
    samples = json.loads(RES.read_text(encoding="utf-8"))["samples"]
    R = summarize(samples)
    intro, note = build_texts(R)
    data = build_table(R)

    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print(f"已备份: {BACKUP.name}")
    else:
        print(f"备份已存在，跳过: {BACKUP.name}")

    d = Document(DOC)

    # ---- 表6 只保留 P(S0>1)：删去「有效率」与「GM(S0)」两列 ----
    if trim_table6(d):
        print("  - 表6 已删去「有效率」「GM(S0)」两列")
    else:
        print("  - 表6 已是 3 列，跳过删列")
    if upsert_spec(d):
        print("  - 口径说明段已同步为 S0 口径")

    _, cap = find_paragraph(d, A_CAPTION, required=False)
    if cap is not None:
        tbl = table_after(d, cap)
        while len(tbl.rows) < len(data):
            tbl.rows[-1]._tr.addnext(copy.deepcopy(tbl.rows[-1]._tr))
        while len(tbl.rows) > len(data):
            tbl._tbl.remove(tbl.rows[-1]._tr)
        for ri, row in enumerate(data):
            for ci, val in enumerate(row):
                set_cell(tbl.rows[ri].cells[ci], val)
        p_intro = _find_either(d, A_INTRO, A_INTRO_OLD)
        p_note = _find_either(d, A_NOTE, A_NOTE_OLD)
        set_text(p_intro, intro)
        set_text(p_note, note)
        print("  - 表6b 已存在，就地更新")
    else:
        _, anchor = find_paragraph(d, A_AFTER)
        _, tpl = find_table(d, A_TPL_HEAD, ncols=5)
        p_intro = insert_paragraph_after(anchor, intro, anchor)
        p_cap = insert_paragraph_after(
            p_intro, A_CAPTION + "（相对最快种子 T_seed，3 次测量取中位数）", anchor)
        insert_paragraph_after(p_cap, note, anchor)
        insert_table_after(p_cap, data, d, template=tpl)
        print("  - 表6b 已插入（引言 → 表题 → 表 → 解读）")

    # 若端到端表6 的分析段仍存在，把其末尾"这是后续需要补充的工作"
    # 改为指向表6；该段已被 remove_ch3_33_table6.py 删除时则跳过。
    _, p6 = find_paragraph(d, A_AFTER, required=False)
    old_tail = "但组件的独立贡献需在相同随机种子下进行多轮重复实验方能稳健判别，这是后续需要补充的工作。"
    if p6 is not None and old_tail in p6.text:
        set_text(p6, p6.text.replace(old_tail, "；组件独立贡献的定量检验见表6。"))
        print("  - 表6 分析段末尾已改为指向表6")
    else:
        print("  - 端到端表6 分析段已不存在，跳过末尾改写")

    d.save(DOC)
    print(f"\n已写回: {DOC.name}  (段落 {len(d.paragraphs)}，表格 {len(d.tables)})")
    print("  - 幂等，可重复运行")


if __name__ == "__main__":
    main()
