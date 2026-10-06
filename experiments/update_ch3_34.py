#!/usr/bin/env python3
"""把 3.4 节的最新结果写入第三章 Word。

数据来源（保证与表7/8/9 一致）：
  experiments/results/table7_selection.json
  experiments/results/table8_crossover.json
  experiments/results/table9_mutation.json
GM(S) 口径：复测中位数后再跨算子取几何平均（analyze_logs.gm_s_for）。

用法：
    python experiments/update_ch3_34.py
"""

import json
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.docx_util import (                     # noqa: E402
    find_paragraph, find_table, insert_paragraph_after, set_cell, set_text,
)

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"
RES = ROOT / "experiments" / "results"


def load(name):
    return {r["config"]: r for r in json.load(open(RES / name, encoding="utf-8"))}


def main():
    d = Document(DOC)
    t7 = load("table7_selection.json")
    t8 = load("table8_crossover.json")
    t9 = load("table9_mutation.json")

    # ---------- 1. 更新三张表 ----------
    # 3.4.1（表5 选择策略 + [40]/[41]/[44] 正文）统一交由 update_ch3_341 处理，
    # 那边维护的是四种选择策略与显著性检验的口径，此处不再单独实现以免互相覆盖。
    from experiments.update_ch3_341 import apply_341, PROBE_JSON

    probe = (json.loads(PROBE_JSON.read_text(encoding="utf-8"))
             if PROBE_JSON.exists() else None)
    apply_341(d, t7, probe)

    # 表6：交叉策略（按表头首列定位）
    _, tbl = find_table(d, "交叉策略")
    for row, cfg in ((1, "cross_unconstrained"), (2, "cross_protected")):
        r = t8[cfg]
        set_cell(tbl.rows[row].cells[1], f"{r['valid_rate'] * 100:.1f}%")
        set_cell(tbl.rows[row].cells[2], f"{r['better_ratio'] * 100:.1f}%")
        set_cell(tbl.rows[row].cells[3], f"{r['gm_g_cross']:.4f}")

    # 表7：变异策略（按表头首列定位）
    _, tbl = find_table(d, "变异策略")
    for row, cfg in ((1, "mut_adaptive"), (2, "mut_uniform"), (3, "mut_aggressive")):
        r = t9[cfg]
        set_cell(tbl.rows[row].cells[1], f"{r['valid_rate'] * 100:.1f}%")
        set_cell(tbl.rows[row].cells[2], f"{r['success_rate'] * 100:.1f}%")
        set_cell(tbl.rows[row].cells[3], f"{r['gm_g_mut']:.4f}")
        set_cell(tbl.rows[row].cells[4], f"{r['gm_s']:.4f}")

    # ---------- 2. 更新正文分析 ----------
    # P44（3.4.1 分析）已在 update_ch3_341.py 中按四种策略的结论维护，此处不再定义。
    P52 = (
        "由表8可见，保护交叉的有效子代中优于主干父代的比例为 "
        f"{t8['cross_protected']['better_ratio'] * 100:.1f}%，明显高于无约束交叉的 "
        f"{t8['cross_unconstrained']['better_ratio'] * 100:.1f}%，接近后者的 2.7 倍；"
        f"保护交叉的 GM(G_cross) 为 {t8['cross_protected']['gm_g_cross']:.4f}，"
        f"即子代平均快于主干父代，而无约束交叉为 {t8['cross_unconstrained']['gm_g_cross']:.4f}，"
        "子代平均反而慢于主干父代。这表明在多处局部优化同时存在时，无约束地重组多个部分容易破坏父代已成立的 "
        "shape、stride、对齐与 Mask 条件；而以性能较优父代为骨架、每次只移植一个相容的局部优化，"
        "更能在有限预算内稳定产生有效改进。最终 GM(S) 方面，保护交叉为 "
        f"{t8['cross_protected']['gm_s']:.4f}，高于无约束交叉的 {t8['cross_unconstrained']['gm_s']:.4f}，"
        "方向与操作层面指标一致，但该端到端差异未达统计显著（p = 0.762），"
        "提示交叉策略的价值主要体现在提高单次交叉的有效性，而非改变最终加速比的量级。"
    )

    P60 = (
        "由表9可见，三种变异策略的有效变异率均较高"
        f"（{min(t9[c]['valid_rate'] for c in t9) * 100:.1f}% 至 "
        f"{max(t9[c]['valid_rate'] for c in t9) * 100:.1f}%），"
        "但优化成功率存在差异：激进变异为 "
        f"{t9['mut_aggressive']['success_rate'] * 100:.1f}%，均匀变异为 "
        f"{t9['mut_uniform']['success_rate'] * 100:.1f}%，自适应变异为 "
        f"{t9['mut_adaptive']['success_rate'] * 100:.1f}%。更值得注意的是，"
        f"三种策略的 GM(G_mut) 分别为 {t9['mut_adaptive']['gm_g_mut']:.4f}、"
        f"{t9['mut_uniform']['gm_g_mut']:.4f} 与 {t9['mut_aggressive']['gm_g_mut']:.4f}，"
        "均接近 1，说明单次变异操作平均而言几乎不带来净改进。最终 GM(S) 方面，均匀变异为 "
        f"{t9['mut_uniform']['gm_s']:.4f}，激进变异为 {t9['mut_aggressive']['gm_s']:.4f}，"
        f"自适应变异为 {t9['mut_adaptive']['gm_s']:.4f}；"
        "三者两两比较均不显著（p = 0.345 至 1.000），与 FULL 配置的差异亦不显著（p = 0.561 至 0.804）。"
        "可见变异策略的主要区别不在于能否生成可运行代码，而在于能否把有限的评测次数集中到更可能带来改进的方向上，"
        "但这种方向性差异在单轮搜索下尚不足以形成稳定的端到端优劣排序。"
    )

    SUMMARY = (
        "综合 3.4.1 至 3.4.3 的结果，可以得到三点认识。第一，父代选择策略对最终结果存在统计显著的影响："
        f"四种策略的 GM(S) 由高到低依次为 UCB 式选择 {t7['sel_ucb']['gm_s']:.4f}、"
        f"轮盘赌 {t7['sel_roulette']['gm_s']:.4f}、均匀选择 {t7['sel_uniform']['gm_s']:.4f} 与"
        f"三元锦标赛 {t7['sel_tournament']['gm_s']:.4f}，Friedman 检验达到统计显著"
        "（chi2 = 11.0878，df = 3，p = 0.0113），且逐个算子剔除均不改变最优策略，结论稳健。"
        "UCB 之所以最优，在于其评分同时包含利用项与探索项，既避免像均匀选择那样"
        "把评测浪费在未经证实的候选上（其被选父代中约一半适应度为 0），"
        "也为搜索保留了跳出当前最优父代邻域的机会，因而在每算子仅 13 次评测的紧预算下最为有效。第二，交叉策略的差异主要体现在操作层面而非端到端量级："
        f"主干保持型保护交叉将有效子代的优化成功率由 {t8['cross_unconstrained']['better_ratio'] * 100:.1f}% 提升至 "
        f"{t8['cross_protected']['better_ratio'] * 100:.1f}%，GM(G_cross) 由 "
        f"{t8['cross_unconstrained']['gm_g_cross']:.4f} 提升至 {t8['cross_protected']['gm_g_cross']:.4f}，"
        "表明在 Triton kernel 这类对 shape、stride、对齐与 Mask 条件高度敏感的代码上，"
        "交叉操作必须保留主干的结构约束才能稳定产生有效后代。第三，三组实验中变异操作的平均增益 GM(G_mut) 均接近 1"
        f"（{min(t9[c]['gm_g_mut'] for c in t9):.4f} 至 {max(t9[c]['gm_g_mut'] for c in t9):.4f}），"
        "说明在 13 次的有限预算下后续进化操作的边际贡献有限，端到端加速主要源自第 0 代基于 Profiling 证据的"
        "策略引导初始化与受约束知识库检索（见 3.3）。因此总体而言，本文方法对进化算子的具体选取具有鲁棒性，"
        "实际部署时无需针对交叉或变异策略做精细调参，而应优先保证策略初始化与检索质量。"
    )

    # 3.4.1 的分析段已由 apply_341 写入，此处不得再覆盖。
    # 段落一律按文字锚点定位：3.4.1 中新增/更新表7a 会移动其后所有下标，
    # 若沿用硬索引就会把文字静默写到错误位置。
    _, p52 = find_paragraph(d, "由表8可见")
    _, p60 = find_paragraph(d, "由表9可见")
    set_text(p52, P52)
    set_text(p60, P60)

    # 小结段落必须幂等：原实现每次运行都插入一次，重复运行会不断累积。
    _, existing = find_paragraph(d, "综合 3.4.1 至 3.4.3 的结果", required=False)
    if existing is not None:
        set_text(existing, SUMMARY)
    else:
        insert_paragraph_after(p60, SUMMARY, p60)

    d.save(DOC)
    print(f"已更新: {DOC}")
    print("  - 表5/6/7 数值 -> 取自 table7/8/9 json（复测口径）")
    print("  - 3.4.1 / 3.4.2 / 3.4.3 分析文字已重写")
    print("  - 3.4 小结段落已插入（显著性 + 机制归因）")


if __name__ == "__main__":
    main()
