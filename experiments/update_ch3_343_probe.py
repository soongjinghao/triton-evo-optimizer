#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 3.4.3 的变异机制探测结果（表9a）写入第三章 Word，并同步更新表9 与分析。

数据来源
--------
  experiments/results/mutation_probe.json      —— 270 个变异体（30 父代 × 3 策略 × 3 次）
  experiments/results/table9_mutation.json     —— 15 算子端到端结果（表9）

为什么加表9a
------------
原 3.4.3 只报告端到端 GM(S)，而该指标经过两代进化与精英保留后被稀释，
且三种策略发生变异时的父代各不相同，无法归因。表9a 在**固定父代**上比较
三种策略，父代相同且必定有效，因此：
  * 观测到的差异只能来自策略本身；
  * 每个变异体都带父代延迟，样本可用率接近 100%
    （原端到端实验约 65% 的样本因父代评测失败无法计算增益）。

编号用「表9a」：与文档已有的「表4（a）/（b）」「表6b」「表7a」一致，
避免给表9 之后的表重新编号。

用法
----
    python experiments/update_ch3_343_probe.py
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
    insert_table_after, remove_element, set_cell, set_text, table_after,
)

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"
BACKUP = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分_加表9a前备份.docx"
PROBE = ROOT / "experiments" / "results" / "mutation_probe.json"
T9 = ROOT / "experiments" / "results" / "table9_mutation.json"

# ---- 锚点 ----
A_DESIGN = "变异实验比较自适应变异"              # 3.4.3 实验设计段
A_PRINCIPLE = "具体地，三类变异操作的含义如下"     # 策略原理段（插在设计段之后）
A_ANCHOR = "G_mut = T_parent / T_child"      # 插在其后，正好位于表9 之前
A_CAPTION9A = "表9a 变异策略机制探测"
A_INTRO = "上述三种策略的差异只体现在变异类型的采样方式上"
A_NOTE = "由表9a可见"
A_ANALYSIS = "由表9可见"
A_TABLE9_HEAD = "变异策略"
A_TPL_HEAD = "方法"                           # 表4（a）：7 列，作版式模板

# 旧的端到端表9（已决定删除）：表题、表体、补充说明段、分析段
A_OLD_CAPTION = "表9 三种变异策略总体对比"
A_OLD_EXTRA = "实验日志额外记录每次变异实际采用的类型"
A_OLD_ANALYSIS = "由表9可见，三种变异策略的有效变异率"

# 机制探测表：删除旧表9 后由「表9a」改名为「表9」，
# 否则会出现「有 9a 而无 9」的孤儿编号，且表8 之后直接跳到表10。
A_CAPTION_MECH_OLD = "表9a 变异策略机制探测"
A_CAPTION_MECH_NEW = "表9 变异策略机制探测"

ORDER = ("adaptive", "uniform", "aggressive")
LABEL = {"adaptive": "自适应变异", "uniform": "均匀变异", "aggressive": "激进变异"}


def probe_rows(res):
    """从探测结果汇总出表9a 需要的数字。"""
    import math
    import statistics
    import collections
    import random

    S = [x for x in res["samples"] if not x.get("_failed") and x.get("strategy")]

    def boot(v, fn, B=10000, seed=11):
        rng = random.Random(seed)
        s = sorted(fn([v[rng.randrange(len(v))] for _ in v]) for _ in range(B))
        return s[int(.025 * B)], s[int(.975 * B)]

    out = {}
    for s in ORDER:
        ss = [x for x in S if x["strategy"] == s]
        c = collections.Counter(x["mutation_type"] for x in ss)
        v = [x["g_mut"] for x in ss if x["g_mut"]]
        lo, hi = boot(v, lambda z: sum(1 for y in z if y > 1) / len(z))
        out[s] = {
            "n": len(ss),
            "param": c.get("param_tuning", 0) / len(ss) * 100,
            "arith": c.get("arithmetic_and_mask", 0) / len(ss) * 100,
            "struct": c.get("structure_rewrite", 0) / len(ss) * 100,
            "noop": sum(1 for x in ss if x.get("is_noop")) / len(ss) * 100,
            "rate": sum(1 for x in v if x > 1) / len(v) * 100,
            "lo": lo * 100, "hi": hi * 100,
            "median": statistics.median(v),
            "sim": statistics.fmean([x["similarity"] for x in ss]),
            "lines": statistics.fmean([x["changed_lines"] for x in ss]),
        }
    noise = [x["g_mut"] for x in S if x.get("is_noop") and x["g_mut"]]
    out["_noise"] = {
        "n": len(noise),
        "mean": statistics.fmean(noise),
        "sd": statistics.stdev(noise) if len(noise) > 1 else float("nan"),
    }
    # 方差分解
    v = [math.log(x["g_mut"]) for x in S if x["g_mut"]]
    g0 = statistics.fmean(v)
    grp = {s: [math.log(x["g_mut"]) for x in S if x["strategy"] == s and x["g_mut"]]
           for s in ORDER}
    ssb = sum(len(g) * (statistics.fmean(g) - g0) ** 2 for g in grp.values())
    ssw = sum(sum((y - statistics.fmean(g)) ** 2 for y in g) for g in grp.values())
    bp = collections.defaultdict(list)
    for x in S:
        if x["g_mut"]:
            bp[x["parent_id"]].append(math.log(x["g_mut"]))
    ssb_p = sum(len(g) * (statistics.fmean(g) - g0) ** 2 for g in bp.values())
    tot = ssb + ssw
    out["_var"] = {"strategy": ssb / tot * 100, "noise": ssw / tot * 100,
                   "parent": ssb_p / tot * 100}
    # adaptive 分层
    out["_stratum"] = {}
    for st in ("L1(<1.5)", "L3(>2.5)"):
        sub = [x for x in S if x["strategy"] == "adaptive" and x["parent_stratum"] == st]
        if sub:
            c = collections.Counter(x["mutation_type"] for x in sub)
            out["_stratum"][st] = {
                "n": len(sub),
                "struct": c.get("structure_rewrite", 0) / len(sub) * 100,
                "arith": c.get("arithmetic_and_mask", 0) / len(sub) * 100,
            }
    # 按父代配对比较：同一父代上，先把该策略各次变异的对数增益取均值，
    # 再对两策略之差作配对检验。这样父代质量在差值中被抵消，剩下的是策略差异。
    import itertools
    try:
        from scipy.stats import wilcoxon
    except Exception:                                     # pragma: no cover
        wilcoxon = None
    by_par = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in S:
        if x["g_mut"]:
            by_par[x["parent_id"]][x["strategy"]].append(math.log(x["g_mut"]))
    pairs, ps, ns = [], [], []
    for p1, p2 in itertools.combinations(ORDER, 2):
        dif = [statistics.fmean(m[p1]) - statistics.fmean(m[p2])
               for m in by_par.values() if p1 in m and p2 in m and m[p1] and m[p2]]
        lo_d, hi_d = boot(dif, statistics.fmean)
        p = float(wilcoxon(dif).pvalue) if wilcoxon and len(dif) > 5 else float("nan")
        pairs.append({"a": p1, "b": p2, "n": len(dif), "delta": statistics.fmean(dif),
                      "lo": lo_d, "hi": hi_d, "p": p})
        ps.append(p)
        ns.append(len(dif))
    out["_pair"] = {
        "pairs": pairs, "p_lo": min(ps), "p_hi": max(ps),
        "n_lo": min(ns), "n_hi": max(ns),
        "cover_zero": all(x["lo"] <= 0 <= x["hi"] for x in pairs),
    }
    out["_parents"] = len(res["parents"])
    out["_total"] = len(S)
    out["_kernels"] = len({p["kernel"] for p in res["parents"]})
    return out


def principle_text():
    """三种变异策略的原理说明。

    必须写在表9a 之前：自适应变异的权重**随父代水平分档**，若不加说明，
    读者容易把表9a 中它的实测占比误解为一个固定分布。
    权重与阈值取自 genetic_operators.py:567-585。
    """
    return (
        "具体地，三类变异操作的含义如下：参数调优只调整 num_warps、num_stages、"
        "BLOCK_SIZE 等 Launch 与 constexpr 参数，保持核心计算、Grid、寻址与 Mask 不变；"
        "算术与 Mask 优化在保持 Grid 与输出元素映射的前提下做数学等价替换与谓词、对齐简化；"
        "结构重写则允许重构 Grid 与 program_id 解码、改变任务打包与分块方式。"
        "三种策略的差别即体现在对这三类的采样权重上：均匀变异对三类等概率采样（各 1/3），"
        "且完全不参考父代的实测性能；激进变异以结构重写为主导，三类权重依次为 0.60、0.25、0.15；"
        "自适应变异并非一组固定权重，而是按父代相对参考实现的加速比分为三档——"
        "加速比大于 2.5 时只采用算术与 Mask 优化（0.70）与参数调优（0.30）、"
        "完全不触发结构重写，介于 1.5 与 2.5 之间时以算术与 Mask 优化为主（0.60/0.30/0.10），"
        "小于 1.5 时才以结构重写为主（0.50/0.35/0.15）。"
        "因此在阅读表9a 时需要注意，自适应变异的实测占比并非一个固定分布，"
        "而是上述三档规则在本次父代构成上的混合结果。"
    )


def remove_old_table9(d):
    """删除旧的端到端表9 及其附属段。

    端到端 GM(S) 经过两代进化与精英保留后被稀释，且差异不显著、逐个算子剔除时
    最优者还会翻转，因此不作为判别依据（在表9 的解读段中另行说明）。
    """
    removed = False
    _, cap = find_paragraph(d, A_OLD_CAPTION, required=False)
    if cap is not None:
        tbl = table_after(d, cap)
        if tbl is not None:
            remove_element(tbl)
        remove_element(cap)
        removed = True
    for pref in (A_OLD_EXTRA, A_OLD_ANALYSIS):
        _, p = find_paragraph(d, pref, required=False)
        if p is not None:
            remove_element(p)
            removed = True
    return removed


def rename_mechanism_table(d):
    """把正文中的「表9a」统一改名为「表9」（旧表9 已删除）。"""
    changed = False
    for p in d.paragraphs:
        if "表9a" in p.text:
            set_text(p, p.text.replace("表9a", "表9"))
            changed = True
    return changed


def upsert_principle(d):
    _, existing = find_paragraph(d, A_PRINCIPLE, required=False)
    if existing is not None:
        set_text(existing, principle_text())
        return False
    _, design = find_paragraph(d, A_DESIGN)
    insert_paragraph_after(design, principle_text(), design)
    return True


def build_texts(R):
    nz = R["_noise"]
    var = R["_var"]
    st = R["_stratum"]

    intro = (
        "上述三种策略的差异只体现在变异类型的采样方式上。为直接检验这一差异是否真实成立、"
        "并剥离搜索轨迹的影响，本文在固定父代上做一次机制探测："
        "在 3.3 与 3.4.1、3.4.2 所用的同一批 "
        f"{R['_kernels']} 个算子上，为每个算子选取 2 个父代个体"
        f"（其最快初始种子与已有最优候选各 1 个），共 {R['_parents']} 个父代；"
        "每个算子取两个父代，是因为自适应变异的类型权重依父代水平而定，"
        "只有覆盖不同水平的父代才能检验其分级行为。"
        "在每个父代上分别按三种策略各生成 3 个变异体并完整评测，"
        f"共 {R['_total']} 个变异体。由于父代固定且必定有效，三种策略面对的是同一批父代，"
        "观测到的差异只能来自策略本身；同时每个变异体都携带父代延迟，样本可用率接近 100%"
        "（端到端实验中约 65% 的变异样本因父代评测失败而无法计算增益）。"
        "每个变异体采用多次测量取中位数，量测抖动约为 ±"
        f"{nz['sd'] / nz['mean'] * 100:.1f}%（由未改动代码的无操作变异样本实测得到）。"
        "结果如表9所示。需要说明的是，本探测回答的是“一次变异的策略差异”，"
        "与端到端实验的结果互相补充、不可互相替代。"
    )

    a, u, g = R["adaptive"], R["uniform"], R["aggressive"]
    note = (
        f"由表9可见：激进变异的结构重写占比最高，为 {g['struct']:.1f}%（名义 60%），"
        f"均匀变异三类接近均分（{u['param']:.1f}%、{u['arith']:.1f}%、{u['struct']:.1f}%），"
        "自适应变异则介于两者之间。"
        f"三者的源码相似度（{min(x['sim'] for x in (a, u, g)):.3f} 至 "
        f"{max(x['sim'] for x in (a, u, g)):.3f}）与平均改动行数"
        f"（{min(x['lines'] for x in (a, u, g)):.1f} 至 "
        f"{max(x['lines'] for x in (a, u, g)):.1f} 行）彼此接近，"
        "说明三种策略的差别在于“改什么”而非“改多少”。"
    )
    pr = R["_pair"]
    rate_txt = "、".join(
        f"{LABEL[s]} {R[s]['rate']:.1f}%（95% 置信区间 [{R[s]['lo']:.1f}%, {R[s]['hi']:.1f}%]）"
        for s in ORDER)
    cover = ("任意两者之差的 95% 自助置信区间均包含 0" if pr["cover_zero"]
             else "两者之差在部分组合上不包含 0")
    note += (
        f"在改进率 P(G_mut > 1) 上，三种策略分别为{rate_txt}，"
        "三者的 95% 置信区间相互重叠。"
        "按父代配对比较：同一父代上两策略各生成 3 个变异体，"
        f"取其对数增益均值之差，共 {pr['n_lo']} 至 {pr['n_hi']} 对，"
        f"{cover}，Wilcoxon 符号秩检验亦不显著"
        f"（p 介于 {pr['p_lo']:.3f} 与 {pr['p_hi']:.3f} 之间），"
        "即在一次变异能否带来提升这一点上，三种策略无法区分。"
        "这并不意味着变异环节不重要：变异增益的方差中，策略身份仅解释 "
        f"{var['strategy']:.1f}%，而父代身份解释 {var['parent']:.1f}%"
        "（两者为两次独立的单因素分解，占比不可直接相加），"
        "决定一次变异是否有效的关键是“对哪个父代变异”而非“采用哪种变异策略”，"
        "这与 3.4.2 中主干保持型保护交叉的结论一致。"
        "需要说明的是，本文不以端到端加速比作为判别依据："
        "该指标经过两代进化与精英保留之后初始差异已被稀释，"
        "三种策略的差异不显著（Friedman 检验 p = 0.4371），"
        "逐个算子剔除时最优者还会发生变化，不足以支撑稳定的排序结论。"
        "因此，在固定的 13 次评测预算下，进一步优化的方向不应是设计更复杂的变异算子，"
        "而应是在有限的评测机会中优先安排那些更可能带来改进的子代。"
    )
    return intro, note


def build_table(R):
    head = ["变异策略", "参数调优", "算术与 Mask", "结构重写", "无操作率",
            "改进率 P(G>1)", "改进率 95%CI"]
    data = [head]
    for s in ORDER:
        r = R[s]
        data.append([
            LABEL[s], f"{r['param']:.1f}%", f"{r['arith']:.1f}%",
            f"{r['struct']:.1f}%", f"{r['noop']:.1f}%", f"{r['rate']:.1f}%",
            f"[{r['lo']:.1f}%, {r['hi']:.1f}%]",
        ])
    return data


def update_table9(d, t9):
    """用最新结果刷新表9。"""
    # 必须限定列数：表9a 的首列同样是「变异策略」，只按首列会命中错误的表
    _, tbl = find_table(d, A_TABLE9_HEAD, ncols=5)
    layout = [(1, "mut_adaptive"), (2, "mut_uniform"), (3, "mut_aggressive")]
    for idx, cfg in layout:
        r = t9[cfg]
        row = tbl.rows[idx]
        set_cell(row.cells[1], f"{r['valid_rate'] * 100:.1f}%")
        set_cell(row.cells[2], f"{r['success_rate'] * 100:.1f}%")
        set_cell(row.cells[3], f"{r['gm_g_mut']:.4f}")
        set_cell(row.cells[4], f"{r['gm_s']:.4f}")


def main():
    res = json.loads(PROBE.read_text(encoding="utf-8"))
    t9 = {r["config"]: r for r in json.loads(T9.read_text(encoding="utf-8"))}
    R = probe_rows(res)
    intro, note = build_texts(R)
    data = build_table(R)

    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print(f"已备份: {BACKUP.name}")
    else:
        print(f"备份已存在，跳过: {BACKUP.name}")

    d = Document(DOC)

    # ---- 三种策略的原理（必须在表9a 之前） ----
    if upsert_principle(d):
        print("  - 已插入变异策略原理段（含自适应分档权重）")
    else:
        print("  - 变异策略原理段已存在，就地更新")

    # ---- 删除旧的端到端表9 及其附属段 ----
    if remove_old_table9(d):
        print("  - 已删除旧的端到端表9（表题 / 表体 / 补充说明段 / 分析段）")
    else:
        print("  - 旧的端到端表9 已不存在，跳过")

    # ---- 机制探测表：存在则更新，不存在则插入 ----
    _, cap = find_paragraph(d, A_CAPTION_MECH_NEW, required=False)
    if cap is None:
        _, cap = find_paragraph(d, A_CAPTION_MECH_OLD, required=False)
    if cap is not None:
        tbl = table_after(d, cap)
        while len(tbl.rows) < len(data):
            tbl.rows[-1]._tr.addnext(copy.deepcopy(tbl.rows[-1]._tr))
        while len(tbl.rows) > len(data):
            tbl._tbl.remove(tbl.rows[-1]._tr)
        for ri, row in enumerate(data):
            for ci, val in enumerate(row):
                set_cell(tbl.rows[ri].cells[ci], val)
        _, p_intro = find_paragraph(d, A_INTRO)
        _, p_note = find_paragraph(d, "由表9可见", required=False)
        if p_note is None:
            _, p_note = find_paragraph(d, A_NOTE)
        set_text(p_intro, intro)
        set_text(p_note, note)
        print("  - 机制探测表已存在，就地更新")
    else:
        _, anchor = find_paragraph(d, A_ANCHOR)
        _, tpl = find_table(d, A_TPL_HEAD, ncols=7)
        p_intro = insert_paragraph_after(anchor, intro, anchor)
        p_cap = insert_paragraph_after(
            p_intro, A_CAPTION_MECH_OLD + "（固定父代，每父代每策略 3 次）", anchor)
        insert_paragraph_after(p_cap, note, anchor)
        insert_table_after(p_cap, data, d, template=tpl)
        print("  - 机制探测表已插入（引言 → 表题 → 表 → 解读）")

    # ---- 改名：旧表9 已删，「表9a」升为「表9」，消除孤儿编号 ----
    if rename_mechanism_table(d):
        print("  - 已将正文中的「表9a」改名为「表9」")
    else:
        print("  - 无需改名")

    d.save(DOC)
    print(f"\n已写回: {DOC.name}  (段落 {len(d.paragraphs)}，表格 {len(d.tables)})")
    print("  - 幂等，可重复运行")


if __name__ == "__main__":
    main()
