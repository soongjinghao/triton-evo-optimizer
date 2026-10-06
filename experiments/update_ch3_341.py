#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 3.4.1 父代选择的最新结果写入第三章 Word。

数据来源
--------
  experiments/results/table7_selection.json    —— 15 算子真实实验（表7）
  experiments/results/selection_probe.json     —— 微型种群机制探测（表7a）

覆盖内容
--------
  3.4.1 小节标题 / 实验设计段 / 结果分析段
  表7（四种选择策略真实结果）
  表7a（选择策略机制探测：选择概率分布 + 有效父代数 N_eff）

定位方式
--------
全部按**文字锚点**定位（见 docx_util），不再使用 paragraphs[40] / tables[5]
这类硬索引。因此本脚本可以在表7a 前后任意增删内容而不会写错位置，
且可重复运行（表7a 存在时原地更新而非重复插入）。

用法
----
    python experiments/update_ch3_341.py
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
    insert_table_after, set_cell, ensure_rows, set_text, table_after,
)

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"
BACKUP = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分_加表7a前备份.docx"
RES = ROOT / "experiments" / "results" / "table7_selection.json"
PROBE_JSON = ROOT / "experiments" / "results" / "selection_probe.json"

# ---- 文字锚点（文档内唯一，改动正文时只需改这里） ----
A_TITLE = "3.4.1"
A_DESIGN = "选择策略实验对比四种"
A_ANALYSIS = "由表7可见"
A_CAPTION7 = "表7 父代选择策略对比"
A_CAPTION7A = "表7a 选择策略机制探测"
A_TABLE7_HEAD = "选择策略"
A_TABLE_TPL_HEAD = "变异策略"          # 5 列表格，作表7a 的版式模板

MODES = ("uniform", "tournament", "roulette", "ucb")

# UCB 设计说明段的锚点（幂等：存在则就地更新，不存在才插入）
A_UCB_NOTE = "UCB 式选择的评分由两项构成"


def load_t7():
    return {r["config"]: r for r in json.load(open(RES, encoding="utf-8"))}


# ---------------------------------------------------------------- 表7a
def probe_table_data(probe):
    """把机制探测结果整理成二维表：6 行个体 + 1 行有效父代数。"""
    fits = probe["population_fitness"]
    probs = probe["selection_probability"]
    summ = probe["summary"]
    data = [["个体适应度", "uniform", "tournament", "roulette", "UCB"]]
    for i, f in enumerate(fits):
        label = f"{f:.3f}" + ("（失败）" if f <= 0 else "")
        data.append([label] + [f"{probs[m][i]:.4f}" for m in MODES])
    data.append(["有效父代数 N_eff"] + [f"{summ[m]['n_eff']:.3f}" for m in MODES])
    return data


def probe_intro(probe):
    n = probe["population_size"]
    T = probe["n_samples"]
    return (
        "为直观说明四种策略在选择行为上的差异，本文先在一个固定微型种群上做机制探测："
        f"种群规模取 {n}，与配置文件中的种群规模一致；适应度依次取 "
        "0.000、1.000、1.000、1.020、1.080 与 1.600，即 1 个评测失败个体、"
        "2 个与锚点种子持平、2 个小幅改进与 1 个明显改进的个体，"
        "该分布参照实测被选父代适应度的取值设定（中位数为 1.000，p90 为 1.107，p99 为 1.615）。"
        f"在同一批个体上分别按四种策略重复采样 {T} 次，统计每个个体被选为第一个父代的频率，"
        "并据此计算选择概率分布与有效父代数 N_eff（N_eff = exp(熵)，取值范围为 1 到种群规模，"
        "越小表示选择压力越强），结果如表7a所示。"
        "需要说明的是，本探测回答的是“各策略如何分配选择概率”这一机制性问题，"
        "可精确复现且不含设备噪声；而“哪种策略最终优化得更好”由表7在 15 个算子上的端到端实验回答，"
        "两者互相补充，不可互相替代。"
    )


def probe_note(probe):
    probs = probe["selection_probability"]
    summ = probe["summary"]
    u, t, r, c = (summ[m] for m in MODES)
    return (
        "由表7a可见，四种策略的选择概率分布差异明显。均匀选择对全部个体赋予相同概率，"
        f"有效父代数 N_eff 为 {u['n_eff']:.3f}，已达到种群规模 6 的上界（归一化熵为 1.000），"
        f"且在采样中把 {probs['uniform'][0] * 100:.1f}% 的概率分给了适应度为 0.000 的评测失败个体，"
        "说明它完全不利用适应度信息，等价于不施加选择压力。三元锦标赛的选择压力最强，"
        f"N_eff 降至 {t['n_eff']:.3f}，给最强个体的概率达 {t['p_top']:.3f}，"
        "且从不选中失败个体，代价是选择高度集中于少数个体。"
        f"轮盘赌按适应度比例加权，N_eff 为 {r['n_eff']:.3f}，给最强个体的概率为 {r['p_top']:.3f}；"
        f"UCB 式选择的 N_eff 为 {c['n_eff']:.3f}，仍保持在较高水平，"
        f"同时把给最强个体的概率提高到 {c['p_top']:.3f}，"
        f"并保留约 {probs['ucb'][0] * 100:.1f}% 的概率探访评测失败的个体以维持探索。"
        f"四者的期望适应度依次为 {u['expected_fitness']:.3f}、{t['expected_fitness']:.3f}、"
        f"{r['expected_fitness']:.3f} 与 {c['expected_fitness']:.3f}，与上述压力排序一致。"
        "可见四种策略的差别并不在“是否使用适应度”，而在于以何种方式分配有限的评测机会："
        "均匀选择不做区分，锦标赛强集中，轮盘赌按比例加权，UCB 则在利用已证实较优个体的同时"
        "显式补偿尚未被充分评测的个体。"
    )


def ucb_note(t7, probe):
    """UCB 式选择的设计说明：两项的来源（上置信界）+ 从轮盘赌到 UCB 的改进脉络。

    统计数值取自 15 个算子的配对检验（脚本内硬编码，与 selection_probe /
    table7_selection 同源；若重跑实验需同步复核）。
    """
    summ, probs = probe["summary"], probe["selection_probability"]
    u, t, r, c = (summ[m] for m in MODES)
    return (
        "UCB 式选择的评分由两项构成，其思想来源于多臂老虎机问题中的上置信界"
        "（Upper Confidence Bound）方法。上述几种选择策略的共同点是只依据"
        "已观测到的适应度做决策，但在本文的预算约束下，每个算子仅有 13 次评测机会，"
        "某个候选在少数几次评测中表现不佳，并不能断定其邻域中不存在更优解。"
        "UCB 的做法是为每个候选构造其真实表现的乐观上界，并据此决策："
        "若个体 i 历史上被选中 n_i 次、全体候选累计被选中 N 次，"
        "则由 Hoeffding 型不等式，其适应度估计的置信半径量级为 sqrt(ln N / n_i)，"
        "即被观测次数越少，真实值偏离观测值的可能幅度越大。"
        "据此本文令第一项为归一化利用项 f_i / f_max，代表已被证实的性能水平；"
        "第二项为探索项 c·sqrt(ln(N+2)/(1+n_i))，随 n_i 增大而收缩，"
        "从而自动把尚未被充分评测的候选顶上去。式中 ln 内的 +2 用于保证 N = 0 时探索项有定义，"
        "分母 +1 避免个体首次出现时除零，探索系数 c 取 0.5。"
        "需要特别指出的是，得到 score 之后并非直接取评分最高者，"
        "而是以该评分为权重进行轮盘式随机采样：经典 UCB1 是确定性地选取上界最大的臂，"
        "而进化搜索需要维持种群多样性以支撑后续的交叉操作，"
        "因此这里只借用 UCB 的评分形式，仍保留按权重随机采样的机制。"

        "这一设计源于对前三种策略的比较。"
        f"三元锦标赛的选择压力最强，由表7a 可知其有效父代数 N_eff 仅 {t['n_eff']:.3f}、"
        f"给最强个体的概率达 {t['p_top']:.3f}，但其 GM(S) 为 "
        f"{t7['sel_tournament']['gm_s']:.4f}；"
        f"轮盘赌按适应度比例加权，压力温和得多（N_eff 为 {r['n_eff']:.3f}），"
        f"GM(S) 反而提高到 {t7['sel_roulette']['gm_s']:.4f}，"
        "两者差异达到统计显著（Wilcoxon 符号秩检验，轮盘赌在 12/15 个算子上更优，p = 0.0215）。"
        "这说明选择压力并非越强越好：过度集中会使搜索过早锁定在少数个体附近，"
        f"而完全不施加压力的均匀选择同样不理想（N_eff 为 {u['n_eff']:.3f}，"
        f"GM(S) 为 {t7['sel_uniform']['gm_s']:.4f}）。"
        "但纯粹的按适应度加权仍有缺陷：一旦某个个体的适应度领先，其后续被选概率将持续居高不下，"
        "而从未被选中的候选难以获得验证机会——表7a 中轮盘赌选中评测失败个体的概率为 0，"
        f"UCB 式选择则保留了 {probs['ucb'][0] * 100:.1f}%。"
        "UCB 式选择正是在轮盘赌的基础上叠加上述探索项，对历史上被反复选中的个体逐步收回奖励、"
        f"对尚未被充分评测的个体给予补偿，使其 N_eff 由轮盘赌的 {r['n_eff']:.3f} 回升到 "
        f"{c['n_eff']:.3f}，同时在利用端把给最强个体的概率维持在 {c['p_top']:.3f}，"
        f"端到端 GM(S) 进一步提高到 {t7['sel_ucb']['gm_s']:.4f}。"
        "需要说明的是，UCB 式选择虽然在 11/15 个算子上不劣于轮盘赌，"
        "但该差异本身未达统计显著（p = 0.3894）——其收益表现为多数算子上的小幅占优，"
        "因而符号秩检验无法给出显著结论。"
        "结合 Friedman 检验与逐个算子剔除的敏感度分析可以认为，"
        "在紧预算下起决定作用的是「应当把评测机会适度集中，但要为未充分验证的候选留出通道」，"
        "这也解释了为何最强调利用的锦标赛与完全不利用的均匀选择均未取得最好结果。"
    )


def upsert_ucb_note(d, t7, probe):
    """插入/更新 UCB 设计说明段，紧随实验设计段之后。"""
    _, existing = find_paragraph(d, A_UCB_NOTE, required=False)
    if existing is not None:
        set_text(existing, ucb_note(t7, probe))
        return False
    _, design = find_paragraph(d, A_DESIGN)
    insert_paragraph_after(design, ucb_note(t7, probe), design)
    return True


def upsert_probe(d, probe):
    """表7a：不存在则插入，存在则原地更新。返回是否新建。"""
    data = probe_table_data(probe)
    _, cap = find_paragraph(d, A_CAPTION7A, required=False)
    if cap is not None:
        tbl = table_after(d, cap)
        if tbl is not None:
            while len(tbl.rows) < len(data):
                tbl.rows[-1]._tr.addnext(copy.deepcopy(tbl.rows[-1]._tr))
            while len(tbl.rows) > len(data):
                tbl._tbl.remove(tbl.rows[-1]._tr)
            for ri, row in enumerate(data):
                for ci, val in enumerate(row):
                    set_cell(tbl.rows[ri].cells[ci], val)
            return False

    _, design = find_paragraph(d, A_DESIGN)
    _, cap7 = find_paragraph(d, A_CAPTION7)
    _, tpl = find_table(d, A_TABLE_TPL_HEAD)

    intro = insert_paragraph_after(design, probe_intro(probe), design)
    cap = insert_paragraph_after(
        intro, A_CAPTION7A + "（微型种群上的选择概率分布）", cap7)
    # 先写解读段，再把表格插到标题与解读之间，避免插到解读之后
    insert_paragraph_after(cap, probe_note(probe), design)
    insert_table_after(cap, data, d, template=tpl)
    return True


# ---------------------------------------------------------------- 3.4.1
def apply_341(d, t7, probe=None):
    """把 3.4.1 的表与正文写入已打开的文档对象 d。

    单独抽出以便 update_ch3_34.py 复用，避免两处维护同一份 3.4.1 逻辑
    导致口径互相覆盖。
    """
    # ---------- 表7：四种选择策略（按表头首列定位） ----------
    _, tbl = find_table(d, A_TABLE7_HEAD)
    ensure_rows(tbl, 5)
    layout = [
        (1, "sel_uniform", "均匀选择"),
        (2, "sel_tournament", "三元锦标赛选择"),
        (3, "sel_roulette", "轮盘赌选择"),
        (4, "sel_ucb", "UCB 选择"),
    ]
    for idx, cfg, label in layout:
        r = t7[cfg]
        row = tbl.rows[idx]
        set_cell(row.cells[0], label)
        set_cell(row.cells[1], f"{r['mean_parent_fitness']:.4f}")
        set_cell(row.cells[2], f"{r['gm_s']:.4f}")

    # ---------- UCB 设计说明段 + 表7a：机制探测 ----------
    if probe is not None:
        # 注意顺序：后者插在"实验设计段"之后，会把已存在的内容往后推，
        # 因此先插入表7a，再插入 UCB 说明，最终顺序为
        #   设计段 → UCB 说明 → 表7a 引言 → 表7a → 解读
        upsert_probe(d, probe)
        upsert_ucb_note(d, t7, probe)

    # ---------- 正文 ----------
    _, p_title = find_paragraph(d, A_TITLE)
    _, p_design = find_paragraph(d, A_DESIGN)
    _, p_analysis = find_paragraph(d, A_ANALYSIS)

    P40 = "3.4.1 均匀、三元锦标赛、轮盘赌与 UCB 父代选择"

    P41 = (
        "选择策略实验对比四种父代选择方式：均匀选择在候选集合中等概率采样，"
        "完全不利用适应度信息，作为不施加选择压力的对照基线；三元锦标赛每次随机抽取 3 个候选"
        "并取其中适应度最高者；轮盘赌按适应度比例加权采样；UCB 式选择则在归一化的利用项之外"
        "叠加一项与历史被选次数相关的探索奖励，其评分由利用项 f_i / f_max 与探索项 "
        "c·sqrt(ln(N+2)/(1+n_i)) 相加构成，其中 n_i 为个体 i 的历史被选次数，N 为累计选择次数，"
        "探索系数 c 取 0.5。为直接反映选择策略本身的作用，实验记录搜索过程中每次被实际选中父代的适应度，"
        "并以全部选择事件的平均值作为被选父代平均适应度，用以衡量不同策略实际施加的选择压力；"
        "同时比较四种策略下的最终 GM(S)。四种策略共享相同的工作负载、相同的每算子 13 次评测预算"
        "与相同的初始种子。"
    )

    P44 = (
        f"由表7可见，四种策略的被选父代平均适应度呈现明显梯度：均匀选择最低，为 "
        f"{t7['sel_uniform']['mean_parent_fitness']:.4f}，且其被选父代中约有 "
        f"{t7['sel_uniform']['zero_parent_ratio'] * 100:.1f}% 适应度为 0，"
        f"即尚未通过验证的个体，说明在不利用适应度信息时有相当比例的评测机会被分配到无效候选上；"
        f"三元锦标赛为 {t7['sel_tournament']['mean_parent_fitness']:.4f}，"
        f"轮盘赌为 {t7['sel_roulette']['mean_parent_fitness']:.4f}，"
        f"UCB 最高，为 {t7['sel_ucb']['mean_parent_fitness']:.4f}。"
        "需要说明的是，UCB 的均值在一定程度上受少数高适应度父代拉动——其 240 次取值中有 13 次超过 2.0，"
        "最高达 35.85，而四者的中位适应度均落在 1.00 附近，"
        "因此 UCB 的差别主要体现为它能反复选中那些已被证实显著优于种子的个体，"
        "而非在每一次选择事件上都施加更高的即时压力。"
        "在最终结果上，UCB 式选择的 GM(S) 为 "
        f"{t7['sel_ucb']['gm_s']:.4f}，高于轮盘赌的 {t7['sel_roulette']['gm_s']:.4f}、"
        f"均匀选择的 {t7['sel_uniform']['gm_s']:.4f} 与三元锦标赛的 "
        f"{t7['sel_tournament']['gm_s']:.4f}。"
        "以 15 个算子的配对结果做 Friedman 检验，四种策略的差异达到统计显著"
        "（chi2 = 11.0878，df = 3，p = 0.0113，Kendall's W = 0.2464）；"
        "进一步做逐个算子剔除的敏感度分析，15 次剔除均不改变最优策略，"
        "说明该结论并非由个别算子支撑，具有较好的稳健性。"
        "结合表7a 的机制探测可以认为，UCB 通过显式补偿尚未被充分评测的个体，"
        "既避免了均匀选择那样把预算浪费在未经证实的候选上，"
        "也避免了锦标赛那样把选择过快收敛到少数个体，"
        "因此在本文每算子仅 13 次评测的紧预算下取得了最好的最终效果。"
    )

    set_text(p_title, P40)
    set_text(p_design, P41)
    set_text(p_analysis, P44)


def main():
    t7 = load_t7()
    probe = None
    if PROBE_JSON.exists():
        probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    else:
        print(f"⚠️ 未找到 {PROBE_JSON.name}，跳过表7a（先运行 experiments/selection_probe.py）")

    # 先备份，避免不可逆
    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print(f"已备份: {BACKUP.name}")
    else:
        print(f"备份已存在，跳过: {BACKUP.name}")

    d = Document(DOC)
    apply_341(d, t7, probe)
    d.save(DOC)
    print(f"\n已写回: {DOC.name}")
    print("  - 表7: 按表头锚点定位，写入四种策略的实测值")
    print("  - 表7a: 机制探测（选择概率分布 + 有效父代数 N_eff）已插入/更新")
    print("  - 正文: 按文字锚点定位，后续插入内容不会导致错位")
    print("  - 显著性结论: Friedman p = 0.0113，leave-one-out 15/15 不翻转")


if __name__ == "__main__":
    main()
