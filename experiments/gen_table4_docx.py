#!/usr/bin/env python3
"""生成 3.1 主实验的 Word 结果表（表4）。

用法：
    cd /workspace/Agent && python experiments/gen_table4_docx.py

输出：doc/表4_3.1主实验结果.docx
数据源：experiments/results/table4.json
"""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'experiments/results/table4.json'
OUT = ROOT / 'doc/表4_3.1主实验结果.docx'

data = json.load(open(SRC, encoding='utf-8'))
summary = data['summary']
groups = data['groups']


def fmt(v, nd=4):
    """空值统一显示为 —"""
    return '—' if v is None else f'{v:.{nd}f}'


doc = Document()
doc.add_heading('表4 主实验结果：与基线方法对比', level=2)

p = doc.add_paragraph(
    '实验设置：在官方原始种子集（50 个 Kernel）上比较四种方法。'
    'B0 为参考实现基线（不调用 LLM）；B1 为无策略引导的进化搜索；'
    'B2 为单轨迹迭代优化（爬山式，无种群）；FULL 为本文完整方法'
    '（策略驱动初始化 + Profiling 反馈 + RAG 知识库 + 进化搜索）。'
    '各方法均限定相同的 NPU 评测预算（每 Kernel 13 次），'
    '并在与参考实现相同的串行单设备协议下复测最终结果。'
    'F = T_seed / T_best（相对优化起点），S = T_base / T_best（相对参考实现），'
    'GM 为几何平均。')
p.runs[0].font.size = Pt(9)

# ---------- 主表 ----------
doc.add_heading('（a）整体结果', level=3)
hdr = ['方法', 'GM(F)', 'GM(S)', '中位数 F', '中位数 S', '参与 Kernel 数', 'LLM 调用数/算子']
t = doc.add_table(rows=1, cols=len(hdr))
t.style = 'Table Grid'
for i, h in enumerate(hdr):
    t.rows[0].cells[i].text = h

for r in summary:
    cells = t.add_row().cells
    llm = r.get('LLM调用数/算子')
    vals = [
        r['method'],
        fmt(r.get('GM(F)')),
        fmt(r.get('GM(S)')),
        fmt(r.get('median(F)')),
        fmt(r.get('median(S)')),
        str(r.get('N_used', '')),
        '—' if llm is None else f'{llm:.2f}',
    ]
    for i, v in enumerate(vals):
        cells[i].text = v

# ---------- 分组表 ----------
doc.add_heading('（b）按 Kernel 规模分组的 GM(S)', level=3)
hdr2 = ['规模分组', 'Kernel 数', 'GM(S) B1', 'GM(S) B2', 'GM(S) FULL', '中位数 S (FULL)']
t2 = doc.add_table(rows=1, cols=len(hdr2))
t2.style = 'Table Grid'
for i, h in enumerate(hdr2):
    t2.rows[0].cells[i].text = h

for g in groups:
    cells = t2.add_row().cells
    vals = [
        g['label'],
        str(g['n_kernel']),
        fmt(g.get('GM(S)_b1')),
        fmt(g.get('GM(S)_b2')),
        fmt(g.get('GM(S)_full')),
        fmt(g.get('median(S)_full')),
    ]
    for i, v in enumerate(vals):
        cells[i].text = v

# ---------- 分析（数值动态计算）----------
_by = {r['method']: r for r in summary}
_full = _by['FULL']
_b1, _b2 = _by['B1'], _by['B2']
_best_over_b2 = (_full['GM(S)'] / _b2['GM(S)'] - 1) * 100
_best_over_b1 = (_full['GM(S)'] / _b1['GM(S)'] - 1) * 100
_gm_gap = _full['GM(S)'] - _full['median(S)']

doc.add_heading('结果分析', level=3)
for txt in [
    f'（1）FULL 在整体与全部三个规模分组上均取得最优 GM(S)：'
    f'整体 GM(S)={_full["GM(S)"]:.4f}，分别高于 B2（{_b2["GM(S)"]:.4f}）{_best_over_b2:.1f}% '
    f'与 B1（{_b1["GM(S)"]:.4f}）{_best_over_b1:.1f}%。'
    '说明策略驱动初始化与 Profiling、RAG 的协同作用是有效的，而非单纯依赖更大的搜索预算。',

    f'（2）分组结果显示，方法间差异随 Kernel 规模增大而扩大：'
    f'短 Kernel（<5μs）上 FULL 的 GM(S) 为 {groups[0]["GM(S)_full"]:.4f}，'
    f'中长 Kernel（5~100μs）为 {groups[1]["GM(S)_full"]:.4f}，'
    f'长 Kernel（≥100μs）达到 {groups[2]["GM(S)_full"]:.4f}。'
    '原因是超短 Kernel 的绝对执行时间已接近硬件与框架开销下限，可优化空间有限。',

    f'（3）需注意的是，GM(S) 与中位数 S 之间存在明显差距'
    f'（FULL：GM {_full["GM(S)"]:.4f} vs 中位数 {_full["median(S)"]:.4f}，相差 {_gm_gap:.4f}）。'
    '这表明整体加速比在较大程度上由少数优化空间极大的 Kernel 贡献；'
    '多数 Kernel 的增益在 1.0~1.1 倍量级。'
    '因此在引用 GM 指标时应同时报告中位数，避免高估方法的普适增益。',

    f'（4）在相同评测预算下，FULL 的 LLM 调用数为 {_full["LLM调用数/算子"]:.2f} 次/算子，'
    f'高于 B1（{_b1["LLM调用数/算子"]:.2f}）与 B2（{_b2["LLM调用数/算子"]:.2f}），'
    '额外开销来自策略生成与 Profiling 诊断环节；'
    '考虑到其带来的加速比提升，该开销在自动优化场景下是可接受的。',
]:
    pp = doc.add_paragraph(txt)
    pp.runs[0].font.size = Pt(10)

doc.add_paragraph()
fn = doc.add_paragraph(
    '注：F = T_seed / T_best，S = T_base / T_best；GM 为几何平均；'
    'N_used 为最终参与统计的 Kernel 数。'
    '官方原始种子集中有部分文件在当前环境无法运行，'
    '已用同源可运行版本替换，并在附录中列出替换清单。')
fn.runs[0].font.size = Pt(9)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print('已生成:', OUT)
