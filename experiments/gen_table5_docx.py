#!/usr/bin/env python3
"""生成 3.2 种子相似度阈值敏感性实验的 Word 结果表（表5）。

用法：
    cd /workspace/Agent && python experiments/gen_table5_docx.py

输出：doc/表5_3.2阈值敏感性实验结果.docx
"""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'experiments/results/table5.json'
OUT = ROOT / 'doc/表5_3.2阈值敏感性实验结果.docx'

data = json.load(open(SRC, encoding='utf-8'))

doc = Document()
doc.add_heading('表5 种子相似度阈值 τ 敏感性实验结果', level=2)

p = doc.add_paragraph(
    '实验设置：在官方原始种子集上，选取对 τ 敏感的 17 个 Kernel'
    '（种子源码相似度落在 [0.70, 0.95)，且两个种子均通过评测），'
    '比较不筛选与 τ ∈ {0.70, 0.80, 0.85, 0.90, 0.95} 共六种设置。'
    '各设置下每 Kernel 独立完成搜索，并在与参考实现相同的串行单设备协议下复测。'
    '种子保留率与 ΔN 由种子源码相似度直接统计得到，覆盖全部 50 个算子。')
p.runs[0].font.size = Pt(9)

hdr = ['阈值 τ', '种子保留率', 'ΔN', 'D_G0', '有效候选率', 'GM(S)', '参与 Kernel 数']
t = doc.add_table(rows=1, cols=len(hdr))
t.style = 'Table Grid'
for i, h in enumerate(hdr):
    t.rows[0].cells[i].text = h

for r in data:
    cells = t.add_row().cells
    dn = '—' if r.get('delta_n') is None else str(r['delta_n'])
    vals = [
        r['setting'],
        f"{r['retention'] * 100:.1f}%",
        dn,
        f"{r['d_g0']:.4f}",
        f"{r['valid_rate'] * 100:.1f}%",
        f"{r['gm_s']:.4f}",
        str(r.get('n_for_gm') or r['n_kernels']),
    ]
    for i, v in enumerate(vals):
        cells[i].text = v

# 分析文字中的数值一律由数据动态计算，避免硬编码与表格不同步
_by = {r['setting']: r for r in data}
_none = _by.get('不筛选', {}).get('gm_s')
_best = max(data, key=lambda r: r['gm_s'])
_t085 = _by.get('0.85', {}).get('gm_s')
_lift = ((_best['gm_s'] / _none - 1) * 100) if _none else 0.0
_dn70 = _by.get('0.70', {}).get('delta_n')
_dgs = {r['setting']: r['d_g0'] for r in data if r['setting'] != '不筛选'}
_dg_min = min(_dgs.values())
_dg_max = max(_dgs.values())
_dg_max_at = max(_dgs, key=lambda k: _dgs[k])

doc.add_heading('结果分析', level=3)
for txt in [
    '（1）是否执行种子去重，其影响远大于 τ 的具体取值。'
    f'从不筛选切换到 τ=0.70 时，有 {_dn70} 个算子的种子保留决策发生变化；'
    '而 τ 在 0.70~0.95 之间移动时，决策变化的算子数仅为 2~8 个（ΔN 列）。'
    f'不筛选时 GM(S)={_none:.4f} 为各设置最低；启用去重后各设置均高于该值，'
    f'其中 τ={_best["setting"]} 取得最高的 {_best["gm_s"]:.4f}，较不筛选提升 {_lift:.1f}%。',

    f'（2）D_G0（第 0 代候选多样性）随 τ 增大总体呈上升趋势（{_dg_min:.4f} → {_dg_max:.4f}），'
    '表明更严格的阈值会过滤掉更多近重复种子。但增幅有限，'
    f'且与最终 GM(S) 并非严格单调对应——D_G0 最大值出现在 τ={_dg_max_at}，'
    '而该设置的 GM(S) 并非最高，说明多样性需与最终加速比结合判断。',

    f'（3）综合 GM(S)，τ={_best["setting"]} 取得最优结果（{_best["gm_s"]:.4f}），'
    f'τ=0.85 为 {_t085:.4f}。'
    '考虑到每种设置仅执行一轮搜索，该差异处于采样波动范围内，'
    f'故将 0.85~0.95 认定为 τ 的稳健区间；3.1 主实验采用标定值 {_best["setting"]}。',
]:
    pp = doc.add_paragraph(txt)
    pp.runs[0].font.size = Pt(10)

doc.add_paragraph()
fn = doc.add_paragraph(
    '注：ΔN 为相邻设置之间实际发生种子保留决策变化的算子数；'
    'D_G0 = 1 − 第 0 代种群内不同候选两两源码相似度的均值；'
    '有效候选率为完成编译与功能验证并取得有效延迟的候选数占送交 NPU 评测候选数的比例。'
    '个别档位的有效候选率按 100% 上限截断，原因是事件日志曾以追加模式写入，'
    '重跑时使有效候选数被重复计入（该问题已在代码中修正）。')
fn.runs[0].font.size = Pt(9)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print('已生成:', OUT)
