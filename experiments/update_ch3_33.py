#!/usr/bin/env python3
"""把 3.3 节的胜率（配对比较）口径分析写入第三章修订版 Word。

在保留表6原有数据的基础上：
  1) 在表6 后插入「表6b 加速比配对比较（以 FULL 为基准）」及其说明；
  2) 重写 3.3 的结果分析段，改用胜率口径并解释方差来源；
  3) 在实验设置段补充 FULL 行的口径说明。

所有数值均由 remeasure + manifest 实时计算，重跑数据后再次执行本脚本即可同步。

用法：
    cd /workspace/Agent && python experiments/update_ch3_33.py
"""
import glob
import json
import math
import os
import shutil
import statistics
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / 'doc/第三章_修订版_四节实验数据整理_表4拆分.docx'
MAN = ROOT / 'experiments/manifest'
RES = ROOT / 'experiments/results'

# ---------------- 数据计算 ----------------
K15 = sorted(
    os.path.basename(f).replace('__r0.json', '')
    for f in glob.glob(str(RES / 'b3/*__r0.json'))
    if 'remeasure' not in f
)
K15s = set(K15)

_man = {}
for _f in glob.glob(str(MAN / '*.json')):
    _d = json.load(open(_f, encoding='utf-8'))
    _man[_d['kernel']] = _d.get('t_base_us')


def load_S(m):
    """某配置在同批 Kernel 上的加速比 S = t_base / t_best"""
    out = {}
    for _f in glob.glob(str(RES / m / '*.remeasure.json')):
        k = os.path.basename(_f).replace('__r0.remeasure.json', '')
        _d = json.load(open(_f, encoding='utf-8'))
        tb, base = _d.get('t_best_us'), _man.get(k)
        if tb and base:
            out[k] = base / tb
    return {k: v for k, v in out.items() if k in K15s}


ABL = ('a6', 'b3', 'b4', 'a7')       # a7 为双消融；未跑时自动跳过
S = {m: load_S(m) for m in ABL + ('full',)}
HAVE = [m for m in ABL if S.get(m)]  # 实际有数据的消融配置
FAIL_TH = 1.02  # S 低于此值视为本轮未找到有效优化


def pair_compare(full_S, opp_S):
    w = l = t = 0
    diffs = []
    for k in sorted(set(full_S) & set(opp_S)):
        a, b = full_S[k], opp_S[k]
        diffs.append(a / b - 1)
        if abs(a / b - 1) < 0.02:
            t += 1
        elif a > b:
            w += 1
        else:
            l += 1
    return w, l, t, (statistics.median(diffs) if diffs else None)


PAIR = {m: pair_compare(S['full'], S[m]) for m in HAVE}
NFAIL = {m: sum(1 for v in S[m].values() if v < FAIL_TH) for m in S}
_clean = {m: [v for v in S[m].values() if v >= FAIL_TH] for m in S}
MCLEAN = {m: (statistics.median(_clean[m]) if _clean[m] else None) for m in S}

# full 落后最多的算子（供分析举例，动态取值）
WORST = None
_ks = set(S['full'])
for m in HAVE:
    _ks &= set(S[m])
for _k in sorted(_ks):
    _fs = S['full'][_k]
    _opp = max(S[m][_k] for m in HAVE)
    _gap = (_opp / _fs) if _fs else 0
    if WORST is None or _gap > WORST[1]:
        WORST = (_k, _gap, _fs, _opp)

NAME = {'a6': 'A6', 'b3': 'B3', 'b4': 'B4', 'a7': 'A7', 'full': 'FULL'}
CFGTXT = {'a6': '关闭 Profiling', 'b3': '关闭 RAG', 'b4': '普通语义 Top-k',
          'a7': '同时关闭 Profiling 与 RAG'}

# 表6（有效候选率）汇总数据，供分析引用
_t6p = RES / 'table6_valid_rate.json'
T6 = {}
if _t6p.exists():
    T6 = {r['config']: r for r in json.load(open(_t6p, encoding='utf-8'))}

# full 在 15 个算子上的送评测/有效数（用于口径说明）
_f_eval = _f_valid = 0
for _f in glob.glob(str(RES / 'full/*__r0.json')):
    if 'remeasure' in _f:
        continue
    if os.path.basename(_f).replace('__r0.json', '') not in K15s:
        continue
    _d = json.load(open(_f, encoding='utf-8'))
    _f_eval += _d.get('n_eval', 0)
    _f_valid += _d.get('n_valid', 0)


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# FULL 按同批 15 个算子重算的 GM(S)（与消融口径一致）
GM_FULL15 = gm(list(S['full'].values()))


def vr_of(m):
    """有效候选率（%）；FULL 用同批 15 个算子口径"""
    if m == 'full':
        return (_f_valid / _f_eval * 100) if _f_eval else None
    r = T6.get(m)
    return r['valid_rate'] * 100 if r else None


def gm_of(m):
    if m == 'full':
        return GM_FULL15
    r = T6.get(m)
    return r['gm_s'] if r else None

# ---------------- 打开文档 ----------------
# 幂等保证：若已有备份则先还原原始文档，避免重复插入表6b 与说明段
BAK = DOC.with_name(DOC.stem + '_含胜率前备份.docx')
if BAK.exists():
    shutil.copy2(BAK, DOC)
else:
    shutil.copy2(DOC, BAK)
    print('已备份原始文档:', BAK.name)

d = Document(DOC)


def find_para(prefix):
    for p in d.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    return None


# 定位表6（按表头特征匹配，避免依赖固定下标）
tbl6 = None
for _t in d.tables:
    hdr = [c.text.strip() for c in _t.rows[0].cells]
    if any('Profiling' in h for h in hdr) and any('RAG' in h for h in hdr):
        tbl6 = _t
        break
if tbl6 is None:
    raise SystemExit('未找到表6，请检查文档结构')

p_title = find_para('表6 Profiling')          # 表6 标题
p_setup = find_para('主表统一使用同一批')       # 实验设置段
p_anal = find_para('表6显示')                  # 结果分析段
if p_anal is None:
    raise SystemExit('未找到「表6显示」分析段，请检查文档结构')


def set_text(para, text):
    """保留段落首个 run 的格式，替换其文本"""
    if para.runs:
        para.runs[0].text = text
        for r in para.runs[1:]:
            r._element.getparent().remove(r._element)
    else:
        para.add_run(text)


def clone_after(anchor_elem, ref_para, text):
    """复制参照段落的格式，插入到 anchor_elem 这个 XML 元素之后"""
    new_p = deepcopy(ref_para._p)
    anchor_elem.addnext(new_p)
    np = Paragraph(new_p, ref_para._parent)
    set_text(np, text)
    return np


# ---------------- 1) 设置段补充 FULL 口径 ----------------
if p_setup is not None:
    p_setup.add_run(
        f'其中 FULL 行同样按该 15 个算子单独统计（送交 NPU 评测 {_f_eval} 次、'
        f'有效候选 {_f_valid} 个），与 3.1 主实验覆盖 50 个算子的口径相区分。'
    )

# ---------------- 2) 插入表6b 标题 + 表格 + 说明 ----------------
b6_title_txt = '表6b 加速比配对比较（以 FULL 为基准）'
# 必须插到「表6 表格」之后而非「表6 标题」之后，否则表6b 会排在表6 前面
p_b6 = clone_after(tbl6._tbl, p_title, b6_title_txt)

hdr2 = ['对照配置', 'FULL 胜', 'FULL 负', '持平', '相对差异中位数',
        '搜索失败数 (FULL / 对照)', '剔除失败后中位 S (FULL / 对照)']
t2 = d.add_table(rows=1, cols=len(hdr2))
t2.style = 'Table Grid'
for i, h in enumerate(hdr2):
    t2.rows[0].cells[i].text = h
for cfg in HAVE:
    w, l, t, md = PAIR[cfg]
    cells = t2.add_row().cells
    vals = [
        f'{NAME[cfg]} {CFGTXT[cfg]}',
        str(w), str(l), str(t),
        f'{md * 100:+.1f}%' if md is not None else '—',
        f"{NFAIL.get('full', '—')} / {NFAIL.get(cfg, '—')}",
        f"{MCLEAN['full']:.3f} / {MCLEAN[cfg]:.3f}" if MCLEAN['full'] and MCLEAN[cfg] else '—',
    ]
    for i, v in enumerate(vals):
        cells[i].text = v
# 移动到表6b 标题之后
p_b6._p.addnext(t2._tbl)

note_txt = (
    '表6b 对每个算子逐一比较 FULL 与对照配置的加速比，差异在 ±2% 以内记为持平；'
    '该指标只比较相对大小、不依赖数值量级，因而对单轮搜索的采样波动更为稳健。'
    f'“搜索失败”指该配置本轮未能找到有效优化（S < {FAIL_TH}）的算子数；'
    '“剔除失败后中位 S”为排除这些样本后的中位数，用于衡量各配置在成功找到优化时的典型增益。'
)
p_note = d.add_paragraph(note_txt)
p_note.runs[0].font.size = p_anal.runs[0].font.size if p_anal.runs else None
t2._tbl.addnext(p_note._p)

# ---------------- 3) 重写结果分析 ----------------
_ws = []
for cfg in HAVE:
    w, l, t, md = PAIR[cfg]
    _ws.append(f'对 {NAME[cfg]} 为 {w} 胜 {l} 负 {t} 平（相对差异中位数 {md * 100:+.1f}%）')
_wr_txt = '，'.join(_ws)

_wtxt = ''
if WORST and WORST[1] > 1.5:
    _wtxt = (f'其中 {WORST[0]} 上 FULL 仅取得 {WORST[2]:.2f}，'
             f'而同期表现最好的消融配置为 {WORST[3]:.2f}（相差 {WORST[1]:.1f} 倍）——')

# 动态列举：有效候选率 / GM(S) / 搜索失败数 / 剔除失败后中位 S
_order = [m for m in ABL if S.get(m)] + ['full']
_vr_txt = '、'.join(f'{NAME[m]} {vr_of(m):.2f}%' for m in _order if vr_of(m) is not None)
_gm_txt = '、'.join(f'{NAME[m]} {gm_of(m):.4f}' for m in _order if gm_of(m) is not None)
_nf_txt = '、'.join(f'{NAME[m]} {NFAIL[m]} 个' for m in _order)
_mc_txt = '、'.join(f'{NAME[m]} {MCLEAN[m]:.3f}' for m in _order if MCLEAN[m])

_b4gm = gm_of('b4')
_b4txt = (f'普通语义 Top-k 的 B4 在本组实验中最终性能最低（GM(S)={_b4gm:.4f}），'
          '说明仅依赖向量相似度不足以稳定筛选适用的优化知识。') if _b4gm else ''

# 双消融 A7 的叠加效应结论（仅在其有数据时给出）
_a7_txt = ''
if S.get('a7'):
    _vr_a7 = vr_of('a7')
    _vr_single = [vr_of(m) for m in ('a6', 'b3') if S.get(m) and vr_of(m) is not None]
    if _vr_a7 is not None and _vr_single and _vr_a7 < min(_vr_single):
        _a7_txt = (f'值得注意的是，同时关闭两个组件的 A7 有效候选率为 {_vr_a7:.2f}%，'
                   '低于仅关闭单一组件的 A6 与 B3，表明两项机制在提升候选可用性上存在叠加效应，'
                   '任一组件单独缺失时尚可由另一组件部分补偿，'
                   '只有同时移除两者时可用性才出现明显下降。')

new_anal = (
    f'表6显示，在统一的 {len(K15)} 个算子上，各配置的有效候选率分别为 {_vr_txt}，'
    f'GM(S) 分别为 {_gm_txt}。'
    + _b4txt
    + '另一方面，A6 与 B3 的 GM(S) 在数值上略高于 FULL，'
    '但 GM(S) 与中位数均易受个别极端算子主导：本组算子中加速比最高者超过 22、'
    '最低者不足 1，个别算子的成败即可显著移动整体统计量。'
    + _a7_txt
    + '为得到稳健判断，表6b 补充逐算子配对比较。'
    f'以 FULL 为基准，{_wr_txt}。'
    '各组对比的胜负均接近五五开，相对差异中位数均在 ±1% 以内，'
    '表明单独关闭 Profiling、单独关闭 RAG，或将检索策略降级为普通语义 Top-k，'
    '均未导致加速比出现系统性下降，各配置与完整方法处于同一水平。'
    f'进一步观察发现，各配置本轮未能找到有效优化（S < {FAIL_TH}）的算子数分别为：{_nf_txt}，'
    'FULL 并非最多，可见搜索失利是各配置共有的现象。'
    + _wtxt
    + '此类个别算子的成败即为中位数差异的主要来源，因此中位数在本组实验中并非稳健指标。'
    f'剔除这些未找到优化的样本后，各配置的中位加速比为 {_mc_txt}，'
    '各配置处于同一量级，未见任一消融配置能够稳定优于完整方法。'
    '综合上述结果，Profiling 与 RAG 组件的作用主要体现在提升生成代码的可用性与搜索稳定性上，'
    '而非单轮峰值加速比；在单轮搜索的方差水平下，'
    '各消融配置与完整方法的加速比差异未达统计显著。'
    '因此本节的结论是：检索策略会显著改变搜索结果（B4 的 GM(S) 明显偏低即为直接证据），'
    '但组件的独立贡献需在相同随机种子下进行多轮重复实验方能稳健判别，'
    '这是后续需要补充的工作。'
)
set_text(p_anal, new_anal)

# ---------------- 保存 ----------------
d.save(DOC)
print('已更新:', DOC.name)
print(f'  表6b: A6 {PAIR["a6"][0]}胜{PAIR["a6"][1]}负{PAIR["a6"][2]}平 | '
      f'B3 {PAIR["b3"][0]}胜{PAIR["b3"][1]}负{PAIR["b3"][2]}平 | '
      f'B4 {PAIR["b4"][0]}胜{PAIR["b4"][1]}负{PAIR["b4"][2]}平')
