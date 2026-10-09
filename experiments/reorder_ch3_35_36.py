#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.5 与 3.6 交换顺序：案例分析后移作收尾，候选预筛选前移紧接 3.4 消融。

换序理由
--------
* 3.5 案例分析自述"在总体实验和消融实验之后"，是收尾性质，放在最后更合适；
* 3.6 候选预筛选是方法机制，与 3.3 / 3.4 的消融同类，紧接 3.4 更连贯；
* 3.6 的动机引用的是 3.1 的预算约束，并不依赖 3.5，换序无逻辑冲突。

连带改动
--------
表号必须随出现顺序走，因此三张表要旋转编号：
    表10 代表性算子案例          → 表12
    表11 候选预筛选效果          → 表10
    表12 候选预筛选模型离线评估  → 表11
小节号：3.6 / 3.6.1~3.6.5 → 3.5 / 3.5.1~3.5.5；3.5 → 3.6。

安全
----
* 改号用**单遍精确匹配**（先比对完整文本再写入），避免"表10→表12"与
  "表12→表11"互相踩踏产生重复编号。
* 移动的是从 3.5 标题到文末的整个尾部，先整段摘下再按新顺序挂回，
  期间保留 w:sectPr 为最后一个元素。
* 幂等：若已换序（3.5 标题已是"候选预筛选"），直接跳过移动。

执行前请确认已备份；脚本自身也会在 doc/ 下留一份备份。
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.docx_util import set_text                       # noqa: E402

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"

A_35 = "3.5 代表性算子案例分析"
A_36 = "3.6 候选预筛选机制"

# 单遍精确匹配：key 为改前的完整段落文本
RENUMBER = {
    # 小节标题
    A_35:                  "3.6 代表性算子案例分析",
    A_36:                  "3.5 候选预筛选机制",
    "3.6.1 动机":          "3.5.1 动机",
    "3.6.2 训练数据采集":  "3.5.2 训练数据采集",
    "3.6.3 模型设计与集成方式": "3.5.3 模型设计与集成方式",
    "3.6.4 评估指标":      "3.5.4 评估指标",
    "3.6.5 离线验证结果":  "3.5.5 离线验证结果",
    # 表标题（三张表旋转编号）
    "表10 代表性算子案例的可测量数据":     "表12 代表性算子案例的可测量数据",
    "表11 候选预筛选效果":                "表10 候选预筛选效果",
    "表12 候选预筛选模型离线评估结果":    "表11 候选预筛选模型离线评估结果",
}


def renumber(d):
    """单遍改号：逐段比对完整文本，命中即写入新文本。"""
    n = 0
    for p in d.paragraphs:
        t = p.text.strip()
        if t in RENUMBER:
            set_text(p, RENUMBER[t])
            n += 1
    return n


def fix_refs(d):
    """修正正文里指向被重编号表格的引用。

    改号只动了标题，正文里的"由表12可见""表11中…"仍是旧编号，必须同步：
      * "由表12可见"（谈 AUC / P@13）→ 指离线评估结果 → 表11
      * 以"最后需要说明"开头的段落里"表11"（谈预筛选那行的精确率/召回率/GM(S)）
        → 指候选预筛选效果 → 表10
    两条规则命中不同的段落，且改后不再匹配自身条件，故可重复运行。
    """
    n = 0
    for p in d.paragraphs:
        t = p.text
        new = t
        if "由表12可见" in new:
            new = new.replace("由表12可见", "由表11可见")
        if "最后需要说明" in new[:12]:
            new = new.replace("表11", "表10")
        if new != t:
            set_text(p, new)
            n += 1
    return n


def swap(d):
    """把 3.6 整块移到 3.5 之前。已换序则跳过，返回 True 表示实际移动了。"""
    body = d.element.body
    children = list(body.iterchildren())

    i35 = i36 = None
    for idx, el in enumerate(children):
        if el.tag.split('}')[-1] != 'p':
            continue
        t = "".join(n.text or "" for n in el.iter(qn('w:t'))).strip()
        if i35 is None and t.startswith(A_35):
            i35 = idx
        elif i36 is None and t.startswith(A_36):
            i36 = idx

    # 已换序（3.5 是候选预筛选）→ 幂等跳过
    if i35 is None or i36 is None:
        return False
    if i36 < i35:
        return False
    if i35 > i36:
        raise RuntimeError("3.5 标题出现在 3.6 之后，状态异常，请检查或回滚备份")

    tail = [el for el in children[i35:]
            if el.tag.split('}')[-1] != 'sectPr']
    k = i36 - i35                      # 3.6 标题在 tail 中的下标
    block_a, block_b = tail[:k], tail[k:]

    sect = body.find(qn('w:sectPr'))
    for el in tail:
        body.remove(el)
    for el in block_b + block_a:       # 先挂 3.6，再挂 3.5
        if sect is not None:
            sect.addprevious(el)
        else:
            body.append(el)
    return True


def main():
    stamp = datetime.now().strftime("%m%d_%H%M")
    bak = DOC.with_name(DOC.stem + f"_换序备份_{stamp}.docx")
    shutil.copy2(DOC, bak)
    print("备份:", bak.name)

    d = Document(DOC)
    moved = swap(d)
    print("块移动:", "已执行" if moved else "跳过（已换序）")

    n = renumber(d)
    print(f"改号: {n} 处（小节标题 7 + 表标题 3）")

    m = fix_refs(d)
    print(f"正文表号引用修正: {m} 处")

    d.save(DOC)
    print("已保存:", DOC.name)


if __name__ == "__main__":
    main()
