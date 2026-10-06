#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三章 Word 文档操作的公共工具。

存在意义
--------
原先 update_ch3_341.py / update_ch3_34.py 全部用硬编码下标定位
（d.paragraphs[40]、d.tables[5] 等）。一旦往正文中间插入任何段落或表格，
其后所有下标全部错位，脚本会静默地把内容写到错误的位置——这类错误不会报错，
只会污染论文，且极难察觉。

本模块改为**按文字内容定位锚点**（find_paragraph / find_table），
插入操作一律相对锚点进行，因此正文结构变化不会影响写回正确性。
"""
import copy

from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run


# ---------------------------------------------------------------- 定位
def iter_body(doc):
    """按文档真实顺序遍历 body 的子元素，产出 Paragraph / Table 对象。

    d.paragraphs 与 d.tables 是两个互不相干的列表，无法反映"段落在表前还是表后"。
    需要判断相对位置时必须用本函数。
    """
    for child in doc.element.body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield Paragraph(child, doc)
        elif tag == 'tbl':
            yield Table(child, doc)


def find_paragraph(doc, prefix, start=0, required=True):
    """返回第一个文本以 prefix 开头的段落下标与对象。"""
    for i, p in enumerate(doc.paragraphs):
        if i < start:
            continue
        if p.text.strip().startswith(prefix):
            return i, p
    if required:
        raise LookupError(f"未找到以 {prefix!r} 开头的段落")
    return None, None


def find_table(doc, first_row_first_cell, required=True, ncols=None):
    """返回第一个首行首列文本匹配的表格下标与对象。

    表头的首列文字（如"选择策略""变异策略"）通常可作为锚点，但**并不保证
    唯一**：例如 3.4.3 中表9 与表9a 的首列都是"变异策略"。此时必须再限定
    列数（ncols），否则会命中错误的表格并把数据写错位置——这类错误不会报错，
    只会静默污染表格，因此 ncols 应视为默认必填的防御手段。
    """
    for i, t in enumerate(doc.tables):
        if not t.rows:
            continue
        if t.rows[0].cells[0].text.strip() != first_row_first_cell:
            continue
        if ncols is not None and len(t.rows[0].cells) != ncols:
            continue
        return i, t
    if required:
        extra = f"且列数为 {ncols}" if ncols is not None else ""
        raise LookupError(
            f"未找到首行首列为 {first_row_first_cell!r}{extra} 的表格")
    return None, None


def table_after(doc, paragraph):
    """返回紧跟在某段落之后的第一个表格（跳过中间的空段落）。"""
    seen = False
    for item in iter_body(doc):
        if item is paragraph or (isinstance(item, Paragraph)
                                 and item._p is paragraph._p):
            seen = True
            continue
        if not seen:
            continue
        if isinstance(item, Table):
            return item
    return None


# ---------------------------------------------------------------- 写文本
def set_text(p, text):
    """保留首个 run 的样式，替换段落文字。"""
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def set_cell(cell, text):
    set_text(cell.paragraphs[0], text)


def ensure_rows(tbl, need):
    """复制末行版式，补齐到 need 行。"""
    while len(tbl.rows) < need:
        tbl.rows[-1]._tr.addnext(copy.deepcopy(tbl.rows[-1]._tr))


# ---------------------------------------------------------------- 插入
def _copy_run_like(src_run, parent_para, text):
    """按 src_run 的字符格式新建一个 run 并写入 text。"""
    r = copy.deepcopy(src_run._r)
    for t in r.findall(qn('w:t')):
        r.remove(t)
    for br in r.findall(qn('w:br')):
        r.remove(br)
    parent_para._p.append(r)
    run = Run(r, parent_para)
    run.text = text
    return run


def _pick_run(paragraph):
    """挑出段落中真正带格式的 run。

    Word 的段落/单元格常以**空 run** 开头（字号字体都为空），真正承载
    格式的 run 在后面。若直接取 runs[0]，新建内容会丢失字体字号。
    优先取"第一个非空 run"，退而取"第一个带 rPr 的 run"。
    """
    runs = paragraph.runs
    if not runs:
        return None
    for r in runs:
        if r.text.strip():
            return r
    for r in runs:
        if r._r.find(qn('w:rPr')) is not None:
            return r
    return runs[0]


def insert_paragraph_after(anchor, text, template):
    """在 anchor 段落后插入一段文字，格式复制自 template 段落的首个 run。

    template 通常取"同级别的正文段"或"表标题段"，以保证字体字号一致。
    """
    new_p = copy.deepcopy(template._p)
    for r in new_p.findall(qn('w:r')):
        new_p.remove(r)
    for extra in new_p.findall(qn('w:hyperlink')):
        new_p.remove(extra)
    anchor._p.addnext(new_p)
    para = Paragraph(new_p, anchor._parent)
    src_run = _pick_run(template)
    if src_run is not None:
        _copy_run_like(src_run, para, text)
    else:
        para.add_run(text)
    return para


def insert_table_after(anchor, data, doc, template=None):
    """在 anchor 段落后插入一个表格，边框/列宽/字号复制自 template 表格。

    data 为二维列表，第一行为表头。返回新建的 Table。
    """
    n_rows, n_cols = len(data), len(data[0])
    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    anchor._p.addnext(tbl._tbl)          # lxml 的 addnext 会移动元素

    if template is not None and template._tbl is not None:
        # 边框、对齐等表级属性
        tp = tbl._tbl.tblPr
        tp.getparent().replace(tp, copy.deepcopy(template._tbl.tblPr))
        # 列宽（列数一致时才可复用，否则会与 tblGrid 不匹配）
        tpl_grid = template._tbl.find(qn('w:tblGrid'))
        grid = tbl._tbl.find(qn('w:tblGrid'))
        if tpl_grid is not None and grid is not None and \
                len(tpl_grid) == len(grid):
            tbl._tbl.replace(grid, copy.deepcopy(tpl_grid))
        try:
            tbl.style = template.style
        except Exception:
            pass
        header_cell = template.rows[0].cells[0]
        body_cell = (template.rows[1].cells[0]
                     if len(template.rows) > 1 else header_cell)
    else:
        header_cell = body_cell = None

    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.rows[ri].cells[ci]
            para = cell.paragraphs[0]
            for r in list(para.runs):
                r._r.getparent().remove(r._r)
            src = header_cell if ri == 0 else body_cell
            src_run = _pick_run(src.paragraphs[0]) if src is not None else None
            if src_run is not None:
                _copy_run_like(src_run, para, str(val))
            else:
                para.add_run(str(val))
    return tbl


def delete_column(tbl, idx):
    """删除表格的第 idx 列（0 基），并把被删列的宽度按比例分摊给其余列。

    python-docx 没有删除列的接口，需要直接改 XML：逐行删除对应的 w:tc，
    并同步删除 tblGrid 中对应的 w:gridCol，否则列数与列宽定义会不一致。
    """
    grid = tbl._tbl.find(qn('w:tblGrid'))
    removed_w = 0
    if grid is not None and idx < len(grid):
        removed_w = int(grid[idx].get(qn('w:w')) or 0)
        grid.remove(grid[idx])

    for tr in tbl._tbl.findall(qn('w:tr')):
        tcs = tr.findall(qn('w:tc'))
        if idx < len(tcs):
            tr.remove(tcs[idx])

    # 把腾出来的宽度按比例还给剩下的列，保持表格总宽不变
    if grid is not None and removed_w and len(grid):
        ws = [int(g.get(qn('w:w')) or 0) for g in grid]
        total = sum(ws) or 1
        for g, w in zip(grid, ws):
            g.set(qn('w:w'), str(w + int(removed_w * w / total)))


def remove_element(item):
    """从文档中删除一个段落或表格。"""
    el = item._p if isinstance(item, Paragraph) else item._tbl
    el.getparent().remove(el)
