"""删除被判定为假成功（search_ok=False）的实验产物，以便重跑。

只处理明确标记为 False 的记录；没有该字段的旧格式数据一律保留，避免误删。
"""
import json
import glob
import shutil
from pathlib import Path

ROOT = Path('/workspace/Agent')
removed = []

for f in glob.glob(str(ROOT / 'experiments/results/**/*__r[0-9].json'), recursive=True):
    if 'remeasure' in f:
        continue
    try:
        d = json.loads(Path(f).read_text(encoding='utf-8'))
    except Exception:
        continue

    if d.get('search_ok') is not False:
        continue

    p = Path(f)
    stem = p.name.replace('.json', '')
    for pat in (f'{stem}.py', f'{stem}.remeasure.json'):
        q = p.parent / pat
        if q.exists():
            q.unlink()
            removed.append(str(q))
    g = p.parent / f'{stem}_gen0'
    if g.exists():
        shutil.rmtree(g)
        removed.append(str(g))
    p.unlink()
    removed.append(str(p))
    print(f'  删除无效: {p.parent.name}/{stem}')

print(f'\n共删除 {len(removed)} 个文件/目录')
