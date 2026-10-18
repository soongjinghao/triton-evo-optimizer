#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 run_id 从累计样本里切出新采集的那一批（0 NPU）。

`experiments/dataset/samples.jsonl` 是**累计**文件（每跑一次追加），
审计/训练/验收都需要能干净地取出"这一批"。

用法
----
    python experiments/merge_new_samples.py \
        --src experiments/dataset/samples.jsonl \
        --run-ids 905 --out experiments/dataset/samples_new_5op.jsonl
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "experiments" / "dataset" / "samples.jsonl"


def main():
    ap = argparse.ArgumentParser(description="按 run_id 切出新采集样本")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    want = {str(x) for x in a.run_ids}
    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    sel = [r for r in rows if str(r.get("run_id")) in want]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"源 {a.src}: {len(rows)} 条 → run_id {sorted(want)}: {len(sel)} 条")
    print(f"算子 {len({r.get('kernel') for r in sel})} 个: "
          f"{sorted({r.get('kernel') for r in sel})}")
    pools = {r.get("pool_id") for r in sel if r.get("pool_id")}
    print(f"候选池 {len(pools)} 个")
    print(f"✅ 写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
