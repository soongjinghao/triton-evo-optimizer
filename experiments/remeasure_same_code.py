#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一份源码的复测诊断（少量 NPU）。

为什么需要
----------
`analyze_pool_duplicates.py` 发现：**同一份源码**在同一候选池内被多次评测时，
延迟相对极差最大可达 18%（例如 eye_kernel 的 1.28~1.50us）。

在把它归因于"计时噪声"之前，必须先核对评测输入与启动配置 ——
评审明确指出这一点。若复测后同一份代码的延迟仍然稳定在很窄的区间，
说明池内的离散来自别处（例如不同的外部启动配置）；
若复测本身就有同样的波动，那才是测量噪声。

做法
----
1. 从新采集样本里挑出"同一源码被多次评测且极差最大"的若干组；
2. 用**固定的输入与启动配置**（同一个 kernel 工作目录、同一 device）
   对每份代码独立重复计时 `--repeats` 次；
3. 报告复测内部的相对极差 / 变异系数，并与池内原始极差对照。

用法
----
    source set_env/set_api_local.sh
    export EA_NUM_DEVICES=1
    python experiments/remeasure_same_code.py \
        --samples experiments/dataset/samples_new_5op.jsonl \
        --top 4 --repeats 10 \
        --json-out experiments/results/remeasure_same_code.json
"""

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_features_shared import load_codes      # noqa: E402
from experiments.harness import build_executor                            # noqa: E402

DEFAULT_SAMPLES = ROOT / "experiments" / "dataset" / "samples_new_5op.jsonl"
DEFAULT_CODES = ROOT / "experiments" / "dataset" / "codes"


def _pos(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def pick_groups(rows, top, min_n=2):
    """挑出同一源码被多次评测、且池内相对极差最大的若干组。"""
    by_hash = collections.defaultdict(list)
    for r in rows:
        h = r.get("child_code_sha256")
        cl = _pos(r.get("child_latency"))
        if h and cl:
            by_hash[str(h)].append((cl, r))
    groups = []
    for h, items in by_hash.items():
        if len(items) < min_n:
            continue
        lats = [v for v, _ in items]
        spread = (max(lats) - min(lats)) / min(lats)
        groups.append((spread, h, items))
    groups.sort(reverse=True)
    return groups[:top]


def main():
    ap = argparse.ArgumentParser(description="同一源码的复测诊断（少量 NPU）")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--codes-dir", default=str(DEFAULT_CODES))
    ap.add_argument("--top", type=int, default=4, help="取极差最大的前 N 组")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.samples, encoding="utf-8") if l.strip()]
    codes = load_codes(a.codes_dir)
    groups = pick_groups(rows, a.top)
    if not groups:
        print("❌ 没有找到『同一源码被多次评测』的组。")
        return 1

    print("=" * 88)
    print("同一源码复测诊断（固定输入与启动配置）")
    print(f"  样本 {a.samples}：{len(rows)} 条；复测次数 {a.repeats}")
    print("=" * 88)

    out = []
    for spread, h, items in groups:
        r0 = items[0][1]
        kernel = r0.get("kernel")
        cid = str(r0.get("candidate_id"))
        code = codes.get(cid)
        if not code:
            print(f"  ⚠️ 跳过 {h[:12]}：代码库里没有 {cid}")
            continue
        from experiments.harness import make_config
        cfg = make_config("full", kernel, run_id=0, seed=0)
        ex = build_executor(kernel, cfg)
        vals = []
        for _ in range(a.repeats):
            res = ex.measure_repeat(code, repeats=1, device_id=a.device)
            if res and res.success and _pos(res.execution_time):
                vals.append(float(res.execution_time))
        if not vals:
            print(f"  ❌ {kernel} {h[:12]}：复测全部失败")
            continue
        rel = (max(vals) - min(vals)) / min(vals)
        cv = statistics.pstdev(vals) / statistics.fmean(vals)
        rec = dict(kernel=kernel, sha=h, candidate_id=cid,
                   pool_spread=round(spread, 4),
                   pool_latencies=sorted({round(v, 4) for v, _ in items}),
                   remeasure_n=len(vals),
                   remeasure=[round(v, 4) for v in vals],
                   remeasure_spread=round(rel, 4),
                   remeasure_cv=round(cv, 4),
                   remeasure_median=round(statistics.median(vals), 4))
        out.append(rec)
        print(f"\n  {kernel}  sha={h[:12]}  候选 {cid}")
        print(f"    池内实测 {rec['pool_latencies']}  相对极差 {spread:.1%}")
        print(f"    复测 {len(vals)} 次：中位 {rec['remeasure_median']:.4f} us，"
              f"相对极差 {rel:.1%}，CV {cv:.2%}")
        print(f"    → 复测极差 {'≈' if abs(rel - spread) < 0.03 else '≠'} 池内极差"
              f"{'：可归因为测量波动' if abs(rel - spread) < 0.03 else '：需核对启动配置'}")

    print("\n" + "-" * 88)
    if out:
        same = sum(1 for r in out if abs(r["remeasure_spread"] - r["pool_spread"]) < 0.03)
        print(f"结论：{same}/{len(out)} 组的复测波动与池内极差一致 → 测量噪声；")
        print(f"      {len(out) - same}/{len(out)} 组不一致 → 需核对外部启动配置/输入。")
    print("-" * 88)

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"✅ JSON: {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
