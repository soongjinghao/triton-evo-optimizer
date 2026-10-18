#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新采集数据的**验收检查**（0 NPU）——对应评审给出的验收表。

评审 v8.1「新数据验收要求」：

| 指标                    | 建议目标                    |
|-------------------------|---------------------------|
| pool_id 记录率           | 100%                      |
| 已评测候选的源码完整率    | 100%                      |
| 参考父代 ID、源码与延迟一致性 | 100% 可核验（允许计时误差） |
| 特征提取一致性测试        | 全部通过                   |
| 完整池评价               | 用于正式 P@k 的池标签完整   |
| 数据划分                 | 不用参与开发的测试算子训练  |

本脚本把前 5 项做成可执行的断言，任何一项不达标就退出码 1。
第 6 项由 `--exclude-kernels`（留出口径）与 `experiments/splits/dev15.json` 控制，
这里只做提示与核对，不代替划分文件本身。

用法
----
    python experiments/verify_new_samples.py \
        --samples experiments/dataset/samples.jsonl \
        --expect-pool 20 \
        --exclude-kernels $(cat experiments/splits/dev15.json) \
        --json-out experiments/results/verify_new_samples.json
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.candidate_filter_features_shared import (      # noqa: E402
    load_codes, normalize_pids)

DEFAULT_CODES = ROOT / "experiments" / "dataset" / "codes"
DEFAULT_SAMPLES = ROOT / "experiments" / "dataset" / "samples.jsonl"


def _pos(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _sha(text):
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def ref_pid(r):
    """与在线/训练同一套口径：reference_parent_id → backbone_id → parent_ids[0]。"""
    for key in ("reference_parent_id", "backbone_id"):
        v = r.get(key)
        if v:
            return str(v), key
    pids = normalize_pids(r.get("parent_ids"))
    return (pids[0], "parent_ids[0]") if pids else (None, "none")


def main():
    ap = argparse.ArgumentParser(description="新采集样本的验收检查（0 NPU）")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--codes-dir", default=str(DEFAULT_CODES))
    ap.add_argument("--expect-pool", type=int, default=20,
                    help="期望的候选池大小（线上 20 选 4 → 20）")
    ap.add_argument("--lat-tol", type=float, default=0.01,
                    help="父代延迟一致性容差（相对误差，默认 1%）")
    ap.add_argument("--exclude-kernels", nargs="*", default=[],
                    help="留出算子名单：一旦给出，样本里出现这些算子即**硬失败**")
    ap.add_argument("--expected-full-pools", type=int, default=0,
                    help="期望的完整池数量；>0 时不达标即失败"
                         "（例如 3 算子 × 2 代 = 6）")
    ap.add_argument("--min-latency-check-rate", type=float, default=0.0,
                    help="父代延迟交叉核对覆盖率下限（0~1）。注意：覆盖率天生"
                         "受限于'父代是否也在本批样本里'，设得太高会误报")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="任何一项不达标就以退出码 1 结束（CI 用）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.samples, encoding="utf-8") if l.strip()]
    codes = load_codes(a.codes_dir)
    n = len(rows)
    print("=" * 84)
    print("新采集样本验收检查（0 NPU）")
    print(f"  样本 {a.samples}: {n} 条")
    print(f"  代码库 {a.codes_dir}: {len(codes)} 条")
    print("=" * 84)

    # ---- (1) pool_id 记录率 ----
    with_pool = [r for r in rows if r.get("pool_id")]
    print(f"\n[1] pool_id 记录率: {len(with_pool)} / {n} "
          f"({len(with_pool) / max(1, n) * 100:.1f}%)  目标 100%")

    # ---- (2) 已评测候选源码完整率 ----
    evaluated = [r for r in rows if _pos(r.get("child_latency")) is not None]
    child_ok = child_sha_bad = child_sha_checked = 0
    for r in evaluated:
        cid = str(r.get("candidate_id"))
        if cid in codes:
            child_ok += 1
            s = r.get("child_code_sha256")
            if s:
                child_sha_checked += 1
                if _sha(codes[cid]) != s:
                    child_sha_bad += 1
    print(f"[2] 已评测候选源码完整率: {child_ok} / {len(evaluated)} "
          f"({child_ok / max(1, len(evaluated)) * 100:.1f}%)  目标 100%")
    print(f"    源码哈希比对: 不符 {child_sha_bad} / {child_sha_checked}"
          if child_sha_checked else "    源码哈希比对: —（样本无 child_code_sha256）")

    # ---- (3) 参考父代 ID / 源码 / 延迟一致性 ----
    lat_of = {}
    for r in rows:
        cl = _pos(r.get("child_latency"))
        if cl is not None:
            lat_of[(r.get("kernel"), str(r.get("candidate_id")))] = cl
    src_cnt = collections.Counter()
    checked = lat_bad = code_bad = sha_bad = sha_checked = no_ref = 0
    label_bad = 0
    examples = []
    for r in rows:
        ref, src = ref_pid(r)
        src_cnt[src] += 1
        if ref is None:
            no_ref += 1
            continue
        if ref not in codes:
            code_bad += 1
        psha = r.get("parent_code_sha256")
        if psha and ref in codes:
            sha_checked += 1
            if _sha(codes[ref]) != psha:
                sha_bad += 1
        pl, cl = _pos(r.get("parent_latency")), _pos(r.get("child_latency"))
        pl_ref = lat_of.get((r.get("kernel"), ref))
        if pl is not None and pl_ref is not None:
            checked += 1
            if abs(pl - pl_ref) / max(pl, pl_ref) > a.lat_tol:
                lat_bad += 1
                if len(examples) < 3:
                    examples.append(
                        f"{r.get('kernel')} 子={r.get('candidate_id')} 父={ref}: "
                        f"parent_latency={pl:.4f} vs 父代自身={pl_ref:.4f}")
        if pl is not None and cl is not None:
            if int(bool(r.get("label"))) != int(pl > cl):
                label_bad += 1
    print(f"\n[3] 参考父代一致性（目标 100% 可核验）")
    print(f"    参考父代来源分布: {dict(src_cnt)}")
    print(f"    延迟可核对 {checked} 条，不一致 {lat_bad} 条 "
          f"({lat_bad / max(1, checked) * 100:.1f}%)")
    unverified = len(rows) - checked
    if unverified:
        print(f"    ⚠️ 另有 {unverified} 条**未能**按同样方式交叉核对"
              f"（参考父代本身不在本批样本里，或延迟缺失）。")
        print("       这些样本**并非已证实有误**，只是尚未独立核对；")
        print("       要覆盖它们需关联种子评测记录与历史候选档案（见 --min-latency-check-rate）。")
    print(f"    参考父代源码不在库中: {code_bad} / {len(rows) - no_ref}")
    print(f"    父代源码哈希不符: {sha_bad} / {sha_checked}"
          if sha_checked else "    父代源码哈希不符: —（样本无 parent_code_sha256）")
    print(f"    标签与 (parent_latency > child_latency) 不自洽: {label_bad}")
    for e in examples:
        print(f"      例: {e}")

    # ---- (4) 完整池评价 ----
    pools = collections.defaultdict(list)
    for r in with_pool:
        pools[r["pool_id"]].append(r)
    full = [p for p, v in pools.items() if len(v) == a.expect_pool and
            all(_pos(x.get("parent_latency")) and _pos(x.get("child_latency"))
                for x in v)]
    sizes = sorted(len(v) for v in pools.values())
    print(f"\n[4] 完整池评价")
    if pools:
        print(f"    pool_id 分组 {len(pools)} 个，组大小 中位 "
              f"{sizes[len(sizes) // 2]}（min {sizes[0]} / max {sizes[-1]}）")
    else:
        print("    ⚠️ 没有任何 pool_id，无法按真实候选池评价")
    print(f"    大小 == {a.expect_pool} 且全部有标签的完整池: {len(full)} 个 "
          f"→ 这些池才能进正式 P@k")
    orphan = sum(1 for p, v in pools.items() if len(v) != a.expect_pool)
    if orphan:
        print(f"    ⚠️ {orphan} 个池大小不等于 {a.expect_pool}，"
              f"只能作诊断（可能是搜索被预算截断或多批合并）")

    # ---- (5) 划分隔离：一旦给了留出名单，混入即硬失败 ----
    ex = set(a.exclude_kernels)
    leaked = sorted({r.get("kernel") for r in rows if r.get("kernel") in ex})
    print(f"\n[5] 划分隔离")
    print(f"    样本算子 {len({r.get('kernel') for r in rows})} 个；"
          f"留出口径 {len(ex)} 个")
    if ex:
        if leaked:
            print(f"    ❌ 样本中落在留出名单的算子 {len(leaked)} 个: "
                  f"{leaked[:8]}{' ...' if len(leaked) > 8 else ''}")
            print("       → 这些算子参与过模型选择，混入训练集会让离线指标虚高。")
        else:
            print("    ✅ 未发现留出算子混入")
    else:
        print("    ⚠️ 未给 --exclude-kernels，本次不做隔离检查")

    full_ok = (a.expected_full_pools <= 0 or len(full) >= a.expected_full_pools)
    if a.expected_full_pools > 0:
        print(f"\n[6] 完整池数量: {len(full)} / 期望 {a.expected_full_pools} → "
              f"{'✅' if full_ok else '❌'}")
    check_rate_ok = True
    if a.min_latency_check_rate > 0:
        rate = checked / max(1, n)
        check_rate_ok = rate >= a.min_latency_check_rate
        print(f"[7] 父代延迟核对覆盖率: {checked}/{n} = {rate:.1%} "
              f"（下限 {a.min_latency_check_rate:.0%}）→ "
              f"{'✅' if check_rate_ok else '❌'}")

    ok_all = (len(with_pool) == n and child_ok == len(evaluated)
              and lat_bad == 0 and code_bad == 0 and sha_bad == 0
              and label_bad == 0 and bool(pools)
              and not leaked and full_ok and check_rate_ok)
    print("\n" + "=" * 84)
    print("验收结论: " + ("✅ 全部达标，可以进入重训与真实 Top-k 评价"
                          if ok_all else "❌ 未达标，见上（对应项需先修采集链路）"))
    print("=" * 84)

    if a.json_out:
        out = {"samples": str(a.samples), "n_samples": n,
               "pool_id_rate": len(with_pool) / max(1, n),
               "evaluated": len(evaluated),
               "child_code_ok": child_ok, "child_sha_bad": child_sha_bad,
               "ref_source": dict(src_cnt),
               "ref_latency_checked": checked, "ref_latency_bad": lat_bad,
               "ref_code_missing": code_bad, "parent_sha_bad": sha_bad,
               "label_inconsistent": label_bad,
               "n_pools": len(pools),
               "n_full_pools": len(full), "expect_pool": a.expect_pool,
               "expected_full_pools": a.expected_full_pools,
               "latency_check_rate": checked / max(1, n),
               "leaked_kernels": leaked, "pass": bool(ok_all)}
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"✅ JSON: {a.json_out}")
    return 0 if (ok_all or not a.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
