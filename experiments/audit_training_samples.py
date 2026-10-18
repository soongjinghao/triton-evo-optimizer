#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史训练样本审计（0 NPU）——重训预测模型前的必做一步。

评审意见："先审计历史训练样本，修复能够可靠恢复的父子关系，再在固定 35/15
算子隔离方案下重新训练。"

本脚本回答四件事：

  Q1 划分隔离：dev15 的 15 个测试算子是否混在训练样本里？（必须排除）
  Q2 父子关系可恢复性：有多少样本的父代代码能取到？
     取不到 → 8 个父子代码特征全部退化成常数（r_*=0、sim=1），
     这些样本对训练是**噪声**，而且会教模型一个错误的关联。
  Q3 标签口径：parent_latency / child_latency 是否有效，label 与 g_mut 是否自洽。
  Q4 gen0 样本该不该进训练集：在线 gen0 默认不筛选
     （enable_candidate_filter_gen0=False），且 gen0 没有真实父代，
     特征天然退化。留着它们等于用一批"线上永远不会出现的样本"训练。

产出
----
  * 控制台审计报告
  * `--json-out` 审计报告
  * `--clean-out` 清洗后的训练样本（可直接喂给 retrain_candidate_filter.py）

用法
----
    python experiments/audit_training_samples.py \
        --samples experiments/dataset/samples_frozen_20261016.jsonl \
        --split experiments/splits/dev15.json \
        --clean-out experiments/dataset/samples_clean_v7.jsonl
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

DEFAULT_SAMPLES = ROOT / "experiments" / "dataset" / "samples_frozen_20261016.jsonl"
DEFAULT_SPLIT = ROOT / "experiments" / "splits" / "dev15.json"


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
    """样本真正的**参考父代** ID：reference_parent_id → backbone_id → parent_ids[0]。

    评审 v8.1：审计不能只检查旧的第一个父代 —— 交叉场景下 parent_ids[0]
    未必是主干父代，用它核对会把"取错了父代"误报成"数据不一致"。
    """
    for key in ("reference_parent_id", "backbone_id"):
        v = r.get(key)
        if v:
            return str(v), key
    pids = normalize_pids(r.get("parent_ids"))
    return (pids[0], "parent_ids[0]") if pids else (None, "none")


def load_split(path):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("test_kernels", "test", "holdout", "holdout_kernels"):
        if key in s:
            return set(s[key])
    raise KeyError(f"划分文件缺少 test 算子列表，可用键: {list(s)}")


def main():
    ap = argparse.ArgumentParser(description="历史训练样本审计（0 NPU）")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--clean-out", default=None,
                    help="写出清洗后的样本（剔除不可恢复/不参与筛选的样本）")
    ap.add_argument("--keep-gen0", action="store_true",
                    help="保留 gen0 样本（默认剔除：在线不筛选 gen0）")
    ap.add_argument("--keep-code-missing", action="store_true",
                    help="保留父代代码不可恢复的样本（用于对照实验，不要用于正式训练）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.samples, encoding="utf-8") if l.strip()]
    test_kernels = load_split(a.split)
    codes = load_codes()

    print("=" * 88)
    print("历史训练样本审计（0 NPU）")
    print(f"  样本文件: {a.samples}")
    print(f"  划分文件: {a.split}")
    print("=" * 88)
    print(f"\n原始样本 {len(rows)} 条，算子 {len({r.get('kernel') for r in rows})} 个，"
          f"代码库 {len(codes)} 条")

    # ---------------- Q1 划分隔离 ----------------
    print("\n【Q1】划分隔离")
    sample_kernels = {r.get("kernel") for r in rows}
    leak = sample_kernels & test_kernels
    print(f"  dev15 测试算子 {len(test_kernels)} 个，其中出现在样本里: {len(leak)} 个")
    if leak:
        print("  ⚠️  这些算子**必须从训练集剔除**，否则离线评估的 AUC 是泄漏后的乐观值。")
        print(f"      {sorted(leak)}")
    else:
        print("  ✅ 样本里不含测试算子")

    # ---------------- Q2 父子关系可恢复性 ----------------
    print("\n【Q2】父子关系可恢复性（决定 8 个代码特征是不是噪声）")
    stat = collections.Counter()
    by_op = collections.defaultdict(collections.Counter)
    for r in rows:
        op = r.get("operation") or "?"
        pids = normalize_pids(r.get("parent_ids"))
        if not pids:
            st = "no_parent"
        elif pids[0] in codes:
            st = "code_ok"
        else:
            st = "code_missing"
        stat[st] += 1
        by_op[op][st] += 1
    n = len(rows)
    for st in ("code_ok", "code_missing", "no_parent"):
        print(f"  {st:14s} {stat[st]:6d}  ({stat[st] / n * 100:5.1f}%)")
    print("\n  按 operation 拆解：")
    print(f"    {'operation':24s} {'code_ok':>8s} {'code_missing':>14s} {'no_parent':>10s} {'缺失率':>8s}")
    for op, c in sorted(by_op.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(c.values())
        bad = c["code_missing"] + c["no_parent"]
        print(f"    {op[:24]:24s} {c['code_ok']:8d} {c['code_missing']:14d} "
              f"{c['no_parent']:10d} {bad / tot * 100:7.1f}%")
    if stat["code_missing"]:
        print(f"\n  ⚠️  {stat['code_missing']} 条样本有 parent_ids 但父代代码不在库中：")
        print("      这 8 个特征会退化为常数（r_*=0、sim=1），与 gen0 无法区分。")
        print("      它们不是『数据缺失』而是『特征错误』，必须剔除，不能靠 imputer 兜底。")

    # ---------------- Q3 标签口径 ----------------
    print("\n【Q3】标签口径")
    bad_par = sum(1 for r in rows if _pos(r.get("parent_latency")) is None)
    bad_ch = sum(1 for r in rows if _pos(r.get("child_latency")) is None)
    print(f"  parent_latency 无效: {bad_par} / {n} ({bad_par / n * 100:.1f}%)")
    print(f"  child_latency  无效: {bad_ch} / {n} ({bad_ch / n * 100:.1f}%)")
    lab = collections.Counter()
    op_lab = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        y = 1 if r.get("label") else 0
        lab[y] += 1
        op_lab[r.get("operation") or "?"][y] += 1
    print(f"  label=1 占比: {lab[1] / n * 100:.1f}%  (#1={lab[1]}, #0={lab[0]})")
    print("  按 operation：")
    for op, (n0, n1) in sorted(op_lab.items(), key=lambda x: -(x[1][0] + x[1][1])):
        print(f"    {op[:24]:24s} n={n0 + n1:5d}  label=1 {n1 / (n0 + n1) * 100:5.1f}%")

    # ---------------- Q4 gen0 ----------------
    print("\n【Q4】gen0 样本")
    gen0 = [r for r in rows if str(r.get("operation") or "").startswith("gen0")]
    print(f"  gen0 样本 {len(gen0)} 条（{len(gen0) / n * 100:.1f}%），全部无 parent_ids")
    print("  在线默认 enable_candidate_filter_gen0=False，即**从不筛选 gen0**；")
    print("  且 gen0 没有真实父代，代码特征天然退化为常数。")
    print("  → 默认从训练集剔除；如确要保留请加 --keep-gen0。")

    # ---------------- Q5 参考父代一致性 ----------------
    # 评审指出：审计只验证了 parent_latency > 0，没有验证"父代 ID / 源码 / 延迟"
    # 与训练标签指向的是不是同一个参考对象。这里是针对性的交叉核对。
    print("\n【Q5】参考父代一致性（参考父代 ID ↔ 源码 ↔ 延迟 ↔ 标签是否指向同一对象）")
    lat_of = {}            # (kernel, candidate_id) -> 该候选被评测出的延迟
    for r in rows:
        cl = _pos(r.get("child_latency"))
        if cl is not None:
            lat_of[(r.get("kernel"), str(r.get("candidate_id")))] = cl

    src_cnt = collections.Counter()
    checked = checked_old = 0
    mism = mism_old = 0
    ref_code_missing = 0
    sha_checked = sha_bad = 0
    label_bad = 0
    examples = []
    for r in rows:
        pids = normalize_pids(r.get("parent_ids"))
        if not pids:
            continue
        ref, src = ref_pid(r)
        src_cnt[src] += 1
        pl = _pos(r.get("parent_latency"))
        cl = _pos(r.get("child_latency"))
        # 旧口径（只用 parent_ids[0]）作为对照：它能告诉我们
        # 之前的 37.6% 不一致里有多少其实是"取错了父代"。
        pl_old = lat_of.get((r.get("kernel"), pids[0]))
        if pl is not None and pl_old is not None:
            checked_old += 1
            if abs(pl - pl_old) / max(pl, pl_old) > 0.01:
                mism_old += 1
        if ref is None:
            continue
        pl_ref = lat_of.get((r.get("kernel"), ref))
        if pl is None or pl_ref is None:
            continue
        checked += 1
        if abs(pl - pl_ref) / max(pl, pl_ref) > 0.01:
            mism += 1
            if len(examples) < 3:
                examples.append(
                    f"{r.get('kernel')} 子={r.get('candidate_id')} 父={ref}"
                    f"（来源 {src}）：记录的 parent_latency={pl:.4f} "
                    f"vs 父代自身 child_latency={pl_ref:.4f}")
        if ref not in codes:
            ref_code_missing += 1
        if cl is not None and int(bool(r.get("label"))) != int(pl > cl):
            label_bad += 1
        psha = r.get("parent_code_sha256")
        if psha and ref in codes:
            sha_checked += 1
            if _sha(codes[ref]) != psha:
                sha_bad += 1

    print(f"  参考父代来源: {dict(src_cnt)}")
    if set(src_cnt) <= {"parent_ids[0]", "none"}:
        print("  ℹ️  本快照没有 reference_parent_id / backbone_id 字段（v8.1 才加入），")
        print("     所以这里只能退回 parent_ids[0]，与旧口径一致 —— 也就是说")
        print("     现有 7431 条**无法**验证『取错父代』到底占多少。")
        print("     只有用修复后的采集链路重新采数据，这项核对才有意义。")
    if checked:
        print(f"  按**参考父代**核对：可核对 {checked} 条，延迟不一致（>1%）{mism} 条 "
              f"({mism / checked * 100:.1f}%)")
    else:
        print("  按**参考父代**核对：无可核对样本")
    if checked_old:
        print(f"  按**旧口径 parent_ids[0]** 核对：可核对 {checked_old} 条，"
              f"不一致 {mism_old} 条 ({mism_old / checked_old * 100:.1f}%)")
    if checked and checked_old and mism < mism_old:
        print(f"  → 改用参考父代后少了 {mism_old - mism} 条不一致："
              f"这部分是**取错父代**造成的，不是数据错误。")
    for e in examples:
        print(f"     例: {e}")
    print(f"  参考父代源码不在库中: {ref_code_missing} / {checked}"
          if checked else "  参考父代源码不在库中: —")
    print(f"  父代源码哈希对不上: {sha_bad} / {sha_checked}"
          if sha_checked else "  父代源码哈希对不上: —（旧快照无 parent_code_sha256）")
    print(f"  标签与 (parent_latency > child_latency) 不自洽: {label_bad} / {checked}"
          if checked else "  标签自洽性: —")
    if mism:
        print("  ⚠️  仍有延迟不一致：参考父代口径尚未完全统一（交叉主干父代 /")
        print("     评测基线不同源）。新数据必须让 reference_parent_id 与")
        print("     parent_latency 指向同一位已评测父代，做到 100% 可核验。")

    # ---------------- Q6 源码完整性 ----------------
    print("\n【Q6】源码完整性")
    child_missing = sum(1 for r in rows if str(r.get("candidate_id")) not in codes)
    print(f"  子代源码缺失: {child_missing} / {n} ({child_missing / n * 100:.1f}%)")
    has_sha = sum(1 for r in rows if r.get("child_code_sha256"))
    print(f"  带 child_code_sha256 的样本: {has_sha} / {n}"
          f"（旧快照无此字段，需重新采集后才有）")
    csha_checked = csha_bad = 0
    for r in rows:
        s = r.get("child_code_sha256")
        cid = str(r.get("candidate_id"))
        if s and cid in codes:
            csha_checked += 1
            if _sha(codes[cid]) != s:
                csha_bad += 1
    if csha_checked:
        print(f"  子代源码哈希对不上: {csha_bad} / {csha_checked}"
              f"（对不上说明 codes/ 里的文件与样本不是同一份代码）")

    # ---------------- Q7 候选池标识 ----------------
    print("\n【Q7】候选池标识（决定 P@k 能不能算成真实 20 选 4）")
    with_pool = sum(1 for r in rows if r.get("pool_id"))
    print(f"  带 pool_id 的样本: {with_pool} / {n}")
    if with_pool:
        sizes = collections.Counter()
        for r in rows:
            if r.get("pool_id"):
                sizes[r["pool_id"]] += 1
        sz = sorted(sizes.values())
        print(f"  池数 {len(sz)}，池大小 中位 {sz[len(sz)//2]}，"
              f"最小 {sz[0]}，最大 {sz[-1]}")
    else:
        # 旧快照没有 pool_id，用 (kernel, gen) 近似看一下会并成多大的假池
        sizes = collections.Counter((r.get("kernel"), r.get("gen")) for r in rows)
        sz = sorted(sizes.values())
        print("  ⚠️  旧快照无 pool_id。若退化为按 (kernel, gen) 分组：")
        print(f"     组数 {len(sz)}，组内条数 中位 {sz[len(sz)//2]}，"
              f"最小 {sz[0]}，最大 {sz[-1]}")
        print("     线上实际是每代约 20 选 4；中位数远大于 20 说明分组把多批候选")
        print("     并成了假池 —— 此时的 P@4 / 随机基线 / 新旧对比全部失真。")

    # ---------------- 清洗 ----------------
    keep, drop_reason = [], collections.Counter()
    for r in rows:
        op = str(r.get("operation") or "")
        if not a.keep_gen0 and op.startswith("gen0"):
            drop_reason["gen0_never_filtered_online"] += 1
            continue
        if r.get("kernel") in test_kernels:
            drop_reason["holdout_kernel_leak"] += 1
            continue
        pids = normalize_pids(r.get("parent_ids"))
        if not pids:
            drop_reason["no_parent_ids"] += 1
            continue
        # 按**参考父代**判定（与在线特征提取取同一个父代），
        # 不再用 parent_ids[0] —— 交叉场景下它不是主干父代。
        ref, _ = ref_pid(r)
        if (not a.keep_code_missing) and (ref is None or ref not in codes):
            drop_reason["parent_code_unrecoverable"] += 1
            continue
        r["reference_parent_id"] = ref
        if _pos(r.get("parent_latency")) is None or _pos(r.get("child_latency")) is None:
            drop_reason["invalid_latency"] += 1
            continue
        keep.append(r)

    print("\n【清洗结果】")
    print(f"  原始 {n} → 保留 {len(keep)}（{len(keep) / n * 100:.1f}%），"
          f"覆盖算子 {len({r['kernel'] for r in keep})} 个")
    for k, v in drop_reason.most_common():
        print(f"    剔除 {k:32s} {v:6d}")
    if keep:
        y1 = sum(1 for r in keep if r.get("label"))
        print(f"  保留集 label=1 占比 {y1 / len(keep) * 100:.1f}%")
        assert not ({r["kernel"] for r in keep} & test_kernels), "清洗后仍含测试算子"

    if a.clean_out:
        Path(a.clean_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.clean_out, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✅ 清洗后样本已写出: {a.clean_out}")

    if a.json_out:
        out = {
            "samples": a.samples, "n_raw": n,
            "n_kernels_raw": len(sample_kernels),
            "holdout_kernels": sorted(test_kernels),
            "holdout_in_samples": sorted(leak),
            "recoverability": dict(stat),
            "by_operation": {k: dict(v) for k, v in by_op.items()},
            "label_rate": lab[1] / n if n else None,
            "gen0": len(gen0),
            "n_keep": len(keep),
            "drop_reasons": dict(drop_reason),
        }
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"✅ 审计报告已写出: {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
