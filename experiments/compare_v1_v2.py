#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1 / V2 同预算对照（0 NPU）。

对照什么
--------
V2 只改**生成机制**（候选指纹去重 + 有限重试补位 + 目标明确的变异 /
结构化交叉），NPU 预算与候选池规模与 V1 完全相同。因此要对照的是：

  1. 有效多样性：每池不同源码数、重复率 —— **分母必须是生成阶段的全部候选**，
     不能只用评测成功的样本（否则 V2 因失败多而显得更"多样"）；
  2. 失败情况：失败候选数与**失败类型**（超时 / 算错 / 设备 / 编译）；
  3. 成本：NPU 调用数（应相同）、Token、墙钟；
  4. 收益：最终加速比 S。

【口径修正】评审指出两点：
  * 多样性统计要用**生成候选事件**作分母（本脚本直接读事件日志）；
  * 结果 JSON 必须用**显式 run_id** 定位，不能靠文件名排序自动挑。

用法
----
    python experiments/compare_v1_v2.py \
        --v1-config fulleval --v1-run-id 905 \
        --v2-config fulleval_v2 --v2-run-id 906 \
        --kernels eye_kernel compute_identity_kernel moe_align_block_size_stage \
        --json-out experiments/results/compare_v1_v2.json
"""

import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "experiments" / "logs"


def load_pools(config, run_id, kernels):
    """按显式 run_id 读该次运行的全部生成候选（含评测失败的）。"""
    pools = collections.defaultdict(list)
    for p in sorted(glob.glob(str(LOGS / f"filter_online__{config}__*__r{run_id}.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") != "eval" or r.get("gen", 0) < 1:
                continue
            if str(r.get("run_id", run_id)) != str(run_id):
                continue
            if kernels and r.get("kernel") not in kernels:
                continue
            pools[r.get("pool_id") or f"{r.get('kernel')}|r{run_id}|g{r.get('gen')}"].append(r)
    return pools


def fail_kind(r):
    err = (r.get("error") or "").lower()
    for kw, kind in (("timeout", "timeout"),
                     ("accuracy", "accuracy"),
                     ("assert", "accuracy"),
                     ("allclose", "accuracy"),
                     ("mismatch", "accuracy"),
                     ("compile", "compile"),
                     ("resource", "resource"),
                     ("device", "device")):
        if kw in err:
            return kind
    return "other"


def pool_stats(pools):
    n = ok = 0
    distinct = 0
    per_pool = []
    fails = collections.Counter()
    for members in pools.values():
        n += len(members)
        ok += sum(1 for r in members if r.get("correctness_pass"))
        hs = [str(r.get("child_code_sha256") or "") for r in members]
        d = len({h for h in hs if h})
        distinct += d
        per_pool.append(d)
        for r in members:
            if not r.get("correctness_pass"):
                fails[fail_kind(r)] += 1
    return dict(n_candidates=n, n_pools=len(pools), n_success=ok,
                success_rate=ok / max(1, n),
                n_distinct=distinct,
                distinct_per_pool=sum(per_pool) / max(1, len(per_pool)),
                dup_rate=1 - distinct / max(1, n),
                fail_by_kind=dict(fails), n_failed=n - ok)


def run_stats(res_dir, kernels, seed):
    """按**显式** seed 定位结果 JSON，不做文件名排序猜测。"""
    d = Path(res_dir)
    out = {}
    for k in kernels:
        p = d / f"{k}__r{seed}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        a = r.get("audit", {}) or {}
        out[k] = dict(npu_calls=a.get("npu_calls"), eval_budget=a.get("eval_budget"),
                      budget_ok=a.get("budget_ok"), t_best_us=r.get("t_best_us"),
                      S=r.get("S"), seconds=r.get("seconds"),
                      llm_tokens=a.get("llm_tokens"))
    return out


def main():
    ap = argparse.ArgumentParser(description="V1 / V2 同预算对照（0 NPU）")
    ap.add_argument("--v1-config", default="fulleval")
    ap.add_argument("--v1-run-id", required=True)
    ap.add_argument("--v2-config", default="fulleval_v2")
    ap.add_argument("--v2-run-id", required=True)
    ap.add_argument("--v1-seed", default=None, help="V1 结果 JSON 的 seed（显式指定）")
    ap.add_argument("--v2-seed", default=None)
    ap.add_argument("--v1-results", default=str(ROOT / "experiments/results/filter_online/fulleval"))
    ap.add_argument("--v2-results", default=str(ROOT / "experiments/results/filter_online/fulleval_v2"))
    ap.add_argument("--kernels", nargs="*", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    p1 = load_pools(a.v1_config, a.v1_run_id, set(a.kernels) if a.kernels else None)
    p2 = load_pools(a.v2_config, a.v2_run_id, set(a.kernels) if a.kernels else None)
    if not p1 or not p2:
        print("❌ 找不到指定 run_id 的候选池事件。")
        return 1
    kernels = a.kernels or sorted({r.get("kernel") for v in p1.values() for r in v})
    s1, s2 = pool_stats(p1), pool_stats(p2)

    print("=" * 92)
    print("V1 / V2 同预算对照（0 NPU）")
    print(f"  V1: config={a.v1_config} run_id={a.v1_run_id}  V2: config={a.v2_config} run_id={a.v2_run_id}")
    print(f"  分母 = 生成阶段的全部候选（含评测失败），共 {len(kernels)} 个算子")
    print("=" * 92)

    print(f"\n【A】有效多样性与失败情况")
    print(f"  {'指标':22s} {'V1':>12s} {'V2':>12s} {'变化':>12s}")
    rows = [("n_pools", "候选池数", "d"), ("n_candidates", "生成候选总数", "d"),
            ("n_success", "评测成功", "d"), ("success_rate", "成功率", "p"),
            ("n_distinct", "不同源码数", "d"),
            ("distinct_per_pool", "每池不同源码", "f"),
            ("dup_rate", "池内重复率", "p"), ("n_failed", "失败候选", "d")]
    for key, label, fmt in rows:
        v1, v2 = s1[key], s2[key]
        if fmt == "p":
            print(f"  {label:22s} {v1:>12.1%} {v2:>12.1%} {v2 - v1:>+12.1%}")
        elif fmt == "f":
            print(f"  {label:22s} {v1:>12.2f} {v2:>12.2f} {v2 - v1:>+12.2f}")
        else:
            print(f"  {label:22s} {v1:>12d} {v2:>12d} {v2 - v1:>+12d}")

    print(f"\n  失败类型分布：V1 {s1['fail_by_kind'] or '—'}   V2 {s2['fail_by_kind'] or '—'}")
    if s2["n_failed"] and set(s2["fail_by_kind"]) <= {"other"}:
        print("    ℹ️ 全部归为 other：这批日志是**修复前**记录的（只有一句 "
              "'Performance test failed'）。")
        print("       现在 executor 已把失败分为 timeout / accuracy / compile / "
              "device / resource，新运行才会带类型。")

    def by_gen(pools):
        """跨 run 对齐：不同运行的 pool_id 不同，按 (kernel, gen) 配对。"""
        out = collections.defaultdict(list)
        for members in pools.values():
            if not members:
                continue
            out[(members[0].get("kernel"), members[0].get("gen"))].extend(members)
        return out

    print(f"\n【B】逐池不同源码数（按 kernel × gen 对齐；两次运行 pool_id 不同）")
    print(f"  {'kernel / gen':50s} {'V1':>6s} {'V2':>6s}")
    g1, g2 = by_gen(p1), by_gen(p2)
    for k in sorted(set(g1) | set(g2)):
        d1 = len({r.get("child_code_sha256") for r in g1.get(k, []) if r.get("child_code_sha256")})
        d2 = len({r.get("child_code_sha256") for r in g2.get(k, []) if r.get("child_code_sha256")})
        print(f"  {str(k)[:50]:50s} {d1:>6d} {d2:>6d}")

    r1 = run_stats(a.v1_results, kernels, a.v1_seed or a.v1_run_id[-1])
    r2 = run_stats(a.v2_results, kernels, a.v2_seed or a.v2_run_id[-1])
    print(f"\n【C】成本与收益（逐算子）")
    print(f"  {'kernel':34s} {'V1 S':>9s} {'V2 S':>9s} {'V1 NPU':>7s} {'V2 NPU':>7s} "
          f"{'V1 tok':>10s} {'V2 tok':>10s}")
    for k in kernels:
        x, y = r1.get(k, {}), r2.get(k, {})
        f = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        print(f"  {k[:34]:34s} {f(x.get('S')):>9s} {f(y.get('S')):>9s} "
              f"{str(x.get('npu_calls')):>7s} {str(y.get('npu_calls')):>7s} "
              f"{str(x.get('llm_tokens')):>10s} {str(y.get('llm_tokens')):>10s}")

    n1 = sorted({v.get("npu_calls") for v in r1.values() if v.get("npu_calls") is not None})
    n2 = sorted({v.get("npu_calls") for v in r2.values() if v.get("npu_calls") is not None})
    print(f"\n  NPU 调用数 V1={n1}  V2={n2} "
          f"{'✅ 同预算' if n1 and n1 == n2 else '⚠️ 不一致，不是同预算对照'}")
    print("  ⚠️ V1/V2 用了不同随机种子，且 V2 同时改了去重/交叉/变异，"
          "不能据此判断是哪一项带来收益。")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"v1": {"config": a.v1_config, "run_id": a.v1_run_id, **s1},
             "v2": {"config": a.v2_config, "run_id": a.v2_run_id, **s2},
             "v1_runs": r1, "v2_runs": r2,
             "per_pool_v1": {str(k): len(v) for k, v in g1.items()},
             "per_pool_v2": {str(k): len(v) for k, v in g2.items()}},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ JSON: {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
