#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.4.3 变异策略的**机制探测**（mutation probe）—— 同父代成对设计。

为什么需要它
------------
原 3.4.3 让三种策略各跑一条独立搜索轨迹，再比较端到端 GM(S)，该设计无法归因：

  1. 三种策略发生变异时的**父代不同**，实际比较的是"不同父代上的变异"，
     父代质量差异被混进策略差异；
  2. adaptive 的行为本身依赖父代水平（genetic_operators.py:577-585：
     speedup > 2.5 完全不做结构重写，speedup < 1.5 才以结构重写主导），
     不按父代水平分层就无法解释；
  3. 端到端 GM(S) 是"策略初始化 + 选择 + 交叉 + 变异 + 精英保留"的乘积，
     精英保留会兜底，差的变异被抹平，端到端差异被稀释。

本脚本的做法
------------
固定一组**已验证有效**的父代，在其上按 uniform / adaptive / aggressive
各生成 R 个变异体，每个变异体完整走一次评测（编译 → 功能正确性 → NPU 计时）。
由于父代固定且有效：

  * 每个变异体的差异只能来自策略本身（消除 1）
  * 可按父代 speedup 分层分析（消除 2）
  * 变异体自带 parent_latency，样本利用率接近 100%
    （原设计因父代评测失败，仅约 35% 的样本能算出增益）

⚠️ 定位说明（写论文时要讲清楚）
------------------------------
这是**机制探测**，回答"**一次变异**的策略差异"，可精确复现；
**不**回答"多轮累积后哪种策略最终更好"——后者仍由 15 算子的端到端实验给出。
与 3.4.1 的 selection_probe 定位完全一致：机制层与结果层互相补充，
不可互相替代。

已知保真度取舍
--------------
默认不给父代做 Profiling 诊断（--profiling 可开启）。profiling_context 不影响
策略选择，只影响 LLM 输出质量，且对三种策略一致作用，因此不影响策略间的比较。

用法
----
    source set_env/set_api_local.sh
    python experiments/mutation_probe.py --kernels eye_kernel --repeats 2   # 标定
    python experiments/mutation_probe.py --repeats 6                       # 全量
"""
import argparse
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from experiments.harness import make_config, build_executor          # noqa: E402
from experiments import common                                        # noqa: E402
from genetic_operators import GeneticOperators, Individual            # noqa: E402
from llm_interface import LLMInterface                                # noqa: E402

STRATEGIES = ("uniform", "adaptive", "aggressive")
OUT_JSON = ROOT / "experiments" / "results" / "mutation_probe.json"


# ------------------------------------------------------------------ 父代
def _seed_candidates(kernel, manifest):
    """manifest 中所有通过验证的初始种子：{name: (path, latency)}"""
    out = {}
    for name, info in (manifest.get("seeds") or {}).items():
        if info.get("success") and info.get("latency_us"):
            p = Path(info["path"])
            if p.exists():
                out[name] = (p, info["latency_us"])
    return out


def _optimized_candidate(kernel):
    """已有搜索结果中的最优候选代码（作高 speedup 父代）。"""
    best = None
    for cfg in ("mut_adaptive", "mut_uniform", "mut_aggressive", "full"):
        py = common.RESULTS_DIR / cfg / f"{kernel}__r0.py"
        rm = common.RESULTS_DIR / cfg / f"{kernel}__r0.remeasure.json"
        if not (py.exists() and rm.exists()):
            continue
        try:
            d = json.loads(rm.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("success") and d.get("t_best_us"):
            if best is None or d["t_best_us"] < best[1]:
                best = (py, d["t_best_us"])
    return best


def build_parents(kernels, per_kernel=2):
    """构造父代候选列表：每个算子取最快种子 + 已有最优候选。"""
    parents = []
    for k in kernels:
        mf = common.MANIFEST_DIR / f"{k}.json"
        if not mf.exists():
            print(f"  [跳过] {k}: 无 manifest")
            continue
        man = json.loads(mf.read_text(encoding="utf-8"))
        cands = []
        seeds = _seed_candidates(k, man)
        if seeds:
            name = min(seeds, key=lambda n: seeds[n][1])
            cands.append(("seed", seeds[name][0]))
        opt = _optimized_candidate(k)
        if opt:
            cands.append(("optimized", opt[0]))
        for source, path in cands[:per_kernel]:
            parents.append({
                "id": f"{k}|{source}", "kernel": k, "source": source,
                "code": common.read_code(path), "path": str(path),
                "t_seed_us": man.get("t_seed_us"), "t_base_us": man.get("t_base_us"),
            })
    return parents


# ------------------------------------------------------------------ 工具
def changed_lines(a, b):
    """用 SequenceMatcher 统计改动的行数（增+删）。"""
    n = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(
            None, a.splitlines(), b.splitlines()).get_opcodes():
        if tag != "equal":
            n += max(i2 - i1, j2 - j1)
    return n


def failure_mode(error, syntax_error):
    if syntax_error:
        return "语法错误"
    if not error:
        return "通过"
    e = error.lower()
    if "syntax" in e:
        return "语法错误"
    if "compil" in e or "import" in e:
        return "编译失败"
    if "timeout" in e or "time" in e:
        return "超时"
    if "correct" in e or "functional" in e or "mismatch" in e or "test" in e:
        return "功能不正确"
    return "其他"


def gm(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def boot_ci_mean(vals, B=4000, seed=0):
    """均值的 bootstrap 95% CI（valss 为空时返回 nan）。"""
    if len(vals) < 3:
        return (float("nan"),) * 2
    rng = random.Random(seed)
    s = sorted(statistics.fmean([vals[rng.randrange(len(vals))]
                                 for _ in vals]) for _ in range(B))
    return s[int(.025 * B)], s[int(.975 * B)]


# ------------------------------------------------------------------ 主流程
def probe(parents, repeats, seed, with_profiling, max_seconds, out_json,
          mreps=3):
    random.seed(seed)
    samples, nominal = [], {}
    t_start = time.time()
    results = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "repeats": repeats, "random_seed": seed,
               "profiling": with_profiling, "parents": [], "samples": [],
               "nominal_weights": {}}

    for pi, spec in enumerate(parents, 1):
        if max_seconds and time.time() - t_start > max_seconds:
            print(f"\n[达到 time budget {max_seconds}s] 在第 {pi} 个父代处停止")
            break
        k = spec["kernel"]
        cfg = make_config("full", k, run_id=0, seed=seed, log=False,
                          mutation_mode="uniform")
        cfg.save_code = False
        executor = build_executor(k, cfg)
        llm = LLMInterface(cfg)
        gops = GeneticOperators(llm, cfg)

        # --- 父代本体测量 ---
        # 1) evaluate 取得与真实搜索一致的 speedup 口径（adaptive 分级依赖它，见 577-585）
        # 2) measure_repeat 取多次中位数作为稳定延迟，抑制短 Kernel 的抖动
        sp_res = executor.evaluate(spec["code"], device_id=0)
        pres = executor.measure_repeat(spec["code"], repeats=max(5, mreps),
                                       device_id=0)
        if not (pres.success and pres.execution_time):
            print(f"  [父代无效] {spec['id']}: {pres.error}")
            continue
        parent = Individual(
            code=spec["code"], generation=0,
            metadata={"execution_time": pres.execution_time,
                      "speedup": sp_res.speedup, "evaluated": True,
                      "success": True, "device_id": 0},
            id=f"p{pi:03d}")
        if with_profiling:
            from evolutionary_algorithm import EvolutionaryAlgorithm
            ea = EvolutionaryAlgorithm.__new__(EvolutionaryAlgorithm)
            ea.config, ea.executor, ea.gops = cfg, executor, gops
            try:
                parent.metadata["profiling_context"] = ea._extract_profiling_context(
                    device_id=0, ind=parent)
            except Exception as e:
                parent.metadata["profiling_context"] = ""
                print(f"  [profiling 失败] {e}")

        # adaptive 分级用的就是该 speedup（见 genetic_operators.py:577-585）
        sp = sp_res.speedup
        stratum = "L3(>2.5)" if sp > 2.5 else ("L2(1.5~2.5)" if sp > 1.5 else "L1(<1.5)")
        pinfo = {"id": parent.id, "kernel": k, "source": spec["source"],
                 "latency_us": pres.execution_time,
                 "speedup_adaptive": sp, "stratum": stratum,
                 "t_seed_us": spec["t_seed_us"], "t_base_us": spec["t_base_us"]}
        results["parents"].append(pinfo)
        print(f"\n=== [{pi}/{len(parents)}] {parent.id} {k} ({spec['source']}) "
              f"T={pres.execution_time:.3f}us speedup={sp:.3f} → {stratum}")

        for strat in STRATEGIES:
            cfg.mutation_mode = strat
            for r in range(repeats):
                t0 = time.time()
                try:
                    mut = gops.mutate(parent)
                except Exception as e:
                    samples.append({"_failed": True, "parent_id": parent.id,
                                    "strategy": strat, "error": repr(e)[:200]})
                    print(f"    {strat} #{r+1} mutate 异常: {e}")
                    continue
                mt = mut.metadata.get("mutation_type", "?")
                nominal.setdefault(strat, mut.metadata.get("mutation_weights"))
                # 多次测量取中位数，抑制单次 15% 量级的计时抖动
                res = executor.measure_repeat(mut.code, repeats=mreps, device_id=0)
                g = (pres.execution_time / res.execution_time
                     if (res.success and res.execution_time) else None)
                fm = failure_mode(res.error, mut.metadata.get("syntax_error"))
                sim = SequenceMatcher(None, spec["code"], mut.code).ratio()
                lines = changed_lines(spec["code"], mut.code)
                samples.append({
                    "parent_id": parent.id, "kernel": k, "parent_source": spec["source"],
                    "parent_stratum": stratum, "strategy": strat, "repeat": r,
                    "mutation_type": mt,
                    # LLM 返回了与父代完全相同的代码 → 实质"无操作变异"，
                    # 这类样本自带"代码没变"，其 G 值天然构成测量噪声的对照组
                    "is_noop": bool(sim >= 0.9999 and lines == 0),
                    "similarity": sim,
                    "changed_lines": lines,
                    "success": res.success, "latency_us": res.execution_time,
                    "g_mut": g, "failure_mode": fm,
                    "error": (res.error or "")[:200],
                    "seconds": round(time.time() - t0, 1),
                })
                mark = f"G={g:.4f}" if g else f"失败({fm})"
                print(f"    {strat:11s} #{r+1} {mt:20s} {mark} "
                      f"[{time.time()-t0:.0f}s]")
                # 逐条落盘，长任务中断不丢数据
                results["samples"] = samples
                out_json.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    results["nominal_weights"] = nominal
    return results


# ------------------------------------------------------------------ 分析
def analyze(res):
    samples = [s for s in res["samples"] if not s.get("_failed")]
    if not samples:
        print("无有效样本")
        return
    print("\n" + "=" * 92)
    print("3.4.3 变异策略机制探测 —— 结果")
    print("=" * 92)
    print(f"样本总数 {len(samples)}  父代数 {len(res['parents'])}  "
          f"每父代每策略 {res['repeats']} 次")

    # 表A：变异类型分布（实测 vs 名义）
    print("\n" + "-" * 92)
    print("[A] 变异类型分布：实测占比 vs 名义权重（检验策略的名义定义是否成立）")
    print("-" * 92)
    types = ["param_tuning", "arithmetic_and_mask", "structure_rewrite"]
    print(f"  {'策略':12s}" + "".join(f"{t[:14]:>16s}" for t in types))
    for s in STRATEGIES:
        sub = [x for x in samples if x["strategy"] == s]
        if not sub:
            continue
        c = Counter(x["mutation_type"] for x in sub)
        row = "".join(f"{c.get(t,0)/len(sub)*100:15.1f}%" for t in types)
        print(f"  {s:12s}{row}")
    print("\n  名义权重（adaptive 随父代水平变化，故不唯一）：")
    for s in STRATEGIES:
        w = res["nominal_weights"].get(s) or {}
        if w:
            print(f"    {s:12s}" + "  ".join(f"{k}={v:.2f}" for k, v in w.items()))

    # 表B：变异幅度
    print("\n" + "-" * 92)
    print("[B] 变异幅度：把「激进/均匀/自适应」变成可测量量")
    print("-" * 92)
    print(f"  {'策略':12s}{'源码相似度':>14s}{'改动行数':>12s}{'有效率':>12s}")
    for s in STRATEGIES:
        sub = [x for x in samples if x["strategy"] == s]
        if not sub:
            continue
        print(f"  {s:12s}"
              f"{statistics.fmean([x['similarity'] for x in sub]):>14.4f}"
              f"{statistics.fmean([x['changed_lines'] for x in sub]):>12.1f}"
              f"{sum(1 for x in sub if x['success'])/len(sub)*100:>11.1f}%")

    # 表C：失败模式
    print("\n" + "-" * 92)
    print("[C] 失败模式构成（有效率之外，看失败集中在哪一步）")
    print("-" * 92)
    modes = sorted({x["failure_mode"] for x in samples})
    print(f"  {'策略':12s}" + "".join(f"{m:>12s}" for m in modes) + f"{'样本':>8s}")
    for s in STRATEGIES:
        sub = [x for x in samples if x["strategy"] == s]
        if not sub:
            continue
        c = Counter(x["failure_mode"] for x in sub)
        print(f"  {s:12s}" + "".join(
            f"{c.get(m,0)/len(sub)*100:>11.1f}%" for m in modes) + f"{len(sub):>8d}")

    # 表C2：无操作变异率 + 测量噪声底
    print("\n" + "-" * 92)
    print("[C2] 无操作变异率与测量噪声底")
    print("-" * 92)
    print("  LLM 返回与父代完全相同的代码时，该样本的 G 值只反映测量噪声，")
    print("  因此「无操作变异」天然构成噪声对照组，不需要额外开销。")
    print(f"\n  {'策略':12s}{'无操作率':>12s}{'真实变异样本':>14s}")
    for s in STRATEGIES:
        sub = [x for x in samples if x["strategy"] == s]
        if not sub:
            continue
        no = sum(1 for x in sub if x.get("is_noop"))
        print(f"  {s:12s}{no/len(sub)*100:>11.1f}%{len(sub)-no:>14d}")
    noise = [x["g_mut"] for x in samples if x.get("is_noop") and x["g_mut"]]
    if len(noise) >= 2:
        print(f"\n  噪声对照组 n={len(noise)}: "
              f"G = {statistics.fmean(noise):.4f} ± {statistics.stdev(noise):.4f} (SD)  "
              f"范围 [{min(noise):.4f}, {max(noise):.4f}]")
        print(f"    相对噪声 ≈ {statistics.stdev(noise)/statistics.fmean(noise)*100:.1f}%  "
              f"→ 策略间的观测差异需大于该幅度才可采信")
    else:
        print("\n  噪声对照组样本不足，无法估计噪声底")

    # 表D：主指标 G_mut（整体 + 分层）
    for label, pred in (("【整体】", lambda x: True),
                        *[(f"【父代 {s}】", lambda x, s=s: x["parent_stratum"] == s)
                          for s in ("L1(<1.5)", "L2(1.5~2.5)", "L3(>2.5)")]):
        print("\n" + "-" * 92)
        print(f"[D] 单步增益 G_mut = T_parent / T_child  {label}")
        print("-" * 92)
        print(f"  {'策略':12s}{'n':>5s}{'GM(g)':>10s}{'95%CI':>20s}"
              f"{'中位数':>10s}{'改进率P(g>1)':>14s}{'显著≠1':>9s}")
        for s in STRATEGIES:
            vals = [x["g_mut"] for x in samples
                    if x["strategy"] == s and x["g_mut"] and pred(x)]
            if not vals:
                print(f"  {s:12s}{0:>5d}")
                continue
            lo, hi = boot_ci_mean(vals)
            flag = "是" if (lo > 1 or hi < 1) else "否"
            print(f"  {s:12s}{len(vals):>5d}{gm(vals):>10.4f}"
                  f"{('[%.3f, %.3f]' % (lo, hi)):>20s}"
                  f"{statistics.median(vals):>10.4f}"
                  f"{sum(1 for v in vals if v > 1)/len(vals)*100:>13.1f}%{flag:>9s}")

    # 表E：同父代配对比较（消除父代异质性）
    print("\n" + "-" * 92)
    print("[E] 同父代配对比较：Δ = 该父代上两策略的平均 G_mut 之差")
    print("-" * 92)
    bypk = defaultdict(lambda: defaultdict(list))
    for x in samples:
        if x["g_mut"]:
            bypk[x["parent_id"]][x["strategy"]].append(x["g_mut"])
    print(f"  {'对比':32s}{'配对数':>8s}{'Δ(均值)':>12s}{'bootstrap95%CI':>22s}{'胜出/配对':>12s}")
    for a, b in (("uniform", "adaptive"), ("uniform", "aggressive"),
                 ("adaptive", "aggressive")):
        ds, wins, tot = [], 0, 0
        for pid, m in bypk.items():
            if m.get(a) and m.get(b):
                da, db = statistics.fmean(m[a]), statistics.fmean(m[b])
                ds.append(da - db)
                tot += 1
                wins += 1 if da > db else 0
        if not ds:
            print(f"  {a + ' − ' + b:32s}{0:>8d}")
            continue
        lo, hi = boot_ci_mean(ds)
        print(f"  {a + ' − ' + b:32s}{tot:>8d}{statistics.fmean(ds):>+12.4f}"
              f"{('[%.3f, %.3f]' % (lo, hi)):>22s}{wins:>9d}/{tot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", "-k", nargs="+", default=None)
    ap.add_argument("--repeats", "-R", type=int, default=6)
    ap.add_argument("--per-kernel", type=int, default=2,
                    help="每个算子取几个父代（种子/最优候选）")
    ap.add_argument("--max-parents", type=int, default=0, help="0=不限")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--measure-repeats", type=int, default=3,
                    help="每个变异体 / 父代的重复测量次数（取中位数）")
    ap.add_argument("--profiling", action="store_true")
    ap.add_argument("--max-seconds", type=int, default=0, help="0=不限")
    a = ap.parse_args()

    from experiments.analyze_logs import K15
    kernels = a.kernels or sorted(K15)
    parents = build_parents(kernels, per_kernel=a.per_kernel)
    if a.max_parents:
        parents = parents[:a.max_parents]
    print(f"父代数 {len(parents)}  策略 3  重复 {a.repeats}  "
          f"→ 预计评测 {len(parents)*(3*a.repeats+1)} 次")

    res = probe(parents, a.repeats, a.seed, a.profiling, a.max_seconds, OUT_JSON,
                a.measure_repeats)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {OUT_JSON}")
    analyze(res)


if __name__ == "__main__":
    main()
