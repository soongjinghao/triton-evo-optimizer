#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.5 候选预筛选：第二轮特征工程（纯 CPU，不占 NPU）。

第一轮（v6）的结论
------------------
最终特征 = log_parent + r_block/r_warps/r_stages + sim，
P@13 0.4603 → 0.4706，池内可区分度 43.8% → 52.2%。
池内可区分度仍偏低：约一半候选打分相同，同分内仍是任意选。

第二轮三个新方向
----------------
1. **训练范围收窄到真正使用的场景**
   第 0 代候选不使用预筛选（P86：gen0 的 AUC 仅 0.53/0.47，接近随机），
   但模型是在**含 gen0 的全量数据**上训练的，等于让一部分容量去拟合
   一个根本不会被使用的场景。本轮同时评测"全量"与"仅变异+交叉"两种范围。

2. **父代相对种子的改进程度 rel_seed = log(seed_time / parent_latency)**
   这是真正的"剩余改进空间"刻画，且与 t_base 口径的 speedup 不同：
   seed_time 是进化起点（最快种子），t_base 是数据集参考实现。
   第一轮试的 parent_speedup（t_base 口径）无效，本轮换 seed 口径再试。

3. **访存/计算规模的父子比（比值而非绝对计数）**
   第一轮测的是"各自 tl.load 出现次数"的比值，无增益；
   本轮改为**总访存规模**与**总算子调用数**的比值，并加入 mask 变化。

评价：按算子 5 折 GroupKFold；主指标 P@13，附 AUC / PR-AUC / R@13
与"池内可区分度"。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import common                                   # noqa: E402
from experiments.candidate_filter_model_v5 import load_codes, SAMPLES  # noqa: E402
from experiments.candidate_filter_features_v6 import (            # noqa: E402
    load_df, add_parent_speedup, add_code_features, CAT,
    p_at_k_perkernel, tie_rate, ratio, jaccard, struct_feats)

from sklearn.compose import ColumnTransformer                    # noqa: E402
from sklearn.impute import SimpleImputer                         # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold                   # noqa: E402
from sklearn.pipeline import Pipeline                            # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler   # noqa: E402

OUT = ROOT / "experiments" / "results" / "candidate_filter_features_v7.json"

RE_TL_CALL = re.compile(r"tl\.[a-z_]+")
RE_MASK = re.compile(r"mask")


def uniq(a):
    out = []
    for x in a:
        if x not in out:
            out.append(x)
    return out


def r_at_k_perkernel(y, s, k, groups):
    tot = []
    for g in uniq(groups):
        m = np.asarray(groups) == g
        yy, ss = np.asarray(y)[m], np.asarray(s)[m]
        pos = int(yy.sum())
        if pos == 0:
            continue
        take = min(k, len(yy))
        tot.append(int(yy[np.argsort(-ss)[:take]].sum()) / pos)
    return float(np.mean(tot)) if tot else np.nan


def add_v7_features(df):
    """第二轮新增特征。"""
    # ---- rel_seed：父代相对最快种子的改进程度（对数）----
    df["seed_time"] = pd.to_numeric(df.get("seed_time"), errors="coerce")
    rel = []
    miss = 0
    for st, pl in zip(df["seed_time"], df["parent_latency"]):
        if st and pl and st > 0 and pl > 0:
            rel.append(np.log(st / pl))
        else:
            rel.append(np.nan)
            miss += 1
    df["rel_seed"] = rel
    print(f"[特征] rel_seed 可算 {len(df) - miss}/{len(df)}")

    # ---- 访存/计算规模的父子比 ----
    code_map = load_codes()
    lat_of = {}
    for cid, cl in zip(df["candidate_id"], df["child_latency"]):
        if isinstance(cl, (int, float)) and not np.isnan(cl):
            lat_of[str(cid)] = float(cl)

    def pick_parent(pids, pl):
        if isinstance(pids, str):
            try:
                pids = json.loads(pids.replace("'", '"'))
            except Exception:
                pids = [pids]
        cands = [str(p) for p in (pids or []) if str(p) in code_map]
        if not cands:
            return ""
        if pl and len(cands) > 1:
            best, bd = None, None
            for c in cands:
                if c in lat_of:
                    d = abs(lat_of[c] - pl)
                    if bd is None or d < bd:
                        best, bd = c, d
            if best is not None and bd < 1e-6:
                return code_map[best]
        return code_map[cands[0]]

    cache = {}

    def f(cid):
        cid = str(cid)
        if cid not in cache:
            cache[cid] = struct_feats(code_map.get(cid, ""))
        return cache[cid]

    recs = []
    for cid, pids, pl in zip(df["candidate_id"], df["parent_ids"],
                             df["parent_latency"]):
        ch = f(cid)
        pc = struct_feats(pick_parent(pids, pl))
        ccode = code_map.get(str(cid), "")
        pcode = pick_parent(pids, pl)
        ops_c = float(len(RE_TL_CALL.findall(ccode)))
        ops_p = float(len(RE_TL_CALL.findall(pcode)))
        mask_c = float(len(RE_MASK.findall(ccode)))
        mask_p = float(len(RE_MASK.findall(pcode)))
        recs.append(dict(
            r_ops=ratio(ops_c, ops_p),
            r_mask=ratio(mask_c, mask_p),
            ops_abs=np.log1p(ops_c),
            block_abs=np.log1p(ch["block"]),
        ))
    for k in recs[0]:
        df[k] = [r[k] for r in recs]
    return df


# ---------------- 特征集 ----------------
BASE6 = ["log_parent", "r_block", "r_warps", "r_stages", "sim"]        # 第一轮最终
V7_SETS = [
    ("P0 第一轮最终(基准)", BASE6),
    ("P1 +rel_seed", BASE6 + ["rel_seed"]),
    ("P2 +r_ops/r_mask", BASE6 + ["r_ops", "r_mask"]),
    ("P3 +ops_abs/block_abs", BASE6 + ["ops_abs", "block_abs"]),
    ("P4 P1+P2", BASE6 + ["rel_seed", "r_ops", "r_mask"]),
    ("P5 全部新特征", BASE6 + ["rel_seed", "r_ops", "r_mask",
                          "ops_abs", "block_abs"]),
]


def ev(data, cols):
    X = data[cols + CAT]
    y = data["y"].to_numpy()
    g = data["kernel"].to_numpy()
    acc = {k: [] for k in ("auc", "pr", "p13", "r13", "p13r", "tie")}
    rng = np.random.default_rng(0)
    for tr, te in GroupKFold(5).split(X, y, g):
        p = Pipeline([("prep", ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                              ("s", StandardScaler())]), cols),
            ("cat", Pipeline([("i", SimpleImputer(strategy="constant",
                                                  fill_value="none")),
                              ("o", OneHotEncoder(handle_unknown="ignore"))]),
             CAT)])),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
        p.fit(X.iloc[tr], y[tr])
        s = p.predict_proba(X.iloc[te])[:, 1]
        yt, gt = y[te], g[te]
        acc["auc"].append(roc_auc_score(yt, s))
        acc["pr"].append(average_precision_score(yt, s))
        acc["p13"].append(p_at_k_perkernel(yt, s, 13, gt))
        acc["r13"].append(r_at_k_perkernel(yt, s, 13, gt))
        acc["p13r"].append(p_at_k_perkernel(yt, rng.random(len(yt)), 13, gt))
        pl = data.iloc[te].assign(score=s)
        v = [tie_rate(x["score"].to_numpy()) for _, x in
             pl.groupby(["kernel", "gen"]) if len(x) >= 5]
        acc["tie"].append(np.mean(v) if v else np.nan)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def report(tag, data):
    print("\n" + "=" * 106)
    print(f"{tag}   n={len(data)}  算子={data['kernel'].nunique()}   "
          f"正样本率={data['y'].mean():.1%}")
    print("=" * 106)
    print(f"  {'变体':24s}{'AUC':>17s}{'PR-AUC':>17s}{'P@13':>17s}"
          f"{'R@13':>15s}{'可区分':>9s}")
    print("-" * 106)
    out = []
    for name, cols in V7_SETS:
        r = ev(data, cols)
        r["variant"] = name
        r["numeric_features"] = cols
        out.append(r)
        print(f"  {name:24s}{'%.4f±%.3f' % r['auc']:>17s}"
              f"{'%.4f±%.3f' % r['pr']:>17s}{'%.4f±%.3f' % r['p13']:>17s}"
              f"{'%.4f' % r['r13'][0]:>15s}{r['tie'][0]:>8.1%}")
    best = max(out, key=lambda r: r["p13"][0])
    base = out[0]
    print("-" * 106)
    print(f"  最优(P@13): {best['variant']}  "
          f"{best['p13'][0]:.4f} vs 基准 {base['p13'][0]:.4f}  "
          f"({(best['p13'][0] / base['p13'][0] - 1) * 100:+.1f}%)")
    print(f"  随机基线 P@13 = {base['p13r'][0]:.4f}")
    return out


def main():
    df = add_v7_features(
        add_code_features(add_parent_speedup(load_df())))
    res = {}
    res["all"] = report("① 全量样本（与现模型同范围）", df)
    mc = df[df["operation"].isin(["mutation", "crossover"])].reset_index(drop=True)
    res["mut_cross_only"] = report("② 仅变异+交叉（实际启用预筛选的范围）", mc)

    # 最优组合的系数
    best = max(res["mut_cross_only"], key=lambda r: r["p13"][0])
    cols = best["numeric_features"]
    pipe = Pipeline([("prep", ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), cols),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant",
                                              fill_value="none")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT)])),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    pipe.fit(mc[cols + CAT], mc["y"].to_numpy())
    names = list(cols) + list(pipe.named_steps["prep"]
                              .named_transformers_["cat"]
                              .named_steps["o"].get_feature_names_out(CAT))
    coef = pipe.named_steps["clf"].coef_[0]
    print(f"\n  最优（{best['variant']}）在②范围下的系数:")
    co = {}
    for n, c in sorted(zip(names, coef), key=lambda x: -abs(x[1])):
        co[n] = float(c)
        print(f"      {n:32s} {c:+.4f}")

    out = dict(n_all=int(len(df)), n_mutcross=int(len(mc)),
               positive_rate_all=float(df["y"].mean()),
               positive_rate_mutcross=float(mc["y"].mean()),
               results_all=res["all"], results_mutcross=res["mut_cross_only"],
               best=dict(variant=best["variant"], numeric_features=cols,
                         categorical=CAT, coefficients=co))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[保存] {OUT}")


if __name__ == "__main__":
    main()
