#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.5 候选预筛选：增强特征的离线评测（纯 CPU，不占 NPU）。

背景与动机
----------
现有最终模型只用 6 个特征（log_parent / gen / pop_size / tau /
operation / mutation_type）。实测发现：

  * gen / pop_size / tau 是**配置级常量**，同一代内所有候选取值相同，
    对"20 选 13"的排序零贡献（代内有变化的比例 0.0% / 3.4% / 7.6%）；
  * 真正随候选变化的只有 parent_latency / operation / mutation_type，
    于是 20 个候选平均只有 7.6 个不同分值，最大同分组达 12 个——
    同分组内部的选择是随机的。

因此本轮补上**候选级、父子相对**的特征：

  1. parent_speedup：父代**未截断**的加速比 t_base/t_parent − 1。
     注意不能直接沿用 Individual.fitness：executor.py:284 把它截断为
     min(speedup, 2.0)，实测 14.1% 的父代撞到该上限而完全无法区分。
  2. 配置数值的父子比：BLOCK_SIZE、num_warps、num_stages 的**实际数值**
     之比。旧版 STRUCT_PATTERNS 只统计 "BLOCK" 出现的**次数**，
     抓不到 param_tuning 真正改动的东西（如 64 → 128）。
  3. 访存与计算操作数的父子比：tl.load / tl.store / tl.dot / tl.sum。
  4. 父子代码 jaccard 相似度、改动行数。

评价协议
--------
与既有模型一致：按算子分组的 5 折交叉验证（GroupKFold on kernel）。
**对外指标仅三项**：AUC、PR-AUC（以 PR-AUC ÷ 正样本率 的 lift 呈现）、
池内可区分度。正样本率（≈0.358）作为统一基准；PR-AUC 取比值形式可消除
该基准对随机分类器的影响，从而具备跨模型可比性。

用法
----
    python experiments/candidate_filter_features_v6.py
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import common                                  # noqa: E402
from experiments.candidate_filter_model_v5 import load_codes, SAMPLES  # noqa: E402

from sklearn.compose import ColumnTransformer                   # noqa: E402
from sklearn.impute import SimpleImputer                        # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold                  # noqa: E402
from sklearn.pipeline import Pipeline                           # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

OUT = ROOT / "experiments" / "results" / "candidate_filter_features_v6.json"

# ---------------- 代码特征抽取 ----------------
RE_BLOCK = re.compile(r"BLOCK[A-Z_]*\s*[:=]\s*(\d+)")
RE_WARPS = re.compile(r"num_warps\s*=\s*(\d+)")
RE_STAGES = re.compile(r"num_stages\s*=\s*(\d+)")
RE_LOAD = re.compile(r"tl\.load")
RE_STORE = re.compile(r"tl\.store")
RE_DOT = re.compile(r"tl\.dot")
RE_SUM = re.compile(r"tl\.sum")
RE_TOK = re.compile(r"[A-Za-z_]\w*")


def _mx(rx, code, default=0.0):
    v = [int(x) for x in rx.findall(code)]
    return float(max(v)) if v else default


def struct_feats(code):
    """从一段 Triton 代码抽取静态特征。空代码返回全 0。"""
    if not code:
        return dict(block=0.0, warps=0.0, stages=0.0, n_load=0.0,
                    n_store=0.0, n_dot=0.0, n_sum=0.0, n_lines=0.0,
                    have=0.0)
    return dict(
        block=_mx(RE_BLOCK, code), warps=_mx(RE_WARPS, code),
        stages=_mx(RE_STAGES, code),
        n_load=float(len(RE_LOAD.findall(code))),
        n_store=float(len(RE_STORE.findall(code))),
        n_dot=float(len(RE_DOT.findall(code))),
        n_sum=float(len(RE_SUM.findall(code))),
        n_lines=float(code.count("\n") + 1),
        have=1.0,
    )


def jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(RE_TOK.findall(a)), set(RE_TOK.findall(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ratio(c, p):
    """父子比值取对数；缺失时用 0（表示无变化）。"""
    if c <= 0 and p <= 0:
        return 0.0
    return math.log((c + 1.0) / (p + 1.0))


# ---------------- 数据装载 ----------------
def load_df():
    rows = [json.loads(l) for l in open(SAMPLES, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    raw = len(df)
    # 与候选预筛选最终模型完全一致的清洗口径
    df = df[df["run_id"].astype(str) == "0"].copy()
    df["mutation_type"] = df["mutation_type"].fillna("none").astype(str)
    for c in ("tau", "gen", "pop_size", "parent_latency", "child_latency"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tau"] = df["tau"].fillna(-1.0)
    ok = (df["parent_latency"] > 0) & (df["child_latency"] > 0)
    df = df[ok].copy()
    df = df[df["child_latency"] / df["parent_latency"] <= 10].copy()
    df["y"] = (df["child_latency"] < df["parent_latency"]).astype(int)
    df["log_parent"] = np.log1p(df["parent_latency"])
    print(f"[数据] 原始 {raw} → 清洗后 {len(df)}；正样本率 {df['y'].mean():.1%}；"
          f"算子 {df['kernel'].nunique()}")
    return df.reset_index(drop=True)


def add_parent_speedup(df):
    """父代未截断加速比。fitness 被 min(speedup,2.0) 截断，不能直接用作特征。"""
    tb = {}
    for p in common.MANIFEST_DIR.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("t_base_us"):
            tb[d["kernel"]] = d["t_base_us"]
    sp = []
    miss = 0
    for k, pl in zip(df["kernel"], df["parent_latency"]):
        if k in tb and pl and pl > 0:
            sp.append(tb[k] / pl - 1.0)
        else:
            sp.append(np.nan)
            miss += 1
    df["parent_speedup"] = sp
    df["log_parent_speedup"] = np.log1p(df["parent_speedup"].clip(lower=0))
    print(f"[特征] parent_speedup 可算 {len(df) - miss}/{len(df)}；"
          f"其中 ≥2.0（即 fitness 被截断）占 "
          f"{np.mean([x >= 2 for x in sp if not np.isnan(x)]) * 100:.1f}%")
    return df


def add_code_features(df):
    code_map = load_codes()
    print(f"[特征] 候选代码库 {len(code_map)} 份")

    # 用"该候选被评测时的延迟"反查父代身份：
    # crossover 的 parent_latency 记的是主干父代，取延迟最接近的那个
    lat_of = {}
    for cid, cl in zip(df["candidate_id"], df["child_latency"]):
        if isinstance(cl, (int, float)) and not np.isnan(cl):
            lat_of[str(cid)] = float(cl)

    cache = {}

    def feats(cid):
        cid = str(cid)
        if cid not in cache:
            cache[cid] = struct_feats(code_map.get(cid, ""))
        return cache[cid]

    def pick_parent(cid, pids, pl):
        if isinstance(pids, str):
            try:
                pids = json.loads(pids.replace("'", '"'))
            except Exception:
                pids = [pids]
        cands = [str(p) for p in (pids or []) if str(p) in code_map]
        if not cands:
            # gen0 种子级候选无进化父代：以候选自身代码作父代 fallback，
            # 比值特征退化为 1.0（log(1)=0），相似度 1.0——表示“相对起点无改动”。
            return code_map.get(str(cid), "")
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

    recs = []
    for cid, pids, pl in zip(df["candidate_id"], df["parent_ids"],
                             df["parent_latency"]):
        ch = feats(cid)
        pc = struct_feats(pick_parent(cid, pids, pl))
        recs.append(dict(
            r_block=ratio(ch["block"], pc["block"]),
            r_warps=ratio(ch["warps"], pc["warps"]),
            r_stages=ratio(ch["stages"], pc["stages"]),
            r_load=ratio(ch["n_load"], pc["n_load"]),
            r_store=ratio(ch["n_store"], pc["n_store"]),
            r_dot=ratio(ch["n_dot"], pc["n_dot"]),
            r_sum=ratio(ch["n_sum"], pc["n_sum"]),
            r_lines=ratio(ch["n_lines"], pc["n_lines"]),
            sim=jaccard(code_map.get(str(cid), ""), pick_parent(cid, pids, pl)),
            have_code=ch["have"] * pc["have"],
        ))
    for k in recs[0]:
        df[k] = [r[k] for r in recs]
    print(f"[特征] 父子代码均可得的比例: {df['have_code'].mean():.1%}")
    return df


# ---------------- 特征集与评估 ----------------
NUM_BASE = ["log_parent", "gen", "pop_size", "tau"]
CAT = ["operation", "mutation_type"]
SPEEDUP = ["log_parent_speedup"]
CFG = ["r_block", "r_warps", "r_stages"]
CODE = ["r_load", "r_store", "r_dot", "r_sum", "r_lines", "sim"]

VARIANTS = [
    ("A 基线（现最终模型）", NUM_BASE),
    ("B 基线+父代未截断speedup", NUM_BASE + SPEEDUP),
    ("C B+配置数值父子比", NUM_BASE + SPEEDUP + CFG),
    ("D C+访存/计算数父子比+相似度", NUM_BASE + SPEEDUP + CFG + CODE),
    ("E 仅新特征（去掉配置常量）", SPEEDUP + CFG + CODE),
]


def make_pipe(num_cols):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                              ("s", StandardScaler())]), num_cols),
            ("cat", Pipeline([("i", SimpleImputer(strategy="constant",
                                                  fill_value="none")),
                              ("o", OneHotEncoder(handle_unknown="ignore"))]),
             CAT),
        ])),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])


def tie_rate(s):
    """不同分值数 / 样本数——衡量模型能否区分候选（1.0 = 完全可区分）。"""
    return len(set(np.round(np.asarray(s), 9))) / max(len(s), 1)


def p_at_k_perkernel(y, s, k, groups):
    """按算子分组各取 top-k 再汇总——v7 复用的兼容函数。"""
    hits = sel = 0
    for g in pd.unique(groups):
        m = np.asarray(groups) == g
        yy = np.asarray(y)[m]
        ss = np.asarray(s)[m]
        take = min(k, len(yy))
        if take == 0:
            continue
        idx = np.argsort(-ss)[:take]
        hits += int(yy[idx].sum())
        sel += take
    return hits / sel if sel else np.nan


def evaluate(df, num_cols, folds=5, positive_rate=None):
    X = df[num_cols + CAT]
    y = df["y"].to_numpy()
    groups = df["kernel"].to_numpy()
    gkf = GroupKFold(n_splits=folds)
    aucs, prs, plifts, tiep = [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipe(num_cols)
        pipe.fit(X.iloc[tr], y[tr])
        s = pipe.predict_proba(X.iloc[te])[:, 1]
        yt, gt = y[te], groups[te]
        aucs.append(roc_auc_score(yt, s))
        pra = average_precision_score(yt, s)
        prs.append(pra)
        if positive_rate:
            plifts.append(pra / positive_rate)
        # 同一 (算子, 代) 池内的可区分度——这才是 20 选 13 真正面对的粒度
        pooled = df.iloc[te].assign(score=s)
        vals = []
        for _, sub in pooled.groupby(["kernel", "gen"]):
            if len(sub) >= 5:
                vals.append(tie_rate(sub["score"].to_numpy()))
        tiep.append(float(np.mean(vals)) if vals else np.nan)
    return dict(auc=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                pr_auc=float(np.mean(prs)),
                pr_lift=float(np.mean(plifts)) if positive_rate else None,
                distinct_in_pool=float(np.nanmean(tiep)))


def main():
    df = load_df()
    df = add_parent_speedup(df)
    df = add_code_features(df)
    positive_rate = float(df["y"].mean())

    print("\n" + "=" * 104)
    print("离线评测：按算子 5 折 GroupKFold；对外指标 AUC / PR-AUC(÷正样本率) / 池内可区分度")
    print("=" * 104)
    print(f"  {'变体':32s}{'AUC':>14s}{'PR-AUC':>12s}{'PR-AUC/P(lift)':>16s}"
          f"{'池内可区分度':>14s}")
    print("-" * 104)
    out = {"n_samples": int(len(df)), "n_kernels": int(df["kernel"].nunique()),
           "positive_rate": positive_rate,
           "protocol": "按算子 5 折 GroupKFold；对外指标仅 AUC / PR-AUC / 池内可区分度",
           "results": []}
    for name, cols in VARIANTS:
        r = evaluate(df, cols, positive_rate=positive_rate)
        r["variant"] = name
        r["numeric_features"] = cols
        out["results"].append(r)
        print(f"  {name:32s}{r['auc']:.4f}±{r['auc_std']:.3f}"
              f"{r['pr_auc']:>12.4f}{r['pr_lift']:>16.3f}"
              f"{r['distinct_in_pool']:>13.1%}")
    print("-" * 104)
    print(f"  说明：正样本率基准 = {positive_rate:.4f}（所有模型相同）。")
    print("        PR-AUC/P(lift) = PR-AUC ÷ 正样本率，表示比随机基线强多少倍"
          "（消除正样本率基准差异）。")
    print("        「池内可区分度」= 同一(算子,代数)池内不同分值数/候选数，"
          "越接近 1 越能真正区分候选。")

    best = max(out["results"], key=lambda r: r["auc"])
    print(f"\n  按 AUC 最优: {best['variant']}  AUC={best['auc']:.4f}"
          f"（PR-AUC={best['pr_auc']:.4f}，lift={best['pr_lift']:.3f}）")
    base = out["results"][0]
    gain = (best["auc"] - base["auc"]) / base["auc"] * 100
    print(f"  相对基线 A 提升: {gain:+.1f}%   "
          f"池内可区分度 {base['distinct_in_pool']:.1%} → {best['distinct_in_pool']:.1%}")

    # 最优变体的系数
    pipe = make_pipe(best["numeric_features"])
    pipe.fit(df[best["numeric_features"] + CAT], df["y"].to_numpy())
    names = list(best["numeric_features"]) + list(
        pipe.named_steps["prep"].named_transformers_["cat"]
        .named_steps["o"].get_feature_names_out(CAT))
    coef = pipe.named_steps["clf"].coef_[0]
    out["best_coefficients"] = {n: float(c) for n, c in
                                sorted(zip(names, coef), key=lambda x: -abs(x[1]))}
    print(f"\n  {best['variant']} 的系数（正=更可能优于父代）:")
    for n, c in sorted(zip(names, coef), key=lambda x: -abs(x[1]))[:12]:
        print(f"      {n:36s} {c:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {OUT}")
    return out


if __name__ == "__main__":
    main()
