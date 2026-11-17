#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run first-pass, operator-aware univariate feature analysis."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
from typing import Optional

DEFAULT_ROOT=Path(__file__).resolve().parents[2]
META={"candidate_id","operator_name","selected_role","label_latency_us","log_latency_us",
      "seed_latency_us","relative_latency_vs_seed","speedup_vs_seed","log_speedup_vs_seed",
      "measurement_count","device_id","compile_status","source_sha256"}

def read_csv(path:Path)->list[dict[str,str]]:
    with path.open("r",newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def to_float(v)->Optional[float]:
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def pearson(xs:list[float],ys:list[float])->Optional[float]:
    if len(xs)<3 or len(xs)!=len(ys): return None
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    dx=[x-mx for x in xs]; dy=[y-my for y in ys]
    vx=sum(x*x for x in dx); vy=sum(y*y for y in dy)
    if vx<=1e-20 or vy<=1e-20: return None
    return sum(a*b for a,b in zip(dx,dy))/math.sqrt(vx*vy)

def average_ranks(values:list[float])->list[float]:
    order=sorted(range(len(values)),key=lambda i:values[i])
    ranks=[0.0]*len(values); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        rank=((i+1)+j)/2.0
        for k in range(i,j): ranks[order[k]]=rank
        i=j
    return ranks

def spearman(xs:list[float],ys:list[float])->Optional[float]:
    if len(xs)<3: return None
    return pearson(average_ranks(xs),average_ranks(ys))

def within_operator_vectors(rows, feature):
    groups=defaultdict(list)
    for r in rows:
        x=to_float(r.get(feature)); y=to_float(r.get("log_speedup_vs_seed"))
        if x is not None and y is not None:
            groups[r["operator_name"]].append((x,y))
    xc=[]; yc=[]; xr=[]; yr=[]
    for group in groups.values():
        if len(group)<2: continue
        xs=[z[0] for z in group]; ys=[z[1] for z in group]
        mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
        xc.extend(x-mx for x in xs); yc.extend(y-my for y in ys)
        rx=average_ranks(xs); ry=average_ranks(ys)
        mxr=sum(rx)/len(rx); myr=sum(ry)/len(ry)
        xr.extend(x-mxr for x in rx); yr.extend(y-myr for y in ry)
    return xc,yc,xr,yr

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default=str(DEFAULT_ROOT))
    ap.add_argument("--input",default="costModel_v2/features/normalized_training_dataset.csv")
    ap.add_argument("--output-dir",default="costModel_v2/features")
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve()
    inpath=Path(args.input); inpath=inpath if inpath.is_absolute() else root/inpath
    outdir=Path(args.output_dir); outdir=outdir if outdir.is_absolute() else root/outdir
    rows=read_csv(inpath)
    if not rows: raise ValueError("Normalized training dataset is empty")
    feature_cols=[c for c in rows[0] if c not in META]
    ranking=[]
    for feature in feature_cols:
        xs=[]; ys=[]
        for r in rows:
            x=to_float(r.get(feature)); y=to_float(r.get("log_speedup_vs_seed"))
            if x is not None and y is not None: xs.append(x); ys.append(y)
        variance=(sum((x-sum(xs)/len(xs))**2 for x in xs)/len(xs)) if xs else 0.0
        xc,yc,xr,yr=within_operator_vectors(rows,feature)
        within_variance=(sum(x*x for x in xc)/len(xc)) if xc else 0.0
        vals=sorted(set(xs))
        ranking.append({
            "feature":feature,"valid_count":len(xs),"unique_values":len(vals),
            "global_variance":variance,"within_operator_variance":within_variance,
            "pearson_global":pearson(xs,ys),"spearman_global":spearman(xs,ys),
            "within_operator_pair_count":len(xc),
            "pearson_within_operator":pearson(xc,yc),
            "spearman_within_operator":pearson(xr,yr),
        })
    ranking.sort(key=lambda r:abs(r["spearman_within_operator"] or 0.0),reverse=True)
    outdir.mkdir(parents=True,exist_ok=True)
    rank_path=outdir/"feature_ranking.csv"
    with rank_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(ranking[0].keys())); w.writeheader(); w.writerows(ranking)

    # Detect highly redundant features using operator-centered Pearson correlation.
    active=[r["feature"] for r in ranking if r["within_operator_variance"]>1e-12]
    vectors={}
    for feature in active:
        xc,yc,xr,yr=within_operator_vectors(rows,feature)
        vectors[feature]=xc
    redundant=[]
    for i,a in enumerate(active):
        for b in active[i+1:]:
            corr=pearson(vectors[a],vectors[b])
            if corr is not None and abs(corr)>=0.98:
                redundant.append({"feature_a":a,"feature_b":b,"within_operator_pearson":corr})
    report={
        "rows":len(rows),"operators":len({r["operator_name"] for r in rows}),
        "candidate_roles":{},"feature_count":len(feature_cols),
        "non_constant_feature_count":len(active),
        "target":"log_speedup_vs_seed = log(seed_latency_us / candidate_latency_us)",
        "ranking_method":"Pearson/Spearman correlations on raw values and within-operator centered values; ranked by absolute within-operator Spearman.",
        "top_20_features":ranking[:20],
        "highly_redundant_feature_pairs_abs_corr_ge_0_98":redundant,
        "caveat":"Exploratory filter-based ranking only. The sample is small and candidates are grouped by operator; do not interpret these correlations as causal importance or final generalization performance."
    }
    for r in rows:
        role=r.get("selected_role","")
        report["candidate_roles"][role]=report["candidate_roles"].get(role,0)+1
    (outdir/"feature_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("[analysis] rows              :",len(rows))
    print("[analysis] operators         :",report["operators"])
    print("[analysis] numeric features  :",len(feature_cols))
    print("[analysis] non-constant      :",len(active))
    print("[analysis] top features:")
    for item in ranking[:15]:
        print("  {:42s} within_spearman={}".format(item["feature"],item["spearman_within_operator"]))
    print("[analysis] ranking           :",rank_path)
    print("[analysis] report            :",outdir/"feature_report.json")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
