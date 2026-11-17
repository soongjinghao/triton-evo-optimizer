#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Join clean profiling labels with TTIR features and build raw/normalized datasets."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT=Path(__file__).resolve().parents[2]

def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open("r",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def read_csv(path:Path)->list[dict[str,str]]:
    with path.open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def resolve_repo_path(root:Path,raw:str)->Path:
    p=Path(str(raw))
    if not p.is_absolute(): return (root/p).resolve()
    try: p.relative_to(root); return p
    except ValueError:
        s=str(p); marker="/workspace/Agent/"
        if marker in s: return (root/s.split(marker,1)[1]).resolve()
        return p

def write_csv(path:Path,rows:list[dict[str,Any]],fields:list[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default=str(DEFAULT_ROOT))
    ap.add_argument("--labels",default="costModel_v2/features/clean_labels.jsonl")
    ap.add_argument("--features",default="costModel_v2/features/ttir_features.csv")
    ap.add_argument("--output-dir",default="costModel_v2/features")
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve()
    labels=read_jsonl(resolve_repo_path(root,args.labels))
    features=read_csv(resolve_repo_path(root,args.features))
    outdir=resolve_repo_path(root,args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    label_by_id={r["candidate_id"]:r for r in labels}
    feature_by_id={r["candidate_id"]:r for r in features}
    if len(label_by_id)!=len(labels) or len(feature_by_id)!=len(features):
        raise ValueError("Duplicate candidate_id in labels or feature CSV")
    feature_meta={"candidate_id","operator_name","selected_role","source_sha256"}
    feature_cols=[c for c in (features[0].keys() if features else []) if c not in feature_meta]
    # Baseline = successfully measured dataset_seed candidate for the same operator.
    seed_values={}
    for r in labels:
        if r.get("selected_role")=="dataset_seed":
            seed_values.setdefault(r["operator_name"],[]).append(float(r["label_latency_us"]))
    seed_latency={op:statistics.median(vals) for op,vals in seed_values.items()}
    all_rows=[]; normalized_rows=[]; missing_features=[]
    for cid,label in label_by_id.items():
        feat=feature_by_id.get(cid)
        if feat is None:
            missing_features.append(cid); continue
        if (feat.get("source_sha256") != label.get("source_sha256")
                or feat.get("operator_name") != label.get("operator_name")):
            missing_features.append(cid + " [feature metadata mismatch]"); continue
        latency=float(label["label_latency_us"])
        op=label["operator_name"]
        base=seed_latency.get(op)
        speedup=(base/latency) if base is not None and latency>0 else None
        row={
            "candidate_id":cid,"operator_name":op,"selected_role":label.get("selected_role",""),
            "label_latency_us":latency,"log_latency_us":math.log(latency),
            "seed_latency_us":base if base is not None else "",
            "relative_latency_vs_seed":(latency/base) if base else "",
            "speedup_vs_seed":speedup if speedup is not None else "",
            "log_speedup_vs_seed":math.log(speedup) if speedup is not None and speedup>0 else "",
            "measurement_count":label.get("measurement_count",""),
            "device_id":label.get("device_id",""),
            "compile_status":label.get("compile_status",""),
            "source_sha256":label.get("source_sha256",""),
        }
        for col in feature_cols:
            value=feat.get(col,"")
            row[col]=value
        all_rows.append(row)
        if speedup is not None:
            normalized_rows.append(row)
    meta_cols=["candidate_id","operator_name","selected_role","label_latency_us","log_latency_us",
               "seed_latency_us","relative_latency_vs_seed","speedup_vs_seed","log_speedup_vs_seed",
               "measurement_count","device_id","compile_status","source_sha256"]
    fields=meta_cols+feature_cols
    write_csv(outdir/"training_dataset.csv",all_rows,fields)
    write_csv(outdir/"normalized_training_dataset.csv",normalized_rows,fields)
    summary={
        "label_rows":len(labels),"feature_rows":len(features),
        "joined_rows":len(all_rows),"normalized_rows":len(normalized_rows),
        "operator_count":len({r["operator_name"] for r in all_rows}),
        "normalized_operator_count":len({r["operator_name"] for r in normalized_rows}),
        "role_counts":dict(Counter(r["selected_role"] for r in all_rows)),
        "missing_feature_candidate_ids":missing_features,
        "operators_without_dataset_seed":sorted({r["operator_name"] for r in all_rows if r["operator_name"] not in seed_latency}),
        "baseline_definition":"median measured latency of the operator's dataset_seed candidate",
        "speedup_definition":"seed_latency_us / candidate_label_latency_us; higher is faster",
        "feature_count":len(feature_cols),
        "outputs":{"all":"training_dataset.csv","normalized":"normalized_training_dataset.csv"}
    }
    (outdir/"dataset_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("[dataset] labels             :",len(labels))
    print("[dataset] features           :",len(features))
    print("[dataset] joined rows        :",len(all_rows))
    print("[dataset] normalized rows    :",len(normalized_rows))
    print("[dataset] operators          :",summary["operator_count"])
    print("[dataset] feature dimensions :",len(feature_cols))
    print("[dataset] output             :",outdir)
    return 0 if not missing_features else 2

if __name__=="__main__":
    raise SystemExit(main())
