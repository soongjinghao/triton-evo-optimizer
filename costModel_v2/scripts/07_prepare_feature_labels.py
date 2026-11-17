#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a clean, hash-verified set of successful profiling labels for CostModel V2."""
from __future__ import annotations
import argparse, json, math, statistics
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROLE_ORDER = {"dataset_seed": 0, "historical_best": 1, "historical_p10": 2,
              "historical_p50": 3, "historical_p90": 4}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                obj=json.loads(line)
                if not isinstance(obj,dict):
                    raise ValueError(f"{path}:{n}: expected JSON object")
                rows.append(obj)
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    tmp.replace(path)

def resolve_repo_path(root: Path, raw: str) -> Path:
    p=Path(str(raw))
    if not p.is_absolute():
        return (root/p).resolve()
    try:
        p.relative_to(root)
        return p
    except ValueError:
        s=str(p)
        marker="/workspace/Agent/"
        if marker in s:
            return (root/s.split(marker,1)[1]).resolve()
        return p

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default=str(DEFAULT_ROOT))
    ap.add_argument("--selected",default="costModel_v2/data/selected_candidates.jsonl")
    ap.add_argument("--ir-manifest",default="costModel_v2/data/candidate_ir_manifest.jsonl")
    ap.add_argument("--records",action="append",default=[],
                    help="Profiling JSONL; repeat to merge several runs. If omitted, main labels and labels_retry* are discovered.")
    ap.add_argument("--output-dir",default="costModel_v2/features")
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve()
    selected_path=resolve_repo_path(root,args.selected)
    manifest_path=resolve_repo_path(root,args.ir_manifest)
    outdir=resolve_repo_path(root,args.output_dir)
    selected=read_jsonl(selected_path)
    manifest=read_jsonl(manifest_path)
    sel={r["candidate_id"]:r for r in selected}
    man={r["candidate_id"]:r for r in manifest}
    if len(sel)!=len(selected) or len(man)!=len(manifest):
        raise ValueError("Duplicate candidate_id in selected candidates or IR manifest")

    if args.records:
        record_paths=[resolve_repo_path(root,x) for x in args.records]
    else:
        record_paths=[]
        main_record=resolve_repo_path(root,"costModel_v2/labels/candidate_profile_records.jsonl")
        if main_record.is_file():
            record_paths.append(main_record)
        retry_dir=resolve_repo_path(root,"costModel_v2")
        record_paths.extend(sorted(retry_dir.glob("labels_retry*/candidate_profile_records.jsonl")))
    record_paths=list(dict.fromkeys(p for p in record_paths if p.is_file()))
    if not record_paths:
        raise FileNotFoundError("No candidate_profile_records.jsonl found")

    all_records=[]
    source_counts={}
    for path in record_paths:
        rows=read_jsonl(path)
        source_counts[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)]=len(rows)
        for row in rows:
            row=dict(row)
            row["_record_file"]=str(path)
            all_records.append(row)

    # Keep the strongest successful measurement for each candidate:
    # more successful repeats first, then latest timestamp.
    best={}
    rejected=Counter()
    for r in all_records:
        cid=str(r.get("candidate_id") or "")
        if r.get("label_status")!="success":
            rejected[str(r.get("label_status") or "missing_status")]+=1
            continue
        if cid not in sel or cid not in man:
            rejected["candidate_not_in_selected_or_manifest"]+=1
            continue
        latency=r.get("label_latency_us")
        try: latency=float(latency)
        except (TypeError,ValueError):
            rejected["invalid_latency"]+=1; continue
        if not math.isfinite(latency) or latency<=0:
            rejected["invalid_latency"]+=1; continue
        values=r.get("measurement_values_us") or []
        try: values=[float(v) for v in values]
        except (TypeError,ValueError):
            rejected["invalid_measurement_values"]+=1; continue
        count=int(r.get("measurement_count") or len(values))
        if values and count!=len(values):
            rejected["measurement_count_mismatch"]+=1; continue
        if values and (not all(math.isfinite(v) and v>0 for v in values)):
            rejected["invalid_measurement_values"]+=1; continue
        details=r.get("run_details") or []
        if details and any(not d.get("success",False) for d in details):
            rejected["unsuccessful_run_detail"]+=1; continue
        s=sel[cid]; m=man[cid]
        hash_ok=True
        for field in ("source_sha256","test_sha256"):
            if r.get(field)!=s.get(field) or r.get(field)!=m.get(field):
                hash_ok=False
        if not hash_ok:
            rejected["source_or_test_hash_mismatch"]+=1; continue
        if r.get("operator_name")!=s.get("operator_name") or r.get("operator_name")!=m.get("operator_name"):
            rejected["operator_mismatch"]+=1; continue
        ttir_paths=(m.get("ir_paths") or {}).get("ttir") or []
        if not ttir_paths:
            rejected["missing_ttir_path"]+=1; continue
        resolved=[resolve_repo_path(root,p) for p in ttir_paths]
        if any(not p.is_file() or p.stat().st_size==0 for p in resolved):
            rejected["missing_or_empty_ttir_file"]+=1; continue
        # If raw repeat values are available, require the saved label to be their median.
        if values:
            med=float(statistics.median(values))
            if abs(med-latency)>max(1e-4,abs(latency)*1e-5):
                rejected["label_not_median_of_measurements"]+=1; continue
        quality=(count,int(r.get("requested_repeats") or 0),str(r.get("updated_at") or ""))
        if cid not in best or quality>best[cid][0]:
            best[cid]=(quality,r)

    output=[]
    for cid,(_,r) in best.items():
        s=sel[cid]; m=man[cid]
        row={**s,**r}
        row["candidate_id"]=cid
        row["operator_name"]=s["operator_name"]
        row["selected_role"]=s.get("selected_role") or s.get("candidate_type")
        row["ir_paths"]=m.get("ir_paths") or {}
        row["compile_pass"]=m.get("compile_pass")
        row["compile_status"]=m.get("compile_status")
        row["ttir_file_count"]=len(row["ir_paths"].get("ttir") or [])
        row["profile_record_file"]=r.get("_record_file","")
        row.pop("_record_file",None)
        output.append(row)
    output.sort(key=lambda r:(r["operator_name"],ROLE_ORDER.get(r["selected_role"],99),r["candidate_id"]))

    outdir.mkdir(parents=True,exist_ok=True)
    clean_path=outdir/"clean_labels.jsonl"
    write_jsonl(clean_path,output)
    summary={
        "selected_candidate_count":len(selected),
        "ir_manifest_candidate_count":len(manifest),
        "profile_record_files":source_counts,
        "profile_record_count":len(all_records),
        "successful_unique_labels":len(output),
        "operator_count":len({r["operator_name"] for r in output}),
        "role_counts":dict(Counter(r["selected_role"] for r in output)),
        "measurement_count_distribution":dict(Counter(str(r.get("measurement_count")) for r in output)),
        "rejected_or_ignored_counts":dict(rejected),
        "missing_seed_operators":sorted({r["operator_name"] for r in output}-
                                         {r["operator_name"] for r in output if r["selected_role"]=="dataset_seed"}),
        "output":str(clean_path),
        "notes":[
            "Only label_status=success records are retained.",
            "Successful profiling labels are accepted even if warmup compile_status says compiled_with_test_error, because the profiling run itself passed the correctness test and benchmark.",
            "Candidate source/test SHA256 values must match both selected_candidates and candidate_ir_manifest.",
            "Every retained candidate must have at least one existing, non-empty TTIR file."
        ]
    }
    (outdir/"label_audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("[labels] selected candidates :",len(selected))
    print("[labels] input profile rows  :",len(all_records))
    print("[labels] successful labels   :",len(output))
    print("[labels] operators           :",summary["operator_count"])
    print("[labels] role counts         :",summary["role_counts"])
    print("[labels] ignored/rejected    :",dict(rejected))
    print("[labels] output              :",clean_path)
    print("[labels] summary             :",outdir/"label_audit_summary.json")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
