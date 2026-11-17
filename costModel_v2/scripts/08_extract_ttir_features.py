#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract fixed-dimension numeric features from candidate TTIR MLIR files."""
from __future__ import annotations
import argparse, csv, json, math, re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT=Path(__file__).resolve().parents[2]
OP_VOCAB=(
"arith.addf","arith.addi","arith.andi","arith.cmpf","arith.cmpi","arith.constant",
"arith.divf","arith.divsi","arith.extf","arith.extsi","arith.extui","arith.fptosi",
"arith.maxnumf","arith.minnumf","arith.minsi","arith.mulf","arith.muli","arith.ori",
"arith.remsi","arith.select","arith.shli","arith.shrsi","arith.sitofp","arith.subf",
"arith.subi","arith.truncf","arith.uitofp","arith.xori","cf.cond_br",
"math.absf","math.ceil","math.exp","math.exp2","math.log","math.log2","math.rsqrt","math.sqrt",
"scf.condition","scf.for","scf.if","scf.while","scf.yield",
"tt.addptr","tt.assert","tt.atomic_rmw","tt.broadcast","tt.clampf","tt.descriptor_load",
"tt.descriptor_store","tt.dot","tt.expand_dims","tt.func","tt.get_num_programs",
"tt.get_program_id","tt.int_to_ptr","tt.load","tt.make_range","tt.make_tensor_descriptor",
"tt.make_tensor_ptr","tt.reduce","tt.reduce.return","tt.reshape","tt.return","tt.scan",
"tt.scan.return","tt.splat","tt.store"
)
OP_RE=re.compile(r'^\s*(?:%[\w.$-]+(?::\d+)?\s*=\s*)?(?:"([A-Za-z_][\w.]*)"|([A-Za-z_][\w.]*))')
TENSOR_RE=re.compile(r"tensor<([^>]+)")
DTYPE_RE=re.compile(r"(?<![A-Za-z0-9_])(bf16|f16|f32|f64|i1|i8|i16|i32|i64)(?![A-Za-z0-9_])")
RANGE_RE=re.compile(r"tt\.make_range")
INT_ARITH={"arith.addi","arith.subi","arith.muli","arith.divsi","arith.remsi","arith.andi",
           "arith.ori","arith.xori","arith.shli","arith.shrsi","arith.cmpi","arith.extsi",
           "arith.extui","arith.minsi"}
FLOAT_ARITH={"arith.addf","arith.subf","arith.mulf","arith.divf","arith.cmpf","arith.maxnumf",
             "arith.minnumf","arith.extf","arith.truncf","arith.sitofp","arith.uitofp","arith.fptosi"}

def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open("r",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def resolve_repo_path(root:Path,raw:str)->Path:
    p=Path(str(raw))
    if not p.is_absolute(): return (root/p).resolve()
    try: p.relative_to(root); return p
    except ValueError:
        s=str(p); marker="/workspace/Agent/"
        if marker in s: return (root/s.split(marker,1)[1]).resolve()
        return p

def parse_module(path:Path)->dict[str,Any]:
    text=path.read_text(encoding="utf-8",errors="replace")
    ops=Counter()
    tensor_ranks=[]; tensor_elements=[]; tensor_dims=[]; dynamic_shapes=0
    dtype_counts=Counter(); range_extents=[]
    loop_indents=[]; max_loop_depth=0
    for line in text.splitlines():
        m=OP_RE.match(line)
        if m:
            op=m.group(1) or m.group(2)
            if "." in op or op in {"return","yield"}: ops[op]+=1
            if op in {"scf.for","scf.while"}:
                indent=len(line)-len(line.lstrip())
                while loop_indents and loop_indents[-1]>=indent: loop_indents.pop()
                loop_indents.append(indent)
                max_loop_depth=max(max_loop_depth,len(loop_indents))
        if RANGE_RE.search(line):
            end=re.search(r"\bend\s*=\s*(\d+)",line)
            start=re.search(r"\bstart\s*=\s*(\d+)",line)
            if end:
                e=int(end.group(1)); s=int(start.group(1)) if start else 0
                range_extents.append(max(0,e-s))
    for tm in TENSOR_RE.finditer(text):
        body=tm.group(1)
        parts=body.split("x")
        dims=[]
        for token in parts:
            token=token.strip()
            if token.isdigit(): dims.append(int(token))
            elif token=="?": dims.append(None)
            else: break
        if dims:
            tensor_ranks.append(len(dims))
            tensor_dims.extend(d for d in dims if d is not None)
            if any(d is None for d in dims):
                dynamic_shapes+=1
            else:
                tensor_elements.append(math.prod(dims))
        dtype_counts.update(DTYPE_RE.findall(body))
    return {
        "ops":ops,"tensor_ranks":tensor_ranks,"tensor_elements":tensor_elements,
        "tensor_dims":tensor_dims,"dynamic_shapes":dynamic_shapes,
        "dtype_counts":dtype_counts,"range_extents":range_extents,
        "line_count":len(text.splitlines()),"byte_count":path.stat().st_size,
        "pointer_type_count":len(re.findall(r"!tt\.ptr<",text)),
        "loop_max_depth":max_loop_depth
    }

def mean(xs:list[float])->float:
    return float(sum(xs)/len(xs)) if xs else 0.0

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default=str(DEFAULT_ROOT))
    ap.add_argument("--input",default="costModel_v2/features/clean_labels.jsonl")
    ap.add_argument("--output",default="costModel_v2/features/ttir_features.csv")
    ap.add_argument("--errors",default="costModel_v2/features/ttir_feature_errors.jsonl")
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve()
    inpath=resolve_repo_path(root,args.input); outpath=resolve_repo_path(root,args.output)
    errpath=resolve_repo_path(root,args.errors)
    labels=read_jsonl(inpath)
    rows=[]; errors=[]
    for index,candidate in enumerate(labels,1):
        cid=candidate["candidate_id"]
        paths=list(dict.fromkeys((candidate.get("ir_paths") or {}).get("ttir") or []))
        try:
            if not paths: raise ValueError("No TTIR paths in IR manifest")
            modules=[]
            for raw in paths:
                p=resolve_repo_path(root,raw)
                if not p.is_file() or p.stat().st_size==0:
                    raise FileNotFoundError(f"Missing/empty TTIR: {p}")
                modules.append(parse_module(p))
            ops=Counter()
            all_ranks=[]; all_elements=[]; all_dims=[]; all_extents=[]; dtypes=Counter()
            line_count=byte_count=dynamic_shapes=loop_depth=pointer_type_count=0
            for mod in modules:
                ops.update(mod["ops"]); all_ranks.extend(mod["tensor_ranks"])
                all_elements.extend(mod["tensor_elements"]); all_dims.extend(mod["tensor_dims"])
                all_extents.extend(mod["range_extents"]); dtypes.update(mod["dtype_counts"])
                line_count+=mod["line_count"]; byte_count+=mod["byte_count"]
                dynamic_shapes+=mod["dynamic_shapes"]; loop_depth=max(loop_depth,mod["loop_max_depth"]); pointer_type_count+=mod["pointer_type_count"]
            known=set(OP_VOCAB)
            row={
                "candidate_id":cid,"operator_name":candidate["operator_name"],
                "selected_role":candidate.get("selected_role",""),
                "source_sha256":candidate.get("source_sha256",""),
                "ttir_file_count":len(modules),"ttir_total_lines":line_count,
                "ttir_total_bytes":byte_count,"op_total_count":sum(ops.values()),
                "unique_op_type_count":len(ops),"unknown_op_count":sum(v for k,v in ops.items() if k not in known),
                "arith_op_count":sum(v for k,v in ops.items() if k.startswith("arith.")),
                "math_op_count":sum(v for k,v in ops.items() if k.startswith("math.")),
                "tt_op_count":sum(v for k,v in ops.items() if k.startswith("tt.")),
                "scf_op_count":sum(v for k,v in ops.items() if k.startswith("scf.")),
                "cf_op_count":sum(v for k,v in ops.items() if k.startswith("cf.")),
                "memory_read_count":ops["tt.load"]+ops["tt.descriptor_load"],
                "memory_write_count":ops["tt.store"]+ops["tt.descriptor_store"],
                "memory_access_count":ops["tt.load"]+ops["tt.descriptor_load"]+ops["tt.store"]+ops["tt.descriptor_store"]+ops["tt.atomic_rmw"],
                "arith_integer_op_count":sum(ops[k] for k in INT_ARITH),
                "arith_float_op_count":sum(ops[k] for k in FLOAT_ARITH),
                "loop_op_count":ops["scf.for"]+ops["scf.while"],
                "loop_max_nesting_depth":loop_depth,
                "if_count":ops["scf.if"],"branch_count":ops["cf.cond_br"],
                "reduce_count":ops["tt.reduce"],"scan_count":ops["tt.scan"],
                "dot_count":ops["tt.dot"],"atomic_count":ops["tt.atomic_rmw"],
                "broadcast_count":ops["tt.broadcast"],"expand_dims_count":ops["tt.expand_dims"],
                "reshape_count":ops["tt.reshape"],"make_range_count":ops["tt.make_range"],
                "program_id_count":ops["tt.get_program_id"]+ops["tt.get_num_programs"],
                "addptr_count":ops["tt.addptr"],"splat_count":ops["tt.splat"],
                "tensor_type_count":len(all_ranks),
                "tensor_rank_max":max(all_ranks,default=0),
                "tensor_rank_mean":mean([float(x) for x in all_ranks]),
                "tensor_dynamic_shape_count":dynamic_shapes,
                "tensor_static_shape_count":len(all_elements),
                "tensor_elements_max":max(all_elements,default=0),
                "tensor_elements_mean":mean([float(x) for x in all_elements]),
                "tensor_dim_count":len(all_dims),
                "tensor_dim_max":max(all_dims,default=0),
                "tensor_dim_mean":mean([float(x) for x in all_dims]),
                "range_extent_count":len(all_extents),
                "range_extent_sum":sum(all_extents),
                "range_extent_max":max(all_extents,default=0),
                "range_extent_mean":mean([float(x) for x in all_extents]),
                "pointer_type_count":pointer_type_count,
                "constant_dense_count":0,"constant_scalar_count":0,
            }
            # Constant kind counts are line-based and aggregated over all modules.
            for raw in paths:
                text=resolve_repo_path(root,raw).read_text(encoding="utf-8",errors="replace")
                for line in text.splitlines():
                    if "arith.constant" in line:
                        if "dense<" in line: row["constant_dense_count"]+=1
                        else: row["constant_scalar_count"]+=1
            for op in OP_VOCAB:
                row["op_count__"+op.replace(".","_")]=ops[op]
            for dtype in ("bf16","f16","f32","f64","i1","i8","i16","i32","i64"):
                row["dtype_count__"+dtype]=dtypes[dtype]
            rows.append(row)
        except Exception as exc:
            errors.append({"candidate_id":cid,"operator_name":candidate.get("operator_name"),
                           "error":f"{type(exc).__name__}: {exc}"})
        if index%25==0 or index==len(labels):
            print(f"[features] processed {index}/{len(labels)} candidates; success={len(rows)} errors={len(errors)}",flush=True)
    outpath.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        fields=list(rows[0].keys())
        with outpath.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    errpath.parent.mkdir(parents=True,exist_ok=True)
    with errpath.open("w",encoding="utf-8") as f:
        for e in errors: f.write(json.dumps(e,ensure_ascii=False)+"\n")
    print("[features] candidates :",len(labels))
    print("[features] extracted  :",len(rows))
    print("[features] errors     :",len(errors))
    print("[features] CSV        :",outpath)
    print("[features] errors     :",errpath)
    return 0 if not errors else 2

if __name__=="__main__":
    raise SystemExit(main())
