#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 fulleval 日志算"候选预筛选"的在线（真机）精确率与召回率。

原理
----
fulleval 配置（filter_evaluate_all=True）在一次运行中同时记录了：
  * score 行：每代全部候选的模型打分与 selected 标记（即"若启用预筛选会选 top-k"）；
  * eval  行：每代全部候选的真机评测结果（child_latency / improved）。
两者通过 candidate_id 同 run 内一一对应，因此可直接算出：
  精确率 = 被选中的候选中真改进的比例
  召回率 = 被选中且真改进的候选 / 全部真改进候选
这与表10"预筛选"行的在线指标口径一致，且不需要跨运行匹配候选。

用法
----
    cd /workspace/Agent
    python experiments/compute_online_pr.py
"""
import json
import glob
import os

LOGS = "experiments/logs"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    total_sel = total_sel_imp = total_imp = 0
    per_kernel = []
    for p in sorted(glob.glob(f"{LOGS}/filter_scores__fulleval__*.jsonl")):
        rows = load(p)
        score = {r["candidate_id"]: r for r in rows if "event" not in r}
        eval_ = {r["candidate_id"]: r for r in rows if r.get("event") == "eval"}
        sel = imp = sel_imp = 0
        for cid, s in score.items():
            e = eval_.get(cid)
            if not e:
                continue
            if s.get("selected"):
                sel += 1
                if e.get("improved"):
                    sel_imp += 1
            if e.get("improved"):
                imp += 1
        if sel == 0:
            continue
        prec = sel_imp / sel
        rec = sel_imp / imp if imp else 0.0
        name = os.path.basename(p).split("__")[2]
        per_kernel.append((name, sel, sel_imp, imp, prec, rec))
        total_sel += sel
        total_sel_imp += sel_imp
        total_imp += imp

    print("各算子（fulleval 同 run 内）：")
    for name, sel, si, imp, prec, rec in per_kernel:
        print(f"  {name:42s} 选中={sel:2d} 选中且改进={si:2d} 总改进={imp:2d} "
              f"精确率={prec:.3f} 召回率={rec:.3f}")
    print(f"\n汇总（pooled）：精确率={total_sel_imp/total_sel:.4f}  "
          f"召回率={total_sel_imp/total_imp:.4f}   "
          f"(选中={total_sel} 选中且改进={total_sel_imp} 总改进={total_imp})")


if __name__ == "__main__":
    main()
