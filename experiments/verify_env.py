#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务1 · 环境连通性验证。

三项检查：
  1. LLM API      : flash / pro 双模型各发起一次真实调用，记录延迟与 token
  2. RAG 索引      : 挂载 FAISS + 本地 BGE 向量模型，执行一次真实检索
  3. msprof 单算子 : 用与正式实验完全相同的 msprof 命令评测一个算子，
                     验证 OpBasicInfo 的 Task Duration(us) 可解析

用法：
    cd /workspace/Agent
    source set_env/set_api_huoshan.sh
    export ENGINE_FLASH=deepseek-v4-flash-ga-260731
    export ENGINE_PRO=deepseek-v4-pro-ga-260813
    /usr/local/python3.11.15/bin/python3.11 experiments/verify_env.py \
        --kernel eye_kernel --repeats 1 --device 0
"""

import os
import sys
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent

# 与 main.py / harness.py 保持一致：必须在导入 langchain / transformers 之前设置
os.environ.setdefault("HF_HOME", "/workspace/user_data/Agent/RAG/.hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(AGENT_DIR))
os.chdir(AGENT_DIR)  # RAG/knowledge 与 baseline.json 使用相对路径

import argparse

from config import EAConfig
from llm_interface import LLMInterface

from experiments import common
from experiments.harness import build_executor, make_config

REPORT_PATH = common.RESULTS_DIR / "env_verification.json"

# 探针用一次极短的代码生成，既验证连通性也验证「返回内容非空」（pro 为思考型模型，
# max_tokens 过小时会把预算消耗在思考上导致 content 为空）。
PROBE_PROMPT = "写一个 Python 函数 add(a, b)，返回 a+b。只输出代码，不要任何解释。"
PROBE_SYSTEM = "你是代码生成模型。"

RAG_QUERY = "MTE2读访存主瓶颈, 行级全宽规约, 尾块mask, group_size较小"
RAG_CODE = """
@triton.jit
def _rms_norm_kernel(X, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(Y + row * stride + cols, x * rstd, mask=mask)
"""


def _mask(s: str, keep: int = 8) -> str:
    """环境/密钥脱敏。"""
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= keep else s[:keep] + "***"


# ============================== 1. LLM API ==============================
def check_llm(models) -> dict:
    api_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")

    result = {
        "api_url": api_url,
        "api_key": _mask(api_key),
        "models": {},
        "ok": False,
    }

    if not api_url or not api_key:
        result["error"] = "API_URL / API_KEY 未设置（请先 source set_env/set_api_huoshan.sh）"
        return result

    cfg = EAConfig()
    cfg.api_url = api_url
    cfg.api_key = api_key
    cfg.max_llm_tokens = 512

    for role, model in models:
        entry = {"model": model, "ok": False}
        t0 = time.time()
        try:
            llm = LLMInterface(cfg, model_name=model)
            resp = llm.generate(PROBE_PROMPT, system_msg=PROBE_SYSTEM, purpose="analysis")
            entry.update({
                "ok": True,
                "latency_s": round(time.time() - t0, 2),
                "prompt_tokens": int(getattr(resp, "prompt_tokens", 0)),
                "completion_tokens": int(getattr(resp, "completion_tokens", 0)),
                "reply_chars": len(str(resp)),
                "reply": str(resp)[:40].replace("\n", " "),
            })
            entry["ok"] = entry["ok"] and entry["reply_chars"] > 0
        except Exception as exc:
            entry.update({
                "latency_s": round(time.time() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}"[:300],
            })
        result["models"][role] = entry

    result["ok"] = all(v.get("ok") for v in result["models"].values())
    return result


# ============================== 2. RAG 索引 ==============================
def check_rag() -> dict:
    result = {"ok": False}
    t0 = time.time()

    try:
        sys.path.insert(0, str(AGENT_DIR / "RAG"))
        from knowledge_node import TritonKnowledgeNode

        node = TritonKnowledgeNode()
    except Exception as exc:
        result["error"] = f"导入/初始化失败: {type(exc).__name__}: {exc}"[:300]
        return result

    if node.vectorstore is None:
        result["error"] = "FAISS 向量库未挂载（索引缺失或 embedding 模型加载失败）"
        return result

    load_s = round(time.time() - t0, 2)
    ntotal = int(node.vectorstore.index.ntotal)

    dbg = node.debug_retrieve(RAG_QUERY, code=RAG_CODE)
    pack = node.format_candidate_pack(dbg)

    result.update({
        "ok": ntotal > 0 and len(dbg["positives"]) + len(dbg["guards"]) > 0,
        "load_seconds": load_s,
        "index_ntotal": ntotal,
        "action_docs": len(node.action_docs),
        "embedding_model": str(node.embedding_model),
        "query": RAG_QUERY,
        "normalized_query": dbg.get("normalized_query", ""),
        "broad_recall": len(dbg.get("broad", [])),
        "positives": [str(i["action"]) for i in dbg.get("positives", [])],
        "blocked": [str(i["action"]) for i in dbg.get("blocked", [])],
        "guards": [str(i["action"]) for i in dbg.get("guards", [])],
        "candidate_pack_chars": len(pack),
    })
    return result


# ============================== 3. msprof 单算子评测 ==============================
def check_msprof(kernel_name: str, repeats: int, device_id: int) -> dict:
    result = {"kernel": kernel_name, "device_id": device_id, "ok": False}

    msprof_bin = shutil.which("msprof")
    if not msprof_bin:
        result["error"] = "PATH 中找不到 msprof"
        return result

    result["msprof"] = msprof_bin
    # msprof 不支持 --version，从工具链安装路径推断 CANN 版本
    toolkit_dir = Path(os.path.realpath(msprof_bin)).resolve().parent
    while toolkit_dir.name in {"bin", "tools", "profiler"} and toolkit_dir.parent != toolkit_dir:
        toolkit_dir = toolkit_dir.parent
    result["msprof_version"] = os.getenv("ASCEND_HOME_PATH", "").rstrip("/").split("/")[-1] \
        or toolkit_dir.name or "unknown"

    try:
        config = make_config("full", kernel_name, log=False)
        baseline = config.baseline_json
        executor = build_executor(kernel_name, config)
        test_file = common.find_test_file(kernel_name)
        code = common.read_code(common.seed_code_paths(kernel_name)[0])
    except Exception as exc:
        result["error"] = f"执行器构造失败: {type(exc).__name__}: {exc}"[:300]
        return result

    result.update({
        "baseline_json": str(baseline),
        "baseline_time_us": round(executor.baseline_time, 3),
        "test_file": str(test_file),
        "code_chars": len(code),
    })

    # 与正式实验完全相同的 evaluate 口径；repeats>1 时串行取中位数（阶段 A/C 协议）
    t0 = time.time()
    samples, last_err = [], None
    for _ in range(max(1, repeats)):
        res = executor.evaluate(code, device_id=device_id)
        if res.success and res.execution_time > 0:
            samples.append(res.execution_time)
        else:
            last_err = res.error or "measurement failed"

    result["samples_us"] = [round(v, 3) for v in samples]
    result["repeats"] = max(1, repeats)
    result["error"] = last_err
    result["elapsed_s"] = round(time.time() - t0, 1)

    if samples:
        s = sorted(samples)
        mid = len(s) // 2
        result["latency_us"] = round(
            s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2, 3)
    else:
        result["latency_us"] = None

    result["ok"] = bool(res.success and res.execution_time > 0)
    return result


# ============================== 汇总 ==============================
def main():
    ap = argparse.ArgumentParser(description="任务1 环境连通性验证")
    ap.add_argument("--kernel", "-k", default="eye_kernel",
                    help="用于 msprof 冒烟评测的算子（默认 eye_kernel）")
    ap.add_argument("--repeats", type=int, default=1,
                    help="msprof 评测重复次数，>1 时走 measure_repeat 取中位数")
    ap.add_argument("--device", type=int, default=0, help="NPU device id")
    ap.add_argument("--skip", default="", help="跳过检查项，逗号分隔：llm,rag,msprof")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    common.ensure_dirs()

    cfg = EAConfig()
    models = [("flash", cfg.flash_model), ("pro", cfg.pro_model)]

    report = {
        "task": "任务1 环境连通性验证",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "checks": {},
    }

    print("=" * 72)
    print("[1/3] LLM API 连通性（flash / pro 双模型）")
    print("=" * 72)
    if "llm" in skip:
        report["checks"]["llm"] = {"ok": None, "skipped": True}
    else:
        report["checks"]["llm"] = check_llm(models)
        for role, e in report["checks"]["llm"].get("models", {}).items():
            flag = "PASS" if e.get("ok") else "FAIL"
            print(f"  [{flag}] {role:5s} {e['model']:32s} "
                  f"{e.get('latency_s', '-')}s "
                  f"tokens={e.get('prompt_tokens', 0)}+{e.get('completion_tokens', 0)} "
                  f"reply={e.get('reply', e.get('error', ''))[:24]}")

    print("=" * 72)
    print("[2/3] RAG 索引加载与检索")
    print("=" * 72)
    if "rag" in skip:
        report["checks"]["rag"] = {"ok": None, "skipped": True}
    else:
        report["checks"]["rag"] = check_rag()
        r = report["checks"]["rag"]
        print(f"  ntotal={r.get('index_ntotal')} action_docs={r.get('action_docs')} "
              f"load={r.get('load_seconds')}s")
        print(f"  positives={r.get('positives')}")
        print(f"  guards={r.get('guards')}")
        if r.get("error"):
            print(f"  error={r['error']}")

    print("=" * 72)
    print(f"[3/3] msprof 单算子评测（{args.kernel}, device {args.device}）")
    print("=" * 72)
    if "msprof" in skip:
        report["checks"]["msprof"] = {"ok": None, "skipped": True}
    else:
        report["checks"]["msprof"] = check_msprof(args.kernel, args.repeats, args.device)
        m = report["checks"]["msprof"]
        print(f"  msprof={m.get('msprof')} ({m.get('msprof_version')})")
        print(f"  baseline={m.get('baseline_time_us')}us  latency={m.get('latency_us')}us  "
              f"samples={m.get('samples_us')}  elapsed={m.get('elapsed_s')}s")
        if m.get("error"):
            print(f"  error={m['error']}")

    report["all_ok"] = all(v.get("ok") for v in report["checks"].values())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    for name, v in report["checks"].items():
        state = "SKIP" if v.get("skipped") else ("PASS" if v.get("ok") else "FAIL")
        print(f"  {name:7s} : {state}")
    print(f"  报告已保存: {REPORT_PATH}")
    print("=" * 72)

    sys.exit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
