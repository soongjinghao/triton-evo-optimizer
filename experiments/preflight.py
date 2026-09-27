#!/usr/bin/env python3
"""实验启动前自检（Preflight Check）。

背景：LLM 端点不可达时，搜索会静默降级——候选生成失败后回退到种子，
产出 fitness=1.0 的"假成功"结果，表面上 50 个 Kernel 全部完成，
实际整轮数据作废且极难察觉。本脚本用于在启动长任务前拦截该类问题。

检查项：
  1. LLM 端点连通（致命）——不通则拒绝启动
  2. NPU + msprof 可用（致命）——不通则拒绝启动
  3. 数据集与 manifest 完整性（警告）

用法：
    source set_env/set_api_local.sh
    python experiments/preflight.py             # 全检，约 1 分钟
    python experiments/preflight.py --skip-npu  # 跳过 NPU 实测，约 10 秒

退出码：0 = 可启动；1 = 存在致命问题，不要启动。
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import common  # noqa: E402


def check_llm(timeout: float = 40.0):
    """检查 LLM 端点能否正常返回。"""
    url = os.environ.get('API_URL')
    key = os.environ.get('API_KEY')
    model = os.environ.get('ENGINE_PRO') or os.environ.get('ENGINE_FLASH')
    if not url or not model:
        return False, f"环境变量缺失 (API_URL={url!r}, ENGINE_PRO={model!r})"
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_base=url,
            openai_api_key=key or 'EMPTY',
            model=model,
            temperature=0.0,
            max_tokens=16,
            timeout=timeout,
            max_retries=0,
        )
        t0 = time.time()
        r = llm.invoke('Reply with OK')
        return True, f"OK 响应={r.content[:24]!r}  耗时={time.time() - t0:.1f}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:170]}"


def check_npu():
    """用一个短 Kernel 跑一次真实评测，确认 NPU 与 msprof 可用。"""
    try:
        from experiments.harness import make_config, build_executor
        k = 'eye_kernel'
        cfg = make_config('b0', k, log=False)
        ex = build_executor(k, cfg)
        paths = common.seed_code_paths(k)
        if not paths:
            return False, f"{k} 无种子文件"
        code = common.read_code(paths[0])
        t0 = time.time()
        res = ex.evaluate(code, timeout=240, device_id=0)
        if res.success and res.execution_time > 0:
            return True, f"{k} 实测 {res.execution_time:.2f} us  耗时={time.time() - t0:.0f}s"
        return False, f"评测失败: {res.error}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:170]}"


def check_data():
    """检查数据集与 manifest 完整性（非致命）。"""
    kernels = common.list_kernels()
    ok_base = ok_seed = 0
    missing = []
    for k in kernels:
        f = common.MANIFEST_DIR / f'{k}.json'
        if not f.exists():
            missing.append(k)
            continue
        d = json.load(open(f, encoding='utf-8'))
        if d.get('t_base_us') is not None:
            ok_base += 1
        if d.get('t_seed_us') is not None:
            ok_seed += 1
    return {
        'kernels': len(kernels),
        'manifest': len(list(common.MANIFEST_DIR.glob('*.json'))),
        't_base_ok': ok_base,
        't_seed_ok': ok_seed,
        'missing': missing,
    }


def main():
    skip_npu = '--skip-npu' in sys.argv

    print("=" * 58)
    print(" 实验启动前自检 Preflight Check")
    print(f" 数据集: {common.DATASETS_DIR}")
    print("=" * 58)

    fatal = False

    print("\n[1] LLM 端点连通性")
    ok, msg = check_llm()
    print(f"    {'✅' if ok else '❌'} {msg}")
    if not ok:
        fatal = True
        print("    -> 致命：LLM 不可达，搜索会静默降级为 fitness=1.0 的空结果")

    if skip_npu:
        print("\n[2] NPU 可用性 —— 已跳过 (--skip-npu)")
    else:
        print("\n[2] NPU + msprof 可用性")
        ok2, msg2 = check_npu()
        print(f"    {'✅' if ok2 else '❌'} {msg2}")
        if not ok2:
            fatal = True
            print("    -> 致命：NPU 评测不可用")

    print("\n[3] 数据集与 manifest 完整性")
    d = check_data()
    print(f"    Kernel 数: {d['kernels']}")
    print(f"    manifest : {d['manifest']}")
    print(f"    T_base 可用: {d['t_base_ok']} / {d['kernels']}")
    print(f"    T_seed 可用: {d['t_seed_ok']} / {d['kernels']}")
    if d['missing']:
        print(f"    ⚠ 缺 manifest: {d['missing'][:8]}{' ...' if len(d['missing']) > 8 else ''}")

    print()
    print("=" * 58)
    if fatal:
        print(" 结论：❌ 存在致命问题，不要启动实验")
        print("=" * 58)
        sys.exit(1)
    print(" 结论：✅ 全部通过，可以启动实验")
    print("=" * 58)
    sys.exit(0)


if __name__ == '__main__':
    main()
