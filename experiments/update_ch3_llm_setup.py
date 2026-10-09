#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.3 实验设置中「大模型调用约定」段的改写。

原表述只提到"轻量模型 / 强模型"两个端点，既未说明部署方式，也未给出
生成参数。本次改写为：明确本地部署的 vLLM 服务与 DeepSeek-V4，并给出
温度、最大生成长度与重试策略；同时如实说明本地两个端点指向同一模型实例。

事实依据（均为实测，勿凭印象修改）
--------------------------------
* curl http://127.0.0.1:13000/v1/models
      id=140aced515cc4236a6141f2b8a77107f  owned_by=vllm
      root=/root/bentoml/models/deepseekv4---955cf8/...   max_model_len=262144
* set_env/set_api_local.sh
      API_URL=http://127.0.0.1:13000/v1
      ENGINE_FLASH = ENGINE_PRO = 140aced515cc4236a6141f2b8a77107f
        （脚本注释：本地只有一个模型，两个角色都用同一个）
      MAX_LLM_TOKENS=8192    LLM_MAX_RETRIES=10    LLM_RETRY_WAIT=60
* config.py
      llm_temperature = 0.2（默认）；max_llm_tokens 由 MAX_LLM_TOKENS 覆盖为 8192

注意：不能写"DeepSeek-V4-Flash"。config.py 里的 deepseek-v4-flash-ga-260731
只是**未设环境变量时的默认值**，本地部署时被 ENGINE_FLASH 覆盖为上述 ID，
且本地只有一个模型实例，无法确认是否为 flash 版本。

幂等：按文字锚点定位，可反复运行。
"""
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.docx_util import find_paragraph, set_text      # noqa: E402

DOC = ROOT / "doc" / "第三章_修订版_四节实验数据整理_表4拆分.docx"

# 锚点取"大模型侧"：改写后的段落仍以它开头，保证重复运行时能再次命中
A_LLM = "大模型侧"

NEW = (
    "大模型侧统一通过本地部署的 vLLM 服务调用，服务地址为 127.0.0.1:13000/v1"
    "（OpenAI 兼容接口），模型为本地部署的 DeepSeek-V4，最大上下文长度 262144。"
    "框架按任务类型区分两个调用端点：分析类任务（事实提取、策略蓝图生成）走轻量端点，"
    "代码生成类任务走强模型端点；本次实验的本地服务仅部署一个模型实例，"
    "两个端点指向同一模型，因此各对比方法在模型能力上完全可比。"
    "所有调用的生成参数保持一致：温度取 0.2，最大生成长度取 8192 tokens，"
    "其余采样参数沿用服务端默认值；调用失败时按指数退避重试，最多 10 次、"
    "首次等待 60 秒。上述参数在所有对比方法中完全相同。"
)


def main():
    d = Document(DOC)
    _, p = find_paragraph(d, A_LLM)
    old = p.text.strip()
    set_text(p, NEW)
    d.save(DOC)
    print("已更新「大模型调用约定」段")
    print("  原文:")
    print("    " + old)
    print("  新文:")
    print("    " + NEW)


if __name__ == "__main__":
    main()
