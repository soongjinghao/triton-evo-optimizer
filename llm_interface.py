#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Interface Module - Encapsulates Large Model Calls (Thread-Safe & Clean Logging)
"""

import os
import time
import logging
import threading
import httpx
import re
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import EAConfig

#相对于官方文件做出的修改：既是原生str，又挂载.text/.prompt_tokens/.completion_tokens属性
class LLMStringResponse(str):
    def __new__(cls, content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        obj = super().__new__(cls, content)
        obj.prompt_tokens = prompt_tokens
        obj.completion_tokens = completion_tokens
        return obj

    @property
    def text(self) -> str:
        return self


# All possible model list (for validation)
ALL_AVAILABLE_MODELS = [
    'deepseek-v3', 'deepseek-v3.1', 'deepseek-r1',
    'deepseek-v4-flash-260425', 'deepseek-v4-pro-260425',
    'qwen3', 'qwen3-coder',
    'kimi-k2-instruct',
    'glm-4.5'
]


class LLMInterface:
    """
    Large Model Unified Interface - Supports Model Evolution

    Key features:
    1. Support model selection from list
    2. Model itself can be part of evolution
    """

    def __init__(self, config: EAConfig, model_name: Optional[str] = None):
        self.config = config
        self.total_tokens_used = 0
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()  #相对于官方文件做出的修改：线程安全锁，防护并发状态更新

        # Select model: prioritize passed-in, otherwise use first in list
        if model_name:
            self.current_model = model_name
        elif config.llm_models:
            self.current_model = config.llm_models[0]
        else:
            raise ValueError("No available LLM, please add to config.llm_models")

        # Validate model
        if self.current_model not in ALL_AVAILABLE_MODELS:
            print(f"Warning: Model {self.current_model} is not in the standard list, but will still try to use it")

        self._init_llm()
        print(f"[LLM] Initialization complete: {self.current_model}")

    def _init_llm(self):
        """Initialize LLM client"""
        api_url = (self.config.api_url or os.getenv("API_URL", "https://api.siliconflow.cn/v1")).strip()
        api_key = self.config.api_key or os.getenv("API_KEY")

        if not api_key:
            raise ValueError("Please set API_KEY environment variable")

        http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
        https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        proxy_url = https_proxy or http_proxy

        if proxy_url:
            print(f"[LLM] Using proxy: {proxy_url}")
        try:
            client_kwargs = {"timeout": 60.0, "verify": False}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
            client = httpx.Client(**client_kwargs)
        except TypeError:
            client_kwargs = {"timeout": 60.0, "verify": False}
            if proxy_url:
                client_kwargs["proxies"] = proxy_url
            client = httpx.Client(**client_kwargs)

        #相对于官方文件做出的修改：隐藏打印调试信息
        #print(f"[DEBUG] api_url = '{api_url}'")
        #print(f"[DEBUG] http_proxy env = {os.getenv('http_proxy')}")
        #print(f"[DEBUG] HTTP_PROXY env = {os.getenv('HTTP_PROXY')}")
        #print(f"[DEBUG] https_proxy env = {os.getenv('https_proxy')}")
        #print(f"[DEBUG] HTTPS_PROXY env = {os.getenv('HTTPS_PROXY')}")
        #print(f"[DEBUG] OPENAI_PROXY env = {os.getenv('OPENAI_PROXY')}")
        #print(f"[DEBUG] proxy_url (constructed) = {proxy_url}")
        self.llm = ChatOpenAI(
            openai_api_base=api_url,
            openai_api_key=api_key,
            model=self.current_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_llm_tokens,
            http_client=client,
            openai_proxy=""
        )

    def switch_model(self, new_model: str):
        """
        Switch to a new model (for model evolution)

        Args:
            new_model: New model name
        """
        with self._lock:
            if new_model != self.current_model:
                print(f"[LLM] Switching model: {self.current_model} -> {new_model}")
                self.current_model = new_model
                self._init_llm()

    def generate(self, prompt: str, system_msg: str = "", 
                 purpose: str = "unknown", **kwargs) -> LLMStringResponse:
        """
        Call LLM to generate text

        Args:
            prompt: User prompt
            system_msg: System prompt
            purpose: Call purpose (mutate/crossover, etc.)
            **kwargs: Extra information
        """
        with self._lock:
            self.call_count += 1
            call_id = self.call_count

        #相对于官方文件做出的修改：增加了一个目的，种群初始化生成
        purpose_types = {
            'initial': 'Initial Strategy Generation',
            'mutate': 'Mutation', 'crossover': 'Crossover', 'analysis': 'Analysis',
            'fix': 'Fix', 'test_gen': 'Test Generation', 'optimize': 'Optimization',
            'model_evolve': 'Model Evolution', 'unknown': 'Unknown'
        }
        purpose_en = purpose_types.get(purpose, purpose)

        extra_info = []
        if 'parent_fitness' in kwargs:
            extra_info.append(f"parent_fitness={kwargs['parent_fitness']:.3f}")
        if 'parents_fitness' in kwargs:
            extra_info.append(f"parents_fitness={kwargs['parents_fitness']}")
        if 'mutation_type' in kwargs:
            extra_info.append(f"mutation_type={kwargs['mutation_type']}")
        if 'generation' in kwargs:
            extra_info.append(f"gen={kwargs['generation']}")
        if 'model' in kwargs:
            extra_info.append(f"model={kwargs['model']}")

        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""

        #相对于官方文件做出的修改：线程安全的单行Start打印，带有明确的[Call #XX]
        start_time = time.time()
        total_prompt_size = len(prompt) + len(system_msg)
        print(f"[LLM] [Call #{call_id:02d}] 🚀 Start | Model={self.current_model} | Purpose={purpose_en}{extra_str} | Prompt={total_prompt_size} chars")

        messages = []
        if system_msg:
            messages.append(("system", system_msg))
        messages.append(("human", prompt))

        chat_prompt = ChatPromptTemplate.from_messages(messages)
        chain = chat_prompt | self.llm

        try:
            response = chain.invoke({})
            generated_text = response.content

            #相对于官方文件做出的修改：提取API返回的精确Token数
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            elif hasattr(response, 'response_metadata'):
                rm = response.response_metadata
                if isinstance(rm, dict) and 'token_usage' in rm:
                    tu = rm['token_usage']
                    prompt_tokens = tu.get('prompt_tokens', 0)
                    completion_tokens = tu.get('completion_tokens', 0)
                prompt_tokens = len(prompt) // 4 + len(system_msg) // 4
            if completion_tokens == 0:
                completion_tokens = len(generated_text) // 4

            total_tokens = prompt_tokens + completion_tokens
            elapsed = time.time() - start_time

            with self._lock:
                self.total_tokens_used += total_tokens
                self.call_history.append({
                    'call_id': call_id,
                    'model': self.current_model,
                    'purpose': purpose,
                    'purpose_en': purpose_en,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'time_seconds': elapsed,
                    'success': True,
                    'extra': kwargs
                })

            #相对于官方文件做出的修改：带耗时与p/c细分的Complete单行打印！
            print(f"[LLM] [Call #{call_id:02d}] ✅ Complete | Tokens={total_tokens} (p:{prompt_tokens}, c:{completion_tokens}) | Time={elapsed:.2f}s")

            cleaned_text = self._clean_code_markers(generated_text)
            return LLMStringResponse(cleaned_text, prompt_tokens, completion_tokens)

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[LLM] [Call #{call_id:02d}] ❌ Failed | Time={elapsed:.2f}s | Error={str(e)}")
            with self._lock:
                self.call_history.append({
                    'call_id': call_id,
                    'model': self.current_model,
                    'purpose': purpose,
                    'purpose_en': purpose_en,
                    'time_seconds': elapsed,
                    'success': False,
                    'error': str(e),
                    'extra': kwargs
                })
            raise

    def _clean_code_markers(self, text: str) -> str:
        """Clean markdown markers"""
        text = re.sub(r'```python\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        return text.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics (Thread-safe)"""
        with self._lock:
            purpose_stats = {}
            for call in self.call_history:
                p = call['purpose_en']
                if p not in purpose_stats:
                    purpose_stats[p] = {'count': 0, 'tokens': 0, 'time_seconds': 0, 'success': 0, 'models': set()}
                purpose_stats[p]['count'] += 1
                purpose_stats[p]['tokens'] += call.get('total_tokens', 0)
                purpose_stats[p]['time_seconds'] += call.get('time_seconds', 0)
                purpose_stats[p]['models'].add(call.get('model', 'unknown'))
                if call['success']:
                    purpose_stats[p]['success'] += 1

            for p in purpose_stats:
                purpose_stats[p]['models'] = list(purpose_stats[p]['models'])

            total_time = sum(p['time_seconds'] for p in purpose_stats.values())

            return {
                'total_tokens': self.total_tokens_used,
                'total_time_seconds': total_time,
                'call_count': self.call_count,
                'current_model': self.current_model,
                'available_models': self.config.llm_models,
                'purpose_breakdown': purpose_stats
            }

    def print_call_summary(self):
        """Print call summary"""
        stats_data = self.get_stats()
        print(f"\n{'='*60}")
        print("LLM Call Statistics")
        print(f"{'='*60}")
        print(f"Current model: {self.current_model}")
        print(f"Available model list: {', '.join(self.config.llm_models)}")
        print(f"Total call count: {stats_data['call_count']}")
        print(f"Total token consumption: {stats_data['total_tokens']}")
        print(f"Total time spent: {stats_data['total_time_seconds']:.2f}s")

        stats = stats_data['purpose_breakdown']
        print(f"\nGrouped by purpose:")
        for purpose, data in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
            success_rate = data['success']/data['count']*100 if data['count'] > 0 else 0
            avg_time = data['time_seconds']/data['count'] if data['count'] > 0 else 0
            models_str = ', '.join(data['models'])
            print(f"  - {purpose}: {data['count']} times, {data['tokens']} tokens, "
                  f"{data['time_seconds']:.2f}s (avg {avg_time:.2f}s), "
                  f"success rate {success_rate:.0f}%, models used: {models_str}")