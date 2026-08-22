#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Operators Module - Decoupled Architecture (Generic Feature RAG + Profiling-Guided Action)
Universal & Fully Generalized Version for Ascend NPU.
"""

import re
import ast
from dataclasses import dataclass, field
import hashlib
import random
import time
import threading
from typing import Optional, List, Tuple

from config import EAConfig
from llm_interface import LLMInterface
from RAG.knowledge_node import TritonKnowledgeNode


@dataclass
class Individual:
    code: str
    fitness: float = 0.0
    generation: int = 0
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    metadata: dict = field(default_factory=dict)
    model_used: str = "unknown"

    def __hash__(self):
        """基于个体 ID 生成哈希值，用于集合和字典操作"""
        return hash(self.id)

    def __eq__(self, other):
        """基于 ID 比较两个个体是否相等"""
        if not isinstance(other, Individual):
            return False
        return self.id == other.id


class GeneticOperators:
    SYSTEM_ANALYST = "你是硬件与算法感知的 Triton 性能分析专家。严格按格式提取代码特征，绝对不输出冗余解释。"

    SYSTEM_CODER = (
        "你是华为 Ascend NPU 的 Triton 高性能内核专家。\n"
        "⚠️【极硬命令】：只输出一个完整的 ```python 代码块，绝对不要输出解释性文字！\n\n"
        "【规则优先级（发生冲突时严格按此顺序）】:\n"
        "A. 接口契约、数学语义、完整输入域覆盖与边界安全；\n"
        "B. 当前任务指定的变异/交叉/重构类型；\n"
        "C. Profiling 与 RAG 给出的候选策略；\n"
        "D. 通用性能建议。低优先级规则不得破坏高优先级规则。\n\n"
        "【Ascend NPU 高性能通用重构规范】:\n"
        "1. 🔒 【契约与语义一致】：外层 Wrapper 函数名、签名、参数默认值、返回值数量、输出 shape/dtype/layout 以及所有功能开关语义必须与基线一致。\n"
        "2. 🛡️ 【完整覆盖与边界安全】：任何逐元素、分组或规约计算都必须覆盖完整逻辑输入域。禁止通过缩小 BLOCK、跳过尾部列/组/Token 获得伪加速。存在尾块、非 2 次幂 GROUP_SIZE、非整除维度或无效 Token 时必须保留正确 mask；只有能从 constexpr、断言和索引范围严格证明冗余时才允许删除。\n"
        "3. 🛡️ 【量化语义绑定】：若基线通过 Kernel/Wrapper 参数传入 FP8_MIN、FP8_MAX、eps、use_ue8m0 等边界或模式，必须原样保留并使用；若基线没有这些参数，不得为迎合提示而擅自修改接口。禁止根据某个公开测试 shape 或数据值硬编码专用路径。\n"
        "4. 🚫 【静态形状约束】：Kernel 内 `tl.view` 的 shape 与 `tl.arange` 的 stop 必须由编译期常量构成；禁止 3D 及以上 `tl.view`。\n"
        "5. ⚡ 【条件式低风险优化】：倒数代除、Stride 简化、`tl.multiple_of`、`tl.max_contiguous`、Mask 消融都只是候选动作。只有在数学等价、连续轴、基地址和行跨度对齐可由代码证明时才能应用，否则保持原实现。\n"
        "6. 🧭 【寻址与布局保护】：不得把任意 stride 输入擅自改写为连续寻址；只有 Wrapper 明确生成连续张量，或 stride/shape 契约能证明最内层 stride 恒为 1 时，才允许将 `* stride_h` 简化为 `+ cols`。\n"
        "7. 🛡️ 【安全 API 约束】：禁止使用当前 Ascend 后端不支持或高风险的 `tl.math.fast_exp`、`fast_log`、`ilog2` 和 `ldexp`。超越函数替换必须保持误差要求。\n"
        "8. 🧩 【控制流约束】：允许使用 `%` 与 `//` 对 `program_id` 做编译期常量除数的网格解码；禁止依赖 Tensor 数据的 Python/JIT 动态分支、非法动态循环边界以及 `break` 提前退出。\n"
        "9. 🧪 【可泛化优化】：禁止识别函数名、固定输入尺寸或公开测试模式来启用专用计算路径；优化必须对合法输入范围保持正确。\n"
        "10. 🔒 【公开 JIT 入口契约】：若基线对外公开的同名入口本身是 `@triton.jit` kernel，并由外部直接通过 `kernel[grid](...)` 调用，则优化后该同名入口必须继续保持 `@triton.jit` 身份、参数契约与外部 Launch 方式；禁止把它改成普通 Python Wrapper，也禁止让优化依赖修改外部测试代码中的 Grid/调用方式。\n"
        "11. 🚫 【Triton 后端 API 合法性】：禁止为 `tl.load` / `tl.store` 发明或添加当前 Ascend 后端未验证的关键字参数。当前静态门禁明确拒绝 `cache=` 与 `max_contiguous=` 作为 `tl.load/tl.store` 的关键字；不得为了执行策略而绕过此限制。\n"
        "12. 🚫 【JIT 内 Python 方法约束】：在 `@triton.jit` 函数内部禁止对 `tl.constexpr`、Triton tensor 或相关表达式调用普通 Python 对象方法（例如 `.bit_length()`）；需要的编译期常量必须使用合法 constexpr 表达式或由外层 Wrapper 计算后传入。\n"
        "13. 🛡️ 【Mask Fast Path 类型约束】：若编译期 fast path 需要在无 Mask 与有 Mask 之间切换，必须使用显式 constexpr 分支分别调用 `tl.load/tl.store`；禁止使用 `mask = None if ... else tensor_mask` 或等价的 None/Tensor 条件合流。\n"
    )

    EXPERT_PREFIX = "你是华为 Ascend NPU 的 Triton 高性能内核专家。"
    REFACTOR_GUIDELINES = "直接输出 ```python 代码块，不要解释，不要注释，不要测试代码。代码完整，无 ... 或 pass。"

    def __init__(self, llm: LLMInterface, config: EAConfig):
        """初始化遗传算子，绑定 LLM 接口、配置和 RAG 知识库节点"""
        self.llm = llm
        self.config = config
        self.knowledge_node = TritonKnowledgeNode(knowledge_dir="RAG/knowledge")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._lock = threading.Lock()

    def _switch_model_for_purpose(self, purpose: str):
        """根据任务目的切换 LLM 模型：分析任务用 flash_model，生成任务用 pro_model"""
        target_model = self.config.flash_model if purpose == 'analysis' else self.config.pro_model
        if target_model != self.llm.current_model:
            self.llm.switch_model(target_model)

    def _escape_code_for_prompt(self, code: str) -> str:
        """将代码中的花括号转义为双花括号，防止与 Prompt 模板的格式化语法冲突"""
        return code.replace("{", "{{").replace("}", "}}")

    def _compress_code_for_prompt(self, code: str) -> str:
        """在不改变任何代码逻辑的前提下，大幅压缩送入 Prompt 的字符数"""
        if not code:
            return ""
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'', '', code)
        lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') and not any(k in stripped for k in ['Pointers', 'Sizes', 'Strides', 'Meta']):
                continue
            lines.append(line)
        
        clean_code = re.sub(r'\n\s*\n', '\n', '\n'.join(lines))
        return clean_code.strip()

    def _track_tokens(self, response, meta_dict: dict = None):
        """追踪 LLM 调用的 Token 消耗量，累计到全局统计并记录到个体元数据"""
        p_tok = getattr(response, 'prompt_tokens', 0)
        c_tok = getattr(response, 'completion_tokens', 0)
        with self._lock:
            self.total_prompt_tokens += p_tok
            self.total_completion_tokens += c_tok
        if isinstance(meta_dict, dict):
            meta_dict['usage'] = {'prompt_tokens': p_tok, 'completion_tokens': c_tok}

    class ASTSanitizer(ast.NodeVisitor):
        """AST 静态门禁：在送 NPU 前拦截已知高风险/非法 Triton 代码模式。"""
        def __init__(self):
            self.has_3d_view = False
            self.has_fp32_bitcast_bitwise = False
            self._jit_depth = 0
            self.has_jit_bit_length = False
            self.invalid_tl_memory_kwargs = set()
            self.has_none_tensor_mask_merge = False

        @staticmethod
        def _is_jit_decorator(decorator) -> bool:
            if isinstance(decorator, ast.Attribute):
                return decorator.attr == 'jit'
            if isinstance(decorator, ast.Name):
                return decorator.id == 'jit'
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute):
                    return func.attr == 'jit'
                if isinstance(func, ast.Name):
                    return func.id == 'jit'
            return False

        @staticmethod
        def _call_name(func) -> str:
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                prefix = GeneticOperators.ASTSanitizer._call_name(func.value)
                return f"{prefix}.{func.attr}" if prefix else func.attr
            return ""

        @staticmethod
        def _ifexp_contains_none(node) -> bool:
            return (
                isinstance(node, ast.IfExp)
                and (
                    (isinstance(node.body, ast.Constant) and node.body.value is None)
                    or (isinstance(node.orelse, ast.Constant) and node.orelse.value is None)
                )
            )

        def visit_FunctionDef(self, node):
            is_jit = any(self._is_jit_decorator(d) for d in node.decorator_list)
            if is_jit:
                self._jit_depth += 1
            for stmt in node.body:
                self.visit(stmt)
            if is_jit:
                self._jit_depth -= 1

        def visit_Assign(self, node):
            if self._jit_depth and self._ifexp_contains_none(node.value):
                target_names = [
                    t.id for t in node.targets
                    if isinstance(t, ast.Name)
                ]
                if any('mask' in name.lower() for name in target_names):
                    self.has_none_tensor_mask_merge = True
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if (
                self._jit_depth
                and isinstance(node.target, ast.Name)
                and 'mask' in node.target.id.lower()
                and self._ifexp_contains_none(node.value)
            ):
                self.has_none_tensor_mask_merge = True
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'view':
                if len(node.args) >= 2:
                    shape_arg = node.args[1]
                    if isinstance(shape_arg, ast.Tuple) and len(shape_arg.elts) >= 3:
                        self.has_3d_view = True

            call_name = self._call_name(node.func)

            if call_name in {'tl.load', 'tl.store'}:
                for kw in node.keywords:
                    if kw.arg in {'cache', 'max_contiguous'}:
                        self.invalid_tl_memory_kwargs.add(
                            f"{call_name}(..., {kw.arg}=...)"
                        )
                    if kw.arg == 'mask' and self._ifexp_contains_none(kw.value):
                        self.has_none_tensor_mask_merge = True

            if (
                self._jit_depth
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'bit_length'
            ):
                self.has_jit_bit_length = True

            self.generic_visit(node)

    def _get_return_count_ast(self, code: str) -> int:
        """通过 AST 解析获取外层 Wrapper 函数的返回值数量"""
        try:
            parsed = ast.parse(code)
            for node in reversed(parsed.body):
                if isinstance(node, ast.FunctionDef):
                    is_jit = any(
                        (isinstance(d, ast.Attribute) and d.attr == 'jit') or
                        (isinstance(d, ast.Name) and d.id == 'jit')
                        for d in node.decorator_list
                    )
                    if not is_jit:
                        for stmt in reversed(node.body):
                            if isinstance(stmt, ast.Return) and stmt.value:
                                if isinstance(stmt.value, ast.Tuple):
                                    return len(stmt.value.elts)
                                return 1
        except Exception:
            pass
        return 1

    def _sanitize_code(self, raw_code: str, original_code: str = None) -> Tuple[str, Optional[str]]:
        """代码清洗与验证：清理 Markdown 标记、修正 API 调用、检查硬件约束"""
        if not raw_code:
            return "", "LLM generated an empty text response."

        code = raw_code.strip()
        code = re.sub(r'^```python\s*|^```\s*', '', code, flags=re.IGNORECASE)
        code = re.sub(r'```$', '', code)
        code = re.sub(r'^\s*\d+\.\s*', '', code, flags=re.MULTILINE).strip()

        code = code.replace("torch.cuda", "torch.npu")
        code = code.replace("：", ":").replace("，", ",").replace("（", "(").replace("）", ")")

        code = re.sub(r'\btl\.reshape\b', 'tl.view', code)
        code = re.sub(r'\btl\.math\.fast_log2\b', 'tl.log2', code)
        code = re.sub(r'\btl\.math\.fast_exp2\b', 'tl.exp2', code)
        code = re.sub(r',?\s*align=\d+', '', code)
        code = re.sub(r'@triton\.jit\([^)]*\)', '@triton.jit', code)

        try:
            parsed_ast = ast.parse(code)
        except SyntaxError as e:
            return "", f"SyntaxError: {e.msg} at line {e.lineno}"

        sanitizer = self.ASTSanitizer()
        sanitizer.visit(parsed_ast)

        if sanitizer.has_3d_view:
            return "", "Contract Error: Prohibited 3D (or higher) tl.view detected via AST."

        if sanitizer.invalid_tl_memory_kwargs:
            bad = ", ".join(sorted(sanitizer.invalid_tl_memory_kwargs))
            return "", f"Preflight Error: unsupported Ascend tl.load/tl.store keyword detected: {bad}"

        if sanitizer.has_jit_bit_length:
            return "", "Preflight Error: .bit_length() inside @triton.jit is prohibited on the current backend."

        if sanitizer.has_none_tensor_mask_merge:
            return "", "Preflight Error: None/Tensor mask ternary merge detected inside @triton.jit; use explicit constexpr branches."

        if re.search(r'tl\.math\.ilog2|tl\.math\.ldexp', code):
            return "", "Contract Error: ilog2/ldexp are unsupported on Ascend NPU backend."

        if original_code:
            orig_ret_count = self._get_return_count_ast(original_code)
            new_ret_count = self._get_return_count_ast(code)
            if orig_ret_count != new_ret_count:
                return "", f"Contract Error: Original returns {orig_ret_count} items, but new code returns {new_ret_count} items."

        return code, None

    def _get_rag_code_analysis(self, code: str, profiling_context: str = "") -> str:
        """提取用于 RAG 检索的事实型结构/瓶颈标签，未知结构允许保留真实新词汇"""
        prompt = [
            "你是 Triton 编译器性能分析专家。请阅读代码和【硬件 Profiling 实测诊断】，提炼 3-5 个用于 RAG 检索的【事实型结构/瓶颈特征短语】。",
            "只描述能够从代码、Wrapper 契约或 Profiling 直接证明的事实；禁止直接给优化方案，禁止输出 ACTION 名称，禁止猜测对齐、整除、连续性或循环长度。",
            "至少包含 1 个代码结构特征；Profiling 存在明确主瓶颈时，至少包含 1 个 Scalar/Vector/MTE 特征。",
            "知识库词汇仅在能够准确描述当前结构时优先使用；如果当前代码存在知识库词汇无法准确覆盖的新结构，应保留真实的新结构事实短语，不得为了命中已有 ACTION 强行归类。",
            "仅仅出现 for/range 循环不能直接标记为‘单program长串行循环’；只有迭代次数能由 constexpr、shape、代码结构或 Profiling 明确证明为多次且形成显著串行工作时，才允许使用‘长串行循环’标签。",
            "若公开入口本身是 @triton.jit 且外部直接 kernel[grid](...) 调用，可将‘外部Grid固定/直接JIT入口’作为事实结构标签；不要假设优化代码能修改外部 launch site。",
            "",
            "重点从以下维度提取事实：",
            "1. 规约/量化：整行规约、局部/分组规约、GroupQuant/Block-scale、FP8/scale、尾组或其它真实规约结构；不要因为出现 tl.max/tl.sum 就强行归为整行规约。",
            "2. Grid/任务粒度：一维/二维 grid、每 Token/Group/Head 独立 program、program 内循环、海量微小 program、持久化处理；只有证据充分时才使用‘长循环/过细grid’等带程度判断的词。",
            "3. 覆盖/Mask：尾块、无效 Token、动态边界、mask 必要性；只有完整覆盖可严格证明时才标记 mask 可冗余。",
            "4. 访存/布局：连续轴、动态 stride、间接索引、random gather、重复 metadata load、不变量 load；未证明 alignment 时不要输出对齐机会。",
            "5. 算术/管线：整数地址计算、除法/取模、超越函数、重复标量计算，以及 Profiling 中 Scalar/Vector/MTE2/MTE3 的主导项。",
            "",
            profiling_context if profiling_context else "（无有效 Profiling 数据）",
            "",
            "只输出一行，使用逗号分隔，不要解释，不要 Markdown，不要输出优化动作。",
            f"```python\n{self._escape_code_for_prompt(code)}\n```",
        ]

        self._switch_model_for_purpose('analysis')
        response = self.llm.generate(
            "\n".join(prompt),
            system_msg=self.SYSTEM_ANALYST,
            purpose='analysis'
        )
        llm_tags = re.sub(r'[\r\n]+', ' ', response.text.strip()).strip()
        self._track_tokens(response)
        return llm_tags

    def analyze_kernel(
        self, seed_codes: List[str], num_directions: int, profiling_context: str = ""
    ) -> Tuple[List[str], str, str]:
        """分析内核代码，结合代码感知 RAG 生成多个互补优化方向"""
        compressed_seed = self._compress_code_for_prompt(seed_codes[0])
        seed_prompt = f"```python\n{self._escape_code_for_prompt(compressed_seed)}\n```"
        code_features_for_rag = self._get_rag_code_analysis(compressed_seed, profiling_context)

        # 新版 KnowledgeNode 会同时利用事实 Query 与真实代码结构做过滤；
        # Positive ACTION 可以为 0，此时 Planner 进入 evidence-only 模式。
        retrieved_hardware_rules = self.knowledge_node.execute(
            code_features_for_rag,
            code=compressed_seed,
        )

        # 只有 Candidate Pack 中显式的 [P1]/[P2]/... 才属于本轮真实 Positive ACTION。
        # Fallback 区域即使出现 ACTION 名称，也不能算作 RAG 正向命中。
        has_positive_action = bool(
            re.search(
                r'^\s*\[P\d+\]\s+ACTION_',
                retrieved_hardware_rules,
                flags=re.MULTILINE,
            )
        )

        positive_mode_rule = (
            "本轮存在真实 Positive ACTION：Planner 只能把 Candidate Pack 中带 [P1]/[P2]/[P3] "
            "标记的 ACTION 声明为 Positive；Fallback 和 Guard 均不属于 Positive。"
            if has_positive_action
            else
            "本轮没有真实 Positive ACTION：所有 Strategy 的 Positive ACTION 必须严格写 NONE。"
            "Fallback 区域中的 ACTION 名称仅是保守参考，不属于本轮检索命中，"
            "不得在 Strategy 中声明采用这些 ACTION。"
        )

        print(f"\n[RAG] 检索到的候选优化工具箱：")
        print("-" * 80)
        print(retrieved_hardware_rules)
        print("-" * 80)

        prompt_parts = [
            "你是华为 Ascend NPU 的顶级 Triton 编译器架构师。",
            f"【任务】结合基线代码、硬件 Profiling 与 RAG Candidate Pack，设计 {num_directions} 套高度互补、可落地、可验证的重构方案。",
            "",
            "【硬件 Profiling】",
            profiling_context if profiling_context else "（无有效 Profiling 数据）",
            "",
            "🧰【RAG Candidate Pack】",
            self._escape_code_for_prompt(retrieved_hardware_rules),
            "",
            "【RAG 使用原则】",
            f"0. 【本轮 Positive 状态】：{positive_mode_rule}",
            "1. Candidate Pack 中只有位于【高置信候选优化动作】区域、且显式带有 [P1]/[P2]/[P3] 标记的 ACTION，才属于本轮 Positive ACTION。",
            "2. Guard 是正确性/适用性约束，不属于 Positive ACTION；任何 Strategy 与 Guard 冲突时必须服从 Guard。",
            "3. 【Fallback】区域中的 ACTION 名称不属于本轮 Positive，不得因为它出现在 Candidate Pack 中就声明采用该 ACTION。",
            "4. 如果本轮没有 [P1]/[P2]/[P3]，则所有 Strategy 的 Positive ACTION 必须严格写 NONE。",
            "5. Positive=NONE 时，可以仅根据基线代码与 Profiling 独立推导出与某个已有 ACTION 类似的优化思想，但必须使用普通技术描述，不得声称该策略来自该 ACTION，也不得填写 ACTION 名称。",
            "6. 不得因为想凑策略数量而虚构 Candidate Pack 中不存在的 Positive ACTION。",
            "7. 若真实 Positive ACTION 已明确给出核心动作、必须前提或禁用条件，不得改变其核心语义；前提不成立时应放弃，而不是强行套用。",
            "",
            "🎯【组合方案原则】",
            "1. 每套 Strategy 围绕一个明确瓶颈假设组织 1-3 个彼此兼容的修改；主方向优先来自有效 Positive ACTION，没有 Positive 时使用 evidence-only。",
            "2. 不同 Strategy 应体现不同的整体路线、任务粒度、数据布局或资源权衡，而不是只修改一个参数。",
            "3. 所有 Grid、Mask、stride、alignment、bitwise、intrinsic 优化都必须先证明适用条件；证明不足则保持基线实现。",
            "4. 如果某优化只对部分 shape/mode 成立，必须保留通用 fallback。",
            "",
            "⚠️【实现合法性约束】",
            "1. 函数接口、数学语义、输出 shape/dtype/layout、功能开关和完整输入域覆盖必须与基线一致。",
            "2. 禁止 3D 及以上 tl.view；所有 tl.view/tl.arange 静态 shape 必须能够在编译期确定。",
            "3. 独立规约域在任何 packing/Grid 重构后仍必须保持独立，禁止为了性能混合不同输出的 reduction。",
            "4. 未证明完整覆盖时不得删除 mask；未证明 base/stride/dynamic offset 对齐时不得添加 tl.multiple_of 或 tl.max_contiguous。",
            "5. 位运算替代 // 或 % 必须证明除数是 2 的幂；不得从公开 testcase 硬编码 SHIFT/MASK。",
            "6. 不得使用未验证的后端 intrinsic，也不得识别函数名、固定 testcase shape 或数据值做专用答案。",
            "7. 若待优化的公开同名入口本身是 @triton.jit kernel，且由外部直接通过 kernel[grid](...) 调用，则 Strategy 必须保持这个直接 JIT 入口和外部 Launch 契约；不得设计必须新增 Python Wrapper、修改外部 Grid 维度或改变调用方式才成立的方案。",
            "8. 不得因为一个已选 Positive ACTION 顺带引入另一类高风险优化；例如位运算替代、alignment hint、Mask 删除、backend intrinsic 或未验证的 cache hint，若 Candidate Pack 中没有对应证据/Guard，则必须保持基线写法。",
            "9. Strategy 本身不得要求 `tl.load/tl.store(..., cache=...)`、`tl.load/tl.store(..., max_contiguous=...)`、JIT 内 `.bit_length()` 或 None/Tensor Mask 条件合流；这些模式会被送 NPU 前的确定性静态门禁直接拒绝。",
            "",
            "【每个 Strategy 必须写清】",
            "- 当前代码/Profiling 支持的瓶颈假设；",
            "- 实际采用的 Positive ACTION；若无则写 NONE；",
            "- 当前 kernel 上具体要怎么改；",
            "- 成立所需的 shape/stride/layout/资源前提；",
            "- 需要避免的错误实现或 fallback 条件。",
            "",
            "🎯【输出格式】",
            f"严格输出恰好 {num_directions} 行，每个方案一行：",
            "序号. 策略名称 | 现有瓶颈诊断 | 实施动作与具体代码修改方案",
            "不要输出额外解释。",
            "",
            "【待优化的基线代码】",
            seed_prompt,
        ]

        self._switch_model_for_purpose("generation")
        raw_analysis = self.llm.generate(
            "\n".join(prompt_parts),
            system_msg="你是 Ascend NPU Triton 顶级架构师。结合 Profiling 与 RAG 输出互补、具体、可验证的重构蓝图。",
            purpose="analysis",
        )
        self._track_tokens(raw_analysis)

        strategies = []
        for line in raw_analysis.text.split("\n"):
            line = line.strip()
            if line and line.count("|") >= 2:
                strategies.append(re.sub(r"^\s*\d+\.\s*", "", line))

        result = strategies[:num_directions]

        # Fallback 不引用旧版/已废弃 ACTION 名，避免在 0 Positive 时凭空制造知识。
        fallback_templates = [
            "网格与任务粒度保守优化 | 当前 program 粒度可能与单 program 工作量不匹配 | Positive ACTION=NONE；仅依据真实 Grid、循环次数和访存类型调整任务粒度，若外部 Grid 不可修改则保持 Launch 契约",
            "访存与地址计算保守优化 | Scalar/MTE 开销可能来自重复地址计算或 metadata load | Positive ACTION=NONE；仅外提可证明的不变量并简化可证明连续的寻址，不猜测 alignment/stride",
            "算术与谓词保守优化 | Scalar/Vector 指令或 mask 可能存在局部冗余 | Positive ACTION=NONE；仅做数学等价且满足误差要求的局部简化，Mask/bitwise/alignment 必须有严格证明",
            "Launch 与片上资源平衡 | BLOCK 与 program 数可能不匹配 | Positive ACTION=NONE；保持计算域、接口和外部 Launch 契约不变，小范围探索 BLOCK_SIZE/num_warps/num_stages",
        ]

        while len(result) < num_directions:
            idx = len(result)
            template = fallback_templates[idx] if idx < len(fallback_templates) else f"通用保守寻优 {idx+1} | 综合硬件瓶颈 | Positive ACTION=NONE；保持语义与边界不变，仅做有证据的局部优化"
            result.append(template)

        return result, code_features_for_rag, retrieved_hardware_rules

    def generate_initial_individual_with_strategy(
        self, baseline_code: str, strategy_str: str, rag_rules_text: str = ""
    ) -> Individual:
        """根据指定策略蓝图，从基线代码生成初始个体"""
        self._switch_model_for_purpose('generation')
        ret_count = self._get_return_count_ast(baseline_code)

        # 保留 RAG 对策略的支撑，但限制长度，避免整库规则压过当前策略。
        rag_excerpt = self._escape_code_for_prompt((rag_rules_text or "")[:5000])

        template_anchor = (
            "💡【Ascend Triton 高性能通用重构范式库（根据待优化算子类型自适应匹配）】:\n"
            "```python\n"
            "# 范式一：片上分组/块级局部规约算子（如 GroupQuant, SwiGLU-Quant）\n"
            "# x_2d = tl.view(x, (BLOCK_M * N_GROUPS, group_size))\n"
            "# 范式二：多 Head / 行打包 2D 向量化算子（如 Attention Output Correct）\n"
            "# 范式三：2D 矩阵块与 Tile 乘算算子（如 GEMM, Matmul）\n"
            "# 范式四：逐元素与直通拷贝算子（如 Elementwise Add, Relu）\n"
            "# ```"
        )

        prompt_parts = [
            "你是华为 Ascend NPU 的 Triton 高性能内核专家。",
            "【任务】请严格结合【重构策略蓝图】，对【基线代码】进行深度重构。",
            "若策略中的 Positive ACTION 为 NONE，则只执行 strategy 中已经明确写出的 evidence-only 修改，不得自行补充或虚构 RAG ACTION。",
            "",
            "🎯【必须贯彻的重构策略蓝图】",
            f"{self._escape_code_for_prompt(strategy_str)}",
            "",
            "🧰【与该策略相关的 RAG 候选规则】",
            rag_excerpt if rag_excerpt else "（无额外 RAG 规则）",
            "RAG 仅用于补充实现细节：若其触发条件在基线代码中无法证明，或与策略、接口、完整覆盖、边界安全冲突，必须忽略该条规则。",
            "",
            template_anchor,
            "上述范式仅用于结构匹配，禁止为了套用范式而强行引入 tl.view、改变规约域或混合不同输出行/组。",
            "",
            f"🔒【绝对死线 - 接口与语义契约】：",
            f"1. 外层 Wrapper 函数名、签名、参数默认值与返回值数量必须与基线一致（当前必须恰好返回 {ret_count} 个变量）。",
            "2. 输出 shape、dtype、物理布局，以及 use_ue8m0、FP8 边界、stride、有效 Token 等功能语义必须保持一致。",
            "3. 所有逐元素、分组与规约计算必须覆盖完整逻辑输入域；禁止固定 BLOCK 截断剩余数据。",
            "4. 只实施策略蓝图中的主优化方向，不要顺手叠加未经证明的 Mask 删除、连续寻址或对齐声明。",
            "5. 若基线公开同名入口本身是 @triton.jit kernel，则生成后的同名入口必须继续保持 @triton.jit；不得改成普通 Python Wrapper，也不得依赖修改外部 test/launch grid 才能实现策略。",
            "6. 保持 Python 严格缩进，直接输出 ```python 代码块。",
            "",
            f"```python\n{self._escape_code_for_prompt(self._compress_code_for_prompt(baseline_code))}\n```",
            "请直接输出重构后的代码：",
            "```python"
        ]

        metadata = {'operation': 'gen0_strategy_guided', 'applied_strategy': strategy_str}
        response = self.llm.generate("\n".join(prompt_parts), system_msg=self.SYSTEM_CODER, purpose='initial', max_tokens=16888)
        self._track_tokens(response, metadata)

        sanitized_code, syntax_error = self._sanitize_code(response.text, original_code=baseline_code)
        if syntax_error:
            metadata['syntax_error'] = syntax_error

        return Individual(code=sanitized_code, generation=0, metadata=metadata, model_used=self.llm.current_model)

    def mutate(self, individual: Individual) -> Individual:
        """对个体进行变异操作，支持超参调优、算术降级和结构重构三种变异类型"""
        if random.random() < 0.2 and hasattr(self.config, 'available_models') and len(self.config.available_models) > 1:
            other_models = [m for m in self.config.available_models if m != self.llm.current_model]
            if other_models:
                new_model = random.choice(other_models)
                self.llm.switch_model(new_model)
        else:
            self._switch_model_for_purpose('generation')

        ret_count = self._get_return_count_ast(individual.code)
        speedup = individual.metadata.get('speedup', 0.0)

        if speedup > 2.5:
            mutation_types = ['arithmetic_and_mask', 'param_tuning']
            weights = [0.70, 0.30]
        elif speedup > 1.5:
            mutation_types = ['arithmetic_and_mask', 'param_tuning', 'structure_rewrite']
            weights = [0.60, 0.30, 0.10]
        else:
            mutation_types = ['structure_rewrite', 'arithmetic_and_mask', 'param_tuning']
            weights = [0.50, 0.35, 0.15]

        code_mutation_type = random.choices(mutation_types, weights=weights, k=1)[0]
        profiling_context = individual.metadata.get('profiling_context', '')
        compressed_code = self._compress_code_for_prompt(individual.code)

        type_instructions = {
            'param_tuning': (
                "📌 【变异类型：超参与 Launch 调优】\n"
                "- 核心数学计算、Grid 映射、指针寻址和 Mask 逻辑必须保持不变。\n"
                "- 每次只调整 1-2 个参数；围绕当前配置探索后端支持的 `num_warps`（常见 1/2/4/8）、`num_stages`（常见 1-4）、BLOCK_SIZE 或 program 数上限。\n"
                "- 不得把完整规约 BLOCK 缩小到无法覆盖 n_cols/group_size。"
            ),
            'arithmetic_and_mask': (
                "📌 【变异类型：算术等价化与低风险指令精简】\n"
                "- 保持 Grid、输出元素映射与指针寻址公式不变。\n"
                "- 仅在分母非零语义、舍入误差和 dtype 行为一致时，将除法改为倒数乘法；不要强制替换所有除法。\n"
                "- Mask 只有在 constexpr、断言和索引范围严格证明冗余时才能删除。`tl.multiple_of`/`tl.max_contiguous` 只有在基址、行跨度与偏移对齐可证明时才能添加。"
            ),
            'structure_rewrite': (
                "📌 【变异类型：网格维度与分块结构重构】\n"
                "- 允许修改 Grid、program_id 解码、任务打包方式和合法的 1D/2D 分块，以减少 Launch 或串行循环。\n"
                "- 必须保持每个输出元素的唯一映射、完整 Token/列/Group 覆盖、尾块 Mask、规约域独立性和输出布局。\n"
                "- 禁止为了降维而混合不同独立规约行，禁止 3D 及以上 tl.view。"
            )
        }

        mutation_hardline = {
            'param_tuning': "保持 Kernel 计算、Grid、寻址与 Mask 完全不变，只修改 Launch/constexpr 参数。",
            'arithmetic_and_mask': "保持 Grid 与输出映射不变；仅允许有证明的数学等价和谓词/对齐简化。",
            'structure_rewrite': "允许重构 Grid 与 program_id 解码，但必须保持完整计算域、边界安全、规约域和输出布局。",
        }[code_mutation_type]

        prompt_parts = [
            "你是华为 Ascend NPU 的 Triton 高性能内核专家。",
            f"【任务】请阅读下方已跑通的 Triton 代码及其【硬件 Profiling 实测诊断】，进行【{code_mutation_type} 局部定向变异擦亮】，直接输出优化后的完整 Python 代码。\n",
            "",
            f"{profiling_context}",
            "",
            type_instructions[code_mutation_type],
            "\n🔒【本次变异硬边界】：",
            f"1. {mutation_hardline}",
            f"2. 外层 Wrapper 函数名、签名、输出 shape/dtype/layout 与 Return 数量（当前必须恰好返回 {ret_count} 个变量）必须一致。",
            "3. 只执行本次指定的一个主变异方向，不要叠加无关的 Grid、Mask、Stride、对齐或超越函数改写。",
            "4. 若某项优化前提无法从代码中证明，则保留父代写法，不得猜测。",
            f"```python\n{self._escape_code_for_prompt(compressed_code)}\n```",
            self.REFACTOR_GUIDELINES,
            "```python"
        ]

        metadata = {
            'parent': individual.id,
            'operation': 'mutation',
            'mutation_type': code_mutation_type,
            'mutation_weights': dict(zip(mutation_types, weights))
        }

        response = self.llm.generate(
            "\n".join(prompt_parts),
            system_msg=self.SYSTEM_CODER,
            purpose='mutate',
            max_tokens=8192
        )
        self._track_tokens(response, metadata)

        sanitized_code, syntax_error = self._sanitize_code(response.text, original_code=individual.code)
        if syntax_error:
            metadata['syntax_error'] = syntax_error

        return Individual(
            code=sanitized_code,
            generation=individual.generation,
            metadata=metadata,
            model_used=self.llm.current_model
        )

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """交叉操作：以加速比更高的父代为骨架，嫁接另一父代的高效指令或超参"""
        sp1 = parent1.metadata.get('speedup', parent1.fitness)
        sp2 = parent2.metadata.get('speedup', parent2.fitness)

        dominant, donor = (parent1, parent2) if sp1 >= sp2 else (parent2, parent1)
        dom_sp = dominant.metadata.get('speedup', dominant.fitness)
        don_sp = donor.metadata.get('speedup', donor.fitness)

        dom_prof = dominant.metadata.get('profiling_context', '')
        don_prof = donor.metadata.get('profiling_context', '')

        ret_count = self._get_return_count_ast(dominant.code)
        compressed_dom = self._compress_code_for_prompt(dominant.code)
        compressed_don = self._compress_code_for_prompt(donor.code)

        prompt_parts = [
            self.EXPERT_PREFIX,
            "【任务】以【主干父代】为物理骨架，从【供体父代】中识别并嫁接一个兼容、可验证的高效基因，生成更强的子代。",
            "🔒【单基因兼容性交叉约束】:",
            "1. 【主干优先】：完整继承主干父代的数学计算、Grid、指针偏移、Mask、输出布局和功能开关语义。",
            "2. 【一次只移植一个基因】：优先选择一个明确差异，例如单个 Launch 参数组合、一个经证明的算术等价式、一个兼容的加载提示；不要同时混合 Grid、寻址、Mask 和超越函数。",
            "3. 【前提验证】：只有供体基因的 shape、stride、对齐、规约域和 dtype 前提在主干中同样成立时才能移植。`tl.multiple_of`、Mask 删除和倒数代除都不能无条件复制。",
            "4. 【Profiling 使用】：主干父代的 Profiling 决定当前瓶颈；供体 Profiling 只用于解释被移植基因，不得驱动无关重写。",
            f"5. 【契约一致】：外层 Wrapper 函数名、签名、输出 shape/dtype/layout 与返回值数量必须保持一致，当前返回数量为 {ret_count}。",
            "",
            f"### 【主干父代】(ID: {dominant.id} | 加速比: {dom_sp:.4f}x)",
            f"{dom_prof}",
            f"```python\n{self._escape_code_for_prompt(compressed_dom)}\n```",
            "",
            f"### 【供体父代】(ID: {donor.id} | 加速比: {don_sp:.4f}x)",
            f"{don_prof}",
            f"```python\n{self._escape_code_for_prompt(compressed_don)}\n```",
            "",
            self.REFACTOR_GUIDELINES,
            "```python"
        ]

        metadata = {
            'parents': [parent1.id, parent2.id],
            'dominant_parent': dominant.id,
            'operation': 'crossover',
            'dominant_fitness': dominant.fitness,
            'donor_fitness': donor.fitness
        }

        response = self.llm.generate(
            "\n".join(prompt_parts),
            system_msg=self.SYSTEM_CODER,
            purpose='crossover',
            max_tokens=8192
        )
        self._track_tokens(response, metadata)

        sanitized_code, syntax_error = self._sanitize_code(response.text, original_code=dominant.code)
        if syntax_error:
            metadata['syntax_error'] = syntax_error

        return Individual(
            code=sanitized_code,
            generation=max(parent1.generation, parent2.generation),
            metadata=metadata,
            model_used=self.llm.current_model
        )

    def model_crossover(self, model1: str, model2: str) -> str:
        """模型交叉选择：从两个模型中随机选择一个作为变异或者交叉的目标模型"""
        if model1 == model2:
            return model1
        return random.choice([model1, model2])