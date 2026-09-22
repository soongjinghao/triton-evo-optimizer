#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG/knowledge_node.py

Triton Knowledge Node
Evidence-aware + Guard-aware Hybrid RAG Retriever

保持现有项目接口：
    node = TritonKnowledgeNode(...)
    rules = node.execute(llm_analysis_query, code=kernel_code)

检索协议：
    LLM 瓶颈短语
      -> Query 规范化
      -> FAISS Top-K 广召回
      -> Semantic + Anchor lexical + knowledge tier 重排
      -> 显式冲突过滤 + Positive 分段门槛
      -> Positive ACTION Top-N
      -> Guard 向量命中 + 确定性强制联动
      -> Candidate Pack
"""

import os

# ============================================================
# 限制 CPU BLAS / OpenMP 线程数
# 必须在 numpy / torch / transformers / faiss / langchain 导入前设置
# 该项目知识库规模很小，单线程更稳定，也避免影响算子评测。
# ============================================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ["HF_HOME"] = "/workspace/user_data/Agent/RAG/.hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/user_data/Agent/RAG/.hf_cache"

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None


BASE_DIR = Path(__file__).resolve().parent

SYSTEM_ACTIONS = {
    "ACTION_QUERY_VOCABULARY_ALIGNMENT",
    "ACTION_RETRIEVE_POSITIVE_WITH_GUARD",
}

# Query 原词不会被删除；这些只是补充 canonical vocabulary。
QUERY_ALIASES = {
    # 规约 / 量化
    "分组量化": "GroupQuant",
    "组量化": "GroupQuant",
    "group quant": "GroupQuant",
    "小group": "group_size较小",
    "小组": "group_size较小",
    "block scale": "Block-scale",
    "全行规约": "行级全宽规约",
    "整行规约": "行级全宽规约",
    "全宽规约": "行级全宽规约",

    # Grid / 粒度
    "很多小program": "海量微小program",
    "大量小program": "海量微小program",
    "微小program": "海量微小program",
    "program太碎": "海量微小program",
    "一个program干太多活": "单program工作量过大",
    "单program太重": "单program工作量过大",
    "长串行循环": "单program长串行循环",
    "grid太细": "二维grid过细",
    "grid太小": "grid覆盖不足",

    # 访存 / layout
    "随机访存": "random gather",
    "随机gather": "random gather",
    "间接访问": "间接索引",
    "间接寻址": "间接索引",
    "重复读weight": "weight重复加载",
    "重复读取weight": "weight重复加载",
    "重复读scale": "scale重复加载",
    "stride连续性未知": "动态stride连续性未证明",
    "连续性无法证明": "动态stride连续性未证明",
    "连续性未证明": "动态stride连续性未证明",
    "对齐未知": "alignment未证明",
    "对齐无法证明": "alignment未证明",

    # Mask
    "尾部mask": "尾块mask",
    "尾mask": "尾块mask",
    "mask无法证明": "mask证明不足",

    # Profiling
    "mte读瓶颈": "MTE2读访存主瓶颈",
    "mte2瓶颈": "MTE2读访存主瓶颈",
    "mte2读瓶颈": "MTE2读访存主瓶颈",
    "读访存瓶颈": "MTE2读访存主瓶颈",
    "mte写瓶颈": "MTE3写访存主瓶颈",
    "mte3瓶颈": "MTE3写访存主瓶颈",
    "vector瓶颈": "Vector向量主瓶颈",
    "vector主瓶颈": "Vector向量主瓶颈",
    "scalar瓶颈": "Scalar标量主瓶颈",
    "scalar主瓶颈": "Scalar标量主瓶颈",

    # 算术
    "小k": "小K矩阵累加",
    "小k矩阵": "小K矩阵累加",
    "倒平方根": "sqrt/rsqrt",
    "除法瓶颈": "向量除法",

    # V2: Launch / Atomic / Alias / Value-dependent Mask
    "grid重映射": "pid重映射",
    "pid映射": "pid重映射",
    "启动契约": "wrapper launch",
    "launch contract": "wrapper launch",
    "program id": "program_id",
    "原子累加": "atomic_add",
    "原子加": "atomic_add",
    "原子规约": "reduction并行化",
    "原地拷贝": "in-place copy",
    "原地复制": "in-place copy",
    "原地更新": "cache update",
    "地址别名": "src/tgt alias",
    "值相关mask": "value-dependent mask",
    "数据相关mask": "value-dependent mask",
    "非零mask": "nonzero mask",
}

# Query / code 命中这些特征时，即使 Guard 没排进向量 Top-K，也强制补充。
GUARD_TRIGGERS = {
    "ACTION_GUARD_ALIGNMENT_PROOF": (
        "alignment未证明",
        "动态stride连续性未证明",
        "tl.multiple_of",
        "tl.max_contiguous",
        "动态base",
    ),
    "ACTION_GUARD_DTYPE_VIEW_STRIDE": (
        "多dtype view",
        "fp8/fp32/bf16混合",
        "subview",
        "scale区域",
        "rope区域",
    ),
    "ACTION_GUARD_NO_OVERFETCH": (
        "num_valid小于capacity",
        "capacity",
        "over-fetch",
        "无效元素over-fetch",
        "block_table_stride",
    ),
    "ACTION_GUARD_POWER_OF_TWO_BITWISE": (
        "整数除法取模",
        "动态page size",
        "bitwise",
        "位运算",
        "page_size",
    ),
    "ACTION_GUARD_RANDOM_GATHER_OCCUPANCY": (
        "random gather",
        "间接索引",
        "单program工作量过大",
        "occupancy不足",
    ),
    "ACTION_GUARD_BACKEND_INTRINSIC": (
        "tl.math intrinsic",
        "backend支持",
        "reciprocal intrinsic",
        "exp2/log2",
        "runtime error",
    ),
    "ACTION_GUARD_TINY_KERNEL_AUTOTUNE": (
        "极短内核",
        "profiling计数器未捕获",
        "autotune多配置",
    ),
    "ACTION_GUARD_MASK_COVERAGE_PROOF": (
        "尾块mask",
        "mask证明不足",
        "mask可证明冗余",
        "非2次幂逻辑组",
        "无效token",
        "full tile",
    ),
    # V2: hidden-test failures exposed launch / atomic / alias contracts.
    "ACTION_GUARD_LAUNCH_CONTRACT": (
        "program_id",
        "grid重映射",
        "pid重映射",
        "grid expand",
        "grid flatten",
        "grid维度",
        "二维grid",
        "三维grid",
        "wrapper launch",
        "direct jit launch",
        "external launch",
    ),
    "ACTION_GUARD_ALIAS_INPLACE_SEMANTICS": (
        "in-place copy",
        "cache relocation",
        "cache update",
        "src/tgt alias",
        "原地写",
        "kv cache",
    ),
    "ACTION_GUARD_ATOMIC_REDUCTION_EQUIVALENCE": (
        "atomic_add",
        "atomic_max",
        "histogram",
        "多program累加",
        "reduction并行化",
        "scatter reduction",
    ),
}

# 选中高风险 Positive 后联动 Guard。
ACTION_GUARD_BINDINGS = {
    "ACTION_ALIGNED_MASKLESS_FASTPATH": {
        "ACTION_GUARD_MASK_COVERAGE_PROOF",
        "ACTION_GUARD_ALIGNMENT_PROOF",
    },
    "ACTION_MASK_REMOVAL": {
        "ACTION_GUARD_MASK_COVERAGE_PROOF",
    },
    "ACTION_ALIGNMENT_HINTING": {
        "ACTION_GUARD_ALIGNMENT_PROOF",
    },
    "ACTION_CONTIGUOUS_INNER_STRIDE_STRIP": {
        "ACTION_GUARD_ALIGNMENT_PROOF",
    },
    "ACTION_STRIDE_STRIPPING": {
        "ACTION_GUARD_ALIGNMENT_PROOF",
    },
    "ACTION_BITWISE_ADDRESSING": {
        "ACTION_GUARD_POWER_OF_TWO_BITWISE",
    },
    "ACTION_RSQRT_FUSION": {
        "ACTION_GUARD_BACKEND_INTRINSIC",
    },
    "ACTION_TRANSCENDENTAL_APPROX": {
        "ACTION_GUARD_BACKEND_INTRINSIC",
    },

    # V2: only add bindings for newly strengthened knowledge cards.
    "ACTION_GRID_CONSOLIDATE_TINY_PROGRAMS": {
        "ACTION_GUARD_GRID_DIRECTION_CONTEXTUAL",
        "ACTION_GUARD_LAUNCH_CONTRACT",
    },
    "ACTION_GRID_EXPAND_LONG_SERIAL_LOOP": {
        "ACTION_GUARD_GRID_DIRECTION_CONTEXTUAL",
        "ACTION_GUARD_LAUNCH_CONTRACT",
    },
    "ACTION_INDEPENDENT_AXIS_PACKING": {
        "ACTION_GUARD_GRID_DIRECTION_CONTEXTUAL",
        "ACTION_GUARD_LAUNCH_CONTRACT",
        "ACTION_GUARD_MASK_COVERAGE_PROOF",
    },
    "ACTION_GROUP_PACK_RESHAPE_REDUCE": {
        "ACTION_GUARD_MASK_COVERAGE_PROOF",
    },
    "ACTION_VALUE_DEPENDENT_MASK_ELISION": {
        "ACTION_GUARD_MASK_COVERAGE_PROOF",
    },
    "ACTION_RECIPROCAL_MUL_WHEN_SAFE": {
        "ACTION_GUARD_BACKEND_INTRINSIC",
    },
}

CATEGORY_BONUS = {
    "evidence": 0.06,
    "pattern": 0.04,
    "profile_guard": 0.03,
    "generic": 0.00,
}


# ---------------------------------------------------------------------------
# Positive ACTION 冲突规则
#
# 这些不是 Guard，而是“当前 Query 已经明确否定了该 ACTION 的必要前提”。
# 命中后该 ACTION 不进入 Positive Candidate Pack，但仍可留在 Broad/Rerank
# 调试结果中，便于观察向量召回是否存在语义混淆。
# ---------------------------------------------------------------------------
ACTION_CONFLICT_RULES = {
    # random gather / 单 program 已经过重时，不再继续收拢更多工作。
    "ACTION_GRID_CONSOLIDATE_TINY_PROGRAMS": (
        "random gather",
        "单program工作量过大",
        "occupancy不足",
        # 已明确属于“少量重 Program”时，不应再向 consolidation 方向推荐。
        "单program长串行循环",
        "grid覆盖不足",
        "program数量偏少",
    ),

    # 连续性尚未证明时不能把 stride stripping 当成正向建议。
    "ACTION_CONTIGUOUS_INNER_STRIDE_STRIP": (
        "动态stride连续性未证明",
    ),
    "ACTION_STRIDE_STRIPPING": (
        "动态stride连续性未证明",
    ),

    # Mask 完整覆盖无法证明时，不推荐无 mask fast path。
    "ACTION_ALIGNED_MASKLESS_FASTPATH": (
        "mask证明不足",
    ),
    "ACTION_MASK_REMOVAL": (
        "mask证明不足",
    ),

    # 已经是海量微小 program 时，不继续扩展 grid。
    "ACTION_GRID_EXPAND_LONG_SERIAL_LOOP": (
        "海量微小program",
        "二维grid过细",
    ),

    # 单 program 已经过重 / occupancy 不足时，不再做进一步 axis packing。
    "ACTION_INDEPENDENT_AXIS_PACKING": (
        "单program工作量过大",
        "occupancy不足",
    ),
    "ACTION_GROUP_PACK_RESHAPE_REDUCE": (
        "单program工作量过大",
        "occupancy不足",
    ),
}


class TritonKnowledgeNode:
    """
    大模型抽象语义感知 RAG 检索节点。

    Production 检索不再是单纯 Top-5 L2：
      1. Query 词表归一化；
      2. FAISS Top-K 广召回；
      3. 检索锚点 lexical + normalized L2 semantic + knowledge tier 重排；
      4. Positive 显式冲突过滤；
      5. Positive 分段置信门槛；
      6. Positive / Guard 分离；
      7. Guard deterministic override。
    """

    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        broad_top_k: int = 10,
        positive_top_k: int = 3,
        guard_top_k: int = 3,
        retrieval_mode: str = "hybrid",   # hybrid=混合重排+Guard | semantic=普通语义Top-k
    ):
        self.knowledge_dir = self._resolve_knowledge_dir(knowledge_dir)
        self.embedding_model = embedding_model
        self.vectorstore = None

        self.broad_top_k = max(5, int(broad_top_k))
        self.positive_top_k = max(1, int(positive_top_k))
        self.guard_top_k = max(1, int(guard_top_k))
        self.retrieval_mode = retrieval_mode

        self.action_docs: Dict[str, object] = {}

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_OFFLINE"] = "1"

        self._find_local_model_path()
        self._load_vectorstore()

    @staticmethod
    def _resolve_knowledge_dir(knowledge_dir: str) -> Path:
        path = Path(knowledge_dir)

        if path.is_absolute():
            return path

        # 与 RAG/kb_cli.py 的 BASE_DIR / "knowledge" 保持一致。
        if str(path) in {"knowledge", "./knowledge"}:
            return BASE_DIR / "knowledge"

        # 如果调用方显式传了 RAG/knowledge，优先尊重 cwd 下真实存在路径。
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return cwd_candidate

        # 否则相对本模块解析。
        module_candidate = BASE_DIR / path
        if module_candidate.exists():
            return module_candidate

        return cwd_candidate

    def _find_local_model_path(self):
        """查找项目 RAG/.hf_cache 中的本地 snapshot。"""
        hf_cache = self.knowledge_dir.parent / ".hf_cache" / "hub"

        if not hf_cache.exists():
            return

        model_dir_name = f"models--{self.embedding_model.replace('/', '--')}"
        model_dirs = list(hf_cache.glob(model_dir_name))

        if not model_dirs:
            return

        snapshots_dir = model_dirs[0] / "snapshots"

        if not snapshots_dir.exists():
            return

        snapshot_dirs = sorted(
            [p for p in snapshots_dir.iterdir() if p.is_dir()]
        )

        if snapshot_dirs:
            local_path = snapshot_dirs[-1]
            print(f"[ℹ️ KnowledgeNode] 发现本地模型: {local_path}")
            self.embedding_model = str(local_path)

    def _load_vectorstore(self):
        """加载本地 FAISS，并建立 ACTION -> Document 快速索引。"""
        if FAISS is None or HuggingFaceEmbeddings is None:
            print(
                "[⚠️ KnowledgeNode] LangChain/FAISS 未安装，"
                "节点将退化为默认保守规则。"
            )
            return

        if not (self.knowledge_dir / "index.faiss").exists():
            print(
                f"[⚠️ KnowledgeNode] 未在 {self.knowledge_dir} 找到索引，"
                "节点将使用默认保守规则。"
            )
            return

        try:
            embeddings = HuggingFaceEmbeddings(
                # 不显式传 cache_folder，统一交给 HF_HOME / 本地 snapshot 解析。
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            self.vectorstore = FAISS.load_local(
                str(self.knowledge_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )

            self._build_action_lookup()

            print(
                "[✅ KnowledgeNode] 泛化 RAG 本地专家知识库成功挂载！"
                f" ACTION={len(self.action_docs)}"
            )
        except Exception as exc:
            print(f"[❌ KnowledgeNode] 加载向量库失败: {exc}")
            self.vectorstore = None
            self.action_docs = {}

    def _build_action_lookup(self):
        """从 FAISS docstore 建立 action -> Document 映射，用于 Guard Override。"""
        self.action_docs = {}

        if self.vectorstore is None:
            return

        try:
            doc_ids = self.vectorstore.index_to_docstore_id.values()

            for doc_id in doc_ids:
                doc = self.vectorstore.docstore.search(doc_id)

                if doc is None:
                    continue

                action = str(doc.metadata.get("action", "")).strip()

                if not action:
                    action = self._extract_action(doc.page_content)

                if action:
                    self.action_docs[action] = doc
        except Exception as exc:
            print(f"[⚠️ KnowledgeNode] ACTION lookup 构建失败: {exc}")

    @staticmethod
    def _extract_action(text: str) -> str:
        match = re.search(r"\bACTION_[A-Z0-9_]+\b", text or "")
        return match.group(0) if match else ""

    @staticmethod
    def _split_query_tags(query: str) -> List[str]:
        return [
            part.strip()
            for part in re.split(r"[,，;；|\n]+", query or "")
            if part.strip()
        ]

    @staticmethod
    def _text_key(text: str) -> str:
        """用于 lexical matching 的轻量规范化。"""
        return re.sub(
            r"[\s`*_:/（）()\[\]<>]+",
            "",
            (text or "").casefold(),
        )

    def _normalize_query(self, query: str) -> Tuple[str, List[str]]:
        """
        保留原始事实短语，同时补充受控词表 canonical tags。
        """
        raw_tags = self._split_query_tags(query)
        tags: List[str] = []
        seen: Set[str] = set()

        def add_tag(tag: str):
            cleaned = tag.strip()
            if not cleaned:
                return

            key = cleaned.casefold()
            if key in seen:
                return

            seen.add(key)
            tags.append(cleaned)

        for raw_tag in raw_tags:
            add_tag(raw_tag)
            raw_key = raw_tag.casefold()

            for alias, canonical in QUERY_ALIASES.items():
                if alias.casefold() in raw_key:
                    add_tag(canonical)

        normalized = ", ".join(tags)
        return normalized, tags

    @staticmethod
    def _extract_anchor_text(content: str) -> str:
        """
        lexical score 优先只匹配“检索锚点”，避免：
        实测证据/失败证据中的弱相关词把一个 ACTION 错误抬高。
        """
        match = re.search(
            r"\*\*检索锚点\*\*[：:]\s*([^\n]+)",
            content or "",
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

        # 旧知识卡没有“检索锚点”时退化为标题 + 前若干字符。
        return (content or "")[:800]

    @staticmethod
    def _distance_to_semantic(distance: float) -> float:
        """
        normalize_embeddings=True + FAISS L2 时：
            squared_l2 = 2 - 2*cosine
            cosine = 1 - squared_l2/2

        LangChain FAISS 返回的是 L2 distance score（越小越近）。
        这里映射到 [0,1] 便于与 lexical score 混合。
        """
        score = 1.0 - float(distance) / 2.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _confidence_bonus(confidence: str) -> float:
        value = (confidence or "").upper()

        if "A" in value:
            return 0.04

        if "B" in value:
            return 0.02

        return 0.0

    def _lexical_score(
        self,
        query_tags: Sequence[str],
        doc_content: str,
    ) -> float:
        if not query_tags:
            return 0.0

        anchor_text = self._extract_anchor_text(doc_content)
        anchor_key = self._text_key(anchor_text)

        hits = 0

        for tag in query_tags:
            tag_key = self._text_key(tag)

            if not tag_key:
                continue

            if tag_key in anchor_key:
                hits += 1

        return hits / max(len(query_tags), 1)

    def _rerank(
        self,
        docs_and_scores,
        query_tags: Sequence[str],
    ) -> List[Dict[str, object]]:
        """
        Hybrid score:
            0.62 semantic
          + 0.30 lexical anchor overlap
          + source tier bonus
          + confidence bonus

        System rule 永久过滤。
        """
        ranked: List[Dict[str, object]] = []
        seen_actions: Set[str] = set()

        for doc, distance in docs_and_scores:
            action = str(doc.metadata.get("action", "")).strip()

            if not action:
                action = self._extract_action(doc.page_content)

            if not action:
                continue

            if action in SYSTEM_ACTIONS:
                continue

            if bool(doc.metadata.get("is_system_rule", False)):
                continue

            if action in seen_actions:
                continue

            seen_actions.add(action)

            semantic = self._distance_to_semantic(float(distance))
            lexical = self._lexical_score(query_tags, doc.page_content)

            category = str(doc.metadata.get("category", "generic"))
            confidence = str(doc.metadata.get("confidence", "Unknown"))

            if self.retrieval_mode == "semantic":
                # 普通语义 Top-k：仅按向量语义相似度排序，不引入词法/类别重排
                final = semantic
            else:
                final = (
                    0.62 * semantic
                    + 0.30 * lexical
                    + CATEGORY_BONUS.get(category, 0.0)
                    + self._confidence_bonus(confidence)
                )

            ranked.append(
                {
                    "doc": doc,
                    "action": action,
                    "distance": float(distance),
                    "semantic_score": semantic,
                    "lexical_score": lexical,
                    "final_score": final,
                    "category": category,
                    "confidence": confidence,
                    "source": doc.metadata.get("source", "Unknown"),
                    "is_guard": bool(
                        doc.metadata.get("is_guard", False)
                        or action.startswith("ACTION_GUARD_")
                    ),
                    "forced": False,
                }
            )

        ranked.sort(
            key=lambda item: float(item["final_score"]),
            reverse=True,
        )

        return ranked

    def _action_doc_to_item(
        self,
        action: str,
        forced: bool = True,
    ) -> Optional[Dict[str, object]]:
        doc = self.action_docs.get(action)

        if doc is None:
            return None

        return {
            "doc": doc,
            "action": action,
            "distance": 999.0,
            "semantic_score": 0.0,
            "lexical_score": 0.0,
            "final_score": 0.0,
            "category": doc.metadata.get("category", "generic"),
            "confidence": doc.metadata.get("confidence", "Unknown"),
            "source": doc.metadata.get("source", "Unknown"),
            "is_guard": True,
            "forced": forced,
        }

    def _forced_guard_actions(
        self,
        normalized_query: str,
        code: Optional[str],
        positives: Sequence[Dict[str, object]],
    ) -> Set[str]:
        if self.retrieval_mode == "semantic":
            return set()   # 普通语义 Top-k 不带 Guard 确定性联动
        forced: Set[str] = set()
        query_key = self._text_key(normalized_query)
        code_text = (code or "").casefold()

        # 1) Query-driven deterministic Guard.
        for action, triggers in GUARD_TRIGGERS.items():
            for trigger in triggers:
                if self._text_key(trigger) in query_key:
                    forced.add(action)
                    break

        # 2) 少量明确 code pattern，避免依赖 LLM 恰好说出 Guard 词。
        if "tl.multiple_of" in code_text or "tl.max_contiguous" in code_text:
            forced.add("ACTION_GUARD_ALIGNMENT_PROOF")

        if (
            "tl.math.rcp" in code_text
            or "tl.math.rsqrt" in code_text
            or "tl.math.exp2" in code_text
            or "tl.math.log2" in code_text
        ):
            forced.add("ACTION_GUARD_BACKEND_INTRINSIC")

        if ">>" in code_text or "<<" in code_text:
            forced.add("ACTION_GUARD_POWER_OF_TWO_BITWISE")

        # V2: atomic / launch contract are high-risk enough for deterministic guards.
        if re.search(r"tl\.atomic_(?:add|max|min|cas)\s*\(", code_text):
            forced.add("ACTION_GUARD_ATOMIC_REDUCTION_EQUIVALENCE")

        if re.search(
            r"tl\.program_id\s*\(\s*(?:axis\s*=\s*)?2\s*\)",
            code_text,
        ):
            forced.add("ACTION_GUARD_LAUNCH_CONTRACT")

        if (
            "in-place" in code_text
            or "inplace" in code_text
            or "cache relocation" in code_text
        ):
            forced.add("ACTION_GUARD_ALIAS_INPLACE_SEMANTICS")

        # 3) Positive -> Guard binding。
        for item in positives:
            action = str(item.get("action", ""))
            forced.update(ACTION_GUARD_BINDINGS.get(action, set()))

        # Random gather 情况下，如果选中了 consolidate，再强制 occupancy Guard。
        selected_actions = {
            str(item.get("action", ""))
            for item in positives
        }

        if (
            "ACTION_GRID_CONSOLIDATE_TINY_PROGRAMS" in selected_actions
            and (
                "randomgather" in query_key
                or self._text_key("间接索引") in query_key
            )
        ):
            forced.add("ACTION_GUARD_RANDOM_GATHER_OCCUPANCY")

        return forced

    def _conflict_reasons(
        self,
        action: str,
        normalized_query: str,
    ) -> List[str]:
        """
        检查 Query 是否已经明确否定某个 ACTION 的必要前提。

        注意：
        - 这里只处理“显式冲突”，不根据缺失信息做负向推理。
        - 例如 Query 没说 stride 连续，不等于 stride 一定不连续；
          只有出现“动态stride连续性未证明”才阻止 STRIDE_STRIP。
        """
        reasons: List[str] = []
        query_key = self._text_key(normalized_query)

        for phrase in ACTION_CONFLICT_RULES.get(action, ()):
            if self._text_key(phrase) in query_key:
                reasons.append(phrase)

        return reasons

    @staticmethod
    def _positive_threshold_pass(
        lexical: float,
        semantic: float,
    ) -> bool:
        """
        Positive 的分段门槛。

        目的：
        1. 完全没有 anchor 命中时，必须有非常强的语义相似度；
        2. 只有一个弱 anchor 命中时，仍要求较强 semantic；
        3. 多个明确 anchor 命中时可以适度放宽 semantic。

        对当前 3~8 个短语 Query：
        - lexical == 0        : semantic >= 0.74
        - 0 < lexical < .25   : semantic >= 0.70
        - .25 <= lexical < .5 : semantic >= 0.64
        - lexical >= .5       : semantic >= 0.50
        """
        if lexical <= 0.0:
            return semantic >= 0.74

        if lexical < 0.25:
            return semantic >= 0.70

        if lexical < 0.50:
            return semantic >= 0.64

        return semantic >= 0.50

    def _select_positives(
        self,
        reranked: Sequence[Dict[str, object]],
        normalized_query: str,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        """
        选择最终 Positive，并返回被冲突/门槛过滤的候选，供调试终端展示。
        """
        positives: List[Dict[str, object]] = []
        blocked: List[Dict[str, object]] = []

        for item in reranked:
            if item.get("is_guard"):
                continue

            action = str(item.get("action", ""))

            conflict_reasons = self._conflict_reasons(
                action,
                normalized_query,
            )

            if conflict_reasons:
                blocked_item = dict(item)
                blocked_item["blocked_reason"] = (
                    "显式冲突: " + ", ".join(conflict_reasons)
                )
                blocked.append(blocked_item)
                continue

            lexical = float(item.get("lexical_score", 0.0))
            semantic = float(item.get("semantic_score", 0.0))

            if not self._positive_threshold_pass(
                lexical=lexical,
                semantic=semantic,
            ):
                blocked_item = dict(item)
                blocked_item["blocked_reason"] = (
                    "Positive门槛不足: "
                    f"lexical={lexical:.4f}, semantic={semantic:.4f}"
                )
                blocked.append(blocked_item)
                continue

            positives.append(item)

            if len(positives) >= self.positive_top_k:
                break

        return positives, blocked

    def _select_guards(
        self,
        reranked: Sequence[Dict[str, object]],
        normalized_query: str,
        code: Optional[str],
        positives: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        selected: Dict[str, Dict[str, object]] = {}

        # 先保留向量/Hybrid 已经命中的 Guard。
        for item in reranked:
            if not item.get("is_guard"):
                continue

            action = str(item.get("action", ""))

            if action and action not in selected:
                selected[action] = dict(item)

            if len(selected) >= self.guard_top_k:
                break

        # 再用确定性规则覆盖，forced Guard 优先级更高。
        forced_actions = self._forced_guard_actions(
            normalized_query,
            code,
            positives,
        )

        for action in forced_actions:
            item = self._action_doc_to_item(action, forced=True)

            if item is not None:
                selected[action] = item

        # V2: forced Guard 是确定性安全约束，不能再被 guard_top_k 截断。
        guards = list(selected.values())
        forced_guards = [item for item in guards if item.get("forced")]
        recalled_guards = [item for item in guards if not item.get("forced")]

        # Set 产生的 forced_actions 无稳定顺序，这里按 ACTION 名排序保证可复现。
        forced_guards.sort(key=lambda item: str(item.get("action", "")))
        recalled_guards.sort(
            key=lambda item: float(item.get("final_score", 0.0)),
            reverse=True,
        )

        # guard_top_k 只限制普通向量召回 Guard；forced Guard 全部保留。
        remaining_slots = max(0, self.guard_top_k - len(forced_guards))
        return forced_guards + recalled_guards[:remaining_slots]

    def debug_retrieve(
        self,
        llm_analysis_query: str,
        code: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        返回完整检索中间状态，供 kb_cli.py --search 和日志验证。
        """
        raw_query = (llm_analysis_query or "").strip()
        normalized_query, query_tags = self._normalize_query(raw_query)

        empty_result = {
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "query_tags": query_tags,
            "broad": [],
            "reranked": [],
            "positives": [],
            "blocked": [],
            "guards": [],
        }

        if self.vectorstore is None or not normalized_query:
            return empty_result

        k = self.broad_top_k

        try:
            if hasattr(self.vectorstore, "index"):
                total = int(self.vectorstore.index.ntotal)
                if total > 0:
                    k = min(k, total)

            docs_and_scores = self.vectorstore.similarity_search_with_score(
                normalized_query,
                k=k,
            )
        except Exception as exc:
            print(f"[❌ KnowledgeNode] FAISS 检索失败: {exc}")
            return empty_result

        broad = []

        for doc, distance in docs_and_scores:
            action = str(doc.metadata.get("action", "")).strip()
            if not action:
                action = self._extract_action(doc.page_content)

            if action in SYSTEM_ACTIONS:
                continue

            broad.append(
                {
                    "doc": doc,
                    "action": action,
                    "distance": float(distance),
                    "source": doc.metadata.get("source", "Unknown"),
                    "category": doc.metadata.get("category", "generic"),
                }
            )

        reranked = self._rerank(docs_and_scores, query_tags)
        positives, blocked = self._select_positives(
            reranked,
            normalized_query,
        )

        guards = self._select_guards(
            reranked,
            normalized_query,
            code,
            positives,
        )

        return {
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "query_tags": query_tags,
            "broad": broad,
            "reranked": reranked,
            "positives": positives,
            "blocked": blocked,
            "guards": guards,
        }

    def format_candidate_pack(
        self,
        result: Dict[str, object],
    ) -> str:
        """
        格式化为后续 LLM strategy/coder 可直接使用的知识包。
        """
        positives = list(result.get("positives", []))
        guards = list(result.get("guards", []))
        normalized_query = str(result.get("normalized_query", ""))

        sections: List[str] = []

        if normalized_query:
            sections.append(
                "【RAG Query】\n"
                f"{normalized_query}"
            )

        if positives:
            positive_blocks = []

            for idx, item in enumerate(positives, 1):
                doc = item["doc"]
                positive_blocks.append(
                    f"[P{idx}] {item['action']}\n"
                    f"{doc.page_content}"
                )

            sections.append(
                "【高置信候选优化动作】\n\n"
                + "\n\n".join(positive_blocks)
            )

        if guards:
            guard_blocks = []

            for idx, item in enumerate(guards, 1):
                doc = item["doc"]
                forced = "（强制联动）" if item.get("forced") else ""

                guard_blocks.append(
                    f"[G{idx}] {item['action']}{forced}\n"
                    f"{doc.page_content}"
                )

            sections.append(
                "【必须同时检查的安全 Guard】\n"
                "Guard 是正确性/适用条件约束，不是普通性能建议；"
                "若与激进优化冲突，以 Guard 为准。\n\n"
                + "\n\n".join(guard_blocks)
            )

        if not positives:
            sections.append(self._get_default_rules())

        return "\n\n".join(sections)

    def execute(
        self,
        llm_analysis_query: str,
        code: str = None,
    ) -> str:
        """
        执行完整 RAG 检索并返回 Candidate Pack。

        保持原项目 execute(query, code=None) 接口不变。
        """
        if self.vectorstore is None or not llm_analysis_query:
            return self._get_default_rules()

        print(
            "[🔍 RAG Retriever] "
            f"Raw Query: '{llm_analysis_query}'"
        )

        result = self.debug_retrieve(
            llm_analysis_query,
            code=code,
        )

        print(
            "[🔍 RAG Retriever] "
            f"Normalized: '{result['normalized_query']}'"
        )

        positives = result["positives"]
        blocked = result.get("blocked", [])
        guards = result["guards"]

        if blocked:
            conflict_blocked = [
                item for item in blocked
                if str(item.get("blocked_reason", "")).startswith("显式冲突")
            ]
            if conflict_blocked:
                print(
                    "[⛔ RAG Conflict Filter] "
                    + "; ".join(
                        f"{item['action']} -> {item['blocked_reason']}"
                        for item in conflict_blocked[:5]
                    )
                )

        if positives:
            print(
                "[✅ RAG Positive] "
                + ", ".join(
                    str(item["action"])
                    for item in positives
                )
            )
        else:
            print("[⚠️ RAG Positive] 无可靠正向 ACTION，使用保守 fallback。")

        if guards:
            print(
                "[🛡️ RAG Guard] "
                + ", ".join(
                    (
                        str(item["action"])
                        + ("*" if item.get("forced") else "")
                    )
                    for item in guards
                )
            )

        return self.format_candidate_pack(result)

    def _get_default_rules(self) -> str:
        """
        RAG 未可靠命中时使用的保守底线。
        不再默认“删 mask / 强制 reciprocal / 固定 warp”。
        """
        return """
【Fallback：仅在 RAG 无可靠正向命中时使用】

1. ACTION_INVARIANT_LOAD_HOIST
   检查循环内是否存在与迭代无关的 weight / scale / metadata；
   只有能证明为循环不变量时才外提并片上复用。

2. ACTION_GUARD_GRID_DIRECTION_CONTEXTUAL
   根据 program 数量、单 program 工作量、访存类型和片上资源，
   决定扩大 Grid 还是收拢 Grid；禁止固定方向地 flatten/consolidate。

3. ACTION_TILE_AREA_BUDGET
   调整 BLOCK/二维 tile 时同时评估寄存器/UB 占用与 occupancy；
   禁止单纯追求更大的 BLOCK。

【强制正确性底线】
- 不能证明完整覆盖时必须保留 mask；
- 不能证明连续性/对齐时禁止添加 tl.multiple_of 等真实性承诺；
- random gather / 间接索引不假设连续对齐；
- // 与 % 只有 divisor 已证明为 2 的幂时才允许改写为 bitwise；
- 未经当前 Ascend Triton backend testcase 验证的 intrinsic 不直接使用；
- 数学语义、Wrapper/API 契约、完整输入域与 Accuracy 优先于性能。
""".strip()