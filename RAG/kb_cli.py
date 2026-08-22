#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG/kb_cli.py

本地 FAISS 知识库构建 / 检索调试工具。

与项目现有调用方式兼容：
    python3 RAG/kb_cli.py --build
    python3 RAG/kb_cli.py --search

主要改进：
1. 建库与查询统一使用 normalize_embeddings=True。
2. ACTION 级结构化切分，不再产生孤立 "####" 残片。
3. 为每个 ACTION 写入 action/category/confidence/is_guard/is_system_rule metadata。
4. 同名 ACTION 去重，优先保留 evidence > pattern/profile > generic。
5. --search 直接复用 production TritonKnowledgeNode，展示：
   Query 规范化 -> FAISS 广召回 -> Hybrid 重排 -> 冲突/门槛过滤
   -> Positive/Guard Candidate Pack。
"""

import argparse
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
from typing import Dict, List, Optional

os.environ["HF_HOME"] = "/workspace/user_data/Agent/RAG/.hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/user_data/Agent/RAG/.hf_cache"

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
HF_CACHE_DIR = BASE_DIR / ".hf_cache"

EMBEDDING_MODEL = os.environ.get(
    "FAISS_EMBEDDING_MODEL",
    "BAAI/bge-small-zh-v1.5",
)

ALLOWED_SUFFIXES = {
    ".txt", ".md", ".pdf", ".docx", ".doc", ".pptx", ".ppt"
}

SYSTEM_ACTIONS = {
    "ACTION_QUERY_VOCABULARY_ALIGNMENT",
    "ACTION_RETRIEVE_POSITIVE_WITH_GUARD",
}

CATEGORY_PRIORITY = {
    "evidence": 40,
    "profile_guard": 30,
    "pattern": 30,
    "generic": 10,
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_embeddings():
    """获取与 production KnowledgeNode 完全一致的 Embedding 模型。"""
    print(f"[*] 正在加载 Embedding 模型: {EMBEDDING_MODEL} ...")
    return HuggingFaceEmbeddings(
        # 不传 cache_folder：当前比赛容器已通过 HF_HOME 管理缓存，
        # 显式 cache_folder 会改变 sentence-transformers 的查找路径并在离线模式下失败。
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_single_file(file_path: Path):
    """根据文件类型选择加载器；不支持的依赖缺失时安全跳过。"""
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return TextLoader(str(file_path), encoding="utf-8")

    if suffix == ".pdf":
        return PyPDFLoader(str(file_path))

    if suffix in {".docx", ".doc"}:
        try:
            return Docx2txtLoader(str(file_path))
        except Exception:
            return UnstructuredWordDocumentLoader(str(file_path))

    if suffix in {".pptx", ".ppt"}:
        try:
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            return UnstructuredPowerPointLoader(str(file_path))
        except Exception as exc:
            print(f"  - [跳过] {file_path.name}: PowerPoint loader 不可用 ({exc})")
            return None

    return None


def classify_source(source_name: str) -> str:
    """
    给知识来源分层。
    - evidence: 当前正式评测提炼的 evidence-based tips1
    - pattern: 算子模式 recipe
    - profile_guard: profiling / safety 规则
    - generic: 其他通用 ACTION
    """
    name = source_name.lower()

    if name == "tips1.txt" or "evidence" in name:
        return "evidence"

    if "operator_pattern" in name or "pattern_recipe" in name:
        return "pattern"

    if "profiling" in name or "safety" in name or "guard" in name:
        return "profile_guard"

    return "generic"


def split_action_sections(raw_text: str) -> List[str]:
    """
    只在 ACTION 标题行起点切分。

    兼容：
        #### ⚛️ `ACTION_XXX`
        #### ACTION_XXX
        ⚛️ `ACTION_XXX`
        ACTION_XXX

    不再使用 "(?=(?:⚛️|####...))" 双触发正则，
    避免前一块尾部残留孤立的 "####"。
    """
    heading_re = re.compile(
        r"(?mi)^[ \t]*"
        r"(?=(?:#{1,6}[ \t]*)?"
        r"(?:⚛️[ \t]*)?"
        r"`?ACTION_[A-Z0-9_]+)"
    )

    starts = [m.start() for m in heading_re.finditer(raw_text)]
    if not starts:
        return []

    sections: List[str] = []

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(raw_text)
        section = raw_text[start:end].strip()

        if section and "ACTION_" in section:
            sections.append(section)

    return sections


def extract_action_name(section: str) -> Optional[str]:
    match = re.search(r"\bACTION_[A-Z0-9_]+\b", section)
    return match.group(0) if match else None


def extract_confidence(section: str) -> str:
    match = re.search(
        r"\*\*可信度\*\*[：:]\s*([^\n]+)",
        section,
        flags=re.IGNORECASE,
    )
    if not match:
        return "Unknown"

    return match.group(1).strip().rstrip("。")


def parse_action_metadata(section: str, source_name: str) -> Dict[str, object]:
    action = extract_action_name(section) or ""
    confidence = extract_confidence(section)
    category = classify_source(source_name)

    is_guard = action.startswith("ACTION_GUARD_")
    is_system_rule = action in SYSTEM_ACTIONS

    return {
        "source": source_name,
        "action": action,
        "category": category,
        "confidence": confidence,
        "is_guard": is_guard,
        "is_system_rule": is_system_rule,
    }


def _prefer_new_chunk(old_doc: Document, new_doc: Document) -> bool:
    """同名 ACTION 时决定是否用新块覆盖旧块。"""
    old_category = old_doc.metadata.get("category", "generic")
    new_category = new_doc.metadata.get("category", "generic")

    old_priority = CATEGORY_PRIORITY.get(str(old_category), 0)
    new_priority = CATEGORY_PRIORITY.get(str(new_category), 0)

    if new_priority != old_priority:
        return new_priority > old_priority

    # 同级时优先信息更完整的知识块。
    return len(new_doc.page_content) > len(old_doc.page_content)


def build_knowledge_base():
    """
    构建 FAISS 向量知识库。

    规则：
    1. 扫描 RAG/data。
    2. ACTION 级切块。
    3. System governance ACTION 保留在文本中，但不进入业务向量索引。
    4. 同名 ACTION 去重。
    5. 使用 normalize_embeddings=True 建库。
    """
    print(f"[*] 正在扫描数据目录: {DATA_DIR}")

    all_docs: List[Document] = []

    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        loader = load_single_file(path)
        if loader is None:
            continue

        try:
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = path.name

            all_docs.extend(docs)
            print(f"  - 成功加载: {path.name} (找到 {len(docs)} 页/段)")
        except Exception as exc:
            print(f"  - [错误] 加载 {path.name} 失败: {exc}")

    if not all_docs:
        print("[!] data 目录下没有找到有效文档，构建中止。")
        return

    print("[*] 正在进行 ACTION 级结构化切分...")

    action_docs: Dict[str, Document] = {}
    skipped_system = 0
    skipped_no_action = 0
    duplicate_actions = 0

    for doc in all_docs:
        raw_text = doc.page_content
        source = str(doc.metadata.get("source", "Unknown"))

        sections = split_action_sections(raw_text)

        for section in sections:
            metadata = parse_action_metadata(section, source)
            action = str(metadata.get("action", ""))

            if not action:
                skipped_no_action += 1
                continue

            # 系统治理规则是“怎么构建 RAG”的规则，不应参与算子业务召回。
            if bool(metadata.get("is_system_rule")):
                skipped_system += 1
                continue

            new_doc = Document(
                page_content=section,
                metadata=metadata,
            )

            if action not in action_docs:
                action_docs[action] = new_doc
                continue

            duplicate_actions += 1
            if _prefer_new_chunk(action_docs[action], new_doc):
                old_src = action_docs[action].metadata.get("source", "Unknown")
                action_docs[action] = new_doc
                print(
                    f"  - [去重覆盖] {action}: "
                    f"{old_src} -> {metadata.get('source')}"
                )

    chunks = list(action_docs.values())

    print(
        f"[*] 切分完成：{len(chunks)} 个独立 ACTION；"
        f"跳过 system={skipped_system}，"
        f"无 ACTION={skipped_no_action}，"
        f"同名去重={duplicate_actions}。"
    )

    if not chunks:
        print("[!] 没有可索引 ACTION，构建中止。")
        return

    category_counter: Dict[str, int] = {}
    guard_count = 0

    for chunk in chunks:
        category = str(chunk.metadata.get("category", "generic"))
        category_counter[category] = category_counter.get(category, 0) + 1
        guard_count += int(bool(chunk.metadata.get("is_guard")))

    print(f"[*] 知识层分布: {category_counter} | Guard={guard_count}")

    embeddings = get_embeddings()

    print("[*] 正在计算向量并构建 FAISS 索引...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 清理旧文件，避免误以为仍在使用旧索引。
    for stale_name in ("index.faiss", "index.pkl"):
        stale_path = KNOWLEDGE_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    vectorstore.save_local(str(KNOWLEDGE_DIR))

    print(f"[*] 知识库构建成功！已保存至 {KNOWLEDGE_DIR}")
    print("[*] 建库/查询 Embedding 均启用 normalize_embeddings=True。")


def _print_candidate(prefix: str, item: Dict[str, object]):
    print(
        f"{prefix} {item.get('action')} | "
        f"final={float(item.get('final_score', 0.0)):.4f} | "
        f"semantic={float(item.get('semantic_score', 0.0)):.4f} | "
        f"lexical={float(item.get('lexical_score', 0.0)):.4f} | "
        f"category={item.get('category')} | "
        f"source={item.get('source')}"
    )


def interactive_search():
    """
    使用 production TritonKnowledgeNode 做交互检索。

    这样 --search 测试的就是实际 Agent 会使用的检索协议，
    而不是另一套只做 raw FAISS Top-5 的逻辑。
    """
    if not (KNOWLEDGE_DIR / "index.faiss").exists():
        print("[!] 找不到 FAISS 索引，请先运行 --build。")
        return

    try:
        # 直接执行脚本时，RAG/ 会自动位于 sys.path 中。
        from knowledge_node import TritonKnowledgeNode
    except ImportError:
        from RAG.knowledge_node import TritonKnowledgeNode

    node = TritonKnowledgeNode(
        knowledge_dir=str(KNOWLEDGE_DIR),
        embedding_model=EMBEDDING_MODEL,
    )

    print("=" * 72)
    print("欢迎使用 RAG 完整检索协议测试终端 (输入 quit / exit 退出)")
    print("=" * 72)

    while True:
        query = input("\n[🔍 请输入结构特征描述] > ").strip()

        if query.lower() in {"quit", "exit", "q"}:
            print("退出测试。")
            break

        if not query:
            continue

        result = node.debug_retrieve(query)

        print("\n" + "=" * 72)
        print("[1] Query")
        print("=" * 72)
        print(f"Raw       : {result['raw_query']}")
        print(f"Normalized: {result['normalized_query']}")

        print("\n" + "=" * 72)
        print("[2] FAISS Broad Recall")
        print("=" * 72)

        for idx, item in enumerate(result["broad"], 1):
            print(
                f"{idx:>2}. {item.get('action')} | "
                f"L2={float(item.get('distance', 0.0)):.4f} | "
                f"source={item.get('source')} | "
                f"category={item.get('category')}"
            )

        print("\n" + "=" * 72)
        print("[3] Hybrid Rerank")
        print("=" * 72)

        for idx, item in enumerate(result["reranked"][:10], 1):
            _print_candidate(f"{idx:>2}.", item)

        blocked = result.get("blocked", [])
        if blocked:
            print("\n" + "=" * 72)
            print("[3.5] Positive Filter")
            print("=" * 72)

            for idx, item in enumerate(blocked[:10], 1):
                print(
                    f"{idx:>2}. BLOCK {item.get('action')} | "
                    f"{item.get('blocked_reason', 'Unknown reason')}"
                )

        print("\n" + "=" * 72)
        print("[4] Final Candidate Pack")
        print("=" * 72)

        print("\nPositive ACTION:")
        if result["positives"]:
            for idx, item in enumerate(result["positives"], 1):
                _print_candidate(f"  P{idx}.", item)
        else:
            print("  (无可靠正向 ACTION，运行期将使用保守 fallback)")

        print("\nGuard:")
        if result["guards"]:
            for idx, item in enumerate(result["guards"], 1):
                forced = " [forced]" if item.get("forced") else ""
                print(
                    f"  G{idx}. {item.get('action')}{forced} | "
                    f"source={item.get('source')}"
                )
        else:
            print("  (无额外 Guard)")

        print("\n" + "-" * 72)
        print("[最终注入文本]")
        print("-" * 72)
        print(node.format_candidate_pack(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FAISS 知识库构建 / 完整 RAG 协议调试工具"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="读取 RAG/data 构建知识库",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="启动完整检索协议测试",
    )

    args = parser.parse_args()

    if args.build:
        build_knowledge_base()
    elif args.search:
        interactive_search()
    else:
        parser.print_help()