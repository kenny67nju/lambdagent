"""
lambdagent.rag — RAG (Retrieval-Augmented Generation) 工具

内置向量检索，让 Agent 能从知识库中获取上下文。

Lambda 语义:
    RAGTool(store, k)     = Tool("rag", λx. retrieve(store, x, k))
    AgenticRAG(agent, rag) = λx. let ctx = rag(x) in agent(x + ctx)
    RAGStore               = 向量化的文档集合

不引入新的 Lambda 构造 — RAG 是 Tool 的特化版本。

设计原则:
    1. 零外部依赖的基础实现（TF-IDF / BM25 近似）
    2. 可选接入 ChromaDB / FAISS（如果已安装）
    3. Agentic RAG: Agent 自行决定何时检索（Route 模式）

支持的后端:
    - "simple":  内置 TF-IDF 余弦相似度（零依赖）
    - "chroma":  ChromaDB 向量数据库（需要 pip install chromadb）
"""

from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .core import Term, Context, LambdagentError


# ════════════════════════════════════════════════════════════
# 文档数据结构
# ════════════════════════════════════════════════════════════


@dataclass
class Document:
    """一个可检索的文档"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class SearchResult:
    """检索结果"""

    document: Document
    score: float
    rank: int = 0


# ════════════════════════════════════════════════════════════
# SimpleVectorStore: 零依赖的 TF-IDF 检索
# ════════════════════════════════════════════════════════════


class SimpleVectorStore:
    """
    基于 TF-IDF 余弦相似度的向量存储。

    零外部依赖，适合轻量级 RAG。
    对于生产环境，建议用 ChromaDB 后端。
    """

    def __init__(self):
        self.documents: List[Document] = []
        self._tfidf_cache: Optional[Dict] = None

    def add(self, content: str, metadata: Optional[Dict] = None) -> str:
        """添加文档"""
        doc = Document(content=content, metadata=metadata or {})
        self.documents.append(doc)
        self._tfidf_cache = None  # 清除缓存
        return doc.doc_id

    def add_many(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """批量添加"""
        ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            ids.append(self.add(text, meta))
        return ids

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """TF-IDF 余弦相似度检索"""
        if not self.documents:
            return []

        # 构建或使用缓存的 TF-IDF
        if self._tfidf_cache is None:
            self._build_tfidf()

        query_vec = self._text_to_tfidf(query)
        results = []

        for i, doc in enumerate(self.documents):
            doc_vec = self._tfidf_cache["vectors"][i]
            score = self._cosine_similarity(query_vec, doc_vec)
            results.append(SearchResult(document=doc, score=score))

        results.sort(key=lambda r: -r.score)
        for i, r in enumerate(results[:top_k]):
            r.rank = i + 1

        return results[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（英文空格 + 中文字符）"""
        # 英文: 小写 + 按非字母分割
        text = text.lower()
        tokens = re.findall(r"[a-z]+|[\u4e00-\u9fff]", text)
        # 过滤停用词
        stops = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "but",
            "not",
            "it",
            "this",
            "that",
            "with",
            "from",
            "by",
            "as",
            "be",
            "has",
            "have",
            "had",
            "do",
            "does",
        }
        return [t for t in tokens if t not in stops and len(t) > 1]

    def _build_tfidf(self):
        """构建 TF-IDF 索引"""
        doc_tokens = [self._tokenize(d.content) for d in self.documents]
        n_docs = len(doc_tokens)

        # IDF
        df = Counter()
        for tokens in doc_tokens:
            for token in set(tokens):
                df[token] += 1

        idf = {}
        for token, count in df.items():
            idf[token] = math.log(n_docs / (1 + count)) + 1

        # TF-IDF 向量
        vectors = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            vec = {}
            for token, count in tf.items():
                vec[token] = (count / max(1, len(tokens))) * idf.get(token, 1.0)
            vectors.append(vec)

        self._tfidf_cache = {"idf": idf, "vectors": vectors}

    def _text_to_tfidf(self, text: str) -> Dict[str, float]:
        """将文本转为 TF-IDF 向量"""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        idf = self._tfidf_cache["idf"]
        vec = {}
        for token, count in tf.items():
            vec[token] = (count / max(1, len(tokens))) * idf.get(token, 1.0)
        return vec

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        """稀疏向量余弦相似度"""
        common = set(a.keys()) & set(b.keys())
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __len__(self):
        return len(self.documents)

    def __repr__(self):
        return f"SimpleVectorStore({len(self.documents)} docs)"


# ════════════════════════════════════════════════════════════
# ChromaStore: ChromaDB 后端（可选）
# ════════════════════════════════════════════════════════════


class ChromaStore:
    """
    ChromaDB 向量存储后端。

    需要: pip install chromadb

    用法:
        store = ChromaStore("my_collection")
        store.add("文档内容")
        results = store.search("查询", top_k=3)
    """

    def __init__(
        self, collection_name: str = "default", persist_directory: Optional[str] = None
    ):
        try:
            import chromadb
        except ImportError:
            raise LambdagentError(
                "ChromaDB not installed. Run: pip install chromadb\n"
                "Or use SimpleVectorStore (zero dependencies)."
            )

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, content: str, metadata: Optional[Dict] = None) -> str:
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        self._collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
        return doc_id

    def add_many(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        ids = [hashlib.md5(t.encode()).hexdigest()[:12] for t in texts]
        self._collection.add(
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            ids=ids,
        )
        return ids

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        search_results = []
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for i, (doc_text, dist, meta, doc_id) in enumerate(
            zip(docs, distances, metadatas, ids)
        ):
            search_results.append(
                SearchResult(
                    document=Document(content=doc_text, metadata=meta, doc_id=doc_id),
                    score=1.0 / (1.0 + dist),  # 距离转相似度
                    rank=i + 1,
                )
            )
        return search_results

    def __len__(self):
        return self._collection.count()

    def __repr__(self):
        return f"ChromaStore({self._collection.name!r}, {len(self)} docs)"


# ════════════════════════════════════════════════════════════
# RAGTool: 检索工具 (lambdagent Term)
# ════════════════════════════════════════════════════════════


class RAGTool(Term):
    """
    RAG 检索工具 — 从知识库中检索相关文档。

    Lambda 语义:
        RAGTool(store, k) = Tool("rag", λx. retrieve(store, x, k))
        = tool[f] where f(x) = top_k_documents(store, x)

    输出格式: 检索到的文档片段，用 [Source N] 标注。

    用法:
        store = SimpleVectorStore()
        store.add("Python 是一种编程语言...")
        store.add("Lambda 演算是...")

        rag = RAGTool(store, top_k=3)
        context = rag("什么是 Lambda 演算？")
        # → "[Source 1] Lambda 演算是..."
    """

    def __init__(
        self, store, top_k: int = 3, min_score: float = 0.0, format: str = "numbered"
    ):
        """
        Args:
            store: 向量存储（SimpleVectorStore 或 ChromaStore）
            top_k: 返回前 k 个结果
            min_score: 最低相似度阈值
            format: 输出格式 "numbered" | "plain" | "json"
        """
        super().__init__("RAG")
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.format = format

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """检索 = β-规约"""
        ctx = ctx or Context()
        import time

        t0 = time.time()

        results = self.store.search(str(input), top_k=self.top_k)
        results = [r for r in results if r.score >= self.min_score]

        output = self._format_results(results)
        elapsed = (time.time() - t0) * 1000
        ctx.log(
            "RAG", self._trace_id, str(input)[:100], f"{len(results)} results", elapsed
        )
        return output

    def _format_results(self, results: List[SearchResult]) -> str:
        if not results:
            return "[No relevant documents found]"

        if self.format == "plain":
            return "\n\n".join(r.document.content for r in results)
        elif self.format == "json":
            import json

            return json.dumps(
                [
                    {
                        "rank": r.rank,
                        "score": round(r.score, 4),
                        "content": r.document.content,
                        "metadata": r.document.metadata,
                    }
                    for r in results
                ],
                ensure_ascii=False,
                indent=2,
            )
        else:  # numbered
            parts = []
            for r in results:
                meta = ""
                if r.document.metadata:
                    meta = f" ({', '.join(f'{k}={v}' for k, v in r.document.metadata.items())})"
                parts.append(
                    f"[Source {r.rank}, score={r.score:.3f}{meta}]\n{r.document.content}"
                )
            return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════
# AgenticRAG: Agent 自行决定何时检索
# ════════════════════════════════════════════════════════════


class AgenticRAG(Term):
    """
    Agentic RAG — Agent 自行决定何时检索。

    Lambda 语义:
        AgenticRAG(agent, rag, decider) =
            λx. let need_rag = decider(x) in
                 IF need_rag
                    THEN agent(x + rag(x))     ← 带检索
                    ELSE agent(x)               ← 不检索

    与普通 RAG 的区别:
        普通 RAG: 每次都检索（浪费 token）
        Agentic RAG: 只在需要时检索（Agent 自主决定）

    用法:
        agentic = AgenticRAG(
            agent=my_llm,
            rag=RAGTool(store),
            decider=lambda x: "?" in x,  # 简单规则: 有问号就检索
        )
    """

    def __init__(
        self, agent: Term, rag: RAGTool, decider=None, always_retrieve: bool = False
    ):
        """
        Args:
            agent: 主 Agent
            rag: RAG 检索工具
            decider: 决策函数/Agent (input → bool)，None 时总是检索
            always_retrieve: True 时忽略 decider，总是检索
        """
        super().__init__(f"AgenticRAG({agent._name})")
        self.agent = agent
        self.rag = rag
        self.decider = decider
        self.always_retrieve = always_retrieve

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        import time

        t0 = time.time()

        input_str = str(input)
        need_rag = self.always_retrieve

        if not need_rag and self.decider is not None:
            if isinstance(self.decider, Term):
                decision = self.decider.apply(input_str, ctx)
                need_rag = (
                    bool(decision)
                    if not isinstance(decision, str)
                    else any(
                        w in str(decision).lower()
                        for w in ["yes", "true", "retrieve", "search"]
                    )
                )
            elif callable(self.decider):
                need_rag = bool(self.decider(input_str))
        elif self.decider is None:
            need_rag = True  # 无 decider 时默认检索

        if need_rag:
            # 检索 + 增强
            context = self.rag.apply(input_str, ctx)
            augmented = f"[Retrieved Context]\n{context}\n\n[Question]\n{input_str}"
            result = self.agent.apply(augmented, ctx)
        else:
            result = self.agent.apply(input_str, ctx)

        elapsed = (time.time() - t0) * 1000
        ctx.log(
            f"AgenticRAG({'with_rag' if need_rag else 'no_rag'})",
            self._trace_id,
            input_str[:100],
            str(result)[:100],
            elapsed,
        )
        return result


# ════════════════════════════════════════════════════════════
# 便利函数
# ════════════════════════════════════════════════════════════


def create_rag(
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
    top_k: int = 3,
    backend: str = "simple",
    **kwargs,
) -> RAGTool:
    """
    一行创建 RAG 工具。

    用法:
        rag = create_rag([
            "Python 是一种编程语言",
            "Lambda 演算是计算理论的基础",
            "AI Agent 是自主执行任务的系统",
        ])
        result = rag("什么是 Lambda?")
    """
    if backend == "simple":
        store = SimpleVectorStore()
    elif backend == "chroma":
        store = ChromaStore(**kwargs)
    else:
        raise LambdagentError(f"Unknown RAG backend: {backend}")

    store.add_many(documents, metadatas)
    return RAGTool(store, top_k=top_k)
