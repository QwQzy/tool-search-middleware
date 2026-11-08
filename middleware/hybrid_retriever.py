"""工具检索的混合召回器。

包含 ToolProfile 的定义与构建，以及 BM25 + 向量的混合召回与 RRF 融合。
不依赖 LangChain 的 EnsembleRetriever，因为它的 RRF 常数 k=60 硬编码，
且权重在 RRF 下的行为在不同版本中不一致。

依赖：
    pip install rank-bm25

BM25 使用 rank_bm25 的 BM25Okapi 直接构建，不经过 LangChain 的
BM25Retriever。原因是需要在单次检索中按 scene 做元数据过滤，而
BM25Retriever 的公共 API 不保证支持动态过滤。直接持有文档列表和
BM25 索引，可以在评分前完成过滤。

已知限制：
    BM25 路径的 scene 过滤在评分之后执行。这意味着被隔离的工具仍然
    参与了 BM25 打分，只是结果被丢弃。正确的做法是在索引层做过滤，
    但 rank_bm25 没有倒排索引的概念。对一万量级的工具库，后过滤的
    计算开销可以接受；如果工具规模继续增长，应该换用支持过滤的
    BM25 实现（如 Whoosh 或 Elasticsearch）。

    向量路径的 scene 过滤通过底层 VectorStore 的 filter 参数传入，
    是否真正在召回阶段生效取决于具体的 VectorStore 实现。FAISS 等
    内存向量库不支持 filter，此时同样退化为后过滤。部署时应优先选择
    支持元数据过滤的向量库（Chroma、Qdrant、pgvector 等）。
"""
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ============ 工具检索档案 ============

@dataclass(frozen=True)
class ToolProfile:
    """工具检索档案。不可变，构建后不应修改。

    把检索所需的字段从 BaseTool 上剥离出来，让检索层不依赖工具对象的
    完整结构。

    字段：
    - tool：原始 BaseTool 对象，最终返回给调用方
    - name：工具名，作为去重和反查的键
    - scene：场景标签，供硬过滤使用
    - retrieval_text：送入 BM25 和向量的文本
    - keywords：关键词列表，供调试和后续扩展使用
    """

    tool: BaseTool
    name: str
    scene: str
    retrieval_text: str
    keywords: tuple[str, ...]

    @classmethod
    def from_tool(cls, tool: BaseTool) -> "ToolProfile":
        """从 BaseTool 提取检索档案。

        extras 中的字段由工具注册方负责填充。缺失的字段用保守默认值，
        不抛异常——工具可以没有 scene，但不能因为缺 scene 就无法被检索。
        注册阶段的必填字段校验由 ToolSearchMiddleware 负责，不在这一层。
        """
        extras = getattr(tool, "extras", None) or {}

        scene = extras.get("scene") or "default"
        raw_keywords = extras.get("keywords")
        if raw_keywords:
            keywords = tuple(str(k) for k in raw_keywords)
        else:
            keywords = (tool.name,)

        custom_text = extras.get("retrieval_text")
        if custom_text:
            retrieval_text = str(custom_text)
        else:
            retrieval_text = _build_default_text(
                name=tool.name,
                description=tool.description or "",
                keywords=keywords,
            )

        return cls(
            tool=tool,
            name=tool.name,
            scene=str(scene),
            retrieval_text=retrieval_text,
            keywords=keywords,
        )


def build_profiles(tools: list[BaseTool]) -> list[ToolProfile]:
    """批量构建档案。ToolSearchMiddleware 在 __init__ 中调用它。"""
    return [ToolProfile.from_tool(t) for t in tools]


def _build_default_text(
    name: str, description: str, keywords: tuple[str, ...]
) -> str:
    """拼接默认检索文本。

    顺序按重要性排：名称在前，描述在后，关键词在最后。BM25 对词频敏感，
    名称重复出现一次会增加它的权重，这对工具检索是有利的——名称通常是
    最精确的信号。
    """
    parts = [name, description]
    if keywords:
        parts.append(" ".join(keywords))
    return " ".join(p for p in parts if p)


# ============ 重排序协议 ============

@runtime_checkable
class Reranker(Protocol):
    """重排序器的协议。实现者只需提供 async rerank 方法。

    candidates 是 (tool, fused_score) 列表，按 RRF 融合分数降序。
    返回值应该是重排后的 (tool, score) 列表，长度不超过 top_k。
    """

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[BaseTool, float]],
        top_k: int,
    ) -> list[tuple[BaseTool, float]]:
        ...


# ============ 内部候选结构 ============

@dataclass(frozen=True)
class _Candidate:
    """一路召回产生的候选。"""

    tool: BaseTool
    rank: int          # 在该路召回中的排名，从 1 开始
    raw_score: float   # 原始分数，BM25 分数或向量相似度，仅用于调试


# ============ 混合召回器 ============

class ToolHybridRetriever:
    """工具检索的混合召回器。

    构造时接收工具档案和向量检索器，在 __init__ 中构建 BM25 索引。
    search 方法并行执行 BM25 和向量召回，按 RRF 融合，可选重排序。
    """

    def __init__(
        self,
        profiles: list[ToolProfile],
        vector_retriever: BaseRetriever,
        *,
        k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 0.7,
        recall_k: int = 30,
        reranker: Reranker | None = None,
        rerank_threshold: float = 0.0,
    ) -> None:
        if not profiles:
            raise ValueError("ToolHybridRetriever 至少需要一个 ToolProfile")
        if k <= 0:
            raise ValueError(f"RRF 常数 k 必须为正数，实际 {k}")
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("召回权重不能为负数")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("至少一路召回的权重必须大于 0")
        if recall_k < 1:
            raise ValueError(f"recall_k 必须 >= 1，实际 {recall_k}")

        self._profiles = profiles
        self._by_name: dict[str, ToolProfile] = {p.tool.name: p for p in profiles}
        self._vector_retriever = vector_retriever
        self._k = k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._recall_k = recall_k
        self._reranker = reranker
        self._rerank_threshold = rerank_threshold

        # BM25 索引：对每个工具的检索文本做分词
        self._bm25_docs: list[ToolProfile] = list(profiles)
        self._bm25_corpus: list[list[str]] = [
            _tokenize(p.retrieval_text) for p in self._bm25_docs
        ]
        self._bm25 = BM25Okapi(self._bm25_corpus)

    # ---------- 公开接口 ----------

    async def search(
        self,
        query: str,
        *,
        scene: str | None = None,
        k: int = 10,
    ) -> list[tuple[BaseTool, float]]:
        """在工具库上检索，返回 (tool, fused_score) 列表，按分数降序。

        scene 不为 None 时，只返回 scene 匹配的工具。过滤在召回层各自
        执行：向量路径尝试传入 filter，BM25 路径在评分后过滤。
        """
        if not query.strip():
            return []

        bm25_task = asyncio.create_task(self._bm25_recall(query, scene))
        vector_task = asyncio.create_task(self._vector_recall(query, scene))

        results = await asyncio.gather(
            bm25_task, vector_task, return_exceptions=True
        )

        bm25_candidates: list[_Candidate] = []
        vector_candidates: list[_Candidate] = []

        for label, result in zip(("bm25", "vector"), results):
            if isinstance(result, BaseException):
                logger.warning("%s 召回失败，降级为单路结果: %s", label, result)
                continue
            if label == "bm25":
                bm25_candidates = result
            else:
                vector_candidates = result

        if not bm25_candidates and not vector_candidates:
            return []

        fused = self._rrf_fuse(bm25_candidates, vector_candidates)

        if self._reranker is not None and self._should_rerank(fused):
            try:
                reranked = await self._reranker.rerank(query, fused, k)
                if reranked:
                    return reranked[:k]
            except Exception:
                logger.exception("重排序失败，回退到 RRF 融合结果")

        return fused[:k]

    # ---------- BM25 召回 ----------

    async def _bm25_recall(
        self, query: str, scene: str | None
    ) -> list[_Candidate]:
        """BM25 召回。评分是同步 CPU 计算，放到线程池。"""
        return await asyncio.to_thread(self._bm25_recall_sync, query, scene)

    def _bm25_recall_sync(
        self, query: str, scene: str | None
    ) -> list[_Candidate]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        ranked = []
        for profile, score in zip(self._bm25_docs, scores):
            if score <= 0:
                continue
            if scene is not None and profile.scene != scene:
                continue
            ranked.append((profile, float(score)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [
            _Candidate(tool=profile.tool, rank=i + 1, raw_score=score)
            for i, (profile, score) in enumerate(ranked[: self._recall_k])
        ]

    # ---------- 向量召回 ----------

    async def _vector_recall(
        self, query: str, scene: str | None
    ) -> list[_Candidate]:
        """向量召回。

        优先走底层 VectorStore 的 asimilarity_search_with_score，以便传入
        filter。如果 vector_retriever 没有暴露 vectorstore 属性，回退到
        ainvoke 并在结果上后过滤。
        """
        vectorstore = getattr(self._vector_retriever, "vectorstore", None)

        if vectorstore is not None and hasattr(
            vectorstore, "asimilarity_search_with_score"
        ):
            filter_arg = _build_scene_filter(vectorstore, scene)
            try:
                pairs = await vectorstore.asimilarity_search_with_score(
                    query,
                    k=self._recall_k,
                    filter=filter_arg,
                )
            except TypeError:
                # 某些 VectorStore 不接受 filter 参数
                pairs = await vectorstore.asimilarity_search_with_score(
                    query, k=self._recall_k
                )
            candidates = []
            for i, (doc, score) in enumerate(pairs):
                tool = self._tool_from_doc(doc)
                if tool is None:
                    continue
                if scene is not None and self._scene_of(tool) != scene:
                    continue
                candidates.append(
                    _Candidate(tool=tool, rank=i + 1, raw_score=float(score))
                )
            return candidates[: self._recall_k]

        # 回退路径：用 retriever 的 ainvoke，结果后过滤
        docs = await self._vector_retriever.ainvoke(query)
        candidates = []
        for i, doc in enumerate(docs):
            tool = self._tool_from_doc(doc)
            if tool is None:
                continue
            if scene is not None and self._scene_of(tool) != scene:
                continue
            candidates.append(
                _Candidate(
                    tool=tool,
                    rank=i + 1,
                    raw_score=float(doc.metadata.get("score", 0.0)),
                )
            )
        return candidates[: self._recall_k]

    # ---------- RRF 融合 ----------

    def _rrf_fuse(
        self,
        bm25: list[_Candidate],
        vector: list[_Candidate],
    ) -> list[tuple[BaseTool, float]]:
        scores: dict[str, float] = {}
        tools: dict[str, BaseTool] = {}

        for candidates, weight in (
            (bm25, self._bm25_weight),
            (vector, self._vector_weight),
        ):
            if weight == 0:
                continue
            for c in candidates:
                name = c.tool.name
                tools[name] = c.tool
                scores[name] = scores.get(name, 0.0) + weight / (
                    self._k + c.rank
                )

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(tools[name], score) for name, score in ordered]

    # ---------- 重排序触发 ----------

    def _should_rerank(self, fused: list[tuple[BaseTool, float]]) -> bool:
        if not fused:
            return False
        if self._rerank_threshold <= 0:
            return True
        return fused[0][1] < self._rerank_threshold

    # ---------- 辅助 ----------

    def _tool_from_doc(self, doc: Document) -> BaseTool | None:
        name = doc.metadata.get("tool_name") or doc.metadata.get("id")
        if not name:
            return None
        profile = self._by_name.get(str(name))
        return profile.tool if profile else None

    def _scene_of(self, tool: BaseTool) -> str | None:
        profile = self._by_name.get(tool.name)
        return profile.scene if profile else None


# ============ 模块级辅助 ============

def _tokenize(text: str) -> list[str]:
    """轻量分词。生产环境如果需要中文分词，替换为 jieba 或类似实现。

    这里的默认实现按空白和非字母数字字符切分，对英文工具名和描述足够；
    中文会被逐字切分，BM25 仍能工作，但效果不如专业分词器。
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    return tokens


def _build_scene_filter(vectorstore: Any, scene: str | None) -> Any:
    """根据 VectorStore 的类型构造 filter 参数。

    Chroma 使用 {"field": {"$eq": value}}；Qdrant 使用 Payload 过滤；
    不同实现差异很大。这里只处理最常见的 Chroma 风格，其余返回 None，
    由调用方在结果上后过滤。
    """
    if scene is None:
        return None
    store_name = type(vectorstore).__name__.lower()
    if "chroma" in store_name:
        return {"scene": {"$eq": scene}}
    return None