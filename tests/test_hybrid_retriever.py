"""ToolHybridRetriever 的单元测试。

覆盖四个关键行为：
1. RRF 融合让两路共识的工具排在第一
2. scene 过滤生效
3. 单路失败时另一路结果保留
4. 空 query 返回空列表

用假的向量检索器替身，不依赖向量库和 embedding 模型。
"""

import pytest
from langchain_core.documents import Document
from langchain_core.tools import tool

from middleware.hybrid_retriever import ToolHybridRetriever, ToolProfile

pytestmark = pytest.mark.asyncio


# ---------- 测试替身 ----------

def make_tool(name: str, description: str, scene: str = "default"):
    @tool(name_or_callable=name, description=description, extras={"scene": scene})
    async def _impl(x: str = "") -> str:
        return f"{name}:{x}"

    return _impl


class FakeVectorRetriever:
    """假向量检索器。返回预置文档或抛预置异常。"""

    def __init__(self, docs=None, error=None):
        self.docs = docs or []
        self.error = error

    async def ainvoke(self, query):
        if self.error is not None:
            raise self.error
        return list(self.docs)


def doc_for(name: str) -> Document:
    return Document(page_content=name, metadata={"tool_name": name})


# ---------- 测试用例 ----------

async def test_rrf_prefers_consensus():
    """两路都命中的工具应排在只有单路命中的工具之前。"""
    t_refund = make_tool("refund_order", "退款订单")
    t_cancel = make_tool("cancel_order", "取消订单")
    t_weather = make_tool("weather_query", "天气查询")

    profiles = [ToolProfile.from_tool(t) for t in [t_refund, t_cancel, t_weather]]

    # 向量只命中 refund_order
    retriever = FakeVectorRetriever(docs=[doc_for("refund_order")])

    hybrid = ToolHybridRetriever(profiles=profiles, vector_retriever=retriever)
    results = await hybrid.search("refund order", k=3)

    names = [t.name for t, _ in results]
    assert names[0] == "refund_order"
    assert "cancel_order" in names
    assert "weather_query" not in names


async def test_scene_filter_excludes_other_scenes():
    """scene 参数应只返回匹配的工具。"""
    t_cn = make_tool("weather_cn", "中国天气", scene="weather")
    t_us = make_tool("weather_us", "美国天气", scene="weather_us")

    profiles = [ToolProfile.from_tool(t) for t in [t_cn, t_us]]
    retriever = FakeVectorRetriever(
        docs=[doc_for("weather_cn"), doc_for("weather_us")]
    )

    hybrid = ToolHybridRetriever(profiles=profiles, vector_retriever=retriever)
    results = await hybrid.search("weather", scene="weather", k=10)

    names = [t.name for t, _ in results]
    assert "weather_cn" in names
    assert "weather_us" not in names


async def test_vector_failure_falls_back_to_bm25():
    """向量一路抛异常时，BM25 的结果应保留。"""
    t_refund = make_tool("refund_order", "退款订单")
    profiles = [ToolProfile.from_tool(t_refund)]

    retriever = FakeVectorRetriever(error=RuntimeError("vector service down"))

    hybrid = ToolHybridRetriever(profiles=profiles, vector_retriever=retriever)
    results = await hybrid.search("refund", k=5)

    names = [t.name for t, _ in results]
    assert "refund_order" in names


async def test_bm25_failure_falls_back_to_vector():
    """BM25 一路抛异常时，向量的结果应保留。"""
    t_refund = make_tool("refund_order", "退款订单")
    profiles = [ToolProfile.from_tool(t_refund)]

    retriever = FakeVectorRetriever(docs=[doc_for("refund_order")])
    hybrid = ToolHybridRetriever(profiles=profiles, vector_retriever=retriever)

    # 直接破坏 BM25 索引模拟失败
    hybrid._bm25 = None  # type: ignore[assignment]

    results = await hybrid.search("refund", k=5)
    names = [t.name for t, _ in results]
    assert "refund_order" in names


async def test_empty_query_returns_empty():
    """空 query 不应触发任何召回。"""
    t = make_tool("x", "y")
    profiles = [ToolProfile.from_tool(t)]
    hybrid = ToolHybridRetriever(
        profiles=profiles, vector_retriever=FakeVectorRetriever()
    )

    for q in ["", "   ", "\t\n"]:
        results = await hybrid.search(q, k=5)
        assert results == []