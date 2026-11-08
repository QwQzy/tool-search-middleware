"""ToolSearchMiddleware 的集成测试。

覆盖两个关键行为：
1. awrap_model_call：根据历史中的 search_tools 调用，把命中工具加入 visible
2. awrap_tool_call：从中间件注册表取出业务工具并执行

不依赖真实 LLM。ModelRequest 和 ToolCallRequest 用 MagicMock 模拟，
只设置中间件代码实际访问的属性。
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from middleware.search_middleware import ToolSearchMiddleware

pytestmark = pytest.mark.asyncio


# ---------- 测试替身 ----------

def make_tool(
    name: str,
    description: str,
    scene: str = "default",
    read_only: bool = True,
):
    @tool(
        name_or_callable=name,
        description=description,
        extras={"scene": scene, "readOnlyHint": read_only},
    )
    async def _impl(x: str = "") -> str:
        return f"{name}:{x}"

    return _impl


class FakeVectorRetriever:
    def __init__(self, hits=None):
        self.hits = hits or []

    async def ainvoke(self, query):
        from langchain_core.documents import Document

        return [
            Document(page_content=h, metadata={"tool_name": h}) for h in self.hits
        ]


def make_model_request(tools, messages):
    """构造 awrap_model_call 可用的最小 request。"""
    req = MagicMock()
    req.tools = list(tools)
    req.state = {"messages": list(messages)}

    def override(**kwargs):
        new = MagicMock()
        new.tools = kwargs.get("tools", req.tools)
        return new

    req.override = override
    return req


def make_tool_call_request(name, args, call_id="call_1"):
    req = MagicMock()
    req.tool_call = {"name": name, "args": args, "id": call_id}
    req.tool = None
    req.state = {}
    return req


# ---------- awrap_model_call ----------

async def test_no_history_only_core_visible():
    """没有 search_tools 历史时，只有 search_tools 和核心工具可见。"""
    biz = make_tool("refund_order", "退款订单")
    core = make_tool("calculator", "计算器")

    mw = ToolSearchMiddleware(
        all_tools=[biz, core],
        vector_retriever=FakeVectorRetriever(),
        core_tool_names={"calculator"},
    )

    request = make_model_request(
        tools=[mw.tools[0], core],
        messages=[HumanMessage(content="你好")],
    )

    captured = {}

    async def handler(req):
        captured["visible"] = [t.name for t in req.tools]
        return MagicMock()

    await mw.awrap_model_call(request, handler)

    assert "search_tools" in captured["visible"]
    assert "calculator" in captured["visible"]
    assert "refund_order" not in captured["visible"]


async def test_search_history_discloses_matched_tool():
    """历史中有 search_tools 调用时，命中的业务工具应加入可见集合。"""
    biz = make_tool("refund_order", "退款订单")
    core = make_tool("calculator", "计算器")

    mw = ToolSearchMiddleware(
        all_tools=[biz, core],
        vector_retriever=FakeVectorRetriever(hits=["refund_order"]),
        core_tool_names={"calculator"},
    )

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_tools", "args": {"query": "refund"}, "id": "call_1"}
        ],
    )
    request = make_model_request(
        tools=[mw.tools[0], core],
        messages=[HumanMessage(content="我要退款"), ai_msg],
    )

    captured = {}

    async def handler(req):
        captured["visible"] = [t.name for t in req.tools]
        return MagicMock()

    await mw.awrap_model_call(request, handler)

    assert "refund_order" in captured["visible"]
    assert "search_tools" in captured["visible"]


async def test_search_history_below_threshold_not_disclosed():
    """检索分数低于阈值时，业务工具不应被披露。"""
    biz = make_tool("refund_order", "退款订单")

    mw = ToolSearchMiddleware(
        all_tools=[biz],
        vector_retriever=FakeVectorRetriever(hits=["refund_order"]),
        score_threshold=999.0,
    )

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_tools", "args": {"query": "refund"}, "id": "call_1"}
        ],
    )
    request = make_model_request(
        tools=[mw.tools[0]],
        messages=[HumanMessage(content="我要退款"), ai_msg],
    )

    captured = {}

    async def handler(req):
        captured["visible"] = [t.name for t in req.tools]
        return MagicMock()

    await mw.awrap_model_call(request, handler)

    assert "refund_order" not in captured["visible"]


# ---------- awrap_tool_call ----------

async def test_awrap_tool_call_executes_business_tool():
    """业务工具调用应由中间件从注册表取出并执行。"""
    biz = make_tool("refund_order", "退款订单", read_only=False)

    mw = ToolSearchMiddleware(
        all_tools=[biz],
        vector_retriever=FakeVectorRetriever(),
    )

    request = make_tool_call_request("refund_order", {"x": "abc"})

    async def handler(req):
        raise AssertionError("业务工具不应走框架默认 handler")

    result = await mw.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert result.content == "refund_order:abc"


async def test_awrap_tool_call_unknown_tool():
    """未注册的工具名返回 protocol 错误。"""
    mw = ToolSearchMiddleware(
        all_tools=[make_tool("known", "已知工具")],
        vector_retriever=FakeVectorRetriever(),
    )

    request = make_tool_call_request("unknown", {})

    async def handler(req):
        raise AssertionError("未知工具不应走 handler")

    result = await mw.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "unknown" in result.content


async def test_awrap_tool_call_passes_through_search_tools():
    """search_tools 自身应走 handler。"""
    mw = ToolSearchMiddleware(
        all_tools=[make_tool("x", "y")],
        vector_retriever=FakeVectorRetriever(),
    )

    request = make_tool_call_request("search_tools", {"query": "y"})

    called = {}

    async def handler(req):
        called["hit"] = True
        return ToolMessage(
            content="ok", tool_call_id="call_1", name="search_tools"
        )

    result = await mw.awrap_tool_call(request, handler)

    assert called.get("hit")
    assert isinstance(result, ToolMessage)