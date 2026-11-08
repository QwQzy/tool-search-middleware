"""ToolSearchMiddleware 实现。

实现两个 wrap 方法：

- awrap_model_call：在模型调用前决定它这一轮看见哪些工具。
- awrap_tool_call：接管中间件持有的业务工具的执行。业务工具不注册到
  create_agent(tools=...)，模型通过 search_tools 发现它们、通过
  awrap_model_call 把它们披露出去，执行时由这个中间件从注册表里取出
  实例并调用。

可见性规则：
1. search_tools 自身永远可见，它是发现机制。
2. core_tool_names 中的核心工具永远可见（基础工具、calculator 等）。
3. 业务工具只有在被 search_tools 命中过之后才可见。

search_tools 的返回值：
返回命中工具的完整 schema（名称、描述、参数 schema、分数），模型可以
直接从中判断是否调用。完整 schema 在 awrap_model_call 把工具加入可见
集合后不再重复披露，两者不重叠。

历史扫描策略：
每次 awrap_model_call 都重新扫描消息历史中的 search_tools 调用，
重新执行检索，把命中的工具加入可见集合。这是最直接的做法，
对典型对话（1-3 次搜索）的开销可以接受。如果对话很长且搜索频繁，
可以把"已发现工具"存入 agent state 以避免重复检索。

鉴权与写操作确认不在这里：当前版本没有接入 ExecGatewayMiddleware，
awrap_tool_call 只做"找到工具实例并执行"。接入网关后，这个中间件
应放在网关的内层，由网关在外层做鉴权、幂等和错误分类。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool, ToolRuntime
from pydantic import BaseModel, Field

from .hybrid_retriever import Reranker, ToolHybridRetriever, build_profiles

logger = logging.getLogger(__name__)


class SearchToolsInput(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Search query. Use capability keywords (e.g. 'weather', "
            "'refund order') or specific tool names."
        ),
    )


class ToolSearchMiddleware(AgentMiddleware):
    """在模型调用前决定可见工具集、在执行前接管工具调用的中间件。"""

    def __init__(
        self,
        all_tools: list[BaseTool],
        vector_retriever: BaseRetriever,
        *,
        core_tool_names: set[str] | None = None,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 0.7,
        recall_k: int = 30,
        reranker: Reranker | None = None,
        rerank_threshold: float = 0.0,
        score_threshold: float = 0.0,
    ) -> None:
        if not all_tools:
            raise ValueError("ToolSearchMiddleware 至少需要一个业务工具")

        self._core: set[str] = set(core_tool_names or set())
        self._core.add("search_tools")

        self._all_tools: list[BaseTool] = list(all_tools)
        self._by_name: dict[str, BaseTool] = {t.name: t for t in all_tools}

        profiles = build_profiles(all_tools)
        self._retriever = ToolHybridRetriever(
            profiles=profiles,
            vector_retriever=vector_retriever,
            k=rrf_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            recall_k=recall_k,
            reranker=reranker,
            rerank_threshold=rerank_threshold,
        )
        self._score_threshold = score_threshold

        self.tools: list[BaseTool] = [self._create_search_tool()]

    @property
    def retriever(self) -> ToolHybridRetriever:
        """暴露检索器供外部访问（调试、评测）。"""
        return self._retriever

    # ---------- search_tools ----------

    def _create_search_tool(self) -> BaseTool:
        """构造 search_tools。返回命中工具的完整 schema。"""
        retriever = self._retriever
        score_threshold = self._score_threshold

        async def search_impl(
            runtime: ToolRuntime[None, dict],
            query: str,
        ) -> ToolMessage:
            results = await retriever.search(query, k=5)
            if not results:
                return _no_match(runtime.tool_call_id)

            payload = []
            for tool, score in results:
                if score < score_threshold:
                    continue
                payload.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": _json_schema_of(tool),
                        "score": round(float(score), 3),
                    }
                )

            if not payload:
                return _no_match(runtime.tool_call_id, reason="below_threshold")

            return ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                name="search_tools",
                tool_call_id=runtime.tool_call_id,
                status="success",
            )

        return StructuredTool.from_function(
            name="search_tools",
            description=(
                "Search available business tools by keyword or capability. "
                "Use when you need a capability not currently visible. "
                "Returns matching tools with name, description, parameter schema, "
                "and a confidence score. Matching tools become callable in the next turn."
            ),
            func=search_impl,
            coroutine=search_impl,
            infer_schema=False,
            args_schema=SearchToolsInput,
        )

    # ---------- 模型调用拦截 ----------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        active: set[str] = set(self._core)

        for query in self._collect_search_queries(request):
            results = await self._retriever.search(query, k=10)
            for tool, score in results:
                if score >= self._score_threshold:
                    active.add(tool.name)

        visible = _merge_visible(request.tools, self._all_tools, active)

        return await handler(request.override(tools=visible))

    # ---------- 工具执行拦截 ----------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Any]],
    ) -> ToolMessage | Any:
        name = request.tool_call["name"]

        # search_tools 与核心工具走框架默认执行路径
        if name in self._core:
            return await handler(request)

        # 业务工具从中间件注册表里取
        tool = self._by_name.get(name)
        if tool is None:
            return ToolMessage(
                content=(
                    f"ERROR: [protocol] unknown tool {name!r}. "
                    "This tool was not discovered by search_tools."
                ),
                name=name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        try:
            result = await tool.ainvoke(request.tool_call["args"])
        except Exception as e:
            logger.exception("工具 %s 执行失败", name)
            return ToolMessage(
                content=f"ERROR: [protocol] tool {name} failed: {e}",
                name=name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        if isinstance(result, ToolMessage):
            return result

        return ToolMessage(
            content=result if isinstance(result, str) else str(result),
            name=name,
            tool_call_id=request.tool_call["id"],
            status="success",
        )

    # ---------- 历史扫描 ----------

    def _collect_search_queries(self, request: ModelRequest) -> list[str]:
        messages = _get_state_messages(request.state)

        queries: list[str] = []
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                if tc.get("name") != "search_tools":
                    continue
                args = tc.get("args", {})
                query = args.get("query") if isinstance(args, dict) else None
                if query and isinstance(query, str):
                    queries.append(query)
        return queries


# ============ 模块级辅助 ============

def _no_match(tool_call_id: str, reason: str = "no_result") -> ToolMessage:
    content = {
        "no_result": (
            "NO_MATCH: no tool matched this query. "
            "Do not guess. Consider answering directly or asking for clarification."
        ),
        "below_threshold": (
            "NO_MATCH: results below confidence threshold. "
            "Consider a different query or answering directly."
        ),
    }[reason]
    return ToolMessage(
        content=content,
        name="search_tools",
        tool_call_id=tool_call_id,
        status="success",
    )


def _json_schema_of(tool: BaseTool) -> dict:
    """提取工具的 JSON schema。兼容 pydantic model 和已序列化的 dict。"""
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return {}
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    return {}


def _merge_visible(
    request_tools: list[BaseTool],
    middleware_tools: list[BaseTool],
    active: set[str],
) -> list[BaseTool]:
    """合并 request.tools 和中间件持有的工具，按 active 过滤。

    request.tools 包含 search_tools 和核心工具；业务工具由中间件持有，
    不在 request.tools 里。两者合并后去重。
    """
    visible: list[BaseTool] = []
    seen: set[str] = set()

    for t in request_tools:
        if t.name in active:
            visible.append(t)
            seen.add(t.name)

    for t in middleware_tools:
        if t.name in active and t.name not in seen:
            visible.append(t)
            seen.add(t.name)

    return visible


def _get_state_messages(state: Any) -> list:
    if state is None:
        return []
    if isinstance(state, dict):
        return state.get("messages") or []
    if hasattr(state, "messages"):
        return getattr(state, "messages") or []
    if hasattr(state, "get"):
        return state.get("messages") or []
    return []