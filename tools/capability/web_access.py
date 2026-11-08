"""网络访问能力工具。

包含 web_search 和 web_fetch 两个工具，共同构成“访问互联网”这一项
capability。宿主打开这项能力时两者一起注入，关闭时两者都不可见。

设计边界见同目录 README.md。这里只强调三条实现约束：

1. 搜索后端可替换。web_search 不绑定具体搜索引擎，通过 SearchBackend
   协议注入，默认实现使用 DuckDuckGo（无需 API key）。
2. web_fetch 返回 Markdown 正文，不返回原始 HTML。正文提取失败时返回
   ERROR，不静默降级为 HTML。
3. 所有工具为 async 协程，网络 I/O 不阻塞事件循环。trafilatura 的
   extract 是同步 CPU 调用，通过 asyncio.to_thread 移出主线程。
"""

import asyncio
import json
from typing import Protocol

import httpx
import trafilatura
from langchain.tools import tool
from pydantic import BaseModel, Field


# ============ 搜索后端协议 ============

class SearchResult(BaseModel):
    title: str = Field(description="搜索结果标题")
    url: str = Field(description="结果链接")
    snippet: str = Field(description="摘要，由搜索后端提供")


class SearchBackend(Protocol):
    """搜索后端的协议。实现者只需提供 async search 方法。"""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        ...


class DuckDuckGoBackend:
    """DuckDuckGo 搜索后端，无需 API key。

    使用 duckduckgo_search 库的异步接口。如果部署环境不允许访问
    DuckDuckGo，替换成自建搜索服务或商业 API 的 SearchBackend 实现即可，
    web_search 工具本身不感知后端差异。
    """

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RuntimeError(
                "web_search 的默认后端需要 duckduckgo_search。"
                "请 pip install duckduckgo-search，"
                "或注入自定义 SearchBackend。"
            )

        def _sync_search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        rows = await asyncio.to_thread(_sync_search)
        return [
            SearchResult(
                title=row.get("title", ""),
                url=row.get("href", ""),
                snippet=row.get("body", ""),
            )
            for row in rows
        ]


# 模块级后端实例。在组装 agent 时可以替换：
#   import web_access
#   web_access._backend = MyBackend()
_backend: SearchBackend = DuckDuckGoBackend()


# ============ web_search ============

class WebSearchInput(BaseModel):
    query: str = Field(description="搜索查询字符串")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回结果数量上限，默认 5，最大 20",
    )


@tool(
    name_or_callable="web_search",
    description=(
        "搜索互联网，返回标题、URL 和摘要列表，以 JSON 数组字符串表示。"
        "摘要由搜索后端提供，不包含完整页面内容；"
        "需要完整内容时，用 web_fetch 抓取结果中的 URL。"
    ),
    args_schema=WebSearchInput,
    infer_schema=False,
)
async def web_search(query: str, max_results: int = 5) -> str:
    if not query.strip():
        return "ERROR: web_search 的 query 不能为空"
    try:
        results = await _backend.search(query, max_results)
    except Exception as e:
        return f"ERROR: web_search 失败: {e}"
    return json.dumps(
        [r.model_dump() for r in results],
        ensure_ascii=False,
    )


# ============ web_fetch ============

class WebFetchInput(BaseModel):
    url: str = Field(description="要抓取的 URL，必须包含 http:// 或 https:// 协议")
    max_chars: int = Field(
        default=20000,
        ge=1000,
        le=200_000,
        description="返回正文的最大字符数，默认 20000。超出时截断并附加截断标记",
    )


@tool(
    name_or_callable="web_fetch",
    description=(
        "抓取指定 URL 的网页正文，以 Markdown 格式返回。"
        "不返回原始 HTML，导航栏、脚本、样式、广告等样版内容会被移除。"
        "正文提取失败时返回 ERROR，不降级为返回 HTML。"
        "只支持只读抓取，不执行页面脚本，不维持登录态。"
    ),
    args_schema=WebFetchInput,
    infer_schema=False,
)
async def web_fetch(url: str, max_chars: int = 20000) -> str:
    if not url.startswith(("http://", "https://")):
        return f"ERROR: web_fetch 的 url 必须以 http:// 或 https:// 开头，实际 {url!r}"

    # 1. 抓取
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; tool-agent/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        return f"ERROR: web_fetch 请求超时: {url}"
    except httpx.HTTPStatusError as e:
        return f"ERROR: web_fetch 返回 HTTP {e.response.status_code}: {url}"
    except httpx.RequestError as e:
        return f"ERROR: web_fetch 请求失败: {e}"

    content_type = response.headers.get("content-type", "")
    raw = response.text

    # 2. 非 HTML 内容直接返回文本（如纯文本、JSON、Markdown）
    if "html" not in content_type.lower():
        if not raw.strip():
            return f"ERROR: web_fetch 返回空内容: {url}"
        return _truncate(raw, max_chars)

    # 3. HTML → Markdown 正文提取
    # trafilatura.extract 是同步 CPU 调用，放到线程池避免阻塞事件循环
    try:
        markdown = await asyncio.to_thread(
            trafilatura.extract,
            raw,
            output_format="markdown",
            include_links=True,
            include_formatting=True,
            fast=True,
        )
    except Exception as e:
        return f"ERROR: web_fetch 正文提取失败: {e}"

    if not markdown:
        return (
            f"ERROR: web_fetch 未能从 {url} 提取到正文。"
            "页面可能是纯 JavaScript 渲染、需要登录、或正文不在标准 HTML 结构中。"
            "需要浏览器自动化能力时，请开启对应的 capability。"
        )

    return _truncate(markdown, max_chars)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... 已截断，原文超过 {max_chars} 字符]"