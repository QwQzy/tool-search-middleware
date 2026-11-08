from .web_access import web_fetch, web_search

__all__ = [
    "web_fetch",
    "web_search",
    "CAPABILITY_TOOLS",
]


CAPABILITY_TOOLS = [web_search, web_fetch]