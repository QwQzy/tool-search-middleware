# tool-search-middleware

面向万级工具规模的 LangChain 工具发现与执行中间件。BM25 + 向量混合召回、场景硬过滤、受控执行路径。

[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 方法论背景：[当一万个工具涌向模型：一次关于 Tool Search 与 Tool Exec 的拆解](https://generalzy.blog.csdn.net/article/details/170284963)

## 这是什么

当 agent 接入的工具从几十个增长到上万个，两个问题会同时出现：把所有工具的 schema 塞进上下文不现实，而模型在 30–50 个工具以上选对工具的能力就开始退化。这个项目把工具的**发现**和**执行**拆成两个原语：

- **Tool Search**：在工具库上做混合检索，返回一小批带置信度的候选 schema，供模型选择。
- **Tool Exec**：在模型选定工具之后，由一个受控的中间件接管执行，把鉴权、幂等、错误分类、结果清洗放在模型之外。

两者都以 LangChain `AgentMiddleware` 的形式实现，业务工具不注册到 `create_agent(tools=...)`，由中间件持有、按需披露、按名执行。

方法论和设计取舍记录在博客文章里（[当一万个工具涌向模型：一次关于 Tool Search 与 Tool Exec 的拆解](https://generalzy.blog.csdn.net/article/details/170284963)），本 README 只说明项目怎么用、当前做到哪一步、有哪些已知限制。

## 核心概念

### 工具的三层分类

| 类别 | 判定条件 | 发现方式 | 执行方式 |
|---|---|---|---|
| 基础工具 | 只读、无副作用、可确定性验证 | 直接暴露，常驻上下文 | 直接执行 |
| 可选工具 | 通用能力，可见性由宿主控制 | 宿主配置开关 | 经执行网关 |
| 业务工具 | 发现成本高，或依赖外部状态 | `Tool Search` 检索披露 | 经执行网关 |

本仓库的 `tools/basic/`、`tools/capability/`、`tools/business/` 分别对应这三类。`ToolSearchMiddleware` 只处理业务工具。

### 发现与执行分离

```
模型调用前  ──awrap_model_call──▶  决定披露集合
                                    ├─ 核心工具常驻
                                    ├─ 扫描历史中的 search_tools 调用
                                    └─ 命中工具加入可见集合

工具调用前  ──awrap_tool_call──▶   决定执行路径
                                    ├─ search_tools / 核心工具 → 框架 handler
                                    └─ 业务工具 → 中间件注册表 → 执行
```

## 快速开始

### 安装

```bash
pip install -e .
```

主要依赖：`langchain`、`langchain-core`、`rank-bm25`、`pydantic`。评测脚本额外依赖 `langchain-community`、`langchain-openai`、`chromadb`。

### 最小示例

```python
from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from middleware.search_middleware import ToolSearchMiddleware
from tools.basic import BASIC_TOOLS
from tools.business import BUSINESS_TOOLS

# 1. 构建向量库（生产环境可替换为任意 BaseRetriever）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts = [f"{t.name}: {t.description}" for t in BUSINESS_TOOLS]
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="tools")

# 2. 构建中间件
search_mw = ToolSearchMiddleware(
    all_tools=BUSINESS_TOOLS,
    vector_retriever=vectorstore.as_retriever(),
    core_tool_names={t.name for t in BASIC_TOOLS},
)

# 3. 组装 agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[*BASIC_TOOLS, *search_mw.tools],
    middleware=[search_mw],
)
```

`tools=` 里只有基础工具和 `search_mw.tools`（即 `search_tools`）。业务工具由中间件持有，模型通过 `search_tools` 发现它们，通过 `awrap_tool_call` 执行它们。

### 一次对话的流程

```
用户: 我要退款订单 ORD-ABC123

模型 → search_tools("refund order")
     ← [{"name": "refund_order", "parameters": {...}, "score": 0.032}]

模型 → refund_order(order_id="ORD-ABC123", amount=259, reason="...")
     ← ToolMessage(status="success", content="{...}")
```

## 目录结构

```
middleware/
├── hybrid_retriever.py    # ToolProfile、BM25 + 向量 + RRF
└── search_middleware.py   # ToolSearchMiddleware

tools/
├── basic/                 # 基础工具，直接暴露
│   ├── math_tools.py
│   ├── encoding_tools.py
│   ├── unit_convert.py
│   ├── text_tools.py
│   ├── time_tools.py
│   ├── validation_tools.py
│   └── random_tools.py
├── capability/            # 可选能力，宿主开关
│   └── web_access.py
└── business/              # 业务工具，经 search / exec
    ├── weather.py
    ├── geo.py
    ├── ip_intel.py
    ├── currency.py
    ├── order.py
    └── private.py

benchmarks/
├── datasets/retrieval.jsonl
├── metrics.py
└── run_retrieval.py

tests/
├── test_hybrid_retriever.py
└── test_search_agent.py
```

## 核心组件

### `ToolHybridRetriever`

BM25 和向量两路并行召回，RRF 融合，可选重排序。

```python
async def search(
    self, query: str, *, scene: str | None = None, k: int = 10
) -> list[tuple[BaseTool, float]]:
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `k` | 60 | RRF 平滑常数。降低会放大头部排名差异 |
| `vector_weight` | 1.0 | 向量一路的 RRF 权重 |
| `bm25_weight` | 0.7 | BM25 一路的 RRF 权重 |
| `recall_k` | 30 | 每路召回的候选数 |
| `reranker` | `None` | 可选的重排序器，满足 `Reranker` 协议 |
| `rerank_threshold` | 0.0 | 融合最高分低于此值时触发重排 |

### `ToolSearchMiddleware`

```python
class ToolSearchMiddleware(AgentMiddleware):
    def __init__(
        self,
        all_tools: list[BaseTool],
        vector_retriever: BaseRetriever,
        *,
        core_tool_names: set[str] | None = None,
        ...
    ) -> None:
```

除检索参数外，`score_threshold` 控制披露门槛：检索分数低于此值的工具不会进入 `search_tools` 的返回值，也不会被加入可见集合。默认 `0.0` 表示不设阈值，生产环境需要在标注数据上校准。

### `search_tools`

中间件内置的发现工具。输入 `query: str`，返回 JSON 数组：

```json
[
  {
    "name": "refund_order",
    "description": "对订单发起退款。只允许对 paid、shipped、delivered 状态的订单退款...",
    "parameters": {"type": "object", "properties": {...}, "required": [...]},
    "score": 0.032
  }
]
```

无匹配时返回 `NO_MATCH` 字符串，而不是空列表。模型看到 `NO_MATCH` 会转向基础工具、请求澄清或直接回答，而不会在一个空结果上继续规划。

## 评测

### 数据集

`benchmarks/datasets/retrieval.jsonl`，48 条，按 scene 分布：

```jsonl
{"query": "北京现在多少度", "expected": ["query_current_weather"], "scene": "weather"}
{"query": "这个订单不要了，帮我取消", "expected": ["cancel_order"], "scene": "order"}
{"query": "帮我查下订单 ORD-ABC123 的详情，顺便看看快递到哪了", "expected": ["query_order", "query_order_logistics"], "scene": "order"}
```

### 运行

```bash
python -m benchmarks.run_retrieval \
    --dataset benchmarks/datasets/retrieval.jsonl \
    --vector-index ./chroma_tools
```

输出：

```
     hit@1      hit@5     hit@10   recall@5  recall@10        mrr
     0.625      0.875      0.938      0.833      0.917      0.741
```

### 指标

- **hit@k**：top-k 中至少有一个目标工具的查询占比
- **recall@k**：top-k 命中的目标工具数 / 目标工具总数，按查询平均
- **MRR**：第一个命中目标工具的排名倒数，按查询平均

`hit@k` 和 `recall@k` 在单工具查询上相同，在复合查询上分叉。两个都看，才能发现复合查询的召回短板。

## 当前状态

这是一个参考实现，核心逻辑可以运行，但离生产还有明确的距离。

**已实现并测试：**

- `ToolHybridRetriever` 的 BM25 + 向量并行召回、RRF 融合、场景过滤、单路失败降级
- `ToolSearchMiddleware` 的 `awrap_model_call`（披露）和 `awrap_tool_call`（执行接管）
- `search_tools` 的 schema 返回、`NO_MATCH` 处理
- `tools/basic/` 全部工具，`tools/business/` 六个业务域的结构和接口
- 检索器单元测试、中间件集成测试、48 条评测数据集

**未实现或需要替换：**

| 项目 | 当前状态 | 生产方向 |
|---|---|---|
| 鉴权与幂等 | `awrap_tool_call` 直接执行，无权限检查 | 接入 `ExecGatewayMiddleware`，顺序调整为 `[ExecGateway, ToolSearch]` |
| 历史扫描 | 每次 `awrap_model_call` 重新扫描全部消息 | 自定义 state 维护 `discovered_tools: set[str]` |
| 场景硬过滤 | BM25 路径在评分后过滤 | 换用支持索引层过滤的 BM25（Whoosh / Elasticsearch） |
| `score_threshold` | 默认 0.0 | 在评测集上扫描校准 |
| 向量库 | Chroma 占位 | 任意支持元数据过滤的 `BaseRetriever` |
| 数据集 | 48 条人工构造 | 补充 ToolRet 公开基准 |
| 业务工具实现 | mock 数据，带 `"source": "mock"` 标记 | 替换为实际 API 调用 |

**已知限制（代码注释里同样标注）：**

- `ToolCallRequest.tool` 字段未被使用，执行完全依赖中间件注册表。LangChain 版本变化可能影响这个假设。
- `rank_bm25` 没有倒排索引，`scene` 过滤在评分之后执行。工具规模超过十万时应换实现。
- `awrap_model_call` 的披露逻辑对每个历史 `search_tools` 调用重新执行检索，长对话中开销线性增长。
- 重排序器接口已定义，但仓库内没有提供 `Reranker` 的实现，需要调用方自行注入。

## 路线图

1. 接入 `ExecGatewayMiddleware`：鉴权、幂等键、错误分类、结果脱敏
2. 租户过滤从语义层（`scene`）移到安全层（构造时的 `tool_filter`）
3. 在 ToolRet 公开基准上跑一次，和 48 条本地数据集的指标对照
4. 把 BM25 从 `rank_bm25` 换成支持索引层过滤的实现
5. 补充延迟 benchmark：p50 / p95 / p99，覆盖 `search` 和历史扫描两条路径

## 参考

- 博客原文：[当一万个工具涌向模型：一次关于 Tool Search 与 Tool Exec 的拆解](https://generalzy.blog.csdn.net/article/details/170284963)
- [ToolRet: A Benchmark for Tool Retrieval](https://arxiv.org/abs/2503.01763)
- [Tool-REX: Tool Retrieval with LLM-based Document Expansion](https://arxiv.org/abs/2503.02904)（ICLR 2026）
- [LiveMCPBench: Evaluating LLM Agents in Real-World MCP Environments](https://arxiv.org/abs/2505.23193)
- [LangChain Agent Middleware](https://python.langchain.com/docs/concepts/agents/)

## License

MIT

