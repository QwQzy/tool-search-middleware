"""检索 benchmark 运行脚本。

用法：
    python -m benchmarks.run_retrieval --dataset benchmarks/datasets/retrieval.jsonl

输出：
    hit@1  hit@5  hit@10  recall@5  recall@10  mrr
    0.62   0.88   0.94    0.85      0.91       0.74
"""

import argparse
import asyncio
import json

from benchmarks.metrics import evaluate
from middleware.hybrid_retriever import ToolHybridRetriever, build_profiles
from tools.business import BUSINESS_TOOLS


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vector-index", required=True, help="向量库路径或 URI")
    args = parser.parse_args()

    cases = [json.loads(line) for line in open(args.dataset, encoding="utf-8")]

    # 构造检索器
    retriever = _build_retriever(args.vector_index)

    results: list[list[str]] = []
    for case in cases:
        hits = await retriever.search(case["query"], k=10)
        results.append([t.name for t, _ in hits])

    metrics = evaluate(cases, results, ks=(1, 5, 10))
    _print_table(metrics)


def _build_retriever(vector_index: str) -> ToolHybridRetriever:
    """根据部署环境构造检索器。向量库的选择由调用方决定，不写死在脚本里。"""
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="tools",
        embedding_function=embeddings,
        persist_directory=vector_index,
    )

    profiles = build_profiles(BUSINESS_TOOLS)
    return ToolHybridRetriever(
        profiles=profiles,
        vector_retriever=vectorstore.as_retriever(),
    )


def _print_table(m: dict) -> None:
    cols = list(m.keys())
    print("  ".join(f"{c:>10}" for c in cols))
    print("  ".join(f"{m[c]:>10.3f}" for c in cols))


if __name__ == "__main__":
    asyncio.run(main())