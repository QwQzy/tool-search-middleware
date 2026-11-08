"""检索指标。纯函数，输入是标注数据和检索结果，输出是标量。"""


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top = set(retrieved[:k])
    return 1.0 if top & set(expected) else 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(expected)) / len(set(expected))


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    for i, name in enumerate(retrieved, start=1):
        if name in expected_set:
            return 1.0 / i
    return 0.0


def evaluate(
    cases: list[dict],
    results: list[list[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """聚合评估。cases[i] 是标注条目，results[i] 是对应 query 的检索结果（工具名列表）。"""
    assert len(cases) == len(results)

    out = {}
    for k in ks:
        out[f"hit@{k}"] = sum(
            hit_at_k(r, c["expected"], k) for c, r in zip(cases, results)
        ) / len(cases)
        out[f"recall@{k}"] = sum(
            recall_at_k(r, c["expected"], k) for c, r in zip(cases, results)
        ) / len(cases)
    out["mrr"] = sum(
        reciprocal_rank(r, c["expected"]) for c, r in zip(cases, results)
    ) / len(cases)
    return out