"""业务工具：实时汇率查询。

和 basic/unit_convert.py 里的 currency_convert 分工明确：

- currency_convert：使用固定汇率快照，输出可确定性验证，适合"给定金额
  按已知汇率做精确换算"的场景。
- query_exchange_rate：从外部数据源拉取实时汇率，结果反映调用时刻的外部
  状态，不满足可确定性验证，适合"我现在要按最新汇率换算"的场景。

两者在检索时共享 scene="currency"，模型需要在一个查询里看清两个工具的
区别，因此 query_exchange_rate 的描述里明确指向 currency_convert。

每个工具携带 extras：
- readOnlyHint: True
- scene: "currency"

实时汇率数据源按天或按小时更新，不保证与交易市场的实时报价一致。
工具返回值包含 timestamp 字段，标明汇率的采集时间，调用方应据此判断
时效性。

工具函数返回 mock 数据，带 "source": "mock" 标记。
"""

import json
from datetime import datetime, timezone
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


type CurrencyCode = Literal["USD", "CNY", "EUR", "JPY", "GBP", "HKD", "AUD", "CAD", "SGD"]


# mock 汇率表：1 单位基础货币 = X 单位目标货币
# 结构为 {(base, target): rate}，只覆盖部分货币对，其余按 USD 中转
_MOCK_RATES_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "CNY": 0.1381,
    "EUR": 1.0870,
    "JPY": 0.006689,
    "GBP": 1.2658,
    "HKD": 0.1279,
    "AUD": 0.6543,
    "CAD": 0.7421,
    "SGD": 0.7463,
}


def _mock_rate(base: str, target: str) -> float:
    """通过 USD 中转计算汇率。真实实现会直接查询数据源的货币对接口。"""
    return _MOCK_RATES_TO_USD[base] / _MOCK_RATES_TO_USD[target]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============ 实时汇率查询 ============

class QueryExchangeRateInput(BaseModel):
    base: CurrencyCode = Field(
        description="基础货币，如 USD。汇率表示 1 单位基础货币等于多少目标货币",
    )
    target: CurrencyCode = Field(
        description="目标货币，如 CNY",
    )


@tool(
    name_or_callable="query_exchange_rate",
    description=(
        "查询两种货币之间的实时汇率，返回 1 单位基础货币等于多少目标货币，"
        "以及汇率的采集时间。"
        "结果依赖外部数据源，反映调用时刻的状态，不保证与交易市场实时报价一致。"
        "如果是按已知汇率做精确换算、不要求实时性，请用 currency_convert"
        "（basic 工具集中的固定快照换算）。"
    ),
    args_schema=QueryExchangeRateInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "currency"},
)
async def query_exchange_rate(base: CurrencyCode, target: CurrencyCode) -> str:
    try:
        rate = _mock_rate(base, target)
    except KeyError:
        return (
            f"ERROR: [business] 数据源不覆盖 {base} 到 {target} 的汇率。"
            f"当前支持的货币: {', '.join(_MOCK_RATES_TO_USD.keys())}"
        )

    return json.dumps(
        {
            "base": base,
            "target": target,
            "rate": round(rate, 6),
            "timestamp": _now_iso(),
            "source": "mock",
        },
        ensure_ascii=False,
    )


# ============ 货币换算 ============

class ConvertCurrencyInput(BaseModel):
    amount: float = Field(
        gt=0,
        description="要换算的金额，必须大于 0",
    )
    base: CurrencyCode = Field(description="基础货币")
    target: CurrencyCode = Field(description="目标货币")


@tool(
    name_or_callable="convert_currency",
    description=(
        "按实时汇率把金额从基础货币换算到目标货币，返回换算后的金额、"
        "使用的汇率和汇率采集时间。"
        "与 basic 工具集中的 currency_convert 的区别：本工具使用实时汇率，"
        "currency_convert 使用固定快照。需要精确的已知汇率换算时用 currency_convert。"
        "只做单步换算；只需要汇率不需要换算金额时用 query_exchange_rate。"
    ),
    args_schema=ConvertCurrencyInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "currency"},
)
async def convert_currency(
    amount: float,
    base: CurrencyCode,
    target: CurrencyCode,
) -> str:
    if amount <= 0:
        return "ERROR: [param] convert_currency 的 amount 必须大于 0"

    try:
        rate = _mock_rate(base, target)
    except KeyError:
        return (
            f"ERROR: [business] 数据源不覆盖 {base} 到 {target} 的汇率。"
            f"当前支持的货币: {', '.join(_MOCK_RATES_TO_USD.keys())}"
        )

    return json.dumps(
        {
            "amount": amount,
            "base": base,
            "target": target,
            "rate": round(rate, 6),
            "converted_amount": round(amount * rate, 2),
            "timestamp": _now_iso(),
            "source": "mock",
        },
        ensure_ascii=False,
    )