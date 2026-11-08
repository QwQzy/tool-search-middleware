"""基础随机工具。

这个目录的随机工具在接口层面强制要求 seed 参数：相同 seed + 相同输入
产生相同输出。不带种子的随机数依赖系统熵源，每次调用结果不同，
不满足可确定性验证，因此不在此目录提供。

seed 缺失时由 Pydantic 在参数校验阶段直接拒绝（字段为必填），
不会回退到系统熵源。这和 power 的 exponent 用 int 类型强制整数、
divide 在函数内部检查除数为 0 是同一个思路：能在 schema 层拦下的
错误，就不留到运行时。
"""

import json
import random

from langchain.tools import tool
from pydantic import BaseModel, Field


def _rng(seed: int) -> random.Random:
    """用 seed 创建一个独立的 Random 实例，不影响全局 random 状态。"""
    return random.Random(seed)


# ============ 整数 ============

class RandomIntInput(BaseModel):
    a: int = Field(description="区间下界（含）")
    b: int = Field(description="区间上界（含）")
    seed: int = Field(description="随机种子。相同 seed 与相同参数产生相同结果")


@tool(
    name_or_callable="random_int",
    description=(
        "返回 [a, b] 闭区间内的均匀随机整数。"
        "要求 a <= b，否则返回 ERROR。"
        "相同 seed 与相同 (a, b) 必然产生相同结果。"
    ),
    args_schema=RandomIntInput,
    infer_schema=False,
)
async def random_int(a: int, b: int, seed: int) -> str:
    if a > b:
        return f"ERROR: random_int 要求 a <= b，实际 a={a}, b={b}"
    return str(_rng(seed).randint(a, b))


# ============ 浮点数 ============

class RandomFloatInput(BaseModel):
    a: float = Field(description="区间下界（含）")
    b: float = Field(description="区间上界（含）")
    seed: int = Field(description="随机种子。相同 seed 与相同参数产生相同结果")


@tool(
    name_or_callable="random_float",
    description=(
        "返回 [a, b] 区间内的均匀随机浮点数，以 Python float 的最短 repr 形式返回。"
        "要求 a <= b，否则返回 ERROR。"
        "结果是一个未经舍入的 float，需要展示精度时由调用方自行格式化。"
    ),
    args_schema=RandomFloatInput,
    infer_schema=False,
)
async def random_float(a: float, b: float, seed: int) -> str:
    if a > b:
        return f"ERROR: random_float 要求 a <= b，实际 a={a}, b={b}"
    return str(_rng(seed).uniform(a, b))


# ============ 布尔 ============

class RandomBoolInput(BaseModel):
    seed: int = Field(description="随机种子。相同 seed 必然产生相同结果")


@tool(
    name_or_callable="random_bool",
    description=(
        "以 50% 概率返回 \"true\" 或 \"false\"（小写，与 JSON 布尔字面量一致）。"
        "相同 seed 必然产生相同结果。"
    ),
    args_schema=RandomBoolInput,
    infer_schema=False,
)
async def random_bool(seed: int) -> str:
    return "true" if _rng(seed).random() < 0.5 else "false"


# ============ 序列选择 ============

class RandomChoiceInput(BaseModel):
    seq: list[str] = Field(description="候选序列，不能为空")
    seed: int = Field(description="随机种子。相同 seed 与相同序列产生相同结果")


@tool(
    name_or_callable="random_choice",
    description=(
        "从序列中均匀随机选择一个元素并返回。"
        "序列元素按字符串处理；数字元素请先转为字符串再传入。"
        "空序列返回 ERROR。"
    ),
    args_schema=RandomChoiceInput,
    infer_schema=False,
)
async def random_choice(seq: list[str], seed: int) -> str:
    if not seq:
        return "ERROR: random_choice 的序列不能为空"
    return _rng(seed).choice(seq)


class RandomSampleInput(BaseModel):
    seq: list[str] = Field(description="候选序列，不能为空")
    k: int = Field(description="抽取数量，必须满足 0 <= k <= len(seq)")
    seed: int = Field(description="随机种子。相同 seed 与相同参数产生相同结果")


@tool(
    name_or_callable="random_sample",
    description=(
        "从序列中无放回地抽取 k 个元素，返回 JSON 数组字符串。"
        "元素顺序是抽取顺序，不是原序列顺序。"
        "要求 0 <= k <= len(seq)，否则返回 ERROR。"
    ),
    args_schema=RandomSampleInput,
    infer_schema=False,
)
async def random_sample(seq: list[str], k: int, seed: int) -> str:
    if k < 0:
        return f"ERROR: random_sample 要求 k >= 0，实际 k={k}"
    if k > len(seq):
        return f"ERROR: random_sample 的 k={k} 超过序列长度 {len(seq)}"
    picked = _rng(seed).sample(seq, k)
    return json.dumps(picked, ensure_ascii=False)


class ShuffleInput(BaseModel):
    seq: list[str] = Field(description="要打乱的序列")
    seed: int = Field(description="随机种子。相同 seed 与相同序列产生相同结果")


@tool(
    name_or_callable="shuffle",
    description=(
        "返回序列的随机排列，以 JSON 数组字符串表示。"
        "不修改输入序列，返回的是新列表。"
        "相同 seed 与相同序列必然产生相同排列。"
    ),
    args_schema=ShuffleInput,
    infer_schema=False,
)
async def shuffle(seq: list[str], seed: int) -> str:
    result = list(seq)
    _rng(seed).shuffle(result)
    return json.dumps(result, ensure_ascii=False)