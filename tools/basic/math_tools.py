"""基础数学工具。

设计约束：
- 内部计算统一使用 Decimal，避免 float 的二进制误差
- float 输入经 str() 中转进入 Decimal，利用 Python float 的最短 repr 表示
- 返回值统一为字符串，既保证精度不丢失，也避免 JSON 序列化问题
- 错误以 "ERROR: ..." 返回，不抛异常，模型可以据此修正参数后重试
- 工具 schema 由 Pydantic model 显式定义，infer_schema=False，不依赖函数签名推断
- 所有工具均为 async 协程，LangChain 直接 await，不进入线程池

运行环境要求 Python >= 3.12。
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from langchain.tools import tool
from pydantic import BaseModel, Field


type NumberInput = int | float | str


def _dec(x: NumberInput) -> Decimal:
    """统一转换为 Decimal。float 经 str() 中转，避免 0.1 这类数的二进制污染。"""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _fmt(d: Decimal) -> str:
    """格式化为字符串，去掉尾零，不使用科学计数法。"""
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


# ============ 四则运算 ============

class AddInput(BaseModel):
    a: NumberInput = Field(description="第一个加数")
    b: NumberInput = Field(description="第二个加数")


@tool(
    name_or_callable="add",
    description="返回 a + b 的精确结果。",
    args_schema=AddInput,
    infer_schema=False,
)
async def add(a: NumberInput, b: NumberInput) -> str:
    try:
        return _fmt(_dec(a) + _dec(b))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: add 无法计算: {e}"


class SubtractInput(BaseModel):
    a: NumberInput = Field(description="被减数")
    b: NumberInput = Field(description="减数")


@tool(
    name_or_callable="subtract",
    description="返回 a - b 的精确结果。",
    args_schema=SubtractInput,
    infer_schema=False,
)
async def subtract(a: NumberInput, b: NumberInput) -> str:
    try:
        return _fmt(_dec(a) - _dec(b))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: subtract 无法计算: {e}"


class MultiplyInput(BaseModel):
    a: NumberInput = Field(description="第一个因数")
    b: NumberInput = Field(description="第二个因数")


@tool(
    name_or_callable="multiply",
    description="返回 a * b 的精确结果。",
    args_schema=MultiplyInput,
    infer_schema=False,
)
async def multiply(a: NumberInput, b: NumberInput) -> str:
    try:
        return _fmt(_dec(a) * _dec(b))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: multiply 无法计算: {e}"


class DivideInput(BaseModel):
    a: NumberInput = Field(description="被除数")
    b: NumberInput = Field(description="除数，不能为 0")


@tool(
    name_or_callable="divide",
    description="返回 a / b 的精确结果。除数为 0 时返回 ERROR。",
    args_schema=DivideInput,
    infer_schema=False,
)
async def divide(a: NumberInput, b: NumberInput) -> str:
    try:
        bd = _dec(b)
        if bd == 0:
            return "ERROR: divide 的除数不能为 0"
        return _fmt(_dec(a) / bd)
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: divide 无法计算: {e}"


# ============ 幂与开方 ============

class PowerInput(BaseModel):
    base: NumberInput = Field(description="底数")
    exponent: int = Field(description="指数，必须是整数")


@tool(
    name_or_callable="power",
    description="返回 base ** exponent。指数由 Pydantic 强制为整数，非整数会在参数校验阶段被拒绝。",
    args_schema=PowerInput,
    infer_schema=False,
)
async def power(base: NumberInput, exponent: int) -> str:
    try:
        return _fmt(_dec(base) ** exponent)
    except (InvalidOperation, ValueError, OverflowError) as e:
        return f"ERROR: power 无法计算: {e}"


class SqrtInput(BaseModel):
    x: NumberInput = Field(description="被开方数，必须 >= 0")


@tool(
    name_or_callable="sqrt",
    description="返回 x 的平方根，保留 Decimal 精度（50 位）。",
    args_schema=SqrtInput,
    infer_schema=False,
)
async def sqrt(x: NumberInput) -> str:
    try:
        d = _dec(x)
        if d < 0:
            return "ERROR: sqrt 的参数不能为负数"
        return _fmt(d.sqrt())
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: sqrt 无法计算: {e}"


# ============ 取余、取整与舍入 ============

class ModInput(BaseModel):
    a: NumberInput = Field(description="被除数")
    b: NumberInput = Field(description="除数，不能为 0")


@tool(
    name_or_callable="mod",
    description="返回 a % b 的结果。",
    args_schema=ModInput,
    infer_schema=False,
)
async def mod(a: NumberInput, b: NumberInput) -> str:
    try:
        bd = _dec(b)
        if bd == 0:
            return "ERROR: mod 的除数不能为 0"
        return _fmt(_dec(a) % bd)
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: mod 无法计算: {e}"


class FloorInput(BaseModel):
    x: NumberInput = Field(description="输入数值")


@tool(
    name_or_callable="floor",
    description="返回 <= x 的最大整数。",
    args_schema=FloorInput,
    infer_schema=False,
)
async def floor(x: NumberInput) -> str:
    try:
        return str(int(_dec(x).to_integral_value(rounding="ROUND_FLOOR")))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: floor 无法计算: {e}"


class CeilInput(BaseModel):
    x: NumberInput = Field(description="输入数值")


@tool(
    name_or_callable="ceil",
    description="返回 >= x 的最小整数。",
    args_schema=CeilInput,
    infer_schema=False,
)
async def ceil(x: NumberInput) -> str:
    try:
        return str(int(_dec(x).to_integral_value(rounding="ROUND_CEILING")))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: ceil 无法计算: {e}"


class RoundNumInput(BaseModel):
    x: NumberInput = Field(description="要四舍五入的数")
    ndigits: int = Field(default=0, description="保留小数位数，可以为负（对整数位四舍五入）")


@tool(
    name_or_callable="round_num",
    description=(
        "对 x 四舍五入到 ndigits 位小数。使用 ROUND_HALF_UP"
        "（0.5 向绝对值大的方向进位），与 Python 内建 round 的银行家舍入不同，"
        "符合大多数人“四舍五入”的直觉。"
    ),
    args_schema=RoundNumInput,
    infer_schema=False,
)
async def round_num(x: NumberInput, ndigits: int = 0) -> str:
    try:
        quant = Decimal(1).scaleb(-ndigits)
        return _fmt(_dec(x).quantize(quant, rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: round_num 无法计算: {e}"


class AbsoluteInput(BaseModel):
    x: NumberInput = Field(description="输入数值")


@tool(
    name_or_callable="absolute",
    description="返回 |x|。",
    args_schema=AbsoluteInput,
    infer_schema=False,
)
async def absolute(x: NumberInput) -> str:
    try:
        return _fmt(abs(_dec(x)))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: absolute 无法计算: {e}"


# ============ 列表聚合 ============

class MaxValueInput(BaseModel):
    values: list[NumberInput] = Field(description="待比较的数值列表，不能为空")


@tool(
    name_or_callable="max_value",
    description="返回列表中的最大值。空列表返回 ERROR。",
    args_schema=MaxValueInput,
    infer_schema=False,
)
async def max_value(values: list[NumberInput]) -> str:
    if not values:
        return "ERROR: max_value 的输入列表不能为空"
    try:
        return _fmt(max(_dec(v) for v in values))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: max_value 无法计算: {e}"


class MinValueInput(BaseModel):
    values: list[NumberInput] = Field(description="待比较的数值列表，不能为空")


@tool(
    name_or_callable="min_value",
    description="返回列表中的最小值。空列表返回 ERROR。",
    args_schema=MinValueInput,
    infer_schema=False,
)
async def min_value(values: list[NumberInput]) -> str:
    if not values:
        return "ERROR: min_value 的输入列表不能为空"
    try:
        return _fmt(min(_dec(v) for v in values))
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: min_value 无法计算: {e}"


class SumValuesInput(BaseModel):
    values: list[NumberInput] = Field(description="待求和的数值列表，空列表返回 0")


@tool(
    name_or_callable="sum_values",
    description="返回列表中所有数值的和。空列表返回 0。",
    args_schema=SumValuesInput,
    infer_schema=False,
)
async def sum_values(values: list[NumberInput]) -> str:
    try:
        total = Decimal(0)
        for v in values:
            total += _dec(v)
        return _fmt(total)
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: sum_values 无法计算: {e}"


# ============ 百分比与比较 ============

class PercentageInput(BaseModel):
    part: NumberInput = Field(description="部分值")
    whole: NumberInput = Field(description="整体值，不能为 0")


@tool(
    name_or_callable="percentage",
    description=(
        "返回 part 占 whole 的百分比数值（不含 % 符号）。"
        "例如 percentage(25, 200) 返回 \"12.5\"，表示 12.5%。"
    ),
    args_schema=PercentageInput,
    infer_schema=False,
)
async def percentage(part: NumberInput, whole: NumberInput) -> str:
    try:
        wd = _dec(whole)
        if wd == 0:
            return "ERROR: percentage 的整体值不能为 0"
        return _fmt(_dec(part) * 100 / wd)
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: percentage 无法计算: {e}"


class CompareInput(BaseModel):
    a: NumberInput = Field(description="第一个数")
    b: NumberInput = Field(description="第二个数")


@tool(
    name_or_callable="compare",
    description='比较两个数，返回 "<"、"==" 或 ">" 之一。用于避免模型自行比大小出错。',
    args_schema=CompareInput,
    infer_schema=False,
)
async def compare(a: NumberInput, b: NumberInput) -> str:
    try:
        da, db = _dec(a), _dec(b)
        if da < db:
            return "<"
        if da > db:
            return ">"
        return "=="
    except (InvalidOperation, ValueError) as e:
        return f"ERROR: compare 无法计算: {e}"