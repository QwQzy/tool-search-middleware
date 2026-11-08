"""基础单位换算工具。

温度换算是仿射变换（有偏移），不是比例变换，因此单独处理；其余三类都可以
分解为 value * (from 单位系数) / (to 单位系数)。

长度和重量的换算系数取自国际定义，是精确值；用 Fraction 表示，
中间计算不产生舍入，只在输出时格式化一次。

货币换算使用内置的固定汇率快照，不反映实时汇率。实时汇率依赖外部系统状态，
不满足可确定性验证；需要实时汇率时应使用 capability/ 或 business/ 中的
联网工具，而不是这里的换算工具。
"""

from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


type NumberInput = int | float | str


def _to_fraction(x: NumberInput) -> Fraction:
    """统一转换为 Fraction。float 经 str() 中转再经 Decimal，避免二进制误差。"""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Decimal):
        return Fraction(x)
    return Fraction(Decimal(str(x)))


def _fmt(f: Fraction) -> str:
    """格式化为字符串，保留 15 位有效数字，去掉尾零，不使用科学计数法。"""
    d = Decimal(f.numerator) / Decimal(f.denominator)
    if d == d.to_integral_value():
        return str(int(d))
    q = Decimal(1).scaleb(d.adjusted() - 14)
    d = d.quantize(q, rounding=ROUND_HALF_UP)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ============ 长度 ============

type LengthUnit = Literal["m", "km", "cm", "mm", "mile", "yard", "foot", "inch"]

# 长度换算系数（以米为基准）。来源为国际定义，均为精确值。
# mile / yard / foot / inch 使用 1959 年国际协议定义的换算。
_LENGTH_TO_METER: dict[LengthUnit, Fraction] = {
    "m": Fraction(1),
    "km": Fraction(1000),
    "cm": Fraction(1, 100),
    "mm": Fraction(1, 1000),
    "mile": Fraction(1609344, 1000),     # 1 mile = 1609.344 m
    "yard": Fraction(9144, 10000),       # 1 yard = 0.9144 m
    "foot": Fraction(3048, 10000),       # 1 foot = 0.3048 m
    "inch": Fraction(254, 10000),        # 1 inch = 0.0254 m
}


class LengthConvertInput(BaseModel):
    value: NumberInput = Field(description="要换算的数值")
    from_unit: LengthUnit = Field(description="源单位")
    to_unit: LengthUnit = Field(description="目标单位")


@tool(
    name_or_callable="length_convert",
    description=(
        "长度单位换算。支持 m（米）、km（千米）、cm（厘米）、mm（毫米）、"
        "mile（英里）、yard（码）、foot（英尺）、inch（英寸）。"
        "换算系数取自国际定义，结果精确。"
    ),
    args_schema=LengthConvertInput,
    infer_schema=False,
)
async def length_convert(
    value: NumberInput,
    from_unit: LengthUnit,
    to_unit: LengthUnit,
) -> str:
    try:
        v = _to_fraction(value)
    except (ValueError, ArithmeticError) as e:
        return f"ERROR: length_convert 无法解析数值 {value!r}: {e}"
    meters = v * _LENGTH_TO_METER[from_unit]
    return _fmt(meters / _LENGTH_TO_METER[to_unit])


# ============ 重量 ============

type WeightUnit = Literal["kg", "g", "mg", "ton", "pound", "ounce"]

# 重量换算系数（以千克为基准）。来源为国际定义，均为精确值。
_WEIGHT_TO_KG: dict[WeightUnit, Fraction] = {
    "kg": Fraction(1),
    "g": Fraction(1, 1000),
    "mg": Fraction(1, 1_000_000),
    "ton": Fraction(1000),                          # 公制吨
    "pound": Fraction(45359237, 100_000_000),       # 1 lb = 0.45359237 kg
    "ounce": Fraction(45359237, 100_000_000 * 16),  # 1 oz = 1/16 lb
}


class WeightConvertInput(BaseModel):
    value: NumberInput = Field(description="要换算的数值")
    from_unit: WeightUnit = Field(description="源单位")
    to_unit: WeightUnit = Field(description="目标单位")


@tool(
    name_or_callable="weight_convert",
    description=(
        "重量单位换算。支持 kg（千克）、g（克）、mg（毫克）、"
        "ton（公制吨）、pound（磅）、ounce（盎司）。"
        "换算系数取自国际定义，结果精确。"
    ),
    args_schema=WeightConvertInput,
    infer_schema=False,
)
async def weight_convert(
    value: NumberInput,
    from_unit: WeightUnit,
    to_unit: WeightUnit,
) -> str:
    try:
        v = _to_fraction(value)
    except (ValueError, ArithmeticError) as e:
        return f"ERROR: weight_convert 无法解析数值 {value!r}: {e}"
    kg = v * _WEIGHT_TO_KG[from_unit]
    return _fmt(kg / _WEIGHT_TO_KG[to_unit])


# ============ 温度 ============

type TemperatureUnit = Literal["c", "f", "k"]


def _temperature_to_celsius(unit: TemperatureUnit, v: Fraction) -> Fraction:
    match unit:
        case "c":
            return v
        case "f":
            return (v - 32) * Fraction(5, 9)
        case "k":
            return v - Fraction(27315, 100)  # 273.15 K


def _temperature_from_celsius(unit: TemperatureUnit, c: Fraction) -> Fraction:
    match unit:
        case "c":
            return c
        case "f":
            return c * Fraction(9, 5) + 32
        case "k":
            return c + Fraction(27315, 100)


class TemperatureConvertInput(BaseModel):
    value: NumberInput = Field(description="要换算的数值")
    from_unit: TemperatureUnit = Field(
        description="源单位：c 摄氏度、f 华氏度、k 开尔文",
    )
    to_unit: TemperatureUnit = Field(
        description="目标单位：c 摄氏度、f 华氏度、k 开尔文",
    )


@tool(
    name_or_callable="temperature_convert",
    description=(
        "温度单位换算。支持 c（摄氏度）、f（华氏度）、k（开尔文）。"
        "温度换算是仿射变换，含偏移项，因此系数不能简化为比例。"
        "本工具只做数学换算，不校验物理合理性"
        "（例如低于绝对零度的值不会被拒绝）。"
    ),
    args_schema=TemperatureConvertInput,
    infer_schema=False,
)
async def temperature_convert(
    value: NumberInput,
    from_unit: TemperatureUnit,
    to_unit: TemperatureUnit,
) -> str:
    try:
        v = _to_fraction(value)
    except (ValueError, ArithmeticError) as e:
        return f"ERROR: temperature_convert 无法解析数值 {value!r}: {e}"
    celsius = _temperature_to_celsius(from_unit, v)
    return _fmt(_temperature_from_celsius(to_unit, celsius))


# ============ 货币 ============

type CurrencyUnit = Literal["USD", "CNY", "EUR", "JPY", "GBP", "HKD"]

# 货币换算使用内置的固定汇率快照，不反映实时汇率。
# 这些数值是示例快照，实际部署时应替换为你需要的快照日期对应的值。
# 单位为“1 单位该货币 = X USD”。
_CURRENCY_TO_USD: dict[CurrencyUnit, Fraction] = {
    "USD": Fraction(1),
    "CNY": Fraction(1381, 10_000),        # 1 CNY ≈ 0.1381 USD（约 1 USD = 7.24 CNY）
    "EUR": Fraction(10870, 10_000),       # 1 EUR ≈ 1.0870 USD
    "JPY": Fraction(6689, 1_000_000),     # 1 JPY ≈ 0.006689 USD（约 1 USD = 149.5 JPY）
    "GBP": Fraction(12658, 10_000),       # 1 GBP ≈ 1.2658 USD
    "HKD": Fraction(1279, 10_000),        # 1 HKD ≈ 0.1279 USD（约 1 USD = 7.82 HKD）
}


class CurrencyConvertInput(BaseModel):
    value: NumberInput = Field(description="要换算的金额")
    from_unit: CurrencyUnit = Field(
        description="源货币：USD、CNY、EUR、JPY、GBP、HKD",
    )
    to_unit: CurrencyUnit = Field(
        description="目标货币：USD、CNY、EUR、JPY、GBP、HKD",
    )


@tool(
    name_or_callable="currency_convert",
    description=(
        "货币换算。支持 USD、CNY、EUR、JPY、GBP、HKD。"
        "使用内置的固定汇率快照，不反映实时汇率。"
        "实时汇率请使用联网工具获取后再计算，"
        "或确保业务接受快照的精度限制。"
    ),
    args_schema=CurrencyConvertInput,
    infer_schema=False,
)
async def currency_convert(
    value: NumberInput,
    from_unit: CurrencyUnit,
    to_unit: CurrencyUnit,
) -> str:
    try:
        v = _to_fraction(value)
    except (ValueError, ArithmeticError) as e:
        return f"ERROR: currency_convert 无法解析数值 {value!r}: {e}"
    usd = v * _CURRENCY_TO_USD[from_unit]
    return _fmt(usd / _CURRENCY_TO_USD[to_unit])