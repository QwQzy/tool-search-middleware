"""基础时间工具。
时区处理：
- 时区名使用 IANA 名称（如 "Asia/Shanghai"），不接受 "CST"、"EST" 这类缩写，
  因为同一缩写在不同地区含义不同。
- 依赖 zoneinfo（Python 3.9+ 内置）。Linux 和 macOS 使用系统 tzdata；
  Windows 需要额外安装 tzdata 包。
- naive 时间（不带时区信息）的时区含义不确定。需要时区的工具在遇到
  naive 输入时返回 ERROR，不隐式假设本地时区或 UTC。

夏令时：
- 时区转换落在 DST "缺失小时"时，zoneinfo 会跳到下一个有效时刻。
- 落在 "重复小时"时，默认选择第一次出现（fold=0）。
  这一行为与 Python 标准库一致，不额外处理。
"""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import tool
from pydantic import BaseModel, Field


def _resolve_tz(name: str) -> ZoneInfo:
    """把 IANA 时区名解析为 ZoneInfo 实例。调用方负责捕获 ZoneInfoNotFoundError。"""
    return ZoneInfo(name)


def _parse_iso(s: str) -> datetime:
    """用 datetime.fromisoformat 解析。Python 3.11+ 支持绝大多数 ISO 8601 形式，
    包括 Z 后缀、时区偏移、毫秒、周日期等。非法输入抛 ValueError。"""
    return datetime.fromisoformat(s)


def _require_aware(dt: datetime, arg_name: str) -> datetime | str:
    """工具需要时区感知的时间。naive 时间返回 ERROR 字符串。"""
    if dt.tzinfo is None:
        return (
            f"ERROR: {arg_name} 缺少时区信息。"
            "请使用带时区偏移的 ISO 格式（如 2024-01-15T10:30:00+08:00）"
            "或先用 timezone_convert 转换为带时区的时间。"
        )
    return dt


# ============ 当前时间 ============

class NowInput(BaseModel):
    tz: str = Field(
        default="UTC",
        description="IANA 时区名，如 \"Asia/Shanghai\"、\"America/New_York\"。默认 UTC",
    )


@tool(
    name_or_callable="now",
    description=(
        "返回指定时区的当前时间，ISO 8601 格式，带时区偏移。"
        "本工具的结果依赖调用时刻，同一参数两次调用结果不同，"
        "因此不满足可确定性验证，是基础工具集中的例外。"
    ),
    args_schema=NowInput,
    infer_schema=False,
)
async def now(tz: str = "UTC") -> str:
    try:
        zone = _resolve_tz(tz)
    except ZoneInfoNotFoundError:
        return f"ERROR: now 无法识别时区 {tz!r}，请使用 IANA 名称如 \"Asia/Shanghai\""
    return datetime.now(zone).isoformat()


# ============ 解析时间字符串 ============

class ParseTimeInput(BaseModel):
    text: str = Field(
        description=(
            "要解析的时间字符串。支持 ISO 8601（如 2024-01-15T10:30:00+08:00、"
            "2024-01-15、2024-01-15 10:30:00Z）"
        ),
    )


@tool(
    name_or_callable="parse_time",
    description=(
        "将时间字符串解析为结构化的 JSON 对象，"
        "包含 ISO 表示、时间戳、各字段值以及是否带时区。"
        "无法解析时返回 ERROR。"
        "解析结果不带时区时（is_aware=false），后续需要时区的工具会拒绝它。"
    ),
    args_schema=ParseTimeInput,
    infer_schema=False,
)
async def parse_time(text: str) -> str:
    try:
        dt = _parse_iso(text)
    except ValueError as e:
        return f"ERROR: parse_time 无法解析 {text!r}: {e}"
    return json.dumps(
        {
            "iso": dt.isoformat(),
            "timestamp": dt.timestamp() if dt.tzinfo is not None else None,
            "is_aware": dt.tzinfo is not None,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
            "microsecond": dt.microsecond,
            "weekday": dt.isoweekday(),  # 1=周一, 7=周日
        },
        ensure_ascii=False,
    )


# ============ 格式化 ============

class FormatTimeInput(BaseModel):
    text: str = Field(description="要格式化的时间字符串，需能被 ISO 8601 解析")
    fmt: str = Field(
        default="%Y-%m-%d %H:%M:%S%z",
        description=(
            "strftime 格式字符串。默认 \"%Y-%m-%d %H:%M:%S%z\"，"
            "输出形如 \"2024-01-15 10:30:00+0800\""
        ),
    )


@tool(
    name_or_callable="format_time",
    description=(
        "按 strftime 格式输出时间字符串。"
        "输入必须是可被 ISO 8601 解析的时间字符串。"
        "解析失败时返回 ERROR。"
    ),
    args_schema=FormatTimeInput,
    infer_schema=False,
)
async def format_time(text: str, fmt: str = "%Y-%m-%d %H:%M:%S%z") -> str:
    try:
        dt = _parse_iso(text)
    except ValueError as e:
        return f"ERROR: format_time 无法解析 {text!r}: {e}"
    return dt.strftime(fmt)


# ============ 时区转换 ============

class TimezoneConvertInput(BaseModel):
    text: str = Field(description="要转换的时间字符串，必须带时区信息")
    to_tz: str = Field(description="目标 IANA 时区名，如 \"Asia/Shanghai\"")


@tool(
    name_or_callable="timezone_convert",
    description=(
        "把带时区的时间转换到目标时区，返回 ISO 8601 格式。"
        "输入时间必须带时区信息，否则返回 ERROR。"
        "源时间落在目标时区的 DST 缺失或重复区间时，遵循 zoneinfo 的默认处理。"
    ),
    args_schema=TimezoneConvertInput,
    infer_schema=False,
)
async def timezone_convert(text: str, to_tz: str) -> str:
    try:
        dt = _parse_iso(text)
    except ValueError as e:
        return f"ERROR: timezone_convert 无法解析 {text!r}: {e}"
    checked = _require_aware(dt, "text")
    if isinstance(checked, str):
        return checked
    try:
        target = _resolve_tz(to_tz)
    except ZoneInfoNotFoundError:
        return (
            f"ERROR: timezone_convert 无法识别时区 {to_tz!r}，"
            "请使用 IANA 名称如 \"Asia/Shanghai\""
        )
    return checked.astimezone(target).isoformat()


# ============ 时间差 ============

class TimeDiffInput(BaseModel):
    start: str = Field(description="起始时间，ISO 8601 格式")
    end: str = Field(description="结束时间，ISO 8601 格式")
    unit: str = Field(
        default="seconds",
        description="主单位：seconds、minutes、hours、days 之一",
    )


@tool(
    name_or_callable="time_diff",
    description=(
        "返回 end - start 的时长，以 JSON 对象表示："
        "包含 total_seconds 以及按天/时/分/秒拆分后的结构，"
        "另有 sign 字段表示方向（end 晚于 start 为 1，早于为 -1）。"
        "两个时间必须同时带时区或同时不带时区，否则返回 ERROR。"
    ),
    args_schema=TimeDiffInput,
    infer_schema=False,
)
async def time_diff(start: str, end: str, unit: str = "seconds") -> str:
    if unit not in {"seconds", "minutes", "hours", "days"}:
        return f"ERROR: time_diff 的 unit 必须是 seconds/minutes/hours/days，实际 {unit!r}"
    try:
        dt_start = _parse_iso(start)
        dt_end = _parse_iso(end)
    except ValueError as e:
        return f"ERROR: time_diff 解析失败: {e}"

    aware_start = dt_start.tzinfo is not None
    aware_end = dt_end.tzinfo is not None
    if aware_start != aware_end:
        return (
            "ERROR: time_diff 的两个时间必须同时带时区或同时不带时区。"
            f"实际 start.is_aware={aware_start}, end.is_aware={aware_end}"
        )

    delta: timedelta = dt_end - dt_start
    total_seconds = delta.total_seconds()
    sign = 1 if total_seconds >= 0 else -1

    abs_delta = abs(delta)
    days = abs_delta.days
    seconds_remainder = abs_delta.seconds
    hours = seconds_remainder // 3600
    minutes = (seconds_remainder % 3600) // 60
    secs = seconds_remainder % 60
    micros = abs_delta.microseconds

    main_value = {
        "seconds": total_seconds,
        "minutes": total_seconds / 60,
        "hours": total_seconds / 3600,
        "days": total_seconds / 86400,
    }[unit]

    return json.dumps(
        {
            "total_seconds": total_seconds,
            unit: main_value,
            "sign": sign,
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds_remainder": secs + micros / 1_000_000,
        },
        ensure_ascii=False,
    )


# ============ 时间偏移 ============

class TimeShiftInput(BaseModel):
    text: str = Field(description="基准时间，ISO 8601 格式")
    weeks: int = Field(default=0, description="偏移周数，可为负")
    days: int = Field(default=0, description="偏移天数，可为负")
    hours: int = Field(default=0, description="偏移小时数，可为负")
    minutes: int = Field(default=0, description="偏移分钟数，可为负")
    seconds: int = Field(default=0, description="偏移秒数，可为负")
    milliseconds: int = Field(default=0, description="偏移毫秒数，可为负")


@tool(
    name_or_callable="time_shift",
    description=(
        "把基准时间向前或向后偏移若干周/天/时/分/秒/毫秒，返回 ISO 8601 格式。"
        "各偏移量可同时指定，会相加成一个总偏移量。"
        "不提供月/年偏移：月份和年份的加减有歧义（如 1 月 31 日加一个月），"
        "需要这类计算时请先换算成天数再调用。"
    ),
    args_schema=TimeShiftInput,
    infer_schema=False,
)
async def time_shift(
    text: str,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    milliseconds: int = 0,
) -> str:
    try:
        dt = _parse_iso(text)
    except ValueError as e:
        return f"ERROR: time_shift 无法解析 {text!r}: {e}"
    delta = timedelta(
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )
    return (dt + delta).isoformat()


# ============ 时间戳互转 ============

class TimestampToDatetimeInput(BaseModel):
    timestamp: float = Field(
        description="Unix 时间戳，单位秒。小数部分表示毫秒/微秒（如 1705314600.123 表示 1705314600 秒 + 123 毫秒）",
    )
    tz: str = Field(
        default="UTC",
        description="输出所用时区，IANA 名称。默认 UTC",
    )


@tool(
    name_or_callable="timestamp_to_datetime",
    description=(
        "把 Unix 时间戳（秒，可带小数）转换为指定时区的 ISO 8601 时间字符串。"
        "时间戳在 Python 的合法范围之外时返回 ERROR。"
    ),
    args_schema=TimestampToDatetimeInput,
    infer_schema=False,
)
async def timestamp_to_datetime(timestamp: float, tz: str = "UTC") -> str:
    try:
        zone = _resolve_tz(tz)
    except ZoneInfoNotFoundError:
        return (
            f"ERROR: timestamp_to_datetime 无法识别时区 {tz!r}，"
            "请使用 IANA 名称如 \"Asia/Shanghai\""
        )
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as e:
        return f"ERROR: timestamp_to_datetime 的时间戳 {timestamp} 超出支持范围: {e}"
    return dt.astimezone(zone).isoformat()


class DatetimeToTimestampInput(BaseModel):
    text: str = Field(description="时间字符串，必须带时区信息")


@tool(
    name_or_callable="datetime_to_timestamp",
    description=(
        "把带时区的时间字符串转换为 Unix 时间戳（秒，浮点）。"
        "输入必须带时区信息，否则返回 ERROR——"
        "naive 时间的 Unix 时间戳取决于被假设的时区，含义不确定。"
    ),
    args_schema=DatetimeToTimestampInput,
    infer_schema=False,
)
async def datetime_to_timestamp(text: str) -> str:
    try:
        dt = _parse_iso(text)
    except ValueError as e:
        return f"ERROR: datetime_to_timestamp 无法解析 {text!r}: {e}"
    checked = _require_aware(dt, "text")
    if isinstance(checked, str):
        return checked
    try:
        return str(checked.timestamp())
    except (OverflowError, OSError, ValueError) as e:
        return f"ERROR: datetime_to_timestamp 转换失败: {e}"