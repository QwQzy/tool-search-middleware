"""业务工具：天气查询。

每个工具携带 extras：
- readOnlyHint: True —— 天气查询不改变任何持久状态
- scene: "weather" —— 供 Tool Search 场景硬过滤使用

extras 是业务工具元数据的统一挂载点。ToolSearchMiddleware 的场景过滤
读 extras["scene"]，ExecGatewayMiddleware 的读写判断读 extras["readOnlyHint"]。
这两个字段在工具注册时缺失，会在网关层被拒绝。

工具函数目前返回 mock 数据，全部带 "source": "mock" 标记。接入真实数据源时，
把 mock 返回替换为对数据源 API 的调用，保持返回结构不变。
"""

import json
from datetime import date, timedelta
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


type TemperatureUnit = Literal["celsius", "fahrenheit"]


def _format_temperature(value_celsius: float, unit: TemperatureUnit) -> dict:
    """把摄氏温度转换为目标单位，返回 {value, unit} 结构。"""
    if unit == "celsius":
        return {"value": round(value_celsius, 1), "unit": "celsius"}
    return {"value": round(value_celsius * 9 / 5 + 32, 1), "unit": "fahrenheit"}


def _mock_current(location: str) -> dict:
    """真实实现：调用天气数据源的实况 API。"""
    return {
        "location": location,
        "observed_at": "2024-01-15T10:30:00+08:00",
        "condition": "partly_cloudy",
        "temperature_celsius": 22.0,
        "feels_like_celsius": 24.0,
        "humidity_pct": 68,
        "wind_speed_kmh": 12,
        "wind_direction": "NE",
    }


def _mock_forecast_day(location: str, offset_days: int) -> dict:
    """真实实现：调用天气数据源的预报 API。"""
    d = date(2024, 1, 15) + timedelta(days=offset_days)
    return {
        "location": location,
        "date": d.isoformat(),
        "condition": "sunny" if offset_days % 2 == 0 else "cloudy",
        "temperature_max_celsius": 25.0 - offset_days * 0.5,
        "temperature_min_celsius": 16.0 - offset_days * 0.5,
        "precipitation_pct": 10 + offset_days * 5,
    }


def _mock_history(location: str, target: str) -> dict:
    """真实实现：调用天气数据源的历史 API。"""
    return {
        "location": location,
        "date": target,
        "condition": "sunny",
        "temperature_max_celsius": 28.0,
        "temperature_min_celsius": 18.0,
        "precipitation_mm": 0.0,
    }


# ============ 当前天气 ============

class QueryCurrentWeatherInput(BaseModel):
    location: str = Field(
        description=(
            "地点。推荐格式为 \"城市名, 国家代码\"（如 \"Shenzhen, CN\"），"
            "也接受单城市名（如 \"Shenzhen\"）"
        ),
    )
    unit: TemperatureUnit = Field(
        default="celsius",
        description="温度单位：celsius 或 fahrenheit",
    )


@tool(
    name_or_callable="query_current_weather",
    description=(
        "查询指定地点的当前天气，返回温度、体感温度、湿度、风速风向。"
        "返回的是调用时刻的实况。"
        "历史数据用 query_weather_history，未来预报用 query_weather_forecast。"
    ),
    args_schema=QueryCurrentWeatherInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "weather"},
)
async def query_current_weather(
    location: str,
    unit: TemperatureUnit = "celsius",
) -> str:
    if not location.strip():
        return "ERROR: [param] query_current_weather 的 location 不能为空"

    data = _mock_current(location)
    return json.dumps(
        {
            "location": data["location"],
            "observed_at": data["observed_at"],
            "condition": data["condition"],
            "temperature": _format_temperature(data["temperature_celsius"], unit),
            "feels_like": _format_temperature(data["feels_like_celsius"], unit),
            "humidity_pct": data["humidity_pct"],
            "wind_speed_kmh": data["wind_speed_kmh"],
            "wind_direction": data["wind_direction"],
            "source": "mock",
        },
        ensure_ascii=False,
    )


# ============ 天气预报 ============

class QueryWeatherForecastInput(BaseModel):
    location: str = Field(description="地点，格式同 query_current_weather")
    days: int = Field(
        default=3,
        ge=1,
        le=15,
        description="预报天数，1 到 15 天，默认 3 天",
    )
    unit: TemperatureUnit = Field(
        default="celsius",
        description="温度单位：celsius 或 fahrenheit",
    )


@tool(
    name_or_callable="query_weather_forecast",
    description=(
        "查询指定地点未来若干天的天气预报，返回每天的日期、天气状况、"
        "最高/最低温度和降水概率。"
        "当前实况用 query_current_weather，历史数据用 query_weather_history。"
    ),
    args_schema=QueryWeatherForecastInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "weather"},
)
async def query_weather_forecast(
    location: str,
    days: int = 3,
    unit: TemperatureUnit = "celsius",
) -> str:
    if not location.strip():
        return "ERROR: [param] query_weather_forecast 的 location 不能为空"

    forecasts = []
    for offset in range(days):
        day = _mock_forecast_day(location, offset)
        forecasts.append(
            {
                "date": day["date"],
                "condition": day["condition"],
                "temperature_max": _format_temperature(day["temperature_max_celsius"], unit),
                "temperature_min": _format_temperature(day["temperature_min_celsius"], unit),
                "precipitation_pct": day["precipitation_pct"],
            }
        )
    return json.dumps(
        {"location": location, "days": days, "forecasts": forecasts, "source": "mock"},
        ensure_ascii=False,
    )


# ============ 历史天气 ============

class QueryWeatherHistoryInput(BaseModel):
    location: str = Field(description="地点，格式同 query_current_weather")
    date: str = Field(description="历史日期，ISO 格式 YYYY-MM-DD")
    unit: TemperatureUnit = Field(
        default="celsius",
        description="温度单位：celsius 或 fahrenheit",
    )


@tool(
    name_or_callable="query_weather_history",
    description=(
        "查询指定地点在某个历史日期的天气记录，返回当天最高/最低温度、"
        "天气状况和降水量。"
        "当前实况用 query_current_weather，未来预报用 query_weather_forecast。"
    ),
    args_schema=QueryWeatherHistoryInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "weather"},
)
async def query_weather_history(
    location: str,
    date: str,
    unit: TemperatureUnit = "celsius",
) -> str:
    if not location.strip():
        return "ERROR: [param] query_weather_history 的 location 不能为空"

    try:
        date.fromisoformat(date)
    except ValueError:
        return (
            f"ERROR: [param] query_weather_history 的 date 不是合法的 ISO 日期: "
            f"{date!r}，格式应为 YYYY-MM-DD"
        )

    data = _mock_history(location, date)
    return json.dumps(
        {
            "location": data["location"],
            "date": data["date"],
            "condition": data["condition"],
            "temperature_max": _format_temperature(data["temperature_max_celsius"], unit),
            "temperature_min": _format_temperature(data["temperature_min_celsius"], unit),
            "precipitation_mm": data["precipitation_mm"],
            "source": "mock",
        },
        ensure_ascii=False,
    )