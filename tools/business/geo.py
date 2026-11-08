"""业务工具：地理位置查询与计算。

两类工具混在这个文件里：

1. 依赖外部地理数据源的工具（geocode_address、reverse_geocode）——结果
   反映数据源当前状态，不满足可确定性验证。
2. 纯计算工具（haversine_distance、check_geofence）——给定坐标输入，输出
   是确定性的。它们仍放在这里而非 basic/，因为坐标承载地理语义，且与依赖
   外部数据源的工具共享 scene="geo" 的场景标签，检索时会被同一批查询命中。

每个工具携带 extras：
- readOnlyHint: True
- scene: "geo"

坐标约定：统一使用 WGS84 (lat, lon)，纬度在前。国内业务如果需要 GCJ-02
（火星坐标系），应在数据源层转换后传入；工具本身不做坐标系转换，也不做
WGS84/GCJ-02/BD-09 之间的互转——坐标系混用会产生几百米的偏移，这类错误
应该在最靠近数据源的地方暴露，而不是被工具静默转换掩盖。

工具函数返回 mock 数据，带 "source": "mock" 标记。
"""

import json
import math
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


# ============ 校验与辅助函数 ============

def _validate_coord(lat: float, lon: float) -> str | None:
    if not (-90.0 <= lat <= 90.0):
        return f"ERROR: [param] 纬度 {lat} 超出合法范围 [-90, 90]"
    if not (-180.0 <= lon <= 180.0):
        return f"ERROR: [param] 经度 {lon} 超出合法范围 [-180, 180]"
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点间大圆距离，单位千米。地球半径取 WGS84 平均半径 6371.0088 km。"""
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _point_in_polygon(lat: float, lon: float, vertices: list[tuple[float, float]]) -> bool:
    """射线法判断点是否在多边形内。

    vertices 按 [lat, lon] 顺序给出，首尾不重复闭合（由调用方保证），
    算法内部用最后一个顶点连回第一个顶点的方式隐含闭合。

    边界点（恰好在边上）的行为与射线方向有关，判定为"在一侧"，不保证
    稳定落在内或外。这一行为与 geofence 场景的实际精度需求一致：
    围栏边界本身没有亚米级精度，边界点的归属没有意义。
    """
    inside = False
    n = len(vertices)
    j = n - 1
    for i in range(n):
        lat_i, lon_i = vertices[i]
        lat_j, lon_j = vertices[j]
        if (lon_i > lon) != (lon_j > lon):
            x_intersect = (lon - lon_i) * (lat_j - lat_i) / (lon_j - lon_i) + lat_i
            if lat < x_intersect:
                inside = not inside
        j = i
    return inside


def _mock_geocode(address: str) -> dict:
    return {
        "input": address,
        "formatted_address": f"{address}（mock 结果）",
        "lat": 22.5431,
        "lon": 114.0579,
        "country": "CN",
        "region": "广东",
        "city": "深圳",
        "district": "南山区",
        "confidence": 0.92,
        "level": "district",
    }


def _mock_reverse_geocode(lat: float, lon: float) -> dict:
    return {
        "lat": lat,
        "lon": lon,
        "formatted_address": f"广东省深圳市南山区（{lat:.4f}, {lon:.4f} 附近的 mock 结果）",
        "country": "CN",
        "region": "广东",
        "city": "深圳",
        "district": "南山区",
        "street": "科技园南路",
        "confidence": 0.88,
        "level": "street",
    }


# ============ 地理编码 ============

class GeocodeAddressInput(BaseModel):
    address: str = Field(
        min_length=1,
        description="要解析的地址，如 \"深圳市南山区科技园\" 或 \"1600 Amphitheatre Pkwy, Mountain View, CA\"",
    )
    city: str | None = Field(
        default=None,
        description="可选的限定城市，用于消歧。当街道名或地名在多个城市存在时建议传入",
    )


@tool(
    name_or_callable="geocode_address",
    description=(
        "将地址文本解析为经纬度坐标，返回 WGS84 坐标、标准化地址、行政区和置信度。"
        "只做正向解析（地址 → 坐标）；坐标转地址请用 reverse_geocode。"
        "结果依赖外部地理数据源的覆盖，地址越粗略置信度越低。"
        "行政区级地址（如 \"深圳市\"）通常返回该行政区中心点，不是边界。"
    ),
    args_schema=GeocodeAddressInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "geo"},
)
async def geocode_address(address: str, city: str | None = None) -> str:
    if not address.strip():
        return "ERROR: [param] geocode_address 的 address 不能为空"

    # 真实实现中 city 会作为查询提示传给地理编码服务
    data = _mock_geocode(address)
    return json.dumps({**data, "source": "mock"}, ensure_ascii=False)


# ============ 逆地理编码 ============

class ReverseGeocodeInput(BaseModel):
    lat: float = Field(ge=-90, le=90, description="纬度，WGS84")
    lon: float = Field(ge=-180, le=180, description="经度，WGS84")


@tool(
    name_or_callable="reverse_geocode",
    description=(
        "将经纬度坐标解析为地址，返回标准化地址、行政区、街道和置信度。"
        "只做反向解析（坐标 → 地址）；地址转坐标请用 geocode_address。"
        "返回的 level 字段标明解析的最细粒度，可能为 country、region、city、"
        "district 或 street，取决于数据源在该坐标上的覆盖。"
    ),
    args_schema=ReverseGeocodeInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "geo"},
)
async def reverse_geocode(lat: float, lon: float) -> str:
    err = _validate_coord(lat, lon)
    if err:
        return err

    data = _mock_reverse_geocode(lat, lon)
    return json.dumps({**data, "source": "mock"}, ensure_ascii=False)


# ============ 球面距离 ============

type DistanceUnit = Literal["km", "m", "mi", "nmi"]


_UNIT_TO_KM: dict[DistanceUnit, float] = {
    "km": 1.0,
    "m": 1e-3,
    "mi": 1.609344,
    "nmi": 1.852,
}


class HaversineDistanceInput(BaseModel):
    from_lat: float = Field(ge=-90, le=90, description="起点纬度")
    from_lon: float = Field(ge=-180, le=180, description="起点经度")
    to_lat: float = Field(ge=-90, le=90, description="终点纬度")
    to_lon: float = Field(ge=-180, le=180, description="终点经度")
    unit: DistanceUnit = Field(
        default="km",
        description="距离单位：km 千米、m 米、mi 英里、nmi 海里。默认千米",
    )


@tool(
    name_or_callable="haversine_distance",
    description=(
        "计算两个 WGS84 坐标点之间的大圆距离。"
        "使用 haversine 公式，地球按平均半径球体处理，"
        "对城市级到国家级的距离误差在 0.5% 以内；"
        "需要亚米级精度时应该使用考虑椭球模型或地形起伏的地理计算服务。"
        "输入是坐标，不是地址；地址需要先用 geocode_address 解析。"
    ),
    args_schema=HaversineDistanceInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "geo"},
)
async def haversine_distance(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    unit: DistanceUnit = "km",
) -> str:
    err = _validate_coord(from_lat, from_lon) or _validate_coord(to_lat, to_lon)
    if err:
        return err

    km = _haversine_km(from_lat, from_lon, to_lat, to_lon)
    return json.dumps(
        {
            "from": {"lat": from_lat, "lon": from_lon},
            "to": {"lat": to_lat, "lon": to_lon},
            "distance": km / _UNIT_TO_KM[unit],
            "unit": unit,
            "distance_km": km,
            "source": "computed",
        },
        ensure_ascii=False,
    )


# ============ 地理围栏判断 ============

class CircleFence(BaseModel):
    type: Literal["circle"] = Field(
        description="围栏类型，圆形填 \"circle\"",
    )
    center_lat: float = Field(ge=-90, le=90, description="圆心纬度")
    center_lon: float = Field(ge=-180, le=180, description="圆心经度")
    radius_km: float = Field(gt=0, description="半径，单位千米")


class PolygonFence(BaseModel):
    type: Literal["polygon"] = Field(
        description="围栏类型，多边形填 \"polygon\"",
    )
    vertices: list[list[float]] = Field(
        min_length=3,
        description=(
            "多边形顶点列表，每个顶点为 [lat, lon]，至少 3 个。"
            "按顺序给出，首尾自动闭合，不需要重复第一个顶点"
        ),
    )


class CheckGeofenceInput(BaseModel):
    lat: float = Field(ge=-90, le=90, description="待判断点的纬度")
    lon: float = Field(ge=-180, le=180, description="待判断点的经度")
    fence: CircleFence | PolygonFence = Field(
        description=(
            "围栏定义。用 type 字段区分："
            "圆形填 {\"type\": \"circle\", \"center_lat\": ..., \"center_lon\": ..., \"radius_km\": ...}；"
            "多边形填 {\"type\": \"polygon\", \"vertices\": [[lat, lon], ...]}"
        ),
    )


@tool(
    name_or_callable="check_geofence",
    description=(
        "判断一个坐标点是否在指定的地理围栏内。"
        "支持圆形围栏（圆心 + 半径）和多边形围栏（顶点列表）。"
        "圆形围栏按球面大圆距离判断；多边形围栏按射线法判断，"
        "边界点的归属不保证稳定（在围栏边界本身没有亚米级精度时没有实际意义）。"
        "只做位置判断，不做进出事件检测；事件检测需要状态和时序，属于业务系统的职责。"
    ),
    args_schema=CheckGeofenceInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "geo"},
)
async def check_geofence(
    lat: float,
    lon: float,
    fence: CircleFence | PolygonFence,
) -> str:
    err = _validate_coord(lat, lon)
    if err:
        return err

    if isinstance(fence, CircleFence):
        center_err = _validate_coord(fence.center_lat, fence.center_lon)
        if center_err:
            return center_err
        distance_km = _haversine_km(
            lat, lon, fence.center_lat, fence.center_lon
        )
        inside = distance_km <= fence.radius_km
        return json.dumps(
            {
                "point": {"lat": lat, "lon": lon},
                "fence_type": "circle",
                "inside": inside,
                "distance_km": distance_km,
                "radius_km": fence.radius_km,
                "source": "computed",
            },
            ensure_ascii=False,
        )

    # 多边形
    try:
        vertices = [(v[0], v[1]) for v in fence.vertices]
    except (IndexError, TypeError):
        return "ERROR: [param] check_geofence 的 vertices 每个元素必须是 [lat, lon] 形式"

    for v_lat, v_lon in vertices:
        v_err = _validate_coord(v_lat, v_lon)
        if v_err:
            return v_err

    inside = _point_in_polygon(lat, lon, vertices)
    return json.dumps(
        {
            "point": {"lat": lat, "lon": lon},
            "fence_type": "polygon",
            "inside": inside,
            "vertex_count": len(vertices),
            "source": "computed",
        },
        ensure_ascii=False,
    )