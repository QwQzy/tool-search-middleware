"""业务工具：IP 情报查询。

三类工具：IP 归属地、ASN 查询、威胁情报。三者共享同一个输入形态
（一个 IP 地址），但数据源、置信度和语义差异很大，因此拆成三个工具，
而不是一个"查 IP 所有信息"的合并工具。

拆开的理由：合并工具会让模型每次调用都拿到全量字段，其中大部分与当前
问题无关；同时，不同数据源的失败语义不同——归属地查询失败通常是 IP 不
可路由，威胁情报查询失败通常是供应商配额耗尽，合并后模型无法从错误消息
区分这两种情况。

每个工具携带 extras：
- readOnlyHint: True
- scene: "ip"

只做 IPv4/IPv6 单地址查询，不做 CIDR 批量展开、不做域名解析——域名解析
需要联网解析器，属于 capability 层；CIDR 展开属于批量处理，应由调用方
拆分后逐个查询。

工具函数返回 mock 数据，带 "source": "mock" 标记。
"""

import ipaddress
import json
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


def _validate_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | str:
    """校验并规范化 IP 地址。合法返回 ip_address 对象，非法返回 ERROR 字符串。"""
    if not value.strip():
        return "ERROR: [param] IP 地址不能为空"
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return (
            f"ERROR: [param] {value!r} 不是合法的 IPv4/IPv6 地址。"
            "本工具只接受单地址，不接受 CIDR 表示（如 192.168.0.0/24）"
        )
    if ip.is_private:
        return (
            f"ERROR: [business] {value} 是私有地址，不在 IP 情报数据源的覆盖范围内。"
            "私有地址的归属地需要查询内部网络管理系统的 DHCP/VLAN 记录"
        )
    if ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return (
            f"ERROR: [business] {value} 是保留地址（loopback/link-local/reserved），"
            "没有公网归属信息"
        )
    return ip


def _mock_geo(ip: str) -> dict:
    return {
        "ip": ip,
        "country": "CN",
        "country_name": "中国",
        "region": "广东",
        "city": "深圳",
        "postal_code": "518000",
        "latitude": 22.5431,
        "longitude": 114.0579,
        "timezone": "Asia/Shanghai",
        "isp": "China Telecom",
        "accuracy_radius_km": 50,
    }


def _mock_asn(ip: str) -> dict:
    return {
        "ip": ip,
        "asn": 4134,
        "as_name": "CHINANET-BACKBONE",
        "organization": "No.31 Jin-rong Street",
        "country": "CN",
        "rir": "APNIC",
    }


def _mock_threat(ip: str, verdict: str) -> dict:
    return {
        "ip": ip,
        "verdict": verdict,
        "confidence": 85 if verdict == "malicious" else 10,
        "categories": ["scanner", "brute_force"] if verdict == "malicious" else [],
        "first_seen": "2023-11-02T08:00:00Z" if verdict == "malicious" else None,
        "last_seen": "2024-01-14T22:15:00Z" if verdict == "malicious" else None,
        "sources": ["internal-honeypot"] if verdict == "malicious" else [],
    }


# ============ IP 归属地 ============

class QueryIpGeolocationInput(BaseModel):
    ip: str = Field(description="要查询的 IPv4 或 IPv6 地址，单地址，不接受 CIDR")


@tool(
    name_or_callable="query_ip_geolocation",
    description=(
        "查询 IP 地址的地理归属，返回国家、省/州、城市、经纬度、时区、ISP "
        "和精度半径。"
        "结果依赖 IP 情报数据源的更新频率，通常按天级更新，"
        "对刚分配的 IP 可能返回邻近地址的归属信息。"
        "IP 对应的 ASN 信息用 query_ip_asn，威胁情报用 query_ip_threat_intel。"
    ),
    args_schema=QueryIpGeolocationInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "ip"},
)
async def query_ip_geolocation(ip: str) -> str:
    checked = _validate_ip(ip)
    if isinstance(checked, str):
        return checked

    data = _mock_geo(str(checked))
    return json.dumps({**data, "source": "mock"}, ensure_ascii=False)


# ============ IP ASN ============

class QueryIpAsnInput(BaseModel):
    ip: str = Field(description="要查询的 IPv4 或 IPv6 地址，单地址，不接受 CIDR")


@tool(
    name_or_callable="query_ip_asn",
    description=(
        "查询 IP 地址所属的 AS（自治系统），返回 ASN、AS 名称、所属组织和 RIR。"
        "归属地信息用 query_ip_geolocation，威胁情报用 query_ip_threat_intel。"
    ),
    args_schema=QueryIpAsnInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "ip"},
)
async def query_ip_asn(ip: str) -> str:
    checked = _validate_ip(ip)
    if isinstance(checked, str):
        return checked

    data = _mock_asn(str(checked))
    return json.dumps({**data, "source": "mock"}, ensure_ascii=False)


# ============ IP 威胁情报 ============

type ThreatVerdict = Literal["malicious", "suspicious", "clean", "unknown"]


class QueryIpThreatIntelInput(BaseModel):
    ip: str = Field(description="要查询的 IPv4 或 IPv6 地址，单地址，不接受 CIDR")


@tool(
    name_or_callable="query_ip_threat_intel",
    description=(
        "查询 IP 地址的威胁情报，返回判定结果（malicious/suspicious/clean/unknown）、"
        "置信度、威胁类别、首次和最近出现时间、情报来源。"
        "verdict 为 unknown 时表示数据源没有该 IP 的记录，"
        "不等于该 IP 是安全的——数据源的覆盖面有限。"
        "归属地用 query_ip_geolocation，ASN 用 query_ip_asn。"
    ),
    args_schema=QueryIpThreatIntelInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "ip"},
)
async def query_ip_threat_intel(ip: str) -> str:
    checked = _validate_ip(ip)
    if isinstance(checked, str):
        return checked

    # mock：按 IP 末位奇偶决定 verdict，便于测试两个分支
    last_octet = str(checked).split(".")[-1] if isinstance(checked, ipaddress.IPv4Address) else "0"
    try:
        is_even = int(last_octet) % 2 == 0
    except ValueError:
        is_even = True
    verdict: ThreatVerdict = "clean" if is_even else "malicious"

    data = _mock_threat(str(checked), verdict)
    return json.dumps({**data, "source": "mock"}, ensure_ascii=False)