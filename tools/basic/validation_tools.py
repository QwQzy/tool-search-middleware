"""基础校验工具。

所有工具满足基础工具的判定标准：只读、无副作用、可确定性验证。

返回值约定：
- 布尔型校验（is_uuid、regex_fullmatch 等）返回小写 "true" / "false"。
- JSON 语法或 Schema 校验返回结构化 JSON 对象，含 valid 与 errors 字段。
- 调用方提供的 pattern / schema 本身非法时返回 "ERROR: ..."，
  因为这是调用方错误，不是被校验内容的问题。
"""

import ipaddress
import json
import re
import uuid
from urllib.parse import urlparse

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from langchain.tools import tool
from pydantic import BaseModel, Field


# ============ JSON 语法校验 ============

class JsonValidateInput(BaseModel):
    text: str = Field(description="要校验是否为合法 JSON 的字符串")


@tool(
    name_or_callable="json_validate",
    description=(
        "校验字符串是否为合法 JSON。"
        "返回 JSON 对象，形如 {\"valid\": true, \"error\": null, \"position\": null}；"
        "非法时 valid 为 false，error 为错误描述，position 含 line/column/offset。"
    ),
    args_schema=JsonValidateInput,
    infer_schema=False,
)
async def json_validate(text: str) -> str:
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "valid": False,
                "error": e.msg,
                "position": {"line": e.lineno, "column": e.colno, "offset": e.pos},
            },
            ensure_ascii=False,
        )
    return json.dumps({"valid": True, "error": None, "position": None}, ensure_ascii=False)


# ============ JSON Schema 校验 ============

class JsonSchemaValidateInput(BaseModel):
    text: str = Field(description="要校验的 JSON 字符串")
    schema: str = Field(
        description=(
            "JSON Schema 字符串，本身必须是合法 JSON。"
            "按 JSON Schema Draft 2020-12 解释"
        ),
    )


@tool(
    name_or_callable="json_schema_validate",
    description=(
        "用 JSON Schema (Draft 2020-12) 校验 JSON 字符串。"
        "返回 JSON 对象，形如 {\"valid\": bool, \"errors\": [...]}。"
        "每个错误含 path（JSON Pointer 风格的路径）和 message 字段。"
        "被校验的 JSON 本身不是合法 JSON 时，valid 为 false 并在 errors 中报告。"
        "schema 本身不是合法 JSON 或不符合 JSON Schema 元模式时返回 ERROR。"
    ),
    args_schema=JsonSchemaValidateInput,
    infer_schema=False,
)
async def json_schema_validate(text: str, schema: str) -> str:
    try:
        instance = json.loads(text)
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "valid": False,
                "errors": [{"path": "$", "message": f"被校验的 JSON 解析失败: {e.msg}"}],
            },
            ensure_ascii=False,
        )

    try:
        schema_obj = json.loads(schema)
    except json.JSONDecodeError as e:
        return f"ERROR: json_schema_validate 的 schema 不是合法 JSON: {e.msg}"

    try:
        Draft202012Validator.check_schema(schema_obj)
    except SchemaError as e:
        return f"ERROR: json_schema_validate 的 schema 不符合 JSON Schema 元模式: {e.message}"

    validator = Draft202012Validator(schema_obj)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))

    if not errors:
        return json.dumps({"valid": True, "errors": []}, ensure_ascii=False)

    # 限制错误数量，避免异常输入导致输出过大
    formatted = []
    for err in errors[:100]:
        path = "$" + "".join(
            f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path
        )
        formatted.append({"path": path, "message": err.message})

    return json.dumps({"valid": False, "errors": formatted}, ensure_ascii=False)


# ============ JSON 顶层类型 ============

class JsonTypeInput(BaseModel):
    text: str = Field(description="要检查顶层类型的 JSON 字符串")


@tool(
    name_or_callable="json_type",
    description=(
        "返回 JSON 字符串的顶层类型，取值为以下之一："
        "null、boolean、number、string、array、object。"
        "输入不是合法 JSON 时返回 ERROR。"
    ),
    args_schema=JsonTypeInput,
    infer_schema=False,
)
async def json_type(text: str) -> str:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        return f"ERROR: json_type 输入不是合法 JSON: {e.msg}"
    return _json_type_of(value)


def _json_type_of(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


# ============ 正则整串匹配 ============

class RegexFullmatchInput(BaseModel):
    pattern: str = Field(
        description=(
            "正则表达式（Python re 语法）。"
            "使用 re.fullmatch 语义：整个字符串必须完全匹配 pattern，不接受部分匹配"
        ),
    )
    text: str = Field(description="要校验的字符串")


@tool(
    name_or_callable="regex_fullmatch",
    description=(
        "判断字符串是否完全匹配正则 pattern，返回 \"true\" 或 \"false\"（小写）。"
        "与 regex_search（任意位置匹配）不同，本工具要求整个字符串从头到尾都被 pattern 覆盖。"
        "pattern 非法时返回 ERROR。"
    ),
    args_schema=RegexFullmatchInput,
    infer_schema=False,
)
async def regex_fullmatch(pattern: str, text: str) -> str:
    try:
        matched = re.fullmatch(pattern, text) is not None
    except re.error as e:
        return f"ERROR: regex_fullmatch 的 pattern 非法: {e}"
    return "true" if matched else "false"


# ============ UUID 格式检查 ============

class IsUuidInput(BaseModel):
    text: str = Field(description="要检查的字符串")


@tool(
    name_or_callable="is_uuid",
    description=(
        "判断字符串是否为合法 UUID（RFC 4122 格式），返回 \"true\" 或 \"false\"。"
        "接受带连字符（8-4-4-4-12）的标准形式，也接受无连字符、带花括号、"
        "带 urn:uuid: 前缀等形式。"
    ),
    args_schema=IsUuidInput,
    infer_schema=False,
)
async def is_uuid(text: str) -> str:
    try:
        uuid.UUID(text)
    except (ValueError, AttributeError, TypeError):
        return "false"
    return "true"


# ============ 邮箱格式检查 ============

# 大致对应 HTML5 email 输入的校验模式。不实现完整的 RFC 5322，
# 因为它允许多数实际使用中不会出现的边界情况（引号、注释、IP 字面量域等），
# 对模型和用户的预期帮助不大。带点域名是必须的，这一选择拒绝 a@b 这类地址。
_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)


class IsEmailInput(BaseModel):
    text: str = Field(description="要检查的字符串")


@tool(
    name_or_callable="is_email",
    description=(
        "判断字符串是否为常见的邮箱格式，返回 \"true\" 或 \"false\"。"
        "按 HTML5 email 输入的标准实现，不实现完整的 RFC 5322；"
        "带引号、注释、IP 字面量域的地址会被判为 false。"
        "域部分必须至少包含一个点，a@b 这类无点域名会被判为 false。"
    ),
    args_schema=IsEmailInput,
    infer_schema=False,
)
async def is_email(text: str) -> str:
    if not text:
        return "false"
    return "true" if _EMAIL_PATTERN.match(text) else "false"


# ============ URL 格式检查 ============

_ALLOWED_URL_SCHEMES = {"http", "https", "ftp", "ftps", "ws", "wss"}


class IsUrlInput(BaseModel):
    text: str = Field(description="要检查的字符串")


@tool(
    name_or_callable="is_url",
    description=(
        "判断字符串是否为合法 URL，返回 \"true\" 或 \"false\"。"
        "接受 http、https、ftp、ftps、ws、wss 协议；"
        "必须包含 scheme 和 host；端口必须是合法整数。"
        "不接受 mailto:、file:、data: 等其他协议。"
    ),
    args_schema=IsUrlInput,
    infer_schema=False,
)
async def is_url(text: str) -> str:
    try:
        parsed = urlparse(text)
    except ValueError:
        return "false"
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        return "false"
    if not parsed.netloc:
        return "false"
    try:
        _ = parsed.port  # 触发端口校验；非法端口会抛 ValueError
    except ValueError:
        return "false"
    return "true"


# ============ IP 地址格式检查 ============

class IsIpAddressInput(BaseModel):
    text: str = Field(description="要检查的字符串")


@tool(
    name_or_callable="is_ip_address",
    description=(
        "判断字符串是否为合法 IP 地址，返回 \"true\" 或 \"false\"。"
        "同时接受 IPv4 和 IPv6。不接受带端口、带前缀长度（CIDR）、"
        "带方括号的地址；这些形式需先用其他工具切分。"
    ),
    args_schema=IsIpAddressInput,
    infer_schema=False,
)
async def is_ip_address(text: str) -> str:
    try:
        ipaddress.ip_address(text)
    except (ValueError, TypeError):
        return "false"
    return "true"