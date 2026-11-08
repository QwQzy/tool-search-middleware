"""基础编码工具。
"""

import base64
import hashlib
import urllib.parse
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


# ============ Base64 ============

class Base64EncodeInput(BaseModel):
    text: str = Field(description="要编码的文本，按 utf-8 编码后再转为 base64")
    urlsafe: bool = Field(
        default=False,
        description="是否使用 URL-safe 字母表（+/ 替换为 -_），用于嵌入 URL 或文件名",
    )


@tool(
    name_or_callable="base64_encode",
    description=(
        "将文本按 utf-8 编码后转为 base64 字符串。"
        "默认使用标准字母表；urlsafe=True 时使用 URL-safe 字母表。"
    ),
    args_schema=Base64EncodeInput,
    infer_schema=False,
)
async def base64_encode(text: str, urlsafe: bool = False) -> str:
    raw = text.encode("utf-8")
    if urlsafe:
        return base64.urlsafe_b64encode(raw).decode("ascii")
    return base64.b64encode(raw).decode("ascii")


class Base64DecodeInput(BaseModel):
    encoded: str = Field(
        description="base64 字符串。允许缺少尾部填充 =，工具会自动补齐",
    )
    urlsafe: bool = Field(
        default=False,
        description="是否按 URL-safe 字母表解码（-_ 等价于 +/，两种字符都会接受）",
    )


@tool(
    name_or_callable="base64_decode",
    description=(
        "解码 base64 字符串为 utf-8 文本。"
        "输入缺少尾部 = 时自动补齐。"
        "解码后的字节不是有效 utf-8 时返回 ERROR，并在错误消息中附带字节的 hex 表示。"
    ),
    args_schema=Base64DecodeInput,
    infer_schema=False,
)
async def base64_decode(encoded: str, urlsafe: bool = False) -> str:
    if urlsafe:
        encoded = encoded.replace("-", "+").replace("_", "/")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except ValueError as e:
        return f"ERROR: base64_decode 输入不是合法的 base64: {e}"
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return (
            "ERROR: base64_decode 解码成功，但内容不是有效的 utf-8 文本（可能是二进制）。"
            f"原始字节的 hex 表示: {raw.hex()}"
        )


# ============ URL 编码 ============

class UrlEncodeInput(BaseModel):
    text: str = Field(description="要编码的文本")
    plus: bool = Field(
        default=False,
        description=(
            "True 时空格编码为 +（表单提交风格）；"
            "False 时空格编码为 %20（RFC 3986 风格）"
        ),
    )


@tool(
    name_or_callable="url_encode",
    description=(
        "对文本进行 URL 百分号编码，编码所有保留字符（包括 /）。"
        "plus=True 时使用表单风格（空格 → +），默认使用 RFC 3986 风格（空格 → %20）。"
    ),
    args_schema=UrlEncodeInput,
    infer_schema=False,
)
async def url_encode(text: str, plus: bool = False) -> str:
    if plus:
        return urllib.parse.quote_plus(text, safe="")
    return urllib.parse.quote(text, safe="")


class UrlDecodeInput(BaseModel):
    text: str = Field(description="要解码的 URL 编码文本")
    plus: bool = Field(
        default=False,
        description="True 时将 + 视为空格（表单提交风格）",
    )


@tool(
    name_or_callable="url_decode",
    description=(
        "解码 URL 百分号编码的文本。"
        "plus=True 时将 + 解码为空格。"
        "非法的百分号序列（如孤立的 %）会原样保留，不会报错。"
    ),
    args_schema=UrlDecodeInput,
    infer_schema=False,
)
async def url_decode(text: str, plus: bool = False) -> str:
    if plus:
        return urllib.parse.unquote_plus(text)
    return urllib.parse.unquote(text)


# ============ Hex ============

class HexEncodeInput(BaseModel):
    text: str = Field(description="要编码的文本，按 utf-8 编码后转为 hex")


@tool(
    name_or_callable="hex_encode",
    description="将文本按 utf-8 编码后转为小写 hex 字符串。",
    args_schema=HexEncodeInput,
    infer_schema=False,
)
async def hex_encode(text: str) -> str:
    return text.encode("utf-8").hex()


class HexDecodeInput(BaseModel):
    encoded: str = Field(
        description="hex 字符串。允许空格分隔（如 '48 65 6c'），不接受 0x 前缀",
    )


@tool(
    name_or_callable="hex_decode",
    description=(
        "解码 hex 字符串为 utf-8 文本。"
        "解码后的字节不是有效 utf-8 时返回 ERROR，并在错误消息中附带字节的 hex 表示。"
    ),
    args_schema=HexDecodeInput,
    infer_schema=False,
)
async def hex_decode(encoded: str) -> str:
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as e:
        return f"ERROR: hex_decode 输入不是合法的 hex 字符串: {e}"
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return (
            "ERROR: hex_decode 解码成功，但内容不是有效的 utf-8 文本（可能是二进制）。"
            f"原始字节的 hex 表示: {raw.hex()}"
        )


# ============ 哈希 ============

class HashTextInput(BaseModel):
    text: str = Field(description="要计算哈希的文本，按 utf-8 编码后参与计算")
    algorithm: Literal["md5", "sha1", "sha256", "sha512"] = Field(
        default="sha256",
        description="哈希算法。MD5 和 SHA1 仅用于兼容性或校验和，不应用于安全场景",
    )


@tool(
    name_or_callable="hash_text",
    description=(
        "计算文本的哈希值，返回小写 hex 字符串。"
        "MD5 和 SHA1 仅用于兼容性或校验和；安全用途请使用 SHA256 或 SHA512。"
    ),
    args_schema=HashTextInput,
    infer_schema=False,
)
async def hash_text(
    text: str,
    algorithm: Literal["md5", "sha1", "sha256", "sha512"] = "sha256",
) -> str:
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return h.hexdigest()