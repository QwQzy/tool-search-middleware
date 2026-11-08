"""基础文本工具。

正则类工具使用 Python re 模块的默认 flags（不做 DOTALL、IGNORECASE 等
特殊处理）；需要时由调用方在 pattern 中内联（如 (?i)）。pattern 非法时
返回 ERROR，不抛异常。

长度工具统计的是"字符数 / UTF-8 字节数 / 行数"，不统计 token 数：
token 取决于具体 tokenizer，同一个字符串在不同模型下 token 数不同，
引入 token 计数会让结果依赖"哪个模型"这个外部状态，破坏可确定性验证。
"""

import json
import re
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


# ============ 长度 ============

class TextLengthInput(BaseModel):
    text: str = Field(description="要统计的文本")


@tool(
    name_or_callable="text_length",
    description=(
        "统计文本的字符数、UTF-8 字节数和行数，以 JSON 对象返回，"
        "形如 {\"chars\": 5, \"bytes\": 7, \"lines\": 1}。"
        "字符数按 Unicode 码点计，字节数按 utf-8 编码后的字节数计。"
    ),
    args_schema=TextLengthInput,
    infer_schema=False,
)
async def text_length(text: str) -> str:
    try:
        byte_len = len(text.encode("utf-8"))
    except UnicodeEncodeError as e:
        return f"ERROR: text_length 编码失败，文本包含无法编码为 utf-8 的字符: {e}"
    return json.dumps(
        {
            "chars": len(text),
            "bytes": byte_len,
            "lines": len(text.splitlines()),
        },
        ensure_ascii=False,
    )


# ============ 大小写 ============

class TextCaseInput(BaseModel):
    text: str = Field(description="要转换大小写的文本")
    mode: Literal["lower", "upper", "title", "capitalize", "swapcase"] = Field(
        description=(
            "转换模式：lower 全部小写；upper 全部大写；"
            "title 每个单词首字母大写；capitalize 首字符大写其余小写；"
            "swapcase 大小写互换"
        ),
    )


@tool(
    name_or_callable="text_case",
    description=(
        "按指定模式转换文本大小写。"
        "lower/upper 为全小写/全大写；title 为每词首字母大写；"
        "capitalize 为首字母大写、其余小写；swapcase 为大小写互换。"
    ),
    args_schema=TextCaseInput,
    infer_schema=False,
)
async def text_case(
    text: str,
    mode: Literal["lower", "upper", "title", "capitalize", "swapcase"],
) -> str:
    match mode:
        case "lower":
            return text.lower()
        case "upper":
            return text.upper()
        case "title":
            return text.title()
        case "capitalize":
            return text.capitalize()
        case "swapcase":
            return text.swapcase()


# ============ 去除首尾字符 ============

class TextStripInput(BaseModel):
    text: str = Field(description="要处理的文本")
    chars: str | None = Field(
        default=None,
        description=(
            "要去除的字符集合。None 表示去除首尾空白（空格、制表符、换行等）；"
            "非 None 时去除首尾属于该集合的任意字符"
        ),
    )


@tool(
    name_or_callable="text_strip",
    description=(
        "去除文本首尾字符。"
        "默认去除空白（空格、制表符、换行、回车等 Unicode 空白）；"
        "传入 chars 时去除首尾属于该字符集合的字符。"
    ),
    args_schema=TextStripInput,
    infer_schema=False,
)
async def text_strip(text: str, chars: str | None = None) -> str:
    if chars is None:
        return text.strip()
    return text.strip(chars)


# ============ 截断 ============

class TextTruncateInput(BaseModel):
    text: str = Field(description="要截断的文本")
    max_chars: int = Field(
        ge=0,
        description="截断后的最大字符数（含后缀）。必须 >= 0",
    )
    suffix: str = Field(
        default="...",
        description="截断时追加的后缀，通常为 \"...\"；传空字符串表示不加后缀",
    )


@tool(
    name_or_callable="text_truncate",
    description=(
        "将文本截断到 max_chars 个字符以内（含后缀）。"
        "原文本长度 <= max_chars 时原样返回，不追加后缀。"
        "否则保留前 max_chars - len(suffix) 个字符再拼上 suffix。"
        "max_chars 小于 suffix 长度时返回 ERROR。"
    ),
    args_schema=TextTruncateInput,
    infer_schema=False,
)
async def text_truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    if max_chars < len(suffix):
        return (
            f"ERROR: text_truncate 的 max_chars={max_chars} "
            f"小于 suffix 长度 {len(suffix)}"
        )
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(suffix)
    return text[:keep] + suffix


# ============ 替换 ============

class TextReplaceInput(BaseModel):
    text: str = Field(description="原文本")
    old: str = Field(description="要被替换的子串，不能为空")
    new: str = Field(description="替换后的内容")
    count: int = Field(
        default=-1,
        description="最多替换次数。-1 表示全部替换；0 表示不替换",
    )


@tool(
    name_or_callable="text_replace",
    description=(
        "将文本中的 old 子串替换为 new。"
        "count=-1 时替换所有匹配；count>=0 时最多替换 count 次。"
        "old 为空字符串时返回 ERROR（str.replace 对空串的行为容易引起误解）。"
    ),
    args_schema=TextReplaceInput,
    infer_schema=False,
)
async def text_replace(text: str, old: str, new: str, count: int = -1) -> str:
    if old == "":
        return "ERROR: text_replace 的 old 不能为空字符串"
    return text.replace(old, new, count)


# ============ 拆分 ============

class TextSplitInput(BaseModel):
    text: str = Field(description="要拆分的文本")
    separator: str | None = Field(
        default=None,
        description=(
            "拆分分隔符。None 表示按任意连续空白拆分；"
            "非 None 时按该字面分隔符拆分"
        ),
    )
    maxsplit: int = Field(
        default=-1,
        description="最大拆分次数。-1 表示不限次数，0 表示不拆分",
    )


@tool(
    name_or_callable="text_split",
    description=(
        "按分隔符拆分文本，返回 JSON 数组字符串。"
        "separator 为 None 时按任意连续空白拆分；"
        "为非空字符串时按字面分隔符拆分，且连续分隔符之间会产生空字符串元素。"
    ),
    args_schema=TextSplitInput,
    infer_schema=False,
)
async def text_split(
    text: str,
    separator: str | None = None,
    maxsplit: int = -1,
) -> str:
    if separator is not None and separator == "":
        return "ERROR: text_split 的 separator 不能为空字符串；按字符拆分请用 list(text)"
    parts = text.split(separator, maxsplit)
    return json.dumps(parts, ensure_ascii=False)


# ============ 子串包含 ============

class TextContainsInput(BaseModel):
    text: str = Field(description="被搜索的文本")
    needle: str = Field(description="要查找的子串，不能为空")


@tool(
    name_or_callable="text_contains",
    description=(
        "判断 needle 是否为 text 的子串，返回 \"true\" 或 \"false\"（小写）。"
        "用于避免模型在长文本上判断子串是否出现时出错。"
        "needle 为空字符串时返回 ERROR。"
    ),
    args_schema=TextContainsInput,
    infer_schema=False,
)
async def text_contains(text: str, needle: str) -> str:
    if needle == "":
        return "ERROR: text_contains 的 needle 不能为空字符串"
    return "true" if needle in text else "false"


# ============ 正则：匹配检查 ============

class RegexSearchInput(BaseModel):
    pattern: str = Field(description="正则表达式（Python re 语法）")
    text: str = Field(description="要搜索的文本")


@tool(
    name_or_callable="regex_search",
    description=(
        "判断正则 pattern 是否在 text 中匹配到任意位置，返回 \"true\" 或 \"false\"（小写）。"
        "使用 re.search 语义（可在任意位置匹配），不是 re.match 语义（仅从开头匹配）。"
        "pattern 非法时返回 ERROR，并在消息中附带 re 的原始错误。"
    ),
    args_schema=RegexSearchInput,
    infer_schema=False,
)
async def regex_search(pattern: str, text: str) -> str:
    try:
        matched = re.search(pattern, text) is not None
    except re.error as e:
        return f"ERROR: regex_search 的 pattern 非法: {e}"
    return "true" if matched else "false"


# ============ 正则：提取全部匹配 ============

class RegexFindallInput(BaseModel):
    pattern: str = Field(description="正则表达式（Python re 语法）")
    text: str = Field(description="要搜索的文本")


@tool(
    name_or_callable="regex_findall",
    description=(
        "返回正则在文本中所有匹配的完整内容，以 JSON 数组字符串表示。"
        "只返回每个匹配的完整文本（m.group(0)），不返回捕获组内容。"
        "没有匹配时返回空数组 []。"
        "pattern 非法时返回 ERROR。"
    ),
    args_schema=RegexFindallInput,
    infer_schema=False,
)
async def regex_findall(pattern: str, text: str) -> str:
    try:
        matches = [m.group(0) for m in re.finditer(pattern, text)]
    except re.error as e:
        return f"ERROR: regex_findall 的 pattern 非法: {e}"
    return json.dumps(matches, ensure_ascii=False)


# ============ 正则：替换 ============

class RegexSubInput(BaseModel):
    pattern: str = Field(description="正则表达式（Python re 语法）")
    replacement: str = Field(
        description=(
            "替换字符串。支持反向引用，如 \\\\1、\\\\2 引用捕获组内容；"
            "没有捕获组的简单替换直接写目标字符串即可"
        ),
    )
    text: str = Field(description="要处理的文本")


@tool(
    name_or_callable="regex_sub",
    description=(
        "用 replacement 替换 text 中所有匹配 pattern 的部分，返回替换后的文本。"
        "replacement 支持反向引用（\\\\1、\\\\2 等引用捕获组）。"
        "pattern 或 replacement 非法时返回 ERROR。"
    ),
    args_schema=RegexSubInput,
    infer_schema=False,
)
async def regex_sub(pattern: str, replacement: str, text: str) -> str:
    try:
        return re.sub(pattern, replacement, text)
    except re.error as e:
        return f"ERROR: regex_sub 的正则模式或替换字符串非法: {e}"