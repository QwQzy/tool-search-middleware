"""业务工具：私域工具占位。

private 泛指接入内部系统或私有 API 的业务工具。实际项目中，这个文件通常
会被进一步拆成多个业务域文件（如 ticket.py、user_profile.py、inventory.py），
当前只放三个有代表性的工具，覆盖只读查询、有副作用写操作、以及带业务规则
校验的查询三类形态。

三个工具携带 extras：
- readOnlyHint: True/False —— 决定 Exec 网关是否触发写操作确认和幂等检查
- scene: 业务域标签 —— 供 Tool Search 场景硬过滤使用

create_ticket 是写操作。幂等键由 ExecGatewayMiddleware 从参数中的
external_request_id（优先）或 user_id + title 派生，工具本身不管理幂等。

工具函数返回 mock 数据，带 "source": "mock" 标记。接入真实系统时，把 mock
返回替换为内部 API 调用，保持返回结构和错误分类不变。
"""

import json
import uuid
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


type TicketPriority = Literal["low", "medium", "high", "urgent"]
type TicketStatus = Literal["open", "in_progress", "resolved", "closed"]


def _mock_user_profile(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "display_name": "张三",
        "email": "zhangsan@example.com",
        "plan": "enterprise",
        "created_at": "2022-06-01T09:00:00+08:00",
        "last_active_at": "2024-01-14T18:22:00+08:00",
        "tags": ["vip", "beta_tester"],
    }


def _mock_ticket(ticket_id: str) -> dict:
    return {
        "ticket_id": ticket_id,
        "title": "无法登录控制台",
        "status": "in_progress",
        "priority": "high",
        "assignee": "support-team-a",
        "created_at": "2024-01-12T14:05:00+08:00",
        "updated_at": "2024-01-15T09:30:00+08:00",
    }


# ============ 用户画像查询 ============

class QueryUserProfileInput(BaseModel):
    user_id: str = Field(description="用户 ID，内部系统的稳定标识")


@tool(
    name_or_callable="query_user_profile",
    description=(
        "查询用户画像，返回显示名、邮箱、套餐、注册时间、最近活跃时间、标签。"
        "只按 user_id 精确查询；按邮箱或姓名模糊搜索请使用内部用户检索系统。"
    ),
    args_schema=QueryUserProfileInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "user"},
)
async def query_user_profile(user_id: str) -> str:
    if not user_id.strip():
        return "ERROR: [param] query_user_profile 的 user_id 不能为空"

    profile = _mock_user_profile(user_id)
    return json.dumps({**profile, "source": "mock"}, ensure_ascii=False)


# ============ 工单创建 ============

class CreateTicketInput(BaseModel):
    user_id: str = Field(description="提交工单的用户 ID")
    title: str = Field(
        min_length=1,
        max_length=200,
        description="工单标题，1 到 200 字符",
    )
    description: str = Field(
        default="",
        max_length=5000,
        description="工单详细描述，可选，最多 5000 字符",
    )
    priority: TicketPriority = Field(
        default="medium",
        description="优先级：low、medium、high、urgent",
    )
    external_request_id: str | None = Field(
        default=None,
        description=(
            "外部请求 ID，用于幂等。调用方重试时应传入相同的值，"
            "网关会据此判断是否为同一次逻辑操作。不传时由网关从 user_id + title 派生"
        ),
    )


@tool(
    name_or_callable="create_ticket",
    description=(
        "创建一条内部工单，返回工单 ID、状态和创建时间。"
        "本工具会产生副作用，Exec 网关会要求写操作确认。"
        "重试时请传入相同的 external_request_id 避免重复创建。"
    ),
    args_schema=CreateTicketInput,
    infer_schema=False,
    extras={"readOnlyHint": False, "scene": "ticket"},
)
async def create_ticket(
    user_id: str,
    title: str,
    description: str = "",
    priority: TicketPriority = "medium",
    external_request_id: str | None = None,
) -> str:
    if not user_id.strip():
        return "ERROR: [param] create_ticket 的 user_id 不能为空"
    if priority == "urgent":
        return (
            "ERROR: [business] create_ticket 不支持直接创建 urgent 工单，"
            "urgent 需要通过值班通道人工升级"
        )

    ticket_id = f"TKT-{uuid.uuid4().hex[:12].upper()}"
    return json.dumps(
        {
            "ticket_id": ticket_id,
            "status": "open",
            "priority": priority,
            "created_at": "2024-01-15T10:30:00+08:00",
            "external_request_id": external_request_id,
            "source": "mock",
        },
        ensure_ascii=False,
    )


# ============ 工单状态查询 ============

class QueryTicketInput(BaseModel):
    ticket_id: str = Field(description="工单 ID，格式为 TKT- 开头")


@tool(
    name_or_callable="query_ticket",
    description=(
        "查询工单的状态、优先级、处理人和更新时间。"
        "只按 ticket_id 精确查询。"
        "创建工单用 create_ticket，查询用户信息用 query_user_profile。"
    ),
    args_schema=QueryTicketInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "ticket"},
)
async def query_ticket(ticket_id: str) -> str:
    if not ticket_id.startswith("TKT-"):
        return (
            f"ERROR: [param] query_ticket 的 ticket_id 应以 TKT- 开头，"
            f"实际 {ticket_id!r}"
        )

    ticket = _mock_ticket(ticket_id)
    return json.dumps({**ticket, "source": "mock"}, ensure_ascii=False)