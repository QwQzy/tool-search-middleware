"""业务工具：订单查询与操作。

订单工具覆盖四类形态：按 ID 精确查询、按用户列出订单、查询物流状态、
以及两个有副作用的写操作（取消订单、退款）。写操作携带
external_request_id 或从 order_id 派生幂等键，由 ExecGatewayMiddleware
统一处理。

每个工具携带 extras：
- readOnlyHint: True/False
- scene: "order"

业务规则集中体现为一个设计取向：工具本身不做"状态合理性"判断之外的
任何推断。cancel_order 只检查订单状态是否允许取消，refund_order 只检查
金额是否超过订单总额和订单状态是否允许退款。更复杂的业务规则（如"同一
用户 24 小时内退款超过 N 次需要风控审批"）属于下游系统职责，不放进工具。
工具越简单，失败语义越清晰，模型越容易从错误中修正。

工具函数返回 mock 数据，带 "source": "mock" 标记。
"""

import json
import uuid
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


type OrderStatus = Literal[
    "pending",     # 待支付
    "paid",        # 已支付待发货
    "shipped",     # 已发货
    "delivered",   # 已送达
    "cancelled",   # 已取消
    "refunded",    # 已退款
]

type LogisticsStatus = Literal[
    "not_shipped", "in_transit", "out_for_delivery", "delivered", "returned",
]


def _mock_order(order_id: str) -> dict:
    return {
        "order_id": order_id,
        "user_id": "u_10001",
        "status": "shipped",
        "currency": "CNY",
        "total_amount": 259.00,
        "paid_amount": 259.00,
        "items": [
            {
                "sku": "SKU-A100",
                "name": "无线耳机",
                "quantity": 1,
                "unit_price": 199.00,
            },
            {
                "sku": "SKU-B200",
                "name": "耳机收纳盒",
                "quantity": 1,
                "unit_price": 60.00,
            },
        ],
        "created_at": "2024-01-10T15:20:00+08:00",
        "paid_at": "2024-01-10T15:22:00+08:00",
        "shipped_at": "2024-01-11T09:00:00+08:00",
    }


def _mock_logistics(order_id: str) -> dict:
    return {
        "order_id": order_id,
        "carrier": "SF Express",
        "tracking_number": "SF1234567890123",
        "status": "in_transit",
        "events": [
            {
                "at": "2024-01-11T09:00:00+08:00",
                "location": "深圳",
                "description": "包裹已揽收",
            },
            {
                "at": "2024-01-12T02:30:00+08:00",
                "location": "深圳转运中心",
                "description": "已发往下一站",
            },
        ],
    }


def _validate_order_id(order_id: str) -> str | None:
    """订单号格式校验。返回 None 表示合法，返回字符串表示错误消息。"""
    if not order_id.startswith("ORD-"):
        return (
            f"ERROR: [param] 订单号应以 ORD- 开头，实际 {order_id!r}。"
            "完整订单号从下单确认页或订单列表里获取"
        )
    return None


# ============ 按订单号查询 ============

class QueryOrderInput(BaseModel):
    order_id: str = Field(description="订单号，格式为 ORD- 开头")


@tool(
    name_or_callable="query_order",
    description=(
        "按订单号查询订单详情，返回状态、金额、商品明细、各阶段时间戳。"
        "只做精确匹配，不接受用户 ID 或商品名作为查询条件；"
        "按用户列出订单请用 list_orders，查物流请用 query_order_logistics。"
    ),
    args_schema=QueryOrderInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "order"},
)
async def query_order(order_id: str) -> str:
    err = _validate_order_id(order_id)
    if err:
        return err
    order = _mock_order(order_id)
    return json.dumps({**order, "source": "mock"}, ensure_ascii=False)


# ============ 按用户列出订单 ============

class ListOrdersInput(BaseModel):
    user_id: str = Field(description="用户 ID")
    status: OrderStatus | None = Field(
        default=None,
        description="按状态过滤；不传则返回全部状态的订单",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="返回数量上限，默认 10，最大 50",
    )


@tool(
    name_or_callable="list_orders",
    description=(
        "列出某个用户的订单，按创建时间倒序，返回订单号、状态、金额、创建时间。"
        "可按状态过滤。本工具只返回订单摘要，完整明细请用 query_order。"
        "按用户 ID 查询用户画像请用 query_user_profile。"
    ),
    args_schema=ListOrdersInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "order"},
)
async def list_orders(
    user_id: str,
    status: OrderStatus | None = None,
    limit: int = 10,
) -> str:
    if not user_id.strip():
        return "ERROR: [param] list_orders 的 user_id 不能为空"

    # mock：返回一条占位订单，真实实现应该查询订单库
    items = [
        {
            "order_id": f"ORD-{uuid.uuid4().hex[:10].upper()}",
            "status": status or "shipped",
            "total_amount": 259.00,
            "currency": "CNY",
            "created_at": "2024-01-10T15:20:00+08:00",
        }
    ][:limit]

    return json.dumps(
        {"user_id": user_id, "orders": items, "source": "mock"},
        ensure_ascii=False,
    )


# ============ 查询物流 ============

class QueryOrderLogisticsInput(BaseModel):
    order_id: str = Field(description="订单号，格式为 ORD- 开头")


@tool(
    name_or_callable="query_order_logistics",
    description=(
        "查询订单的物流状态，返回承运商、运单号、当前状态和物流轨迹。"
        "只对已发货（shipped）或已送达（delivered）的订单有意义，"
        "未发货订单返回空的轨迹列表。"
        "订单整体状态请用 query_order。"
    ),
    args_schema=QueryOrderLogisticsInput,
    infer_schema=False,
    extras={"readOnlyHint": True, "scene": "order"},
)
async def query_order_logistics(order_id: str) -> str:
    err = _validate_order_id(order_id)
    if err:
        return err
    logistics = _mock_logistics(order_id)
    return json.dumps({**logistics, "source": "mock"}, ensure_ascii=False)


# ============ 取消订单 ============

class CancelOrderInput(BaseModel):
    order_id: str = Field(description="订单号，格式为 ORD- 开头")
    reason: str = Field(
        default="",
        max_length=500,
        description="取消原因，可选，最多 500 字符",
    )
    external_request_id: str | None = Field(
        default=None,
        description=(
            "外部请求 ID，用于幂等。重试时应传入相同的值。"
            "不传时网关从 order_id + 动作类型派生幂等键，"
            "对同一订单的重复取消请求会被识别为同一次操作"
        ),
    )


@tool(
    name_or_callable="cancel_order",
    description=(
        "取消订单。只允许取消 pending（待支付）和 paid（已支付未发货）状态的订单；"
        "shipped 及之后的状态需走退货退款流程，请用 refund_order。"
        "本工具会产生副作用，Exec 网关会要求写操作确认。"
        "重试时请传入相同的 external_request_id。"
    ),
    args_schema=CancelOrderInput,
    infer_schema=False,
    extras={"readOnlyHint": False, "scene": "order"},
)
async def cancel_order(
    order_id: str,
    reason: str = "",
    external_request_id: str | None = None,
) -> str:
    err = _validate_order_id(order_id)
    if err:
        return err

    order = _mock_order(order_id)
    if order["status"] in {"shipped", "delivered"}:
        return (
            f"ERROR: [business] 订单 {order_id} 当前状态为 {order['status']}，"
            "已发货订单不能取消。如需退货请使用 refund_order"
        )
    if order["status"] in {"cancelled", "refunded"}:
        return (
            f"ERROR: [business] 订单 {order_id} 当前状态为 {order['status']}，"
            "不可重复取消"
        )

    return json.dumps(
        {
            "order_id": order_id,
            "status": "cancelled",
            "cancelled_at": "2024-01-15T10:30:00+08:00",
            "external_request_id": external_request_id,
            "source": "mock",
        },
        ensure_ascii=False,
    )


# ============ 申请退款 ============

class RefundOrderInput(BaseModel):
    order_id: str = Field(description="订单号，格式为 ORD- 开头")
    amount: float = Field(
        gt=0,
        description="退款金额，必须大于 0。部分退款时金额可小于订单总额",
    )
    reason: str = Field(
        min_length=1,
        max_length=500,
        description="退款原因，必填，最多 500 字符",
    )
    external_request_id: str | None = Field(
        default=None,
        description=(
            "外部请求 ID，用于幂等。重试时应传入相同的值。"
            "不传时网关从 order_id + amount 派生幂等键，"
            "对同一订单同一金额的重复退款请求会被识别为同一次操作"
        ),
    )


@tool(
    name_or_callable="refund_order",
    description=(
        "对订单发起退款。只允许对 paid、shipped、delivered 状态的订单退款；"
        "退款金额不能超过订单的已支付金额。"
        "本工具会产生副作用，Exec 网关会要求写操作确认，"
        "并按 order_id + amount 派生幂等键防止重复退款。"
        "取消未支付订单请用 cancel_order，不需要走退款流程。"
    ),
    args_schema=RefundOrderInput,
    infer_schema=False,
    extras={"readOnlyHint": False, "scene": "order"},
)
async def refund_order(
    order_id: str,
    amount: float,
    reason: str,
    external_request_id: str | None = None,
) -> str:
    err = _validate_order_id(order_id)
    if err:
        return err

    order = _mock_order(order_id)
    if order["status"] in {"pending", "cancelled", "refunded"}:
        return (
            f"ERROR: [business] 订单 {order_id} 当前状态为 {order['status']}，"
            "不支持退款。pending 订单请直接取消，已退款订单不可重复退款"
        )
    if amount > order["paid_amount"]:
        return (
            f"ERROR: [business] 退款金额 {amount} 超过订单已支付金额 "
            f"{order['paid_amount']}。需要部分退款时 amount 应 <= 已支付金额"
        )

    refund_id = f"RF-{uuid.uuid4().hex[:10].upper()}"
    return json.dumps(
        {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount": amount,
            "currency": order["currency"],
            "status": "processing",
            "reason": reason,
            "external_request_id": external_request_id,
            "source": "mock",
        },
        ensure_ascii=False,
    )