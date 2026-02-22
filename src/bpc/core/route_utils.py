from __future__ import annotations

"""路线辅助函数：
- 可行性检查
- 列签名生成
- 覆盖向量构建
"""

from typing import Dict

from .types import Arc, InstanceData, RouteColumn


def route_feasible(route: RouteColumn, instance: InstanceData) -> bool:
    """校验路线是否满足基础业务约束。

    检查项：
    1) 容量与续航上限；
    2) 客户不重复访问；
    3) 路线闭环（depot->first 与 last->depot）；
    4) 路线使用弧必须在实例合法弧集中；
    5) 禁止 depot->depot 弧。
    """
    if route.load > instance.capacity_u + 1e-9:
        return False
    if route.dist > instance.range_q + 1e-9:
        return False
    if len(set(route.customer_seq)) != len(route.customer_seq):
        return False
    if route.customer_seq:
        start_arc = (route.depot_id, route.customer_seq[0])
        end_arc = (route.customer_seq[-1], route.depot_id)
        if start_arc not in route.arc_flags or end_arc not in route.arc_flags:
            return False
    for arc in route.arc_flags:
        if arc not in instance.arcs:
            return False
        if arc[0] in instance.depots and arc[1] in instance.depots:
            return False
    return True


def route_signature(route: RouteColumn) -> str:
    """构建列唯一签名，用于列池去重。"""
    return f"{route.vehicle_id}|{route.depot_id}|{'-'.join(route.customer_seq)}"


def build_ai(customer_ids, seq) -> Dict[str, int]:
    """把客户访问序列转换为覆盖向量 a_i。"""
    visited = set(seq)
    return {cid: int(cid in visited) for cid in customer_ids}
