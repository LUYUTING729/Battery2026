from __future__ import annotations

"""初始列生成模块。

当前策略：
- 为每个 (vehicle, depot, customer) 生成单客户闭环列；
- 过滤不满足容量/续航的列；
- 通过签名去重。

该模块为列生成提供可行起始列池，避免空 RMP。
"""

from typing import Dict, List, Tuple

from bpc.core.route_utils import build_ai, route_signature
from bpc.core.types import InstanceData, RouteColumn


def _single_customer_route(instance: InstanceData, vehicle_id: str, depot_id: str, customer_id: str) -> RouteColumn | None:
    """构建单客户闭环路线: depot -> customer -> depot。"""
    a1 = (depot_id, customer_id)
    a2 = (customer_id, depot_id)
    if a1 not in instance.arcs or a2 not in instance.arcs:
        return None
    load = instance.demand[customer_id]
    dist = instance.dist[a1] + instance.dist[a2]
    if load > instance.capacity_u + 1e-9 or dist > instance.range_q + 1e-9:
        return None
    cost = instance.dispatch_cost[(vehicle_id, depot_id)] + instance.cost[a1] + instance.cost[a2]
    seq = (customer_id,)
    route = RouteColumn(
        column_id="",
        vehicle_id=vehicle_id,
        depot_id=depot_id,
        customer_seq=seq,
        a_i=build_ai(instance.customers.keys(), seq),
        arc_flags={a1, a2},
        cost=cost,
        load=load,
        dist=dist,
    )
    route.column_id = route_signature(route)
    return route


def generate_initial_columns(instance: InstanceData) -> List[RouteColumn]:
    """遍历组合生成初始列池。"""
    cols: Dict[str, RouteColumn] = {}
    for vid in instance.vehicles:
        for did in instance.depots:
            for cid in instance.customers:
                route = _single_customer_route(instance, vid, did, cid)
                if route is not None:
                    cols[route.column_id] = route
    return list(cols.values())
