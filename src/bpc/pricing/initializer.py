from __future__ import annotations

"""初始列生成模块。

当前策略（多客户贪心）：
- 对每个 (vehicle, depot, seed_customer) 构造一条贪心闭环路线；
- 从 seed 出发，每步选最近且满足资源约束的下一个客户；
- 过滤不满足容量/续航的路线；
- 通过签名去重。

该模块为列生成提供可行起始列池，避免空 RMP。
"""

from typing import Dict, List

from bpc.core.route_utils import build_ai, route_signature
from bpc.core.types import InstanceData, RouteColumn


def _arc_exists(instance: InstanceData, u: str, v: str) -> bool:
    return (u, v) in instance.arcs and (u, v) in instance.dist and (u, v) in instance.cost


def _greedy_route_from_seed(instance: InstanceData, vehicle_id: str, depot_id: str, seed_customer: str) -> RouteColumn | None:
    """构建一条以 seed_customer 为起点的多客户贪心闭环路线。"""
    if not _arc_exists(instance, depot_id, seed_customer) or not _arc_exists(instance, seed_customer, depot_id):
        return None

    load = instance.demand[seed_customer]
    path_dist = instance.dist[(depot_id, seed_customer)]
    if load > instance.capacity_u + 1e-9:
        return None
    if path_dist + instance.dist[(seed_customer, depot_id)] > instance.range_q + 1e-9:
        return None

    seq: List[str] = [seed_customer]
    visited = {seed_customer}
    current = seed_customer

    # 逐步扩展：每步选“最近且扩展后仍可回仓”的客户。
    while True:
        candidates = []
        for nxt in instance.customers:
            if nxt in visited:
                continue
            if not _arc_exists(instance, current, nxt) or not _arc_exists(instance, nxt, depot_id):
                continue
            nload = load + instance.demand[nxt]
            if nload > instance.capacity_u + 1e-9:
                continue
            ndist = path_dist + instance.dist[(current, nxt)]
            # 保证扩展后还可闭环回 depot。
            if ndist + instance.dist[(nxt, depot_id)] > instance.range_q + 1e-9:
                continue
            candidates.append((instance.dist[(current, nxt)], nxt))

        if not candidates:
            break

        candidates.sort(key=lambda x: (x[0], x[1]))
        _, chosen = candidates[0]
        seq.append(chosen)
        visited.add(chosen)
        load += instance.demand[chosen]
        path_dist += instance.dist[(current, chosen)]
        current = chosen

    if not _arc_exists(instance, current, depot_id):
        return None
    total_dist = path_dist + instance.dist[(current, depot_id)]
    if total_dist > instance.range_q + 1e-9:
        return None

    arc_flags = set()
    travel_cost = instance.dispatch_cost[(vehicle_id, depot_id)]
    prev = depot_id
    for c in seq:
        arc = (prev, c)
        arc_flags.add(arc)
        travel_cost += instance.cost[arc]
        prev = c
    back_arc = (prev, depot_id)
    arc_flags.add(back_arc)
    travel_cost += instance.cost[back_arc]

    route = RouteColumn(
        column_id="",
        vehicle_id=vehicle_id,
        depot_id=depot_id,
        customer_seq=tuple(seq),
        a_i=build_ai(instance.customers.keys(), tuple(seq)),
        arc_flags=arc_flags,
        cost=travel_cost,
        load=load,
        dist=total_dist,
    )
    route.column_id = route_signature(route)
    return route


def generate_initial_columns(instance: InstanceData) -> List[RouteColumn]:
    """按 (vehicle, depot, seed_customer) 生成多客户贪心初始列池。"""
    cols: Dict[str, RouteColumn] = {}
    for vid in instance.vehicles:
        for did in instance.depots:
            for cid in instance.customers:
                route = _greedy_route_from_seed(instance, vid, did, cid)
                if route is not None:
                    cols[route.column_id] = route
    return list(cols.values())
