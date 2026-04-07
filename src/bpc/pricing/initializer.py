from __future__ import annotations

"""初始列生成模块。

当前策略：
- 对每个 (vehicle, depot, seed_customer) 构造一条贪心闭环路线；
- 从 seed 出发，每步选最近且满足资源约束的下一个客户；
- 过滤不满足容量/续航的路线；
- 通过签名去重。

并支持 Clarke-Wright savings 启发式构造初始列。

该模块为列生成提供可行起始列池，避免空 RMP。
"""

from dataclasses import dataclass
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


@dataclass
class _SavingsRouteState:
    seq: List[str]
    load: float
    dist: float


def _route_from_seq(instance: InstanceData, vehicle_id: str, depot_id: str, seq: List[str]) -> RouteColumn | None:
    if not seq:
        return None
    if not _arc_exists(instance, depot_id, seq[0]) or not _arc_exists(instance, seq[-1], depot_id):
        return None

    load = sum(instance.demand[c] for c in seq)
    if load > instance.capacity_u + 1e-9:
        return None

    total_dist = instance.dist[(depot_id, seq[0])]
    arc_flags = {(depot_id, seq[0])}
    travel_cost = instance.dispatch_cost[(vehicle_id, depot_id)] + instance.cost[(depot_id, seq[0])]
    for u, v in zip(seq, seq[1:]):
        if not _arc_exists(instance, u, v):
            return None
        total_dist += instance.dist[(u, v)]
        travel_cost += instance.cost[(u, v)]
        arc_flags.add((u, v))
    total_dist += instance.dist[(seq[-1], depot_id)]
    travel_cost += instance.cost[(seq[-1], depot_id)]
    arc_flags.add((seq[-1], depot_id))

    if total_dist > instance.range_q + 1e-9:
        return None

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


def _clarke_wright_routes_for_vehicle_depot(instance: InstanceData, vehicle_id: str, depot_id: str) -> List[RouteColumn]:
    # 初始化：每个客户一条单客户闭环路线。
    routes: Dict[str, _SavingsRouteState] = {}
    customer_to_route: Dict[str, str] = {}
    rid_seq = 0
    for cid in sorted(instance.customers.keys()):
        if not _arc_exists(instance, depot_id, cid) or not _arc_exists(instance, cid, depot_id):
            continue
        load = instance.demand[cid]
        dist = instance.dist[(depot_id, cid)] + instance.dist[(cid, depot_id)]
        if load > instance.capacity_u + 1e-9 or dist > instance.range_q + 1e-9:
            continue
        rid = f"r{rid_seq}"
        rid_seq += 1
        routes[rid] = _SavingsRouteState(seq=[cid], load=load, dist=dist)
        customer_to_route[cid] = rid

    if not routes:
        return []

    # 有向 savings：尝试把 i 所在路线尾部和 j 所在路线头部相连。
    savings_pairs: List[tuple[float, str, str]] = []
    customer_ids = sorted(customer_to_route.keys())
    for i in customer_ids:
        for j in customer_ids:
            if i == j:
                continue
            if not _arc_exists(instance, i, j):
                continue
            saving = instance.dist[(i, depot_id)] + instance.dist[(depot_id, j)] - instance.dist[(i, j)]
            savings_pairs.append((saving, i, j))
    savings_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    for _, i, j in savings_pairs:
        rid_i = customer_to_route.get(i)
        rid_j = customer_to_route.get(j)
        if rid_i is None or rid_j is None or rid_i == rid_j:
            continue
        ri = routes.get(rid_i)
        rj = routes.get(rid_j)
        if ri is None or rj is None:
            continue

        # 仅允许尾接头，符合经典 C&W 合并条件。
        if ri.seq[-1] != i or rj.seq[0] != j:
            continue
        if not _arc_exists(instance, i, j):
            continue

        merged_load = ri.load + rj.load
        if merged_load > instance.capacity_u + 1e-9:
            continue
        merged_dist = ri.dist + rj.dist - instance.dist[(i, depot_id)] - instance.dist[(depot_id, j)] + instance.dist[(i, j)]
        if merged_dist > instance.range_q + 1e-9:
            continue

        merged_seq = ri.seq + rj.seq
        routes[rid_i] = _SavingsRouteState(seq=merged_seq, load=merged_load, dist=merged_dist)
        del routes[rid_j]
        for c in rj.seq:
            customer_to_route[c] = rid_i

    out: List[RouteColumn] = []
    for state in routes.values():
        route = _route_from_seq(instance, vehicle_id, depot_id, state.seq)
        if route is not None:
            out.append(route)
    return out


def _normalize_strategy(strategy: str) -> str:
    s = strategy.strip().lower()
    if s in {"greedy"}:
        return "greedy"
    if s in {"clarke_wright", "clarke_and_wright", "savings", "cw"}:
        return "clarke_wright"
    if s in {"equal_split", "equal_split_by_vehicle", "balanced_by_vehicle", "vehicle_balanced"}:
        return "equal_split_by_vehicle"
    raise ValueError(
        "unknown initial column strategy: "
        f"{strategy!r}. supported: greedy, clarke_wright, equal_split_by_vehicle"
    )


def _split_customers_by_vehicle(instance: InstanceData) -> Dict[str, List[str]]:
    vehicle_ids = sorted(instance.vehicles.keys())
    customer_ids = sorted(instance.customers.keys())
    assigned: Dict[str, List[str]] = {vid: [] for vid in vehicle_ids}
    if not vehicle_ids:
        return assigned
    for idx, cid in enumerate(customer_ids):
        assigned[vehicle_ids[idx % len(vehicle_ids)]].append(cid)
    return assigned


def generate_initial_columns(instance: InstanceData, strategy: str = "greedy") -> List[RouteColumn]:
    """按策略生成初始列池。

    - greedy: 遍历全部 (vehicle, depot, seed_customer)
    - clarke_wright: 每个 (vehicle, depot) 做一轮 savings 合并
    - equal_split_by_vehicle: 先按车辆均分 customer seeds，再按 (vehicle, depot, seed_customer)
    """
    normalized = _normalize_strategy(strategy)
    cols: Dict[str, RouteColumn] = {}
    vehicle_to_customers = None
    if normalized == "equal_split_by_vehicle":
        vehicle_to_customers = _split_customers_by_vehicle(instance)

    for vid in instance.vehicles:
        seed_customers = (
            vehicle_to_customers.get(vid, [])
            if vehicle_to_customers is not None
            else list(instance.customers.keys())
        )
        for did in instance.depots:
            if normalized == "clarke_wright":
                for route in _clarke_wright_routes_for_vehicle_depot(instance, vid, did):
                    cols[route.column_id] = route
                continue
            for cid in seed_customers:
                route = _greedy_route_from_seed(instance, vid, did, cid)
                if route is not None:
                    cols[route.column_id] = route
    if not cols and normalized in {"equal_split_by_vehicle", "clarke_wright"}:
        # 若策略导致无可行初始列，退回默认贪心，避免空 RMP。
        return generate_initial_columns(instance, strategy="greedy")
    return list(cols.values())
