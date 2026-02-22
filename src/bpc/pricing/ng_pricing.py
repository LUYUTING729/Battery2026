from __future__ import annotations

"""ng-route 定价子问题（RCSPP）实现。

核心目标：在给定对偶值下，为每个 (vehicle, depot) 寻找负化简成本路线。

实现要点：
1) 标签法状态: (当前节点, 累计 rc, load, dist, ng-memory, 访问序列)
2) 资源约束: load <= U, dist + 回仓下界 <= Q
3) ng-route 放松: memory 按 V'=(V∩N_h)∪{h} 更新
4) 剪枝: dominance + 2-cycle elimination
"""

import heapq
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from bpc.core.route_utils import build_ai, route_signature
from bpc.core.types import DualValues, InstanceData, NodeState, RouteColumn, SolverConfig


@dataclass(order=True)
class Label:
    """定价标签状态。

    key 用于优先队列排序（当前取 rc）。
    visited 记录客户访问顺序，用于重构列与计算覆盖系数。
    """
    key: float
    node: str
    rc: float
    load: float
    dist: float
    memory: FrozenSet[str]
    visited: Tuple[str, ...]


def _neighbors_by_distance(instance: InstanceData, ng_size: int) -> Dict[str, Set[str]]:
    """为每个客户构造 ng 邻域（按距离最近的 ng_size 个客户）。"""
    neighbors: Dict[str, Set[str]] = {}
    customers = list(instance.customers.keys())
    for c in customers:
        ranked = sorted(
            [h for h in customers if h != c],
            key=lambda h: instance.dist.get((c, h), float("inf")),
        )
        neighbors[c] = set(ranked[:ng_size])
    return neighbors


def _dominates(l1: Label, l2: Label) -> bool:
    """标签支配判定（同节点下）。

    若 l1 在 rc/load/dist 都不劣，且 memory 更宽松（子集关系），
    则 l1 可支配 l2。
    """
    if l1.rc > l2.rc + 1e-12:
        return False
    if l1.load > l2.load + 1e-12:
        return False
    if l1.dist > l2.dist + 1e-12:
        return False
    if not l1.memory.issubset(l2.memory):
        return False
    return (
        l1.rc < l2.rc - 1e-12
        or l1.load < l2.load - 1e-12
        or l1.dist < l2.dist - 1e-12
        or l1.memory != l2.memory
    )


def _arc_allowed(node: NodeState, arc: Tuple[str, str]) -> bool:
    """检查弧是否被节点分支规则禁用。"""
    if arc in node.branch_rule.banned_arcs:
        return False
    return True


def _contains_2cycle(path: Tuple[str, ...], nxt: str) -> bool:
    """检测 2-cycle（...->i->h 后再回 i）。"""
    if len(path) < 2:
        return False
    return path[-2] == nxt


def _completion_lb(instance: InstanceData, curr: str, depot: str) -> float:
    """返回 curr 到 depot 的完成下界（此处用直接边成本）。"""
    return instance.cost.get((curr, depot), float("inf"))


def _route_from_label(
    instance: InstanceData,
    duals: DualValues,
    vehicle_id: str,
    depot_id: str,
    label: Label,
) -> Optional[RouteColumn]:
    """把可回仓标签转换为路线列。

    计算逻辑：
    - 完整路线成本 = dispatch_cost + 路径弧成本和
    - 判断完整 reduced cost 是否为负（仅负列入池）
    """
    if not label.visited:
        return None
    back_arc = (label.node, depot_id)
    if back_arc not in instance.arcs:
        return None

    final_rc = label.rc + instance.cost[back_arc] - duals.branch_arc_dual.get(back_arc, 0.0)
    if final_rc >= -1e-9:
        return None

    seq = label.visited
    arc_flags: Set[Tuple[str, str]] = set()
    prev = depot_id
    dist = 0.0
    travel_cost = instance.dispatch_cost[(vehicle_id, depot_id)]
    for c in seq:
        arc = (prev, c)
        arc_flags.add(arc)
        dist += instance.dist[arc]
        travel_cost += instance.cost[arc]
        prev = c
    arc_flags.add(back_arc)
    dist += instance.dist[back_arc]
    travel_cost += instance.cost[back_arc]

    route = RouteColumn(
        column_id="",
        vehicle_id=vehicle_id,
        depot_id=depot_id,
        customer_seq=seq,
        a_i=build_ai(instance.customers.keys(), seq),
        arc_flags=arc_flags,
        cost=travel_cost,
        load=sum(instance.demand[c] for c in seq),
        dist=dist,
    )
    route.column_id = route_signature(route)
    return route


def _price_for_vehicle_depot(
    instance: InstanceData,
    node: NodeState,
    duals: DualValues,
    cfg: SolverConfig,
    vehicle_id: str,
    depot_id: str,
    ng_neighbors: Dict[str, Set[str]],
) -> List[RouteColumn]:
    """对固定 (vehicle, depot) 做一次定价。

    返回该子问题发现的负化简成本列（去重后）。
    """
    start_rc = instance.dispatch_cost[(vehicle_id, depot_id)] - duals.vehicle_alpha[vehicle_id] - duals.depot_beta[depot_id]
    start = Label(key=start_rc, node=depot_id, rc=start_rc, load=0.0, dist=0.0, memory=frozenset(), visited=tuple())
    queue: List[Label] = [start]
    labels_by_node: Dict[str, List[Label]] = {depot_id: [start]}
    best_routes: Dict[str, RouteColumn] = {}

    while queue and len(best_routes) < cfg.max_new_columns_per_iter:
        cur = heapq.heappop(queue)
        current_customers = set(cur.visited)

        # 从当前标签扩展到所有客户节点。
        for nxt in instance.customers:
            if nxt in cur.memory:
                continue
            if cfg.k_cycle >= 2 and _contains_2cycle(cur.visited, nxt):
                continue
            arc = (cur.node, nxt)
            if arc not in instance.arcs or not _arc_allowed(node, arc):
                continue

            nload = cur.load + instance.demand[nxt]
            ndist = cur.dist + instance.dist[arc]
            # 资源可行性过滤：容量 + 续航（含完成下界）。
            if nload > instance.capacity_u + 1e-9:
                continue
            if ndist + _completion_lb(instance, nxt, depot_id) > instance.range_q + 1e-9:
                continue

            # 递推 reduced cost:
            # + 边成本 - 覆盖对偶 - 分支弧对偶 - clique 惩罚。
            nrc = cur.rc + instance.cost[arc]
            nrc -= duals.branch_arc_dual.get(arc, 0.0)
            if nxt not in current_customers:
                nrc -= duals.cover_pi[nxt]
                nrc -= duals.clique_customer_bonus.get(nxt, 0.0)

            memory = frozenset((set(cur.memory).intersection(ng_neighbors[nxt])) | {nxt})
            visited = cur.visited + ((nxt,) if nxt not in current_customers else tuple())
            lbl = Label(key=nrc, node=nxt, rc=nrc, load=nload, dist=ndist, memory=memory, visited=visited)

            # 同节点支配剪枝。
            bucket = labels_by_node.setdefault(nxt, [])
            dominated = False
            pruned = []
            for old in bucket:
                if _dominates(old, lbl):
                    dominated = True
                    break
                if _dominates(lbl, old):
                    pruned.append(old)
            if dominated:
                continue
            if pruned:
                bucket[:] = [x for x in bucket if x not in pruned]
            bucket.append(lbl)
            heapq.heappush(queue, lbl)

            route = _route_from_label(instance, duals, vehicle_id, depot_id, lbl)
            if route is not None:
                best_routes[route.column_id] = route

    return list(best_routes.values())


def price_columns(
    instance: InstanceData,
    node: NodeState,
    duals: DualValues,
    cfg: SolverConfig,
) -> List[RouteColumn]:
    """定价总入口：遍历全部 (vehicle, depot) 子问题并汇总新列。"""
    ng_neighbors = _neighbors_by_distance(instance, cfg.ng_size)
    out: Dict[str, RouteColumn] = {}
    for vid in instance.vehicles:
        for did in instance.depots:
            cols = _price_for_vehicle_depot(instance, node, duals, cfg, vid, did, ng_neighbors)
            for c in cols:
                out[c.column_id] = c
    return list(out.values())
