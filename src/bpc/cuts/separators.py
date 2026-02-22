from __future__ import annotations

"""主问题 cuts 分离器（启发式版本）。

包含三类鲁棒 cuts：
- RCC（容量圆整割）
- Clique（冲突团割）
- SRI（Subset Row Inequality，启发式模板）

输入为 RMP 当前活跃列及其分数值，输出 violated cuts 列表。
"""

import itertools
import math
from typing import Dict, Iterable, List, Set, Tuple

from bpc.core.types import InstanceData, RouteColumn, SolverConfig
from .definitions import CutDefinition


def _lhs(routes: List[Tuple[RouteColumn, float]], cut: CutDefinition) -> float:
    """计算 cut 左侧值: Σ lambda_r * coeff(r)。"""
    return sum(val * cut.coefficient(r) for r, val in routes)


def _generate_rcc_candidates(instance: InstanceData, max_size: int = 3) -> Iterable[Set[str]]:
    """生成 RCC 候选客户子集（小规模启发式枚举）。"""
    customers = list(instance.customers.keys())
    for size in range(2, max_size + 1):
        for subset in itertools.combinations(customers, size):
            q = sum(instance.demand[c] for c in subset)
            if q > instance.capacity_u + 1e-9:
                yield set(subset)


def _incompatible(instance: InstanceData, i: str, h: str) -> bool:
    """判定客户对是否不相容。

    充分条件：
    1) q_i + q_h > U（容量不相容）
    2) 对所有中心，访问 i/h 的闭环长度下界都 > Q（续航不相容）
    """
    if instance.demand[i] + instance.demand[h] > instance.capacity_u + 1e-9:
        return True
    depots = list(instance.depots.keys())
    for d in depots:
        a = instance.dist.get((d, i), float("inf")) + instance.dist.get((i, h), float("inf")) + instance.dist.get((h, d), float("inf"))
        b = instance.dist.get((d, h), float("inf")) + instance.dist.get((h, i), float("inf")) + instance.dist.get((i, d), float("inf"))
        if min(a, b) <= instance.range_q + 1e-9:
            return False
    return True


def _greedy_cliques(instance: InstanceData, max_cliques: int = 20) -> List[Set[str]]:
    """在冲突图上用贪心法构造 clique 候选。"""
    customers = list(instance.customers.keys())
    adj: Dict[str, Set[str]] = {c: set() for c in customers}
    for i, h in itertools.combinations(customers, 2):
        if _incompatible(instance, i, h):
            adj[i].add(h)
            adj[h].add(i)

    cliques: List[Set[str]] = []
    for c in sorted(customers, key=lambda x: len(adj[x]), reverse=True):
        clique = {c}
        for n in sorted(adj[c], key=lambda x: len(adj[x]), reverse=True):
            if all(n in adj[m] for m in clique):
                clique.add(n)
        if len(clique) >= 2:
            cliques.append(clique)
        if len(cliques) >= max_cliques:
            break
    return cliques


def _generate_sri_candidates(instance: InstanceData, size: int = 3, limit: int = 20) -> List[Tuple[Set[str], Dict[str, float]]]:
    """生成 SRI 候选（当前使用固定 0.5 权重模板）。"""
    out: List[Tuple[Set[str], Dict[str, float]]] = []
    customers = list(instance.customers.keys())
    for subset in itertools.combinations(customers, size):
        weights = {c: 0.5 for c in subset}
        out.append((set(subset), weights))
        if len(out) >= limit:
            break
    return out


def separate_cuts(
    instance: InstanceData,
    active_routes: List[Tuple[RouteColumn, float]],
    cfg: SolverConfig,
    at_root: bool,
    round_id: int,
) -> List[CutDefinition]:
    """分离入口。

    调用顺序：
    1) RCC（优先收紧容量相关分数结构）
    2) Clique（打击不相容客户同路分数覆盖）
    3) SRI（根节点启发式补强）
    """
    cuts: List[CutDefinition] = []

    if cfg.enable_rcc:
        for subset in _generate_rcc_candidates(instance):
            # RHS = ceil(q(S)/U)
            rhs = math.ceil(sum(instance.demand[c] for c in subset) / instance.capacity_u)
            cut = CutDefinition(
                cut_id=f"rcc_{round_id}_{'_'.join(sorted(subset))}",
                cut_type="rcc",
                sense=">=",
                rhs=float(rhs),
                members=frozenset(subset),
            )
            if _lhs(active_routes, cut) + cfg.cut_violation_tol < cut.rhs:
                cuts.append(cut)
            if len(cuts) >= cfg.max_cuts_per_round:
                return cuts

    if cfg.enable_clique:
        for idx, clique in enumerate(_greedy_cliques(instance)):
            cut = CutDefinition(
                cut_id=f"clique_{round_id}_{idx}",
                cut_type="clique",
                sense="<=",
                rhs=1.0,
                members=frozenset(clique),
            )
            if _lhs(active_routes, cut) > cut.rhs + cfg.cut_violation_tol:
                cuts.append(cut)
            if len(cuts) >= cfg.max_cuts_per_round:
                return cuts

    if cfg.enable_sri and at_root:
        for idx, (subset, weights) in enumerate(_generate_sri_candidates(instance)):
            rhs = math.ceil(sum(weights.values()))
            cut = CutDefinition(
                cut_id=f"sri_{round_id}_{idx}",
                cut_type="sri",
                sense=">=",
                rhs=float(rhs),
                members=frozenset(subset),
                weights=weights,
            )
            if _lhs(active_routes, cut) + cfg.cut_violation_tol < cut.rhs:
                cuts.append(cut)
            if len(cuts) >= cfg.max_cuts_per_round:
                return cuts

    return cuts
