from __future__ import annotations

"""分支规则模块（聚合弧流分支）。

核心思想：
- 由 RMP 分数解计算聚合弧流 x_uv = Σ λ_r b_uvr；
- 选择最接近 0.5 的分数弧作为分支对象；
- 左支: 禁用该弧（x_uv <= 0）；右支: 强制至少一次使用该弧（x_uv >= 1）。
"""

from typing import Dict, Optional, Tuple

from bpc.core.types import BranchRule, NodeState


def pick_branch_arc(arc_flow: Dict[Tuple[str, str], float], tol: float = 1e-6) -> Optional[Tuple[str, str]]:
    """从分数弧中选分支弧。

    选择标准：|x_uv - 0.5| 最小（越接近 0.5 越不确定，分支收益通常更高）。
    """
    candidate = None
    best_dist = 1.0
    for arc, value in arc_flow.items():
        frac = abs(value - round(value))
        if frac <= tol:
            continue
        dist = abs(value - 0.5)
        if dist < best_dist:
            best_dist = dist
            candidate = arc
    return candidate


def split_node(parent: NodeState, arc: Tuple[str, str], left_id: int, right_id: int) -> Tuple[NodeState, NodeState]:
    """根据选中弧生成左右子节点。"""
    left_rule = BranchRule(
        banned_arcs=frozenset(set(parent.branch_rule.banned_arcs) | {arc}),
        forced_arcs=parent.branch_rule.forced_arcs,
    )
    right_rule = BranchRule(
        banned_arcs=parent.branch_rule.banned_arcs,
        forced_arcs=frozenset(set(parent.branch_rule.forced_arcs) | {arc}),
    )
    left = NodeState(node_id=left_id, depth=parent.depth + 1, branch_rule=left_rule)
    right = NodeState(node_id=right_id, depth=parent.depth + 1, branch_rule=right_rule)
    return left, right
