from __future__ import annotations

"""Cut 数据结构与列系数计算。"""

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set

from bpc.core.types import RouteColumn


@dataclass
class CutDefinition:
    """统一 cut 表达。

    字段说明：
    - cut_type: rcc / clique / sri
    - sense: >= 或 <=
    - rhs: 右端常数
    - members/weights: cut 需要的集合参数
    """
    cut_id: str
    cut_type: str
    sense: str
    rhs: float
    members: FrozenSet[str] = frozenset()
    weights: Dict[str, float] = field(default_factory=dict)

    def coefficient(self, route: RouteColumn) -> float:
        """计算某条路线在该 cut 下的系数。

        - RCC: 路线是否触及子集 S
        - Clique: 路线覆盖团内客户个数
        - SRI: ceil(Σ θ_i a_ir)
        """
        if self.cut_type == "rcc":
            return 1.0 if any(c in self.members for c in route.customer_seq) else 0.0
        if self.cut_type == "clique":
            return float(sum(1 for c in route.customer_seq if c in self.members))
        if self.cut_type == "sri":
            val = 0.0
            for c in route.customer_seq:
                val += self.weights.get(c, 0.0)
            return float(math.ceil(val))
        return 0.0
