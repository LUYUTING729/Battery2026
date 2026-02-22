from __future__ import annotations

"""核心数据类型定义。

该模块统一描述：
- 输入实例结构（客户、中心、车辆、成本等）
- 路线列结构（RMP 变量对应的数据）
- 分支节点状态与配置
- 求解输出结构
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

Arc = Tuple[str, str]


@dataclass(frozen=True)
class Customer:
    """客户实体：仅包含 ID 与需求。"""
    customer_id: str
    demand: float


@dataclass(frozen=True)
class Depot:
    """中心实体：包含可派车上限。"""
    depot_id: str
    max_vehicles: int


@dataclass(frozen=True)
class Vehicle:
    """车辆实体：包含车辆 ID 与初始位置 origin。"""
    vehicle_id: str
    origin: str


@dataclass
class RouteColumn:
    """列生成中的路线列。

    - customer_seq: 客户访问顺序
    - a_i: 覆盖向量（客户是否被该路线服务）
    - arc_flags: 该路线使用的弧集合
    - cost/load/dist: 成本与资源消耗
    """
    column_id: str
    vehicle_id: str
    depot_id: str
    customer_seq: Tuple[str, ...]
    a_i: Dict[str, int]
    arc_flags: Set[Arc]
    cost: float
    load: float
    dist: float


@dataclass
class InstanceData:
    """求解所需的完整实例数据对象。"""
    instance_id: str
    customers: Dict[str, Customer]
    depots: Dict[str, Depot]
    vehicles: Dict[str, Vehicle]
    demand: Dict[str, float]
    capacity_u: float
    range_q: float
    cost: Dict[Arc, float]
    dist: Dict[Arc, float]
    dispatch_cost: Dict[Tuple[str, str], float]
    arcs: Set[Arc]


@dataclass
class BranchRule:
    """分支规则：禁弧集合与强制弧集合。"""
    banned_arcs: FrozenSet[Arc] = frozenset()
    forced_arcs: FrozenSet[Arc] = frozenset()


@dataclass
class NodeState:
    """分支树节点状态。"""
    node_id: int
    depth: int
    branch_rule: BranchRule = field(default_factory=BranchRule)
    local_columns: Set[str] = field(default_factory=set)
    cuts: List[str] = field(default_factory=list)
    dual_center: Optional[Dict[str, float]] = None


@dataclass
class StabilizationConfig:
    """对偶稳定化配置。"""
    method: str = "box"
    box_width: float = 2.0
    penalty_weight: float = 0.0
    hybrid_switch_iter: int = 25


@dataclass
class SolverConfig:
    """求解器全局配置（算法开关、限制、阈值）。"""
    time_limit_s: float = 600.0
    mip_gap: float = 1e-4
    max_nodes: int = 500
    max_cg_iters: int = 200
    max_new_columns_per_iter: int = 30
    pricing_rc_epsilon: float = 1e-6
    ng_size: int = 12
    k_cycle: int = 2
    cut_violation_tol: float = 1e-6
    max_cuts_per_round: int = 25
    cut_rounds_root: int = 3
    cut_rounds_node: int = 1
    branch_strategy: str = "best_bound"
    random_seed: int = 42
    enable_rcc: bool = True
    enable_clique: bool = True
    enable_sri: bool = True
    enable_stabilization: bool = True
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    output_dir: str = "outputs"


@dataclass
class DualValues:
    """定价阶段使用的对偶容器。"""
    cover_pi: Dict[str, float]
    vehicle_alpha: Dict[str, float]
    depot_beta: Dict[str, float]
    branch_arc_dual: Dict[Arc, float]
    clique_customer_bonus: Dict[str, float]


@dataclass
class SolveResult:
    """单实例求解结果。"""
    status: str
    obj_primal: float
    obj_dual: float
    gap: float
    routes: List[RouteColumn]
    stats: Dict[str, float]


@dataclass
class ExperimentReport:
    """批量实验结果。"""
    batch_id: str
    results: List[SolveResult]
    summary: Dict[str, float]


CoefficientFn = Callable[[RouteColumn], float]
