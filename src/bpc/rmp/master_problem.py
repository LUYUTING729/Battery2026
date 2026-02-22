from __future__ import annotations

"""Restricted Master Problem (RMP) 构建与求解。

RMP 对应论文中的集合划分主问题（线性松弛/整数化版本）：
- 决策变量: lambda_{k,j,r}
- 约束: 覆盖约束、车辆约束、中心容量约束
- 扩展: 分支约束（禁弧/强制弧）与 cuts 行

该模块提供增量增列接口，供列生成迭代调用。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from bpc.core.types import DualValues, InstanceData, NodeState, RouteColumn
from bpc.cuts.definitions import CutDefinition

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


class GurobiUnavailableError(RuntimeError):
    pass


@dataclass
class RmpSolveInfo:
    """RMP 求解后的核心信息。"""
    status: int
    obj_val: float
    lambda_values: Dict[str, float]


class MasterProblem:
    """Gurobi RMP 封装。

    调用方式：
    1) 初始化模型与基础约束；
    2) 通过 add_columns 按批增列；
    3) solve 得到分数解；
    4) extract_duals 输出定价所需对偶。
    """

    def __init__(self, instance: InstanceData, node: NodeState, relax: bool = True):
        if gp is None:
            raise GurobiUnavailableError("gurobipy is required for RMP solving")

        self.instance = instance
        self.node = node
        self.relax = relax
        self.model = gp.Model(f"RMP_{instance.instance_id}_node_{node.node_id}")
        self.model.Params.OutputFlag = 0

        self.vars: Dict[str, gp.Var] = {}
        self.columns: Dict[str, RouteColumn] = {}
        self.cuts: Dict[str, Tuple[CutDefinition, gp.Constr]] = {}

        self.cover_cons = {
            cid: self.model.addConstr(gp.LinExpr() == 1.0, name=f"cover_{cid}")
            for cid in instance.customers
        }
        self.vehicle_cons = {
            vid: self.model.addConstr(gp.LinExpr() <= 1.0, name=f"vehicle_{vid}")
            for vid in instance.vehicles
        }
        self.depot_cons = {
            did: self.model.addConstr(gp.LinExpr() <= dep.max_vehicles, name=f"depot_{did}")
            for did, dep in instance.depots.items()
        }

        self.branch_cons: Dict[Tuple[str, str, str], gp.Constr] = {}
        # 左分支: x_uv <= 0 (禁用弧)。
        for arc in node.branch_rule.banned_arcs:
            key = ("ban", arc[0], arc[1])
            self.branch_cons[key] = self.model.addConstr(gp.LinExpr() <= 0.0, name=f"ban_{arc[0]}_{arc[1]}")
        # 右分支: x_uv >= 1 (至少一次使用该弧)。
        for arc in node.branch_rule.forced_arcs:
            key = ("force", arc[0], arc[1])
            self.branch_cons[key] = self.model.addConstr(gp.LinExpr() >= 1.0, name=f"force_{arc[0]}_{arc[1]}")

    def add_columns(self, routes: List[RouteColumn]) -> int:
        """增量加入路线列并更新所有约束系数。

        系数映射:
        - cover_i: a_{ir}
        - vehicle_k: 1 (该列属于车辆 k)
        - depot_j: 1 (该列从中心 j 出发)
        - branch/cuts: 由列弧集与 cut 类型计算
        """
        added = 0
        for route in routes:
            if route.column_id in self.vars:
                continue
            if not self._column_feasible_to_node(route):
                continue

            vtype = GRB.CONTINUOUS if self.relax else GRB.BINARY
            var = self.model.addVar(lb=0.0, ub=1.0, obj=route.cost, vtype=vtype, name=f"lam_{route.column_id}")
            self.vars[route.column_id] = var
            self.columns[route.column_id] = route

            for cid, flag in route.a_i.items():
                if flag:
                    self.model.chgCoeff(self.cover_cons[cid], var, 1.0)
            self.model.chgCoeff(self.vehicle_cons[route.vehicle_id], var, 1.0)
            self.model.chgCoeff(self.depot_cons[route.depot_id], var, 1.0)

            for (kind, u, v), con in self.branch_cons.items():
                coeff = 1.0 if (u, v) in route.arc_flags else 0.0
                if coeff:
                    self.model.chgCoeff(con, var, coeff)

            for cut, con in self.cuts.values():
                coeff = cut.coefficient(route)
                if abs(coeff) > 1e-12:
                    self.model.chgCoeff(con, var, coeff)

            added += 1

        if added:
            self.model.update()
        return added

    def add_cuts(self, cuts: List[CutDefinition]) -> int:
        """把分离得到的 cuts 转换成 RMP 线性行并注入。"""
        added = 0
        for cut in cuts:
            if cut.cut_id in self.cuts:
                continue
            expr = gp.LinExpr()
            sense = cut.sense
            if sense == ">=":
                con = self.model.addConstr(expr >= cut.rhs, name=cut.cut_id)
            elif sense == "<=":
                con = self.model.addConstr(expr <= cut.rhs, name=cut.cut_id)
            else:
                raise ValueError(f"Unsupported cut sense: {sense}")

            for cid, route in self.columns.items():
                coeff = cut.coefficient(route)
                if abs(coeff) > 1e-12:
                    self.model.chgCoeff(con, self.vars[cid], coeff)
            self.cuts[cut.cut_id] = (cut, con)
            added += 1

        if added:
            self.model.update()
        return added

    def solve(self) -> RmpSolveInfo:
        """求解当前 RMP。"""
        self.model.optimize()
        status = self.model.Status
        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return RmpSolveInfo(status=status, obj_val=float("inf"), lambda_values={})
        values = {cid: var.X for cid, var in self.vars.items()}
        return RmpSolveInfo(status=status, obj_val=float(self.model.ObjVal), lambda_values=values)

    def is_integral(self, tol: float = 1e-6) -> bool:
        """检查当前 lambda 解是否整数。"""
        for var in self.vars.values():
            val = var.X
            if abs(val - round(val)) > tol:
                return False
        return True

    def extract_duals(self) -> DualValues:
        """提取定价用对偶。

        对偶来源：
        - cover_pi: 覆盖约束
        - vehicle_alpha: 车辆约束
        - depot_beta: 中心容量约束
        - branch_arc_dual: 分支弧约束
        - clique_customer_bonus: clique cuts 对客户的惩罚聚合
        """
        cover_pi = {cid: con.Pi for cid, con in self.cover_cons.items()}
        vehicle_alpha = {vid: con.Pi for vid, con in self.vehicle_cons.items()}
        depot_beta = {did: con.Pi for did, con in self.depot_cons.items()}

        branch_arc_dual: Dict[Tuple[str, str], float] = {}
        for (kind, u, v), con in self.branch_cons.items():
            branch_arc_dual[(u, v)] = con.Pi

        clique_bonus: Dict[str, float] = {cid: 0.0 for cid in self.instance.customers}
        for cut, con in self.cuts.values():
            if cut.cut_type == "clique":
                for cid in cut.members:
                    clique_bonus[cid] += con.Pi

        return DualValues(
            cover_pi=cover_pi,
            vehicle_alpha=vehicle_alpha,
            depot_beta=depot_beta,
            branch_arc_dual=branch_arc_dual,
            clique_customer_bonus=clique_bonus,
        )

    def active_routes(self) -> List[Tuple[RouteColumn, float]]:
        """返回当前解中取值 > 0 的列，用于 cut 分离。"""
        return [(self.columns[cid], var.X) for cid, var in self.vars.items() if var.X > 1e-9]

    def selected_routes(self, threshold: float = 0.5) -> List[RouteColumn]:
        """按阈值提取路线，常用于整数解输出。"""
        return [self.columns[cid] for cid, var in self.vars.items() if var.X >= threshold]

    def aggregated_arc_flow(self) -> Dict[Tuple[str, str], float]:
        """聚合弧流 x_uv = Σ_r lambda_r * b_uvr，用于分支选弧。"""
        arc_flow: Dict[Tuple[str, str], float] = {}
        for cid, var in self.vars.items():
            val = var.X
            if val <= 1e-9:
                continue
            route = self.columns[cid]
            for arc in route.arc_flags:
                arc_flow[arc] = arc_flow.get(arc, 0.0) + val
        return arc_flow

    def _column_feasible_to_node(self, route: RouteColumn) -> bool:
        """节点可行性过滤（当前处理禁弧）。"""
        for arc in self.node.branch_rule.banned_arcs:
            if arc in route.arc_flags:
                return False
        return True
