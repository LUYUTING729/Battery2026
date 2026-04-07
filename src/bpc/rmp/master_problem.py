from __future__ import annotations

"""Restricted Master Problem (RMP) 构建与求解（Gurobi-only）。"""

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
    """RMP 封装（Gurobi）。"""

    def __init__(self, instance: InstanceData, node: NodeState, relax: bool = True, solver_backend: str = "gurobi"):
        self.instance = instance
        self.node = node
        self.relax = relax
        self.solver_backend = (solver_backend or "gurobi").strip().lower()

        self.vars: Dict[str, object] = {}
        self.columns: Dict[str, RouteColumn] = {}
        self.cuts: Dict[str, Tuple[CutDefinition, object]] = {}

        self._lambda_values: Dict[str, float] = {}

        self.backend = self._decide_backend(self.solver_backend)
        self._init_gurobi()

    def _decide_backend(self, backend: str) -> str:
        if backend in {"", "auto", "gurobi"}:
            if gp is None:
                raise GurobiUnavailableError("gurobipy is required (Gurobi-only backend).")
            return "gurobi"
        raise ValueError(f"Unknown solver_backend: {backend}. Only 'gurobi' is supported.")

    def _init_gurobi(self) -> None:
        self.model = gp.Model(f"RMP_{self.instance.instance_id}_node_{self.node.node_id}")
        # 默认开启 Gurobi 日志，便于终端观察求解进度与诊断。
        self.model.Params.OutputFlag = 1

        self.cover_cons = {
            cid: self.model.addConstr(gp.LinExpr() == 1.0, name=f"cover_{cid}")
            for cid in self.instance.customers
        }
        self.vehicle_cons = {
            vid: self.model.addConstr(gp.LinExpr() <= 1.0, name=f"vehicle_{vid}")
            for vid in self.instance.vehicles
        }
        self.depot_cons = {
            did: self.model.addConstr(gp.LinExpr() <= dep.max_vehicles, name=f"depot_{did}")
            for did, dep in self.instance.depots.items()
        }

        self.branch_cons: Dict[Tuple[str, str, str], object] = {}
        for arc in self.node.branch_rule.banned_arcs:
            key = ("ban", arc[0], arc[1])
            self.branch_cons[key] = self.model.addConstr(gp.LinExpr() <= 0.0, name=f"ban_{arc[0]}_{arc[1]}")
        for arc in self.node.branch_rule.forced_arcs:
            key = ("force", arc[0], arc[1])
            self.branch_cons[key] = self.model.addConstr(gp.LinExpr() >= 1.0, name=f"force_{arc[0]}_{arc[1]}")

        # 可行化人工变量：避免“受限列池不可行”被误判为节点真不可行。
        self._artificial_penalty = 1e6
        self._artificial_vars: Dict[str, object] = {}
        for cid, con in self.cover_cons.items():
            av = self.model.addVar(lb=0.0, ub=GRB.INFINITY, obj=self._artificial_penalty, vtype=GRB.CONTINUOUS, name=f"art_cover_{cid}")
            self.model.chgCoeff(con, av, 1.0)
            self._artificial_vars[f"cover:{cid}"] = av
        for (kind, u, v), con in self.branch_cons.items():
            if kind != "force":
                continue
            # force: lhs >= 1，可用 +a 可行化
            av = self.model.addVar(lb=0.0, ub=GRB.INFINITY, obj=self._artificial_penalty, vtype=GRB.CONTINUOUS, name=f"art_force_{u}_{v}")
            self.model.chgCoeff(con, av, 1.0)
            self._artificial_vars[f"force:{u}->{v}"] = av
        self.model.update()

    def add_columns(self, routes: List[RouteColumn]) -> int:
        added = 0
        for route in routes:
            if route.column_id in self.vars:
                continue
            if not self._column_feasible_to_node(route):
                continue

            vtype = GRB.CONTINUOUS if self.relax else GRB.BINARY
            ub = GRB.INFINITY if self.relax else 1.0
            var = self.model.addVar(lb=0.0, ub=ub, obj=route.cost, vtype=vtype, name=f"lam_{route.column_id}")
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

            # 保障 cut 加入后的 RMP 可行（列不足时允许人工变量暂时兜底）。
            if sense == ">=":
                av = self.model.addVar(
                    lb=0.0,
                    ub=GRB.INFINITY,
                    obj=self._artificial_penalty,
                    vtype=GRB.CONTINUOUS,
                    name=f"art_cut_ge_{cut.cut_id}",
                )
                self.model.chgCoeff(con, av, 1.0)
                self._artificial_vars[f"cut:{cut.cut_id}"] = av
            elif sense == "<=":
                av = self.model.addVar(
                    lb=0.0,
                    ub=GRB.INFINITY,
                    obj=self._artificial_penalty,
                    vtype=GRB.CONTINUOUS,
                    name=f"art_cut_le_{cut.cut_id}",
                )
                # <= rhs 通过 -a 保证可行
                self.model.chgCoeff(con, av, -1.0)
                self._artificial_vars[f"cut:{cut.cut_id}"] = av

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
        self.model.optimize()
        status = self.model.Status
        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            self._lambda_values = {}
            return RmpSolveInfo(status=status, obj_val=float("inf"), lambda_values={})
        values = {cid: var.X for cid, var in self.vars.items()}
        self._lambda_values = values
        return RmpSolveInfo(status=status, obj_val=float(self.model.ObjVal), lambda_values=values)

    def is_integral(self, tol: float = 1e-6) -> bool:
        # 无可用 primal 解时，X/Pi 等属性不可访问；按“非整数（不可用于分支收敛）”处理。
        if getattr(self.model, "SolCount", 0) <= 0:
            return False
        values = self._lambda_values or {cid: var.X for cid, var in self.vars.items()}
        for val in values.values():
            if abs(val - round(val)) > tol:
                return False
        if self.artificial_violation(tol=tol) > tol:
            return False
        return True

    def extract_duals(self) -> DualValues:
        cover_pi = {cid: con.Pi for cid, con in self.cover_cons.items()}
        vehicle_alpha = {vid: con.Pi for vid, con in self.vehicle_cons.items()}
        depot_beta = {did: con.Pi for did, con in self.depot_cons.items()}

        branch_arc_dual: Dict[Tuple[str, str], float] = {}
        for (kind, u, v), con in self.branch_cons.items():
            branch_arc_dual[(u, v)] = con.Pi

        clique_bonus: Dict[str, float] = {cid: 0.0 for cid in self.instance.customers}
        cut_terms: List[DualValues.CutDualTerm] = []
        for cut, con in self.cuts.values():
            cut_terms.append(
                DualValues.CutDualTerm(
                    cut_id=cut.cut_id,
                    cut_type=cut.cut_type,
                    dual=float(con.Pi),
                    members=cut.members,
                    weights=cut.weights,
                )
            )
            if cut.cut_type == "clique":
                for cid in cut.members:
                    clique_bonus[cid] += con.Pi

        return DualValues(
            cover_pi=cover_pi,
            vehicle_alpha=vehicle_alpha,
            depot_beta=depot_beta,
            branch_arc_dual=branch_arc_dual,
            clique_customer_bonus=clique_bonus,
            cut_terms=cut_terms,
        )

    def active_routes(self) -> List[Tuple[RouteColumn, float]]:
        if getattr(self.model, "SolCount", 0) <= 0:
            return []
        values = self._lambda_values or {cid: var.X for cid, var in self.vars.items()}
        return [(self.columns[cid], val) for cid, val in values.items() if val > 1e-9]

    def selected_routes(self, threshold: float = 0.5) -> List[RouteColumn]:
        if getattr(self.model, "SolCount", 0) <= 0:
            return []
        values = self._lambda_values or {cid: var.X for cid, var in self.vars.items()}
        return [self.columns[cid] for cid, val in values.items() if val >= threshold]

    def aggregated_arc_flow(self) -> Dict[Tuple[str, str], float]:
        if getattr(self.model, "SolCount", 0) <= 0:
            return {}
        arc_flow: Dict[Tuple[str, str], float] = {}
        values = self._lambda_values or {cid: var.X for cid, var in self.vars.items()}
        for cid, val in values.items():
            if val <= 1e-9:
                continue
            route = self.columns[cid]
            for arc in route.arc_flags:
                arc_flow[arc] = arc_flow.get(arc, 0.0) + val
        return arc_flow

    def _column_feasible_to_node(self, route: RouteColumn) -> bool:
        for arc in self.node.branch_rule.banned_arcs:
            if arc in route.arc_flags:
                return False
        return True

    def artificial_violation(self, tol: float = 1e-9) -> float:
        if getattr(self.model, "SolCount", 0) <= 0:
            return float("inf")
        total = 0.0
        for var in self._artificial_vars.values():
            x = float(var.X)
            if x > tol:
                total += x
        return total
