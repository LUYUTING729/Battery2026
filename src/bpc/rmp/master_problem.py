from __future__ import annotations

"""Restricted Master Problem (RMP) 构建与求解。

支持两类后端：
- gurobi: 增量建模，适合生产规模
- highs: 通过 scipy.linprog(method='highs') 调用开源 HiGHS
"""

from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Tuple

from bpc.core.types import DualValues, InstanceData, NodeState, RouteColumn
from bpc.cuts.definitions import CutDefinition

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

try:
    import numpy as np
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    np = None
    linprog = None


class GurobiUnavailableError(RuntimeError):
    pass


@dataclass
class RmpSolveInfo:
    """RMP 求解后的核心信息。"""
    status: int
    obj_val: float
    lambda_values: Dict[str, float]


class MasterProblem:
    """RMP 封装（Gurobi / HiGHS）。"""

    def __init__(self, instance: InstanceData, node: NodeState, relax: bool = True, solver_backend: str = "auto"):
        self.instance = instance
        self.node = node
        self.relax = relax
        self.solver_backend = (solver_backend or "auto").strip().lower()

        self.vars: Dict[str, object] = {}
        self.columns: Dict[str, RouteColumn] = {}
        self.cuts: Dict[str, Tuple[CutDefinition, object]] = {}

        self._lambda_values: Dict[str, float] = {}
        self._cover_pi: Dict[str, float] = {}
        self._vehicle_alpha: Dict[str, float] = {}
        self._depot_beta: Dict[str, float] = {}
        self._branch_arc_dual: Dict[Tuple[str, str], float] = {}
        self._cut_dual: Dict[str, float] = {}

        backend = self._decide_backend(self.solver_backend)
        self.backend = backend

        if backend == "gurobi":
            self._init_gurobi()
        elif backend == "highs":
            self._init_highs()
        else:  # pragma: no cover
            raise ValueError(f"Unsupported backend: {backend}")

    def _decide_backend(self, backend: str) -> str:
        if backend in {"", "auto"}:
            if gp is not None:
                return "gurobi"
            if linprog is not None and np is not None:
                return "highs"
            raise GurobiUnavailableError("No available backend: install gurobipy or scipy (HiGHS)")
        if backend == "gurobi":
            if gp is None:
                raise GurobiUnavailableError("gurobipy is required when solver_backend='gurobi'")
            return "gurobi"
        if backend == "highs":
            if linprog is None or np is None:
                raise GurobiUnavailableError(
                    "scipy is required when solver_backend='highs'. "
                    f"Current python: {sys.executable}. "
                    "If you installed in .venv, run with '.venv/bin/python' or activate the venv."
                )
            return "highs"
        raise ValueError(f"Unknown solver_backend: {backend}")

    def _init_gurobi(self) -> None:
        self.model = gp.Model(f"RMP_{self.instance.instance_id}_node_{self.node.node_id}")
        self.model.Params.OutputFlag = 0

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

    def _init_highs(self) -> None:
        self.model = None
        self.cover_cons = {cid: None for cid in self.instance.customers}
        self.vehicle_cons = {vid: None for vid in self.instance.vehicles}
        self.depot_cons = {did: None for did in self.instance.depots}
        self.branch_cons = {
            ("ban", arc[0], arc[1]): None for arc in self.node.branch_rule.banned_arcs
        }
        for arc in self.node.branch_rule.forced_arcs:
            self.branch_cons[("force", arc[0], arc[1])] = None

    def add_columns(self, routes: List[RouteColumn]) -> int:
        if self.backend == "gurobi":
            return self._add_columns_gurobi(routes)

        added = 0
        for route in routes:
            if route.column_id in self.vars:
                continue
            if not self._column_feasible_to_node(route):
                continue
            idx = len(self.vars)
            self.vars[route.column_id] = idx
            self.columns[route.column_id] = route
            added += 1
        return added

    def _add_columns_gurobi(self, routes: List[RouteColumn]) -> int:
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
        if self.backend == "gurobi":
            return self._add_cuts_gurobi(cuts)

        added = 0
        for cut in cuts:
            if cut.cut_id in self.cuts:
                continue
            self.cuts[cut.cut_id] = (cut, None)
            added += 1
        return added

    def _add_cuts_gurobi(self, cuts: List[CutDefinition]) -> int:
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
        if self.backend == "gurobi":
            return self._solve_gurobi()
        return self._solve_highs()

    def _solve_gurobi(self) -> RmpSolveInfo:
        self.model.optimize()
        status = self.model.Status
        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            self._lambda_values = {}
            return RmpSolveInfo(status=status, obj_val=float("inf"), lambda_values={})
        values = {cid: var.X for cid, var in self.vars.items()}
        self._lambda_values = values
        return RmpSolveInfo(status=status, obj_val=float(self.model.ObjVal), lambda_values=values)

    def _solve_highs(self) -> RmpSolveInfo:
        col_ids = list(self.columns.keys())
        n = len(col_ids)
        if n == 0:
            self._lambda_values = {}
            return RmpSolveInfo(status=2, obj_val=float("inf"), lambda_values={})

        c = np.array([self.columns[cid].cost for cid in col_ids], dtype=float)

        A_eq = []
        b_eq = []
        cover_order = list(self.instance.customers.keys())
        for cust in cover_order:
            A_eq.append([1.0 if self.columns[cid].a_i.get(cust, 0) else 0.0 for cid in col_ids])
            b_eq.append(1.0)

        A_ub = []
        b_ub = []
        # (kind, key, sign_restore)
        # sign_restore=1: 该行是原始 <= 约束
        # sign_restore=-1: 该行来自原始 >= 约束，经取负转成 <=
        ineq_tags: List[Tuple[str, object, int]] = []

        for vid in self.instance.vehicles:
            A_ub.append([1.0 if self.columns[cid].vehicle_id == vid else 0.0 for cid in col_ids])
            b_ub.append(1.0)
            ineq_tags.append(("vehicle", vid, 1))

        for did, dep in self.instance.depots.items():
            A_ub.append([1.0 if self.columns[cid].depot_id == did else 0.0 for cid in col_ids])
            b_ub.append(float(dep.max_vehicles))
            ineq_tags.append(("depot", did, 1))

        for arc in self.node.branch_rule.banned_arcs:
            A_ub.append([1.0 if arc in self.columns[cid].arc_flags else 0.0 for cid in col_ids])
            b_ub.append(0.0)
            ineq_tags.append(("branch_ban", arc, 1))

        for arc in self.node.branch_rule.forced_arcs:
            row = [-1.0 if arc in self.columns[cid].arc_flags else 0.0 for cid in col_ids]
            A_ub.append(row)
            b_ub.append(-1.0)
            ineq_tags.append(("branch_force", arc, -1))

        cut_order: List[str] = []
        for cut_id, (cut, _) in self.cuts.items():
            coeffs = [float(cut.coefficient(self.columns[cid])) for cid in col_ids]
            if cut.sense == "<=":
                A_ub.append(coeffs)
                b_ub.append(float(cut.rhs))
                ineq_tags.append(("cut", cut_id, 1))
            elif cut.sense == ">=":
                A_ub.append([-x for x in coeffs])
                b_ub.append(float(-cut.rhs))
                ineq_tags.append(("cut", cut_id, -1))
            else:
                raise ValueError(f"Unsupported cut sense: {cut.sense}")
            cut_order.append(cut_id)

        bounds = [(0.0, 1.0) for _ in col_ids]
        res = linprog(
            c,
            A_ub=(np.array(A_ub, dtype=float) if A_ub else None),
            b_ub=(np.array(b_ub, dtype=float) if b_ub else None),
            A_eq=(np.array(A_eq, dtype=float) if A_eq else None),
            b_eq=(np.array(b_eq, dtype=float) if b_eq else None),
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            self._lambda_values = {}
            return RmpSolveInfo(status=int(res.status), obj_val=float("inf"), lambda_values={})

        values = {cid: float(res.x[i]) for i, cid in enumerate(col_ids)}
        self._lambda_values = values

        self._cover_pi = {}
        eq_duals = getattr(getattr(res, "eqlin", None), "marginals", None)
        if eq_duals is not None:
            for i, cid in enumerate(cover_order):
                self._cover_pi[cid] = float(eq_duals[i])
        else:
            self._cover_pi = {cid: 0.0 for cid in cover_order}

        self._vehicle_alpha = {vid: 0.0 for vid in self.instance.vehicles}
        self._depot_beta = {did: 0.0 for did in self.instance.depots}
        self._branch_arc_dual = {}
        self._cut_dual = {cid: 0.0 for cid in cut_order}

        ineq_duals = getattr(getattr(res, "ineqlin", None), "marginals", None)
        if ineq_duals is not None:
            for i, (kind, key, sign_restore) in enumerate(ineq_tags):
                pi = float(sign_restore) * float(ineq_duals[i])
                if kind == "vehicle":
                    self._vehicle_alpha[str(key)] = pi
                elif kind == "depot":
                    self._depot_beta[str(key)] = pi
                elif kind in {"branch_ban", "branch_force"}:
                    self._branch_arc_dual[key] = pi
                elif kind == "cut":
                    self._cut_dual[str(key)] = pi

        return RmpSolveInfo(status=0, obj_val=float(res.fun), lambda_values=values)

    def is_integral(self, tol: float = 1e-6) -> bool:
        values = self._lambda_values if self.backend == "highs" else {cid: var.X for cid, var in self.vars.items()}
        for val in values.values():
            if abs(val - round(val)) > tol:
                return False
        return True

    def extract_duals(self) -> DualValues:
        if self.backend == "gurobi":
            return self._extract_duals_gurobi()
        return self._extract_duals_highs()

    def _extract_duals_gurobi(self) -> DualValues:
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

    def _extract_duals_highs(self) -> DualValues:
        cover_pi = {cid: self._cover_pi.get(cid, 0.0) for cid in self.instance.customers}
        vehicle_alpha = {vid: self._vehicle_alpha.get(vid, 0.0) for vid in self.instance.vehicles}
        depot_beta = {did: self._depot_beta.get(did, 0.0) for did in self.instance.depots}
        branch_arc_dual = dict(self._branch_arc_dual)

        clique_bonus: Dict[str, float] = {cid: 0.0 for cid in self.instance.customers}
        for cut_id, (cut, _) in self.cuts.items():
            if cut.cut_type == "clique":
                pi = self._cut_dual.get(cut_id, 0.0)
                for cid in cut.members:
                    clique_bonus[cid] += pi

        return DualValues(
            cover_pi=cover_pi,
            vehicle_alpha=vehicle_alpha,
            depot_beta=depot_beta,
            branch_arc_dual=branch_arc_dual,
            clique_customer_bonus=clique_bonus,
        )

    def active_routes(self) -> List[Tuple[RouteColumn, float]]:
        if self.backend == "gurobi":
            return [(self.columns[cid], var.X) for cid, var in self.vars.items() if var.X > 1e-9]
        return [(self.columns[cid], val) for cid, val in self._lambda_values.items() if val > 1e-9]

    def selected_routes(self, threshold: float = 0.5) -> List[RouteColumn]:
        if self.backend == "gurobi":
            return [self.columns[cid] for cid, var in self.vars.items() if var.X >= threshold]
        return [self.columns[cid] for cid, val in self._lambda_values.items() if val >= threshold]

    def aggregated_arc_flow(self) -> Dict[Tuple[str, str], float]:
        arc_flow: Dict[Tuple[str, str], float] = {}
        if self.backend == "gurobi":
            value_iter = ((cid, var.X) for cid, var in self.vars.items())
        else:
            value_iter = self._lambda_values.items()

        for cid, val in value_iter:
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
