from __future__ import annotations

"""直接 Gurobi 求解器：按 main.tex 中的 SCF 强模型建模。"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from bpc.core.types import Arc, InstanceData, RouteColumn, SolveResult, SolverConfig
from bpc.data.loader import load_instance
from bpc.data.xlsx_loader import (
    ExcelInstanceBundle,
    dump_excel_model_profile,
    load_instance_bundle_from_excel,
)

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


class GurobiDirectSolverUnavailableError(RuntimeError):
    """缺少 gurobipy 时抛出。"""


def _mkdir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _safe_gap(obj_primal: float, obj_dual: float) -> float:
    if not math.isfinite(obj_primal) or not math.isfinite(obj_dual):
        return float("inf")
    denom = abs(obj_primal)
    if denom <= 1e-9:
        return 0.0 if abs(obj_primal - obj_dual) <= 1e-9 else float("inf")
    return abs(obj_primal - obj_dual) / denom


def _write_routes(output_dir: Path, result: SolveResult) -> None:
    path = output_dir / "routes.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["vehicle_id", "depot_id", "customer_seq", "cost", "load", "dist"],
        )
        writer.writeheader()
        for r in result.routes:
            writer.writerow(
                {
                    "vehicle_id": r.vehicle_id,
                    "depot_id": r.depot_id,
                    "customer_seq": "-".join(r.customer_seq),
                    "cost": r.cost,
                    "load": r.load,
                    "dist": r.dist,
                }
            )


def _write_metrics(output_dir: Path, result: SolveResult) -> None:
    path = output_dir / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in result.stats.items():
            writer.writerow({"metric": k, "value": v})


def _write_solution_json(output_dir: Path, result: SolveResult) -> None:
    path = output_dir / "solution.json"
    payload = {
        "status": result.status,
        "obj_primal": result.obj_primal,
        "obj_dual": result.obj_dual,
        "gap": result.gap,
        "stats": result.stats,
        "routes": [
            {
                "vehicle_id": r.vehicle_id,
                "depot_id": r.depot_id,
                "customer_seq": list(r.customer_seq),
                "cost": r.cost,
                "load": r.load,
                "dist": r.dist,
            }
            for r in result.routes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_trace(output_dir: Path, rows: Iterable[dict]) -> None:
    path = output_dir / "trace.csv"
    fieldnames = ["event", "value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_run_config(output_dir: Path, instance_id: str, cfg: SolverConfig, source_info: dict) -> None:
    path = output_dir / "run_config.json"
    payload = {
        "solver": "gurobi_direct_scf",
        "instance_id": instance_id,
        "gurobi_log_file": str(output_dir / "gurobi.log"),
        "time_limit_s": cfg.time_limit_s,
        "mip_gap": cfg.mip_gap,
        "random_seed": cfg.random_seed,
        "problem": {
            "capacity_u": cfg.problem.capacity_u,
            "range_q": cfg.problem.range_q,
            "cost_per_km": cfg.problem.cost_per_km,
            "vehicle_count": cfg.problem.vehicle_count,
            "customer_sheet_name": cfg.problem.customer_sheet_name,
            "depot_sheet_name": cfg.problem.depot_sheet_name,
            "vehicle_sheet_name": cfg.problem.vehicle_sheet_name,
        },
        "source": source_info,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _incoming_by_customer(arcs: Iterable[Arc], customers: List[str]) -> Dict[str, List[Arc]]:
    res = {cid: [] for cid in customers}
    for u, v in arcs:
        if v in res:
            res[v].append((u, v))
    return res


def _outgoing_by_customer(arcs: Iterable[Arc], customers: List[str]) -> Dict[str, List[Arc]]:
    res = {cid: [] for cid in customers}
    for u, v in arcs:
        if u in res:
            res[u].append((u, v))
    return res


def _outgoing_by_depot(arcs: Iterable[Arc], depots: List[str]) -> Dict[str, List[Arc]]:
    res = {did: [] for did in depots}
    for u, v in arcs:
        if u in res:
            res[u].append((u, v))
    return res


def _incoming_by_depot(arcs: Iterable[Arc], depots: List[str]) -> Dict[str, List[Arc]]:
    res = {did: [] for did in depots}
    for u, v in arcs:
        if v in res:
            res[v].append((u, v))
    return res


def _reconstruct_vehicle_route(
    selected_arcs: Dict[Arc, float],
    depot_id: str,
    customers: set[str],
    tol: float = 1e-6,
) -> Tuple[str, ...]:
    active_succ: Dict[str, str] = {}
    for (u, v), val in selected_arcs.items():
        if val > 1.0 - tol:
            active_succ[u] = v
    seq: List[str] = []
    cur = depot_id
    seen: set[str] = set()
    while True:
        nxt = active_succ.get(cur)
        if nxt is None or nxt == depot_id:
            break
        if nxt in seen:
            raise ValueError(f"cycle reconstruction failed for depot={depot_id}, repeated node={nxt}")
        seen.add(nxt)
        if nxt in customers:
            seq.append(nxt)
        cur = nxt
    return tuple(seq)


def _collect_routes(
    instance: InstanceData,
    vehicles: List[str],
    depots: List[str],
    route_arcs: List[Arc],
    x_vars: Dict[Tuple[str, Arc], gp.Var],
    z_vars: Dict[Tuple[str, str], gp.Var],
) -> List[RouteColumn]:
    customers_set = set(instance.customers)
    routes: List[RouteColumn] = []
    for k in vehicles:
        chosen_depots = [j for j in depots if z_vars[(k, j)].X > 0.5]
        if not chosen_depots:
            continue
        depot_id = chosen_depots[0]
        selected_arcs = {(u, v): x_vars[(k, (u, v))].X for (u, v) in route_arcs if x_vars[(k, (u, v))].X > 1e-6}
        customer_seq = _reconstruct_vehicle_route(selected_arcs, depot_id=depot_id, customers=customers_set)
        arc_flags = {arc for arc, val in selected_arcs.items() if val > 0.5}
        cost = sum(instance.cost[arc] for arc in arc_flags) + instance.dispatch_cost[(k, depot_id)]
        dist = sum(instance.dist[arc] for arc in arc_flags)
        load = sum(instance.demand[cid] for cid in customer_seq)
        routes.append(
            RouteColumn(
                column_id=f"direct_{k}_{depot_id}",
                vehicle_id=k,
                depot_id=depot_id,
                customer_seq=customer_seq,
                a_i={cid: int(cid in customer_seq) for cid in instance.customers},
                arc_flags=arc_flags,
                cost=cost,
                load=load,
                dist=dist,
            )
        )
    return routes


def _load_instance_from_any_source(
    instance_id: str,
    cfg: SolverConfig,
    db_path: str,
    csv_dir: str,
    excel_path: str,
) -> Tuple[InstanceData, ExcelInstanceBundle | None, dict]:
    bundle: ExcelInstanceBundle | None = None
    if excel_path:
        bundle = load_instance_bundle_from_excel(
            xlsx_path=excel_path,
            instance_id=instance_id,
            customer_sheet_index=cfg.problem.customer_sheet_index,
            depot_sheet_index=cfg.problem.depot_sheet_index,
            customer_sheet_name=cfg.problem.customer_sheet_name,
            depot_sheet_name=cfg.problem.depot_sheet_name,
            vehicle_sheet_name=cfg.problem.vehicle_sheet_name,
            override_vehicle_count=cfg.problem.vehicle_count,
            allow_infeasible_vehicle_count=cfg.problem.allow_infeasible_vehicle_count,
            override_capacity_u=cfg.problem.capacity_u,
            override_range_q=cfg.problem.range_q,
            override_cost_per_km=cfg.problem.cost_per_km,
        )
        return bundle.instance, bundle, {"excel_path": excel_path, "db_path": "", "csv_dir": ""}
    instance = load_instance(instance_id=instance_id, db_path=db_path, csv_dir=csv_dir)
    return instance, None, {"excel_path": "", "db_path": db_path, "csv_dir": csv_dir}


def solve_instance(
    instance_id: str,
    cfg: SolverConfig,
    db_path: str = "instances.db",
    csv_dir: str = "",
    excel_path: str = "",
    excel_profile_path: str = "",
) -> SolveResult:
    """按 main.tex 中的 SCF 模型直接求解单实例。"""
    if gp is None or GRB is None:
        raise GurobiDirectSolverUnavailableError("gurobipy is required for src/gurobi direct solver.")

    t0 = time.time()
    instance, bundle, source_info = _load_instance_from_any_source(
        instance_id=instance_id,
        cfg=cfg,
        db_path=db_path,
        csv_dir=csv_dir,
        excel_path=excel_path,
    )
    outdir = _mkdir(cfg.output_dir)
    _write_run_config(outdir, instance.instance_id, cfg, source_info)
    if bundle is not None:
        profile_out = excel_profile_path.strip() if excel_profile_path else str(outdir / "excel_model_profile.json")
        dump_excel_model_profile(bundle, profile_out)

    customers = sorted(instance.customers)
    depots = sorted(instance.depots)
    vehicles = sorted(instance.vehicles)
    route_arcs = sorted(
        (u, v)
        for (u, v) in instance.arcs
        if u in instance.customers or u in instance.depots
        if v in instance.customers or v in instance.depots
    )
    if not route_arcs:
        raise ValueError("No routing arcs available on V = I union J.")

    in_customer = _incoming_by_customer(route_arcs, customers)
    out_customer = _outgoing_by_customer(route_arcs, customers)
    out_depot = _outgoing_by_depot(route_arcs, depots)
    in_depot = _incoming_by_depot(route_arcs, depots)
    demand = instance.demand
    U = instance.capacity_u
    Q = instance.range_q
    big_m_d = sum(demand.values())
    big_m_tau = Q
    total_demand = sum(demand.values())
    total_fleet_capacity = U * len(vehicles)

    model = gp.Model(f"mdvrp_direct_{instance.instance_id}")
    model.Params.TimeLimit = cfg.time_limit_s
    model.Params.MIPGap = cfg.mip_gap
    model.Params.Seed = cfg.random_seed
    model.Params.LogFile = str(outdir / "gurobi.log")

    x = {
        (k, arc): model.addVar(vtype=GRB.BINARY, name=f"x[{k},{arc[0]},{arc[1]}]")
        for k in vehicles
        for arc in route_arcs
    }
    z = {
        (k, j): model.addVar(vtype=GRB.BINARY, name=f"z[{k},{j}]")
        for k in vehicles
        for j in depots
    }
    f = {
        (k, arc): model.addVar(lb=0.0, ub=U, vtype=GRB.CONTINUOUS, name=f"f[{k},{arc[0]},{arc[1]}]")
        for k in vehicles
        for arc in route_arcs
    }
    tau = {
        (k, i): model.addVar(lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name=f"tau[{k},{i}]")
        for k in vehicles
        for i in customers
    }

    model.setObjective(
        gp.quicksum(instance.cost[arc] * x[(k, arc)] for k in vehicles for arc in route_arcs)
        + gp.quicksum(instance.dispatch_cost[(k, j)] * z[(k, j)] for k in vehicles for j in depots),
        sense=GRB.MINIMIZE,
    )

    served_expr = {
        (k, i): gp.quicksum(x[(k, arc)] for arc in in_customer[i])
        for k in vehicles
        for i in customers
    }
    total_demand_expr = {
        k: gp.quicksum(demand[i] * served_expr[(k, i)] for i in customers)
        for k in vehicles
    }

    for k in vehicles:
        model.addConstr(gp.quicksum(z[(k, j)] for j in depots) <= 1, name=f"c1[{k}]")

    for i in customers:
        model.addConstr(gp.quicksum(served_expr[(k, i)] for k in vehicles) == 1, name=f"c2[{i}]")

    for k in vehicles:
        active_vehicle = gp.quicksum(z[(k, j)] for j in depots)
        for i in customers:
            model.addConstr(
                served_expr[(k, i)] == gp.quicksum(x[(k, arc)] for arc in out_customer[i]),
                name=f"c3[{k},{i}]",
            )
            model.addConstr(served_expr[(k, i)] <= active_vehicle, name=f"c6[{k},{i}]")

    for k in vehicles:
        for j in depots:
            model.addConstr(gp.quicksum(x[(k, arc)] for arc in out_depot[j]) == z[(k, j)], name=f"c4out[{k},{j}]")
            model.addConstr(gp.quicksum(x[(k, arc)] for arc in in_depot[j]) == z[(k, j)], name=f"c4in[{k},{j}]")

    for j in depots:
        model.addConstr(
            gp.quicksum(z[(k, j)] for k in vehicles) <= instance.depots[j].max_vehicles,
            name=f"c5[{j}]",
        )

    for k in vehicles:
        for arc in route_arcs:
            model.addConstr(f[(k, arc)] <= U * x[(k, arc)], name=f"c7[{k},{arc[0]},{arc[1]}]")

    for k in vehicles:
        for i in customers:
            model.addConstr(
                gp.quicksum(f[(k, arc)] for arc in in_customer[i]) - gp.quicksum(f[(k, arc)] for arc in out_customer[i])
                == demand[i] * served_expr[(k, i)],
                name=f"c8[{k},{i}]",
            )

    for k in vehicles:
        for j in depots:
            depot_net = gp.quicksum(f[(k, arc)] for arc in out_depot[j]) - gp.quicksum(f[(k, arc)] for arc in in_depot[j])
            model.addConstr(depot_net >= total_demand_expr[k] - big_m_d * (1 - z[(k, j)]), name=f"c9lb[{k},{j}]")
            model.addConstr(depot_net <= total_demand_expr[k] + big_m_d * (1 - z[(k, j)]), name=f"c9ub[{k},{j}]")

    for k in vehicles:
        for i in customers:
            model.addConstr(tau[(k, i)] <= Q * served_expr[(k, i)], name=f"c10[{k},{i}]")

    for k in vehicles:
        for i in customers:
            for h in customers:
                if i == h or (i, h) not in instance.dist:
                    continue
                if (k, (i, h)) not in x:
                    continue
                model.addConstr(
                    tau[(k, h)] >= tau[(k, i)] + instance.dist[(i, h)] - Q * (1 - x[(k, (i, h))]),
                    name=f"c11[{k},{i},{h}]",
                )

    for k in vehicles:
        for j in depots:
            for i in customers:
                if (j, i) in instance.dist and (k, (j, i)) in x:
                    model.addConstr(
                        tau[(k, i)] >= instance.dist[(j, i)] - Q * (1 - x[(k, (j, i))]) - big_m_tau * (1 - z[(k, j)]),
                        name=f"c12[{k},{j},{i}]",
                    )
                if (i, j) in instance.dist and (k, (i, j)) in x:
                    model.addConstr(
                        tau[(k, i)] + instance.dist[(i, j)]
                        <= Q + big_m_tau * (1 - x[(k, (i, j))]) + big_m_tau * (1 - z[(k, j)]),
                        name=f"c13[{k},{i},{j}]",
                    )

    model.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    status = status_map.get(model.Status, f"STATUS_{model.Status}")

    if model.SolCount > 0:
        obj_primal = float(model.ObjVal)
        obj_dual = float(model.ObjBound)
        routes = _collect_routes(instance, vehicles, depots, route_arcs, x, z)
    else:
        obj_primal = float("inf")
        obj_dual = float(model.ObjBound) if hasattr(model, "ObjBound") else float("inf")
        routes = []

    runtime = time.time() - t0
    gap = _safe_gap(obj_primal, obj_dual)
    result = SolveResult(
        status=status,
        obj_primal=obj_primal,
        obj_dual=obj_dual,
        gap=gap,
        routes=routes,
        stats={
            "runtime_sec": runtime,
            "num_customers": float(len(customers)),
            "num_depots": float(len(depots)),
            "num_vehicles": float(len(vehicles)),
            "capacity_u": float(U),
            "range_q": float(Q),
            "total_demand": float(total_demand),
            "total_fleet_capacity": float(total_fleet_capacity),
            "num_route_arcs": float(len(route_arcs)),
            "num_vars": float(model.NumVars),
            "num_constrs": float(model.NumConstrs),
            "node_count": float(model.NodeCount),
            "sol_count": float(model.SolCount),
            "mip_gap_reported": float(model.MIPGap) if model.SolCount > 0 else float("inf"),
            "obj_bound": obj_dual,
        },
    )

    _write_trace(
        outdir,
        [
            {"event": "status", "value": status},
            {"event": "runtime_sec", "value": runtime},
            {"event": "obj_primal", "value": obj_primal},
            {"event": "obj_dual", "value": obj_dual},
            {"event": "sol_count", "value": model.SolCount},
        ],
    )
    _write_routes(outdir, result)
    _write_metrics(outdir, result)
    _write_solution_json(outdir, result)
    return result
