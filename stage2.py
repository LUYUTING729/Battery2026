from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import math
import random
import gurobipy as gp
from gurobipy import GRB


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# -------------------------
# Random instance generator
# -------------------------
def generate_random_instance(
    n_customers: int = 10,
    n_depots: int = 2,
    n_vehicles: int = 2,
    seed: int = 1,
    coord_box: Tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0),  # (xmin, xmax, ymin, ymax)
    depot_box: Optional[Tuple[float, float, float, float]] = None,
    demand_range: Tuple[int, int] = (1, 5),
    capacity_U: Optional[float] = None,
    # Endurance strategy: if Q is None, compute as factor * (max_depot_customer_dist + margin)
    endurance_Q: Optional[float] = None,
    endurance_factor: float = 4.0,
    endurance_margin: float = 20.0,
    # Depot capacities
    depot_capacity: Optional[int] = None,
    # Ensure feasibility by scaling U / Q if needed
    ensure_feasible: bool = True,
) -> Tuple[
    List[Any], List[Any], List[Any], List[Any], List[Tuple[Any, Any]],
    Dict[Any, float], float, float,
    Dict[Tuple[Any, Any], float], Dict[Tuple[Any, Any], float],
    Dict[Tuple[Any, Any], float], Dict[Any, Any], Dict[Any, int]
]:
    """
    Returns:
      I, J, K, P, A, q, U, Q, c, d, w, orig, Kmax
    Compatible with build_and_solve_scf() below.
    """

    rng = random.Random(seed)
    xmin, xmax, ymin, ymax = coord_box
    if depot_box is None:
        depot_box = coord_box
    dxmin, dxmax, dymin, dymax = depot_box

    # Sets
    I = [f"c{i+1}" for i in range(n_customers)]
    J = [f"d{j+1}" for j in range(n_depots)]
    K = [f"v{k+1}" for k in range(n_vehicles)]
    P = [f"p{k+1}" for k in range(n_vehicles)]  # 1 origin per vehicle (simple and common)

    # Coordinates
    coord: Dict[Any, Tuple[float, float]] = {}
    # depots
    for j in J:
        coord[j] = (rng.uniform(dxmin, dxmax), rng.uniform(dymin, dymax))
    # customers
    for i in I:
        coord[i] = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
    # vehicle initial locations
    for p in P:
        coord[p] = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))

    # Demands
    q: Dict[Any, float] = {i: float(rng.randint(demand_range[0], demand_range[1])) for i in I}
    total_demand = sum(q.values())
    max_demand = max(q.values()) if I else 0.0

    # Choose U (capacity) if not provided
    if capacity_U is None:
        # A safe default: allow each vehicle to serve ~ total/n_vehicles, plus buffer, but at least max demand
        capacity_U = max(max_demand, math.ceil(total_demand / max(1, n_vehicles)) + demand_range[1])
    U = float(capacity_U)

    # If ensure_feasible, guarantee: n_vehicles * U >= total_demand and U >= max_demand
    if ensure_feasible:
        if U < max_demand:
            U = float(max_demand)
        if n_vehicles * U < total_demand:
            # scale U up to make aggregate capacity feasible
            U = float(math.ceil(total_demand / n_vehicles))

    # Node set and allowed arcs
    V = I + J
    A: List[Tuple[Any, Any]] = []
    for u in V:
        for v in V:
            if u == v:
                continue
            if (u in J) and (v in J):
                continue  # no depot->depot
            A.append((u, v))

    # Distances and costs
    c: Dict[Tuple[Any, Any], float] = {}
    d: Dict[Tuple[Any, Any], float] = {}
    for (u, v) in A:
        dist = euclid(coord[u], coord[v])
        d[(u, v)] = dist
        c[(u, v)] = dist  # cost = distance; replace if needed

    # Endurance Q
    if endurance_Q is None:
        # Use max depot-customer distance to set a conservative Q
        max_dc = 0.0
        for j in J:
            for i in I:
                max_dc = max(max_dc, euclid(coord[j], coord[i]))
        Q = float(endurance_factor * max_dc + endurance_margin)
    else:
        Q = float(endurance_Q)

    # If ensure_feasible: make Q at least enough to do depot->customer->depot for farthest customer
    if ensure_feasible and I:
        max_roundtrip = 0.0
        for j in J:
            for i in I:
                dist_ji = euclid(coord[j], coord[i])
                max_roundtrip = max(max_roundtrip, 2.0 * dist_ji)
        if Q < max_roundtrip:
            Q = max_roundtrip + endurance_margin

    # Relocation costs w[p,j]
    w: Dict[Tuple[Any, Any], float] = {}
    for p in P:
        for j in J:
            w[(p, j)] = euclid(coord[p], coord[j])

    # origins
    orig = {K[idx]: P[idx] for idx in range(n_vehicles)}  # v_k originates at p_k

    # depot capacity Kmax
    if depot_capacity is None:
        # allow all vehicles at any depot by default
        depot_capacity = n_vehicles
    Kmax = {j: int(depot_capacity) for j in J}

    return I, J, K, P, A, q, U, Q, c, d, w, orig, Kmax


# -------------------------
# SCF solver (fixed indexing)
# -------------------------
def build_and_solve_scf(
    I: List[Any],
    J: List[Any],
    K: List[Any],
    P: List[Any],
    A: List[Tuple[Any, Any]],
    q: Dict[Any, float],
    U: float,
    Q: float,
    c: Dict[Tuple[Any, Any], float],
    d: Dict[Tuple[Any, Any], float],
    w: Dict[Tuple[Any, Any], float],
    orig: Dict[Any, Any],
    Kmax: Dict[Any, int],
    time_limit: float = 30.0,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[gp.Model, Dict[Tuple[Any, Any, Any], int], Dict[Tuple[Any, Any], int]]:

    Iset, Jset, Pset = set(I), set(J), set(P)
    assert Iset.isdisjoint(Jset), "I and J must be disjoint."
    for i in I:
        if q[i] > U:
            raise ValueError(f"Infeasible: q[{i}]={q[i]} > U={U}")
    for k in K:
        if orig[k] not in Pset:
            raise ValueError(f"orig[{k}]={orig[k]} not in P")

    nodes = list(I) + list(J)
    in_arcs = {n: [] for n in nodes}
    out_arcs = {n: [] for n in nodes}
    for (u, v) in A:
        out_arcs[u].append((u, v))
        in_arcs[v].append((u, v))

    M_demand = sum(q[i] for i in I)
    M_tau = Q

    m = gp.Model("SCF_MDVRP")
    if not verbose:
        m.Params.OutputFlag = 0
    m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # parameters for fast incumbents
    m.Params.MIPFocus = 1
    m.Params.Heuristics = 0.2
    m.Params.Symmetry = 2
    m.Params.Cuts = 2
    m.Params.Presolve = 2

    # Variables (IMPORTANT indexing)
    x = m.addVars(K, A, vtype=GRB.BINARY, name="x")     # x[k,u,v]
    z = m.addVars(K, J, vtype=GRB.BINARY, name="z")     # z[k,j]
    f = m.addVars(K, A, lb=0.0, vtype=GRB.CONTINUOUS, name="f")  # f[k,u,v]
    tau = m.addVars(K, I, lb=0.0, vtype=GRB.CONTINUOUS, name="tau")

    visit = {(k, i): gp.quicksum(x[k, u, i] for (u, _) in in_arcs[i]) for k in K for i in I}

    # Objective
    m.setObjective(
        gp.quicksum(c[(u, v)] * x[k, u, v] for k in K for (u, v) in A)
        + gp.quicksum(w[(orig[k], j)] * z[k, j] for k in K for j in J),
        GRB.MINIMIZE
    )

    # (1) depot selection
    m.addConstrs((gp.quicksum(z[k, j] for j in J) <= 1 for k in K), name="choose_depot_le1")

    # (2) serve once
    m.addConstrs((gp.quicksum(visit[k, i] for k in K) == 1 for i in I), name="serve_once")

    # (3) route continuity at customers
    m.addConstrs(
        (
            gp.quicksum(x[k, u, i] for (u, _) in in_arcs[i]) ==
            gp.quicksum(x[k, i, v] for (_, v) in out_arcs[i])
            for k in K for i in I
        ),
        name="route_flow_customers"
    )

    # (4) depot degrees
    m.addConstrs(
        (gp.quicksum(x[k, j, v] for (_, v) in out_arcs[j]) == z[k, j] for k in K for j in J),
        name="depot_out"
    )
    m.addConstrs(
        (gp.quicksum(x[k, u, j] for (u, _) in in_arcs[j]) == z[k, j] for k in K for j in J),
        name="depot_in"
    )

    # (5) depot cap
    m.addConstrs((gp.quicksum(z[k, j] for k in K) <= int(Kmax[j]) for j in J), name="depot_cap")

    # (6) activation
    m.addConstrs((visit[k, i] <= gp.quicksum(z[k, j] for j in J) for k in K for i in I), name="activation")

    # (7) flow link
    m.addConstrs((f[k, u, v] <= U * x[k, u, v] for k in K for (u, v) in A), name="flow_link")

    # (8) customer flow balance
    m.addConstrs(
        (
            gp.quicksum(f[k, u, i] for (u, _) in in_arcs[i]) -
            gp.quicksum(f[k, i, v] for (_, v) in out_arcs[i])
            == q[i] * visit[k, i]
            for k in K for i in I
        ),
        name="flow_balance_customers"
    )

    # (9) depot as source (gated)
    total_demand_served = {k: gp.quicksum(q[i] * visit[k, i] for i in I) for k in K}
    for k in K:
        for j in J:
            net_out = gp.quicksum(f[k, j, v] for (_, v) in out_arcs[j]) - gp.quicksum(f[k, u, j] for (u, _) in in_arcs[j])
            m.addConstr(net_out >= total_demand_served[k] - M_demand * (1 - z[k, j]), name=f"src_lb[{k},{j}]")
            m.addConstr(net_out <= total_demand_served[k] + M_demand * (1 - z[k, j]), name=f"src_ub[{k},{j}]")

    # (10) tau upper bound
    m.addConstrs((tau[k, i] <= Q * visit[k, i] for k in K for i in I), name="tau_ub")

    # (11) tau propagation (customer->customer)
    for k in K:
        for (u, v) in A:
            if (u in Iset) and (v in Iset):
                m.addConstr(tau[k, v] >= tau[k, u] + d[(u, v)] - Q * (1 - x[k, u, v]),
                            name=f"tau_prop[{k},{u},{v}]")

    # (12) depot->customer init (gated)
    for k in K:
        for j in J:
            for i in I:
                if (j, i) in d:
                    m.addConstr(tau[k, i] >= d[(j, i)] - Q * (1 - x[k, j, i]) - M_tau * (1 - z[k, j]),
                                name=f"tau_init[{k},{j},{i}]")

    # (13) customer->depot return (gated)
    for k in K:
        for j in J:
            for i in I:
                if (i, j) in d:
                    m.addConstr(tau[k, i] + d[(i, j)] <= Q + M_tau * (1 - x[k, i, j]) + M_tau * (1 - z[k, j]),
                                name=f"tau_ret[{k},{i},{j}]")

    m.optimize()

    x_sol: Dict[Tuple[Any, Any, Any], int] = {}
    z_sol: Dict[Tuple[Any, Any], int] = {}
    if m.SolCount > 0:
        for k in K:
            for j in J:
                z_sol[(k, j)] = int(z[k, j].X > 0.5)
        for k in K:
            for (u, v) in A:
                if x[k, u, v].X > 0.5:
                    x_sol[(k, u, v)] = 1
    return m, x_sol, z_sol


def reconstruct_routes(I, J, K, x_sol, z_sol):
    routes = {}
    for k in K:
        depots = [j for j in J if z_sol.get((k, j), 0) == 1]
        if len(depots) != 1:
            continue
        depot = depots[0]
        succ = {}
        for (kk, u, v), one in x_sol.items():
            if kk == k and one == 1:
                succ[u] = v

        route = [depot]
        cur = depot
        guard = {depot}
        while True:
            if cur not in succ:
                break
            nxt = succ[cur]
            route.append(nxt)
            if nxt == depot:
                break
            if nxt in guard:
                break
            guard.add(nxt)
            cur = nxt
        routes[k] = route
    return routes


if __name__ == "__main__":
    # Example: parameterized instance
    I, J, K, P, A, q, U, Q, c, d, w, orig, Kmax = generate_random_instance(
        n_customers=10,
        n_depots=2,
        n_vehicles=2,
        seed=42,
        coord_box=(0, 50, 0, 50),
        demand_range=(1, 4),
        capacity_U=None,          # let generator choose
        endurance_Q=None,         # let generator choose based on geometry
        endurance_factor=4.0,
        endurance_margin=10.0,
        depot_capacity=None,      # default: n_vehicles
        ensure_feasible=True
    )

    model, x_sol, z_sol = build_and_solve_scf(
        I, J, K, P, A, q, U, Q, c, d, w, orig, Kmax,
        time_limit=30,
        mip_gap=0.0,
        verbose=True
    )

    print("\n==== Summary ====")
    print("Status:", model.Status, "(2=OPTIMAL, 9=TIME_LIMIT, 3=INFEASIBLE)")
    print("SolCount:", model.SolCount)
    if model.SolCount > 0:
        print("Obj:", model.ObjVal)
        print("BestBd:", model.ObjBound)
        print("Gap:", model.MIPGap)

    print("\nU =", U, " Q =", Q, " total_demand =", sum(q.values()))
    print("Depot caps:", Kmax)

    routes = reconstruct_routes(I, J, K, x_sol, z_sol)
    print("\n==== Routes ====")
    for k, r in routes.items():
        print(k, ":", " -> ".join(r))
