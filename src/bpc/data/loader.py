from __future__ import annotations

"""实例数据加载模块（SQLite + CSV）。

数据组织约定：
- SQLite `instances` 表提供实例索引与全局参数 (U, Q)
- CSV 提供客户、中心、车辆、成本矩阵、距离矩阵

加载流程：
1) 读 DB 元信息；
2) 读 CSV 并做字段/类型校验；
3) 构建 arcs（禁止 depot->depot）；
4) 构建 dispatch_cost；
5) 返回 InstanceData。
"""

import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple

from bpc.core.types import Arc, Customer, Depot, InstanceData, Vehicle


REQUIRED_CUSTOMER_COLS = {"id", "demand"}
REQUIRED_DEPOT_COLS = {"id", "max_vehicles"}
REQUIRED_VEHICLE_COLS = {"id", "origin"}
REQUIRED_MATRIX_COLS = {"from", "to", "value"}


def _read_csv_dict(path: Path) -> Iterable[dict]:
    """按 DictReader 流式读取 CSV。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for row in reader:
            yield row


def _require_columns(path: Path, actual, required):
    """校验 CSV 必要列。"""
    missing = set(required) - set(actual or [])
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")


def _load_matrix(path: Path) -> Dict[Arc, float]:
    """读取边矩阵文件 (from,to,value) -> {(u,v): value}。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _require_columns(path, reader.fieldnames, REQUIRED_MATRIX_COLS)
        matrix: Dict[Arc, float] = {}
        for idx, row in enumerate(reader, start=2):
            u = str(row["from"]).strip()
            v = str(row["to"]).strip()
            try:
                val = float(row["value"])
            except Exception as exc:
                raise ValueError(f"Invalid value at {path}:{idx}") from exc
            matrix[(u, v)] = val
    return matrix


def _load_meta(instance_id: str, db_path: Path) -> dict:
    """读取实例元信息（csv_dir, U, Q）。"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT instance_id, csv_dir, capacity_u, range_q
            FROM instances
            WHERE instance_id = ?
            """,
            (instance_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"instance_id not found in DB: {instance_id}")
        return dict(row)
    finally:
        conn.close()


def _build_dispatch_cost(
    vehicles: Dict[str, Vehicle],
    depots: Dict[str, Depot],
    matrix: Dict[Arc, float],
) -> Dict[Tuple[str, str], float]:
    """构建车辆从 origin 切换到 depot 的调度成本矩阵。"""
    dispatch: Dict[Tuple[str, str], float] = {}
    for vid, v in vehicles.items():
        for did in depots:
            key = (v.origin, did)
            if key not in matrix:
                raise ValueError(f"dispatch cost missing in cost_matrix for origin->depot: {key}")
            dispatch[(vid, did)] = matrix[key]
    return dispatch


def load_instance(instance_id: str, db_path: str, csv_dir: str = "") -> InstanceData:
    """加载并验证一个实例。

    关键验证：
    - 必需文件存在；
    - 列头完整；
    - 需求不超过车辆容量；
    - 中心容量非负；
    - 成本矩阵与距离矩阵边集可对齐；
    - 自动过滤 depot->depot 弧。
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"db_path not found: {db}")

    meta = _load_meta(instance_id, db)
    base = Path(csv_dir) if csv_dir else Path(meta["csv_dir"])
    if not base.is_absolute():
        base = db.parent / base

    customers_path = base / "customers.csv"
    depots_path = base / "depots.csv"
    vehicles_path = base / "vehicles.csv"
    cost_path = base / "cost_matrix.csv"
    dist_path = base / "dist_matrix.csv"

    for p in [customers_path, depots_path, vehicles_path, cost_path, dist_path]:
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    customers: Dict[str, Customer] = {}
    with customers_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _require_columns(customers_path, reader.fieldnames, REQUIRED_CUSTOMER_COLS)
        for idx, row in enumerate(reader, start=2):
            cid = str(row["id"]).strip()
            if not cid:
                raise ValueError(f"empty customer id at {customers_path}:{idx}")
            try:
                demand = float(row["demand"])
            except Exception as exc:
                raise ValueError(f"invalid demand at {customers_path}:{idx}") from exc
            customers[cid] = Customer(customer_id=cid, demand=demand)

    depots: Dict[str, Depot] = {}
    with depots_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _require_columns(depots_path, reader.fieldnames, REQUIRED_DEPOT_COLS)
        for idx, row in enumerate(reader, start=2):
            did = str(row["id"]).strip()
            try:
                max_vehicles = int(float(row["max_vehicles"]))
            except Exception as exc:
                raise ValueError(f"invalid max_vehicles at {depots_path}:{idx}") from exc
            if max_vehicles < 0:
                raise ValueError(f"negative depot capacity at {depots_path}:{idx}")
            depots[did] = Depot(depot_id=did, max_vehicles=max_vehicles)

    vehicles: Dict[str, Vehicle] = {}
    with vehicles_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _require_columns(vehicles_path, reader.fieldnames, REQUIRED_VEHICLE_COLS)
        for idx, row in enumerate(reader, start=2):
            vid = str(row["id"]).strip()
            origin = str(row["origin"]).strip()
            if not vid or not origin:
                raise ValueError(f"invalid vehicle row at {vehicles_path}:{idx}")
            vehicles[vid] = Vehicle(vehicle_id=vid, origin=origin)

    cost = _load_matrix(cost_path)
    dist = _load_matrix(dist_path)

    capacity_u = float(meta["capacity_u"])
    range_q = float(meta["range_q"])

    for cid, c in customers.items():
        if c.demand - capacity_u > 1e-9:
            raise ValueError(f"customer demand exceeds capacity: customer={cid}, demand={c.demand}, U={capacity_u}")

    demand = {cid: c.demand for cid, c in customers.items()}

    nodes = set(customers) | set(depots) | {v.origin for v in vehicles.values()}
    arcs: set[Arc] = set()
    for (u, v), _ in cost.items():
        if u == v:
            continue
        if u in depots and v in depots:
            continue
        if u in nodes and v in nodes and (u, v) in dist:
            arcs.add((u, v))

    if not arcs:
        raise ValueError("No valid arcs constructed after applying depot->depot prohibition")

    dispatch_cost = _build_dispatch_cost(vehicles, depots, cost)

    return InstanceData(
        instance_id=instance_id,
        customers=customers,
        depots=depots,
        vehicles=vehicles,
        demand=demand,
        capacity_u=capacity_u,
        range_q=range_q,
        cost=cost,
        dist=dist,
        dispatch_cost=dispatch_cost,
        arcs=arcs,
    )
