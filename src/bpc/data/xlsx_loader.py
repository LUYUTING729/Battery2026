from __future__ import annotations

"""Excel(.xlsx) 实例加载。

目标：
1) 读取 sheet1(客户节点) 与 sheet2(换电站节点) 的经纬度；
2) 计算任意两节点的直线距离(km)并生成弧成本；
3) 把 Excel 数据映射到 InstanceData，并补全模型所需但未显式给出的参数（虚拟方案）。
"""

import json
import math
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bpc.core.types import Arc, Customer, Depot, InstanceData, Vehicle

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _col_to_idx(col: str) -> int:
    acc = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            break
        acc = acc * 26 + (ord(ch) - ord("A") + 1)
    return max(0, acc - 1)


def _cell_ref_to_col(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref or "")
    return _col_to_idx(m.group(1)) if m else 0


def _parse_number(v: object, *, field: str) -> float:
    try:
        return float(str(v).strip())
    except Exception as exc:
        raise ValueError(f"Invalid numeric value for {field}: {v!r}") from exc


def _norm(h: object) -> str:
    return str(h or "").strip().lower()


def _first_existing(d: Dict[str, int], aliases: List[str]) -> int | None:
    for a in aliases:
        if a in d:
            return d[a]
    return None


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    # 球面直线距离（大圆距离），单位 km
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def _extract_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    name = "xl/sharedStrings.xml"
    if name not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(name))
    out: List[str] = []
    for si in root.findall(f"{{{_MAIN_NS}}}si"):
        texts = [t.text or "" for t in si.findall(f".//{{{_MAIN_NS}}}t")]
        out.append("".join(texts))
    return out


def _extract_sheet_targets(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall(f"{{{_PKG_REL_NS}}}Relationship")}

    sheets = []
    for s in wb.findall(f".//{{{_MAIN_NS}}}sheet"):
        name = s.attrib.get("name", "")
        rid = s.attrib.get(f"{{{_REL_NS}}}id", "")
        target = rid_to_target.get(rid, "")
        if target and not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((name, target))
    return sheets


def _cell_value(cell: ET.Element, shared: List[str]) -> str:
    ctype = cell.attrib.get("t")
    if ctype == "inlineStr":
        is_el = cell.find(f"{{{_MAIN_NS}}}is")
        if is_el is None:
            return ""
        return "".join(t.text or "" for t in is_el.findall(f".//{{{_MAIN_NS}}}t"))

    v = cell.find(f"{{{_MAIN_NS}}}v")
    if v is None:
        return ""
    raw = v.text or ""
    if ctype == "s":
        try:
            idx = int(raw)
            if 0 <= idx < len(shared):
                return shared[idx]
        except Exception:
            pass
    return raw


def _read_sheet_rows(xlsx_path: str, sheet_index: int) -> Tuple[str, List[List[str]]]:
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(f"xlsx file not found: {p}")

    with zipfile.ZipFile(p) as zf:
        sheet_targets = _extract_sheet_targets(zf)
        if sheet_index < 0 or sheet_index >= len(sheet_targets):
            raise ValueError(f"sheet_index out of range: {sheet_index}, total={len(sheet_targets)}")
        sheet_name, target = sheet_targets[sheet_index]
        if not target or target not in zf.namelist():
            raise ValueError(f"sheet XML target missing for index={sheet_index}: {target}")

        shared = _extract_shared_strings(zf)
        root = ET.fromstring(zf.read(target))

        rows: List[List[str]] = []
        for row in root.findall(f".//{{{_MAIN_NS}}}sheetData/{{{_MAIN_NS}}}row"):
            vals: Dict[int, str] = {}
            max_col = 0
            for cell in row.findall(f"{{{_MAIN_NS}}}c"):
                col = _cell_ref_to_col(cell.attrib.get("r", "A1"))
                vals[col] = _cell_value(cell, shared).strip()
                max_col = max(max_col, col)
            rows.append([vals.get(i, "") for i in range(max_col + 1)])

        return sheet_name, rows


def _read_sheet_rows_by_name(xlsx_path: str, sheet_name: str) -> Tuple[str, List[List[str]]]:
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(f"xlsx file not found: {p}")
    target_name = _norm(sheet_name)
    with zipfile.ZipFile(p) as zf:
        sheet_targets = _extract_sheet_targets(zf)
        idx = None
        for i, (name, _) in enumerate(sheet_targets):
            if _norm(name) == target_name:
                idx = i
                break
        if idx is None:
            available = [name for name, _ in sheet_targets]
            raise ValueError(f"sheet not found by name: {sheet_name}, available={available}")
    return _read_sheet_rows(xlsx_path, idx)


def _list_sheet_names(xlsx_path: str) -> List[str]:
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(f"xlsx file not found: {p}")
    with zipfile.ZipFile(p) as zf:
        return [name for name, _ in _extract_sheet_targets(zf)]


def _parse_other_data(xlsx_path: str, sheet_index: int = 2) -> Dict[str, float]:
    try:
        _, rows = _read_sheet_rows(xlsx_path, sheet_index)
    except Exception:
        return {}

    out: Dict[str, float] = {}
    for row in rows:
        if len(row) < 2:
            continue
        k = _norm(row[0])
        if not k:
            continue
        try:
            v = float(row[1])
        except Exception:
            continue
        out[k] = v
    return out


@dataclass
class ExcelInstanceBundle:
    instance: InstanceData
    model_profile: Dict[str, object]


def load_instance_bundle_from_excel(
    xlsx_path: str,
    instance_id: str = "excel_instance",
    customer_sheet_index: int = 0,
    depot_sheet_index: int = 1,
    customer_sheet_name: str = "",
    depot_sheet_name: str = "",
    vehicle_sheet_name: str = "",
    override_vehicle_count: int = 0,
    allow_infeasible_vehicle_count: bool = False,
    override_capacity_u: float = 0.0,
    override_range_q: float = 0.0,
    override_cost_per_km: float = 0.0,
) -> ExcelInstanceBundle:
    """从 Excel 构建 InstanceData，并返回参数映射/虚拟值方案。"""

    names = [_norm(x) for x in _list_sheet_names(xlsx_path)]
    if not customer_sheet_name:
        if "bss_data" in names:
            customer_sheet_name = "BSS_data"
        elif "bbs_data" in names:
            customer_sheet_name = "BBS_data"
    if not depot_sheet_name and "charging_data" in names:
        depot_sheet_name = "Charging_data"
    if not vehicle_sheet_name and "vehicle_data" in names:
        vehicle_sheet_name = "Vehicle_data"

    if customer_sheet_name:
        customer_sheet_real_name, c_rows = _read_sheet_rows_by_name(xlsx_path, customer_sheet_name)
    else:
        customer_sheet_real_name, c_rows = _read_sheet_rows(xlsx_path, customer_sheet_index)
    if depot_sheet_name:
        depot_sheet_real_name, d_rows = _read_sheet_rows_by_name(xlsx_path, depot_sheet_name)
    else:
        depot_sheet_real_name, d_rows = _read_sheet_rows(xlsx_path, depot_sheet_index)

    v_rows: List[List[str]] = []
    vehicle_sheet_real_name = ""
    if vehicle_sheet_name:
        vehicle_sheet_real_name, v_rows = _read_sheet_rows_by_name(xlsx_path, vehicle_sheet_name)

    if not c_rows or not d_rows:
        raise ValueError("empty sheet content")

    c_header = {_norm(h): i for i, h in enumerate(c_rows[0]) if _norm(h)}
    d_header = {_norm(h): i for i, h in enumerate(d_rows[0]) if _norm(h)}
    v_header = {_norm(h): i for i, h in enumerate(v_rows[0]) if _norm(h)} if v_rows else {}

    c_name_i = _first_existing(c_header, ["id", "name", "客户", "客户节点", "换电站", "序号"])
    c_lon_i = _first_existing(c_header, ["lon", "lng", "longitude", "经度"])
    c_lat_i = _first_existing(c_header, ["lat", "latitude", "纬度"])
    c_dem_i = _first_existing(c_header, ["demand", "demands", "需求", "需求量"])

    d_name_i = _first_existing(d_header, ["id", "name", "换电站", "序号"])
    d_lon_i = _first_existing(d_header, ["lon", "lng", "longitude", "经度"])
    d_lat_i = _first_existing(d_header, ["lat", "latitude", "纬度"])
    d_cap_i = _first_existing(d_header, ["max_vehicles", "capacity", "车辆上限", "最大车辆数"])

    v_name_i = _first_existing(v_header, ["id", "name", "车辆", "vehicle", "序号"]) if v_header else None
    v_lon_i = _first_existing(v_header, ["lon", "lng", "longitude", "经度"]) if v_header else None
    v_lat_i = _first_existing(v_header, ["lat", "latitude", "纬度"]) if v_header else None

    if c_lon_i is None or c_lat_i is None:
        raise ValueError(f"customer sheet missing lon/lat columns: {customer_sheet_real_name}")
    if d_lon_i is None or d_lat_i is None:
        raise ValueError(f"depot sheet missing lon/lat columns: {depot_sheet_real_name}")

    other_data = _parse_other_data(xlsx_path)

    capacity_u = float(other_data.get("u", 200.0))
    range_q = float(other_data.get("q", 500.0))
    cost_per_km = float(other_data.get("c", 1.0))
    if override_capacity_u > 0:
        capacity_u = float(override_capacity_u)
    if override_range_q > 0:
        range_q = float(override_range_q)
    if override_cost_per_km > 0:
        cost_per_km = float(override_cost_per_km)
    vehicle_count = int(round(other_data.get("k", 0)))

    customers: Dict[str, Customer] = {}
    customer_xy: Dict[str, Tuple[float, float]] = {}
    customer_name_map: Dict[str, str] = {}
    demand: Dict[str, float] = {}
    default_demand = 1.0

    c_idx = 0
    for row in c_rows[1:]:
        if not any(str(x).strip() for x in row):
            continue
        lon = _parse_number(row[c_lon_i] if c_lon_i < len(row) else "", field="customer lon")
        lat = _parse_number(row[c_lat_i] if c_lat_i < len(row) else "", field="customer lat")

        c_idx += 1
        cid = f"c{c_idx}"
        raw_name = str(row[c_name_i]).strip() if c_name_i is not None and c_name_i < len(row) else cid
        dem_raw = row[c_dem_i] if c_dem_i is not None and c_dem_i < len(row) else ""
        dem = default_demand if str(dem_raw).strip() == "" else _parse_number(dem_raw, field="customer demand")

        customers[cid] = Customer(customer_id=cid, demand=dem)
        customer_xy[cid] = (lon, lat)
        customer_name_map[cid] = raw_name or cid
        demand[cid] = dem

    depots: Dict[str, Depot] = {}
    depot_xy: Dict[str, Tuple[float, float]] = {}
    depot_name_map: Dict[str, str] = {}

    d_idx = 0
    for row in d_rows[1:]:
        if not any(str(x).strip() for x in row):
            continue
        lon = _parse_number(row[d_lon_i] if d_lon_i < len(row) else "", field="depot lon")
        lat = _parse_number(row[d_lat_i] if d_lat_i < len(row) else "", field="depot lat")

        d_idx += 1
        did = f"d{d_idx}"
        raw_name = str(row[d_name_i]).strip() if d_name_i is not None and d_name_i < len(row) else did
        depot_name_map[did] = raw_name or did
        depot_xy[did] = (lon, lat)

        # 若 Excel 未给中心出车上限，则采用虚拟分配：按 K 平均分配。
        if d_cap_i is not None and d_cap_i < len(row) and str(row[d_cap_i]).strip():
            max_vehicles = int(round(_parse_number(row[d_cap_i], field="depot max_vehicles")))
        else:
            max_vehicles = 0  # 稍后按 K 统一分配
        depots[did] = Depot(depot_id=did, max_vehicles=max(0, max_vehicles))

    if not customers:
        raise ValueError("no customer nodes parsed from sheet1")
    if not depots:
        raise ValueError("no depot nodes parsed from sheet2")

    if vehicle_count <= 0:
        vehicle_count = len(depots)
    if override_vehicle_count > 0:
        vehicle_count = int(override_vehicle_count)

    depot_ids = list(depots.keys())

    has_explicit_depot_capacity = any(dep.max_vehicles > 0 for dep in depots.values())

    vehicles: Dict[str, Vehicle] = {}
    vehicle_origin_xy: Dict[str, Tuple[float, float]] = {}
    explicit_vehicle_rows = 0
    vehicle_override_warning = ""
    if v_rows and v_lon_i is not None and v_lat_i is not None:
        parsed_vehicle_rows: List[Tuple[str, float, float]] = []
        for i, row in enumerate(v_rows[1:], start=1):
            if not any(str(x).strip() for x in row):
                continue
            raw_vid = str(row[v_name_i]).strip() if v_name_i is not None and v_name_i < len(row) else str(i)
            lon = _parse_number(row[v_lon_i] if v_lon_i < len(row) else "", field="vehicle lon")
            lat = _parse_number(row[v_lat_i] if v_lat_i < len(row) else "", field="vehicle lat")
            parsed_vehicle_rows.append((raw_vid, lon, lat))

        explicit_vehicle_rows = len(parsed_vehicle_rows)
        if override_vehicle_count > 0:
            parsed_vehicle_rows = parsed_vehicle_rows[:override_vehicle_count]
        for i, (raw_vid, lon, lat) in enumerate(parsed_vehicle_rows, start=1):
            vid = f"v{raw_vid}" if str(raw_vid).isdigit() else str(raw_vid)
            origin = f"o{i}"
            vehicles[vid] = Vehicle(vehicle_id=vid, origin=origin)
            vehicle_origin_xy[origin] = (lon, lat)

    if not vehicles:
        for i in range(vehicle_count):
            vid = f"v{i + 1}"
            origin = depot_ids[i % len(depot_ids)]
            vehicles[vid] = Vehicle(vehicle_id=vid, origin=origin)
            vehicle_origin_xy[origin] = depot_xy[origin]
    else:
        vehicle_count = len(vehicles)
        if override_vehicle_count > len(vehicles):
            # vehicle_data 行数不足时，按 depot 循环补充虚拟车辆。
            need = int(override_vehicle_count - len(vehicles))
            start = len(vehicles)
            for i in range(need):
                vid = f"v{start + i + 1}"
                origin = depot_ids[(start + i) % len(depot_ids)]
                vehicles[vid] = Vehicle(vehicle_id=vid, origin=origin)
                vehicle_origin_xy[origin] = depot_xy[origin]
            vehicle_count = len(vehicles)

    if explicit_vehicle_rows > 0 and override_vehicle_count > 0 and override_vehicle_count != explicit_vehicle_rows:
        vehicle_override_warning = (
            "override_vehicle_count differs from Vehicle_data row count: "
            f"override={override_vehicle_count}, explicit_rows={explicit_vehicle_rows}. "
            "Loader will keep explicit vehicle origins for parsed rows and synthesize or truncate vehicles to match the override."
        )
        warnings.warn(vehicle_override_warning, RuntimeWarning)

    if not has_explicit_depot_capacity:
        base = vehicle_count // len(depot_ids)
        rem = vehicle_count % len(depot_ids)
        depots = {
            did: Depot(depot_id=did, max_vehicles=(base + (1 if i < rem else 0)))
            for i, did in enumerate(depot_ids)
        }

    min_required_vehicles = int(math.ceil(sum(demand.values()) / max(1e-12, capacity_u)))
    if override_vehicle_count > 0 and (not allow_infeasible_vehicle_count) and vehicle_count < min_required_vehicles:
        raise ValueError(
            f"vehicle_count={vehicle_count} is infeasible by capacity lower bound: "
            f"ceil(total_demand/capacity_u)={min_required_vehicles}"
        )

    node_xy: Dict[str, Tuple[float, float]] = {}
    node_xy.update(customer_xy)
    node_xy.update(depot_xy)

    dist: Dict[Arc, float] = {}
    cost: Dict[Arc, float] = {}
    all_nodes = list(node_xy.keys())
    for u in all_nodes:
        lon1, lat1 = node_xy[u]
        for v in all_nodes:
            if u == v:
                continue
            lon2, lat2 = node_xy[v]
            dkm = _haversine_km(lon1, lat1, lon2, lat2)
            dist[(u, v)] = dkm
            cost[(u, v)] = dkm * cost_per_km

    arcs: set[Arc] = set()
    for u, v in dist.keys():
        if u in depots and v in depots:
            continue
        arcs.add((u, v))

    dispatch_cost: Dict[Tuple[str, str], float] = {}
    for vid, veh in vehicles.items():
        for did in depots:
            if veh.origin == did:
                dispatch_cost[(vid, did)] = 0.0
            else:
                o_lon, o_lat = vehicle_origin_xy[veh.origin]
                d_lon, d_lat = depot_xy[did]
                dispatch_cost[(vid, did)] = _haversine_km(o_lon, o_lat, d_lon, d_lat) * cost_per_km

    for cid, c in customers.items():
        if c.demand - capacity_u > 1e-9:
            raise ValueError(f"customer demand exceeds capacity_u: {cid} > U={capacity_u}")

    instance = InstanceData(
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

    profile = {
        "xlsx_path": str(Path(xlsx_path)),
        "sheet_mapping": {
            "customers_sheet": customer_sheet_real_name,
            "depots_sheet": depot_sheet_real_name,
            "vehicles_sheet": vehicle_sheet_real_name,
        },
        "model_parameter_mapping": {
            "customers": "sheet1 rows -> c1..cn",
            "depots": "sheet2 rows -> d1..dm",
            "demand": "sheet1.demands (missing -> 1.0)",
            "capacity_u": "sheet3.U (missing -> 200)",
            "range_q": "sheet3.Q (missing -> 500 km)",
            "cost_per_km": "sheet3.c (missing -> 1.0)",
            "vehicles": "vehicle_data rows -> vehicles (fallback: sheet3.K)",
            "dist[(u,v)]": "haversine(lon,lat), km",
            "cost[(u,v)]": "dist[(u,v)] * cost_per_km",
            "dispatch_cost[(v,d)]": "origin(depot-assigned) -> depot 直线成本",
            "arcs": "all directed pairs except depot->depot",
        },
        "virtual_value_scheme": {
            "default_customer_demand": default_demand,
            "default_capacity_u": 200.0,
            "default_range_q_km": 500.0,
            "default_cost_per_km": 1.0,
            "default_vehicle_count": len(depots),
            "default_depot_capacity_rule": "按 K 在 depots 间均分",
            "vehicle_origin_rule": "v_i origin = d_{(i mod m)+1}",
        },
        "resolved_values": {
            "capacity_u": capacity_u,
            "range_q": range_q,
            "cost_per_km": cost_per_km,
            "vehicle_count": vehicle_count,
            "explicit_vehicle_rows": explicit_vehicle_rows,
            "min_required_vehicles_by_capacity": min_required_vehicles,
            "num_customers": len(customers),
            "num_depots": len(depots),
            "num_arcs": len(arcs),
            "num_cost_entries": len(cost),
            "num_dist_entries": len(dist),
        },
        "id_mapping": {
            "customers": customer_name_map,
            "depots": depot_name_map,
        },
        "model_variables_correspondence": {
            "lambda_r": "RouteColumn（由 pricing 生成）",
            "a_i(r)": "route.a_i[cid]",
            "b_uv(r)": "(u,v) in route.arc_flags",
            "x_uv": "aggregated_arc_flow[(u,v)] = Σ_r lambda_r * b_uv(r)",
        },
        "missing_data_filled": {
            "depot_max_vehicles": not has_explicit_depot_capacity,
            "vehicle_origins": True,
            "customer_demand_defaults_used": any(abs(v - default_demand) < 1e-12 for v in demand.values()),
            "vehicle_count_overridden": override_vehicle_count > 0,
            "vehicle_sheet_used": bool(v_rows),
            "synthetic_vehicles_added": max(0, vehicle_count - explicit_vehicle_rows),
        },
        "warnings": [vehicle_override_warning] if vehicle_override_warning else [],
    }

    return ExcelInstanceBundle(instance=instance, model_profile=profile)


def load_instance_from_excel(
    xlsx_path: str,
    instance_id: str = "excel_instance",
    customer_sheet_index: int = 0,
    depot_sheet_index: int = 1,
    customer_sheet_name: str = "",
    depot_sheet_name: str = "",
    vehicle_sheet_name: str = "",
    override_vehicle_count: int = 0,
    allow_infeasible_vehicle_count: bool = False,
    override_capacity_u: float = 0.0,
    override_range_q: float = 0.0,
    override_cost_per_km: float = 0.0,
) -> InstanceData:
    """只返回 InstanceData 的便捷接口。"""
    return load_instance_bundle_from_excel(
        xlsx_path=xlsx_path,
        instance_id=instance_id,
        customer_sheet_index=customer_sheet_index,
        depot_sheet_index=depot_sheet_index,
        customer_sheet_name=customer_sheet_name,
        depot_sheet_name=depot_sheet_name,
        vehicle_sheet_name=vehicle_sheet_name,
        override_vehicle_count=override_vehicle_count,
        allow_infeasible_vehicle_count=allow_infeasible_vehicle_count,
        override_capacity_u=override_capacity_u,
        override_range_q=override_range_q,
        override_cost_per_km=override_cost_per_km,
    ).instance


def dump_excel_model_profile(bundle: ExcelInstanceBundle, out_path: str) -> None:
    Path(out_path).write_text(json.dumps(bundle.model_profile, ensure_ascii=False, indent=2), encoding="utf-8")


def _instance_to_payload(instance: InstanceData) -> Dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "capacity_u": instance.capacity_u,
        "range_q": instance.range_q,
        "customers": [
            {"id": cid, "demand": c.demand}
            for cid, c in sorted(instance.customers.items(), key=lambda x: x[0])
        ],
        "depots": [
            {"id": did, "max_vehicles": dep.max_vehicles}
            for did, dep in sorted(instance.depots.items(), key=lambda x: x[0])
        ],
        "vehicles": [
            {"id": vid, "origin": v.origin}
            for vid, v in sorted(instance.vehicles.items(), key=lambda x: x[0])
        ],
        "demand": [
            {"id": cid, "value": val}
            for cid, val in sorted(instance.demand.items(), key=lambda x: x[0])
        ],
        "cost": [
            {"from": u, "to": v, "value": val}
            for (u, v), val in sorted(instance.cost.items(), key=lambda x: (x[0][0], x[0][1]))
        ],
        "dist": [
            {"from": u, "to": v, "value": val}
            for (u, v), val in sorted(instance.dist.items(), key=lambda x: (x[0][0], x[0][1]))
        ],
        "dispatch_cost": [
            {"vehicle_id": vid, "depot_id": did, "value": val}
            for (vid, did), val in sorted(instance.dispatch_cost.items(), key=lambda x: (x[0][0], x[0][1]))
        ],
        "arcs": [
            {"from": u, "to": v}
            for (u, v) in sorted(instance.arcs, key=lambda x: (x[0], x[1]))
        ],
    }


def _payload_to_instance(payload: Dict[str, Any]) -> InstanceData:
    customers = {
        str(x["id"]): Customer(customer_id=str(x["id"]), demand=float(x["demand"]))
        for x in payload.get("customers", [])
    }
    depots = {
        str(x["id"]): Depot(depot_id=str(x["id"]), max_vehicles=int(x["max_vehicles"]))
        for x in payload.get("depots", [])
    }
    vehicles = {
        str(x["id"]): Vehicle(vehicle_id=str(x["id"]), origin=str(x["origin"]))
        for x in payload.get("vehicles", [])
    }
    demand = {str(x["id"]): float(x["value"]) for x in payload.get("demand", [])}
    cost = {(str(x["from"]), str(x["to"])): float(x["value"]) for x in payload.get("cost", [])}
    dist = {(str(x["from"]), str(x["to"])): float(x["value"]) for x in payload.get("dist", [])}
    dispatch_cost = {
        (str(x["vehicle_id"]), str(x["depot_id"])): float(x["value"])
        for x in payload.get("dispatch_cost", [])
    }
    arcs = {(str(x["from"]), str(x["to"])) for x in payload.get("arcs", [])}

    return InstanceData(
        instance_id=str(payload["instance_id"]),
        customers=customers,
        depots=depots,
        vehicles=vehicles,
        demand=demand,
        capacity_u=float(payload["capacity_u"]),
        range_q=float(payload["range_q"]),
        cost=cost,
        dist=dist,
        dispatch_cost=dispatch_cost,
        arcs=arcs,
    )


def dump_preprocessed_bundle(bundle: ExcelInstanceBundle, out_path: str) -> None:
    """将 Excel 解析结果一次性落盘，供 solve 阶段直接读取。"""
    payload = {
        "format": "bpc_preprocessed_instance_v1",
        "instance": _instance_to_payload(bundle.instance),
        "model_profile": bundle.model_profile,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_preprocessed_bundle(preprocessed_path: str) -> ExcelInstanceBundle:
    """读取预处理实例包。"""
    p = Path(preprocessed_path)
    if not p.exists():
        raise FileNotFoundError(f"preprocessed file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    fmt = payload.get("format", "")
    if fmt != "bpc_preprocessed_instance_v1":
        raise ValueError(f"unsupported preprocessed format: {fmt}")
    instance = _payload_to_instance(payload["instance"])
    profile = payload.get("model_profile", {})
    return ExcelInstanceBundle(instance=instance, model_profile=profile)


def preprocess_excel_to_file(
    xlsx_path: str,
    out_path: str,
    instance_id: str = "excel_instance",
    customer_sheet_index: int = 0,
    depot_sheet_index: int = 1,
    customer_sheet_name: str = "",
    depot_sheet_name: str = "",
    vehicle_sheet_name: str = "",
    override_vehicle_count: int = 0,
    allow_infeasible_vehicle_count: bool = False,
    override_capacity_u: float = 0.0,
    override_range_q: float = 0.0,
    override_cost_per_km: float = 0.0,
) -> None:
    """Excel -> 预处理文件。"""
    bundle = load_instance_bundle_from_excel(
        xlsx_path=xlsx_path,
        instance_id=instance_id,
        customer_sheet_index=customer_sheet_index,
        depot_sheet_index=depot_sheet_index,
        customer_sheet_name=customer_sheet_name,
        depot_sheet_name=depot_sheet_name,
        vehicle_sheet_name=vehicle_sheet_name,
        override_vehicle_count=override_vehicle_count,
        allow_infeasible_vehicle_count=allow_infeasible_vehicle_count,
        override_capacity_u=override_capacity_u,
        override_range_q=override_range_q,
        override_cost_per_km=override_cost_per_km,
    )
    dump_preprocessed_bundle(bundle, out_path=out_path)
