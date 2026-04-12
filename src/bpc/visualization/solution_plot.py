from __future__ import annotations

"""Solution route plotting utilities based on matplotlib.

This module reads the original Excel input to recover customer / depot /
vehicle-origin coordinates, then overlays the solved routes into PNG figures.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bpc.data.xlsx_loader import (
    _first_existing,
    _norm,
    _parse_number,
    _read_sheet_rows,
    _read_sheet_rows_by_name,
)


Point = Tuple[float, float]


@dataclass(frozen=True)
class RouteRecord:
    vehicle_id: str
    depot_id: str
    customer_seq: Tuple[str, ...]
    cost: float
    load: float
    dist: float


@dataclass
class SpatialBundle:
    customers: Dict[str, Point]
    depots: Dict[str, Point]
    vehicle_origins: Dict[str, Point]
    vehicle_to_origin: Dict[str, str]
    customer_labels: Dict[str, str]
    depot_labels: Dict[str, str]


def _resolve_sheet_rows(xlsx_path: str, sheet_index: int, sheet_name: str) -> Tuple[str, List[List[str]]]:
    if sheet_name:
        return _read_sheet_rows_by_name(xlsx_path, sheet_name)
    return _read_sheet_rows(xlsx_path, sheet_index)


def load_spatial_bundle_from_excel(
    xlsx_path: str,
    *,
    customer_sheet_index: int = 0,
    depot_sheet_index: int = 1,
    customer_sheet_name: str = "",
    depot_sheet_name: str = "",
    vehicle_sheet_name: str = "",
) -> SpatialBundle:
    """Load plotting coordinates from the original Excel workbook."""

    _, c_rows = _resolve_sheet_rows(xlsx_path, customer_sheet_index, customer_sheet_name)
    _, d_rows = _resolve_sheet_rows(xlsx_path, depot_sheet_index, depot_sheet_name)

    v_rows: List[List[str]] = []
    if vehicle_sheet_name:
        _, v_rows = _read_sheet_rows_by_name(xlsx_path, vehicle_sheet_name)

    c_header = {_norm(h): i for i, h in enumerate(c_rows[0]) if _norm(h)}
    d_header = {_norm(h): i for i, h in enumerate(d_rows[0]) if _norm(h)}
    v_header = {_norm(h): i for i, h in enumerate(v_rows[0]) if _norm(h)} if v_rows else {}

    c_name_i = _first_existing(c_header, ["id", "name", "客户", "客户节点", "换电站", "序号"])
    c_lon_i = _first_existing(c_header, ["lon", "lng", "longitude", "经度"])
    c_lat_i = _first_existing(c_header, ["lat", "latitude", "纬度"])

    d_name_i = _first_existing(d_header, ["id", "name", "换电站", "序号"])
    d_lon_i = _first_existing(d_header, ["lon", "lng", "longitude", "经度"])
    d_lat_i = _first_existing(d_header, ["lat", "latitude", "纬度"])

    v_name_i = _first_existing(v_header, ["id", "name", "车辆", "vehicle", "序号"]) if v_header else None
    v_lon_i = _first_existing(v_header, ["lon", "lng", "longitude", "经度"]) if v_header else None
    v_lat_i = _first_existing(v_header, ["lat", "latitude", "纬度"]) if v_header else None

    if c_lon_i is None or c_lat_i is None:
        raise ValueError("customer sheet missing lon/lat columns")
    if d_lon_i is None or d_lat_i is None:
        raise ValueError("depot sheet missing lon/lat columns")

    customers: Dict[str, Point] = {}
    customer_labels: Dict[str, str] = {}
    c_idx = 0
    for row in c_rows[1:]:
        if not any(str(x).strip() for x in row):
            continue
        c_idx += 1
        cid = f"c{c_idx}"
        lon = _parse_number(row[c_lon_i] if c_lon_i < len(row) else "", field="customer lon")
        lat = _parse_number(row[c_lat_i] if c_lat_i < len(row) else "", field="customer lat")
        customers[cid] = (lon, lat)
        raw_name = str(row[c_name_i]).strip() if c_name_i is not None and c_name_i < len(row) else cid
        customer_labels[cid] = raw_name or cid

    depots: Dict[str, Point] = {}
    depot_labels: Dict[str, str] = {}
    d_idx = 0
    for row in d_rows[1:]:
        if not any(str(x).strip() for x in row):
            continue
        d_idx += 1
        did = f"d{d_idx}"
        lon = _parse_number(row[d_lon_i] if d_lon_i < len(row) else "", field="depot lon")
        lat = _parse_number(row[d_lat_i] if d_lat_i < len(row) else "", field="depot lat")
        depots[did] = (lon, lat)
        raw_name = str(row[d_name_i]).strip() if d_name_i is not None and d_name_i < len(row) else did
        depot_labels[did] = raw_name or did

    vehicle_origins: Dict[str, Point] = {}
    vehicle_to_origin: Dict[str, str] = {}
    if v_rows and v_lon_i is not None and v_lat_i is not None:
        origin_idx = 0
        for i, row in enumerate(v_rows[1:], start=1):
            if not any(str(x).strip() for x in row):
                continue
            raw_vid = str(row[v_name_i]).strip() if v_name_i is not None and v_name_i < len(row) else str(i)
            vid = f"v{raw_vid}" if str(raw_vid).isdigit() else str(raw_vid)
            origin_idx += 1
            oid = f"o{origin_idx}"
            lon = _parse_number(row[v_lon_i] if v_lon_i < len(row) else "", field="vehicle lon")
            lat = _parse_number(row[v_lat_i] if v_lat_i < len(row) else "", field="vehicle lat")
            vehicle_origins[oid] = (lon, lat)
            vehicle_to_origin[vid] = oid

    return SpatialBundle(
        customers=customers,
        depots=depots,
        vehicle_origins=vehicle_origins,
        vehicle_to_origin=vehicle_to_origin,
        customer_labels=customer_labels,
        depot_labels=depot_labels,
    )


def load_solution_routes(solution_path: str) -> Tuple[str, List[RouteRecord]]:
    payload = json.loads(Path(solution_path).read_text(encoding="utf-8"))
    status = str(payload.get("status", "UNKNOWN"))
    routes = [
        RouteRecord(
            vehicle_id=str(row["vehicle_id"]),
            depot_id=str(row["depot_id"]),
            customer_seq=tuple(str(x) for x in row.get("customer_seq", [])),
            cost=float(row.get("cost", 0.0)),
            load=float(row.get("load", 0.0)),
            dist=float(row.get("dist", 0.0)),
        )
        for row in payload.get("routes", [])
    ]
    return status, routes


def infer_excel_path(solution_path: str) -> Path | None:
    solution_file = Path(solution_path).resolve()
    result_dir = solution_file.parent
    profile_path = result_dir / "excel_model_profile.json"
    if profile_path.exists():
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        raw = str(payload.get("xlsx_path", "")).strip()
        if raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if candidate.exists():
                return candidate
    return None


def infer_excel_context(solution_path: str) -> Dict[str, object]:
    solution_file = Path(solution_path).resolve()
    result_dir = solution_file.parent
    profile_path = result_dir / "excel_model_profile.json"
    context: Dict[str, object] = {
        "xlsx_path": "",
        "customer_sheet_name": "",
        "depot_sheet_name": "",
        "vehicle_sheet_name": "",
    }
    if not profile_path.exists():
        inferred = infer_excel_path(solution_path)
        if inferred is not None:
            context["xlsx_path"] = str(inferred)
        return context

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    raw = str(payload.get("xlsx_path", "")).strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.exists():
            context["xlsx_path"] = str(candidate)
    sheet_mapping = payload.get("sheet_mapping", {}) if isinstance(payload, dict) else {}
    if isinstance(sheet_mapping, dict):
        context["customer_sheet_name"] = str(sheet_mapping.get("customers_sheet", "") or "")
        context["depot_sheet_name"] = str(sheet_mapping.get("depots_sheet", "") or "")
        context["vehicle_sheet_name"] = str(sheet_mapping.get("vehicles_sheet", "") or "")
    return context


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        return plt, Line2D
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it first with: pip install matplotlib"
        ) from exc


def _collect_points(bundle: SpatialBundle) -> List[Point]:
    points = list(bundle.customers.values()) + list(bundle.depots.values()) + list(bundle.vehicle_origins.values())
    if not points:
        raise ValueError("no coordinates available for plotting")
    return points


def _axis_bounds(points: Sequence[Point], padding_ratio: float = 0.08) -> Tuple[float, float, float, float]:
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)
    lon_pad = lon_span * padding_ratio
    lat_pad = lat_span * padding_ratio
    return (min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad)


def _color_cycle() -> List[str]:
    return [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#ea580c",
        "#7c3aed",
        "#0891b2",
        "#c2410c",
        "#0f766e",
        "#be123c",
        "#4338ca",
    ]


def _route_points(route: RouteRecord, bundle: SpatialBundle) -> List[Point]:
    points = [bundle.depots[route.depot_id]]
    for cid in route.customer_seq:
        if cid not in bundle.customers:
            raise KeyError(f"customer {cid} missing in Excel input")
        points.append(bundle.customers[cid])
    points.append(bundle.depots[route.depot_id])
    return points


def _vehicle_sort_key(vehicle_id: str) -> Tuple[int, object]:
    digits = "".join(ch for ch in vehicle_id if ch.isdigit())
    if digits:
        return (0, int(digits))
    return (1, vehicle_id)


def _augment_missing_vehicle_origins(bundle: SpatialBundle, routes: Sequence[RouteRecord]) -> None:
    """Mirror solver fallback: missing vehicles start from cyclic depot origins."""

    if not bundle.depots:
        return

    route_vehicle_ids = sorted({route.vehicle_id for route in routes}, key=_vehicle_sort_key)
    missing = [vid for vid in route_vehicle_ids if vid not in bundle.vehicle_to_origin]
    if not missing:
        return

    depot_ids = list(bundle.depots.keys())
    start = len(bundle.vehicle_to_origin)
    for offset, vid in enumerate(missing):
        depot_id = depot_ids[(start + offset) % len(depot_ids)]
        origin_id = depot_id
        bundle.vehicle_to_origin[vid] = origin_id
        bundle.vehicle_origins[origin_id] = bundle.depots[depot_id]


def write_solution_plot(
    routes: Sequence[RouteRecord],
    bundle: SpatialBundle,
    out_path: str,
    *,
    title: str,
    subtitle: str = "",
    figsize: Tuple[float, float] = (16.0, 10.0),
    dpi: int = 180,
) -> None:
    plt, Line2D = _require_matplotlib()
    points = _collect_points(bundle)
    min_lon, max_lon, min_lat, max_lat = _axis_bounds(points)
    selected_depots = {route.depot_id for route in routes}
    colors = _color_cycle()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.set_title(title, fontsize=18, color="#0f172a", pad=18)
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color="#475569",
            ha="left",
            va="bottom",
        )

    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        route_pts = _route_points(route, bundle)
        xs = [pt[0] for pt in route_pts]
        ys = [pt[1] for pt in route_pts]
        ax.plot(xs, ys, color=color, linewidth=2.6, alpha=0.9, zorder=2)

        origin_id = bundle.vehicle_to_origin.get(route.vehicle_id, "")
        if origin_id and origin_id in bundle.vehicle_origins:
            origin_pt = bundle.vehicle_origins[origin_id]
            depot_pt = bundle.depots[route.depot_id]
            if abs(origin_pt[0] - depot_pt[0]) > 1e-9 or abs(origin_pt[1] - depot_pt[1]) > 1e-9:
                ax.plot(
                    [origin_pt[0], depot_pt[0]],
                    [origin_pt[1], depot_pt[1]],
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.65,
                    zorder=1,
                )
            ax.scatter(
                [origin_pt[0]],
                [origin_pt[1]],
                marker="^",
                s=80,
                color=color,
                edgecolors="#ffffff",
                linewidths=0.8,
                zorder=4,
            )
            ax.annotate(
                f"{route.vehicle_id} origin",
                xy=origin_pt,
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=color,
                zorder=5,
            )

        if len(route_pts) >= 2:
            first_customer = route_pts[1]
            ax.annotate(
                route.vehicle_id,
                xy=first_customer,
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=9,
                color=color,
                zorder=5,
            )

    customer_xs = [pt[0] for pt in bundle.customers.values()]
    customer_ys = [pt[1] for pt in bundle.customers.values()]
    if customer_xs:
        ax.scatter(
            customer_xs,
            customer_ys,
            s=26,
            marker="o",
            facecolors="#ffffff",
            edgecolors="#1e293b",
            linewidths=0.9,
            zorder=3,
        )
    for cid, pt in bundle.customers.items():
        ax.annotate(cid, xy=pt, xytext=(4, 4), textcoords="offset points", fontsize=7, color="#334155", zorder=4)

    for did, pt in bundle.depots.items():
        if did in selected_depots:
            ax.scatter(
                [pt[0]],
                [pt[1]],
                s=95,
                marker="s",
                color="#f59e0b",
                edgecolors="#7c2d12",
                linewidths=1.0,
                zorder=4,
            )
            ax.annotate(did, xy=pt, xytext=(6, -10), textcoords="offset points", fontsize=9, color="#92400e", zorder=5)
        else:
            ax.scatter(
                [pt[0]],
                [pt[1]],
                s=70,
                marker="s",
                color="#e2e8f0",
                edgecolors="#64748b",
                linewidths=0.9,
                zorder=3,
            )
            ax.annotate(did, xy=pt, xytext=(6, -10), textcoords="offset points", fontsize=8, color="#64748b", zorder=4)

    route_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=2.6, label=f"{route.vehicle_id}: {route.depot_id}")
        for idx, route in enumerate(routes)
    ]
    base_handles = [
        Line2D([0], [0], marker="o", markersize=6, markerfacecolor="#ffffff", markeredgecolor="#1e293b", lw=0, label="customer nodes"),
        Line2D([0], [0], marker="s", markersize=7, markerfacecolor="#f59e0b", markeredgecolor="#7c2d12", lw=0, label="selected depots"),
        Line2D([0], [0], marker="s", markersize=7, markerfacecolor="#e2e8f0", markeredgecolor="#64748b", lw=0, label="candidate depots"),
        Line2D([0], [0], marker="^", markersize=7, color="#334155", lw=1.5, linestyle="--", label="vehicle origin to depot"),
    ]
    ax.legend(
        handles=base_handles + route_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=8,
    )

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.6)
    ax.text(
        0.0,
        -0.08,
        f"lon=[{min_lon:.5f}, {max_lon:.5f}]  lat=[{min_lat:.5f}, {max_lat:.5f}]",
        transform=ax.transAxes,
        fontsize=9,
        color="#64748b",
        ha="left",
        va="top",
    )
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def export_solution_plots(
    *,
    solution_path: str,
    xlsx_path: str,
    out_dir: str,
    customer_sheet_index: int = 0,
    depot_sheet_index: int = 1,
    customer_sheet_name: str = "",
    depot_sheet_name: str = "",
    vehicle_sheet_name: str = "",
) -> Dict[str, object]:
    status, routes = load_solution_routes(solution_path)
    bundle = load_spatial_bundle_from_excel(
        xlsx_path,
        customer_sheet_index=customer_sheet_index,
        depot_sheet_index=depot_sheet_index,
        customer_sheet_name=customer_sheet_name,
        depot_sheet_name=depot_sheet_name,
        vehicle_sheet_name=vehicle_sheet_name,
    )
    _augment_missing_vehicle_origins(bundle, routes)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    overview_path = out_root / "solution_routes_overview.png"
    write_solution_plot(
        routes,
        bundle,
        str(overview_path),
        title="MDVRP Solution Routes",
        subtitle=f"status={status}  routes={len(routes)}  source={Path(solution_path).name}",
    )

    manifest = {
        "solution_path": str(Path(solution_path).resolve()),
        "xlsx_path": str(Path(xlsx_path).resolve()),
        "status": status,
        "route_count": len(routes),
        "selected_depots": sorted({route.depot_id for route in routes}),
        "overview_plot": str(overview_path.resolve()),
    }
    manifest_path = out_root / "solution_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest
