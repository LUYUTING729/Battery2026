from __future__ import annotations

"""需求结构实验 CLI。

目标：
- 对客户需求生成三类分布（uniform / clustered / skewed）；
- 每类分布跑固定组数（默认 5 组）；
- 完整调用现有 solve_instance 流程并汇总统计结果。
"""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, List

from bpc.core.types import Customer, InstanceData, SolveResult, SolverConfig
from bpc.data.xlsx_loader import (
    ExcelInstanceBundle,
    dump_preprocessed_bundle,
    load_instance_bundle_from_excel,
    load_preprocessed_bundle,
)
from bpc.rmp.master_problem import GurobiUnavailableError
from bpc.search.solver import solve_instance


_DISTRIBUTIONS = ("uniform", "clustered", "skewed")


def _load_solver_config(cfg_path: str) -> SolverConfig:
    """读取求解配置（与 solver.py 中逻辑保持一致）。"""
    text = Path(cfg_path).read_text(encoding="utf-8")
    data = json.loads(text)
    cfg = SolverConfig()
    for k, v in data.items():
        if k == "stabilization" and isinstance(v, dict):
            for sk, sv in v.items():
                setattr(cfg.stabilization, sk, sv)
        elif k == "problem" and isinstance(v, dict):
            for pk, pv in v.items():
                if hasattr(cfg.problem, pk):
                    setattr(cfg.problem, pk, pv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _clone_with_demands(base: InstanceData, demand_map: Dict[str, float], instance_id: str) -> InstanceData:
    customers = {
        cid: Customer(customer_id=cid, demand=float(demand_map[cid]))
        for cid in sorted(base.customers.keys())
    }
    return InstanceData(
        instance_id=instance_id,
        customers=customers,
        depots=base.depots,
        vehicles=base.vehicles,
        demand={cid: float(demand_map[cid]) for cid in sorted(base.customers.keys())},
        capacity_u=base.capacity_u,
        range_q=base.range_q,
        cost=base.cost,
        dist=base.dist,
        dispatch_cost=base.dispatch_cost,
        arcs=base.arcs,
    )


def _normalize_to_total(raw: Dict[str, float], target_total: float, min_demand: float) -> Dict[str, float]:
    total_raw = sum(max(min_demand, v) for v in raw.values())
    if total_raw <= 1e-12:
        n = max(1, len(raw))
        avg = max(min_demand, target_total / n)
        return {k: avg for k in raw}
    scale = target_total / total_raw
    return {k: max(min_demand, v * scale) for k, v in raw.items()}


def _cap_demands(demand_map: Dict[str, float], max_demand: float, min_demand: float) -> Dict[str, float]:
    out = {}
    for k, v in demand_map.items():
        vv = max(min_demand, v)
        if max_demand > 0:
            vv = min(max_demand, vv)
        out[k] = vv
    return out


def _gen_uniform(base_demand: Dict[str, float], rng: random.Random, target_total: float, min_demand: float) -> Dict[str, float]:
    vals = list(base_demand.values())
    mean = sum(vals) / max(1, len(vals))
    low = max(min_demand, 0.75 * mean)
    high = max(low + 1e-9, 1.25 * mean)
    raw = {cid: rng.uniform(low, high) for cid in base_demand}
    return _normalize_to_total(raw, target_total=target_total, min_demand=min_demand)


def _gen_clustered(base_demand: Dict[str, float], rng: random.Random, target_total: float, min_demand: float) -> Dict[str, float]:
    cids = sorted(base_demand.keys())
    if not cids:
        return {}
    k = max(2, min(4, len(cids)))
    rng.shuffle(cids)
    clusters: List[List[str]] = [[] for _ in range(k)]
    for idx, cid in enumerate(cids):
        clusters[idx % k].append(cid)

    high_cluster_count = max(1, k // 2)
    high_idx = set(rng.sample(list(range(k)), high_cluster_count))

    mean = sum(base_demand.values()) / len(base_demand)
    raw: Dict[str, float] = {}
    for i, cluster in enumerate(clusters):
        if i in high_idx:
            low, high = 1.2 * mean, 1.8 * mean
        else:
            low, high = 0.4 * mean, 0.9 * mean
        low = max(min_demand, low)
        high = max(low + 1e-9, high)
        for cid in cluster:
            raw[cid] = rng.uniform(low, high)

    return _normalize_to_total(raw, target_total=target_total, min_demand=min_demand)


def _gen_skewed(base_demand: Dict[str, float], rng: random.Random, target_total: float, min_demand: float) -> Dict[str, float]:
    # 长尾：少数客户高需求，多数客户低需求
    cids = sorted(base_demand.keys())
    alpha = 2.2
    weights = sorted((rng.paretovariate(alpha) for _ in cids), reverse=True)
    rng.shuffle(cids)
    wsum = sum(weights)
    if wsum <= 1e-12:
        return {cid: target_total / max(1, len(cids)) for cid in cids}

    raw = {cid: max(min_demand, target_total * (w / wsum)) for cid, w in zip(cids, weights)}
    return _normalize_to_total(raw, target_total=target_total, min_demand=min_demand)


def _build_demand_map(
    distribution: str,
    base_demand: Dict[str, float],
    rng: random.Random,
    target_total: float,
    min_demand: float,
    max_demand: float,
) -> Dict[str, float]:
    if distribution == "uniform":
        dm = _gen_uniform(base_demand, rng=rng, target_total=target_total, min_demand=min_demand)
    elif distribution == "clustered":
        dm = _gen_clustered(base_demand, rng=rng, target_total=target_total, min_demand=min_demand)
    elif distribution == "skewed":
        dm = _gen_skewed(base_demand, rng=rng, target_total=target_total, min_demand=min_demand)
    else:
        raise ValueError(f"unsupported distribution: {distribution}")
    return _cap_demands(dm, max_demand=max_demand, min_demand=min_demand)


def _safe_mean(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def _safe_std(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    return statistics.pstdev(vals)


def _summarize_by_distribution(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for dist in _DISTRIBUTIONS:
        sub = [r for r in rows if r["distribution"] == dist]
        solved = [r for r in sub if r["status"] in {"OPTIMAL", "TIME_LIMIT"}]
        obj = [r["obj_primal"] for r in solved]
        gap = [r["gap"] for r in solved]
        runtime = [r["runtime_sec"] for r in solved]

        out[dist] = {
            "runs": float(len(sub)),
            "solved_like": float(len(solved)),
            "solve_rate": float(len(solved) / max(1, len(sub))),
            "avg_primal": _safe_mean(obj),
            "std_primal": _safe_std(obj),
            "avg_gap": _safe_mean(gap),
            "std_gap": _safe_std(gap),
            "avg_runtime_sec": _safe_mean(runtime),
            "std_runtime_sec": _safe_std(runtime),
        }
    return out


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run demand-structure experiment (uniform/clustered/skewed)")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--excel-path", default="")
    parser.add_argument("--preprocessed-path", default="")
    parser.add_argument("--instance-prefix", default="demand_inst")
    parser.add_argument("--groups-per-distribution", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-demand-scale", type=float, default=1.0)
    parser.add_argument("--min-demand", type=float, default=1.0)
    parser.add_argument(
        "--max-demand-ratio-to-capacity",
        type=float,
        default=0.9,
        help="max customer demand = ratio * capacity_u (<=0 means no cap)",
    )
    parser.add_argument("--rmp-solver", default="", choices=["", "gurobi"])
    args = parser.parse_args()
    if not args.excel_path and not args.preprocessed_path:
        raise ValueError("either --excel-path or --preprocessed-path is required")
    if args.groups_per_distribution <= 0:
        raise ValueError("groups-per-distribution must be positive")
    return args


def main() -> None:
    args = _parse_args()
    cfg = _load_solver_config(args.config)
    if args.rmp_solver:
        cfg.rmp_solver = args.rmp_solver

    if args.preprocessed_path:
        base_bundle = load_preprocessed_bundle(args.preprocessed_path)
        source_info = {"preprocessed_path": args.preprocessed_path, "excel_path": ""}
    else:
        base_bundle = load_instance_bundle_from_excel(
            xlsx_path=args.excel_path,
            instance_id=f"{args.instance_prefix}_base",
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
        source_info = {"preprocessed_path": "", "excel_path": args.excel_path}

    base = base_bundle.instance
    base_total = sum(base.demand.values())
    target_total = base_total * float(args.total_demand_scale)
    max_demand = (
        base.capacity_u * float(args.max_demand_ratio_to_capacity)
        if args.max_demand_ratio_to_capacity > 0
        else 0.0
    )

    report_root = Path(cfg.output_dir) / args.batch_id
    scenario_dir = report_root / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[dict] = []
    run_index = 0
    for dist_idx, distribution in enumerate(_DISTRIBUTIONS):
        for group_idx in range(1, args.groups_per_distribution + 1):
            run_index += 1
            demand_seed = args.seed + dist_idx * 1000 + group_idx
            solve_seed = cfg.random_seed + run_index
            rng = random.Random(demand_seed)

            demand_map = _build_demand_map(
                distribution=distribution,
                base_demand=base.demand,
                rng=rng,
                target_total=target_total,
                min_demand=float(args.min_demand),
                max_demand=max_demand,
            )

            iid = f"{args.instance_prefix}_{distribution}_{group_idx:03d}"
            bundle = ExcelInstanceBundle(
                instance=_clone_with_demands(base, demand_map=demand_map, instance_id=iid),
                model_profile={
                    **(base_bundle.model_profile if isinstance(base_bundle.model_profile, dict) else {}),
                    "demand_experiment": {
                        "distribution": distribution,
                        "group": group_idx,
                        "demand_seed": demand_seed,
                        "target_total": target_total,
                        "min_demand": float(args.min_demand),
                        "max_demand": max_demand,
                    },
                },
            )

            scenario_path = scenario_dir / f"{distribution}_g{group_idx:03d}.preprocessed.json"
            dump_preprocessed_bundle(bundle, str(scenario_path))

            cfg_i = SolverConfig(**{**cfg.__dict__})
            cfg_i.stabilization = cfg.stabilization
            cfg_i.random_seed = solve_seed
            cfg_i.output_dir = str(report_root / distribution / f"group_{group_idx:03d}")

            try:
                res = solve_instance(
                    instance_id=iid,
                    cfg=cfg_i,
                    preprocessed_path=str(scenario_path),
                )
            except GurobiUnavailableError as exc:
                res = SolveResult(
                    status="ERROR_GUROBI",
                    obj_primal=float("inf"),
                    obj_dual=float("inf"),
                    gap=float("inf"),
                    routes=[],
                    stats={"error": str(exc)},
                )

            run_rows.append(
                {
                    "distribution": distribution,
                    "group": group_idx,
                    "instance_id": iid,
                    "demand_seed": demand_seed,
                    "solve_seed": solve_seed,
                    "status": res.status,
                    "obj_primal": res.obj_primal,
                    "obj_dual": res.obj_dual,
                    "gap": res.gap,
                    "runtime_sec": float(res.stats.get("runtime_sec", float("inf"))),
                    "nodes_processed": float(res.stats.get("nodes_processed", float("inf"))),
                    "global_columns": float(res.stats.get("global_columns", float("inf"))),
                    "scenario_path": str(scenario_path),
                    "output_dir": cfg_i.output_dir,
                }
            )

    summary_by_distribution = _summarize_by_distribution(run_rows)
    summary_overall = {
        "runs": float(len(run_rows)),
        "solved_like": float(len([r for r in run_rows if r["status"] in {"OPTIMAL", "TIME_LIMIT"}])),
        "avg_primal": _safe_mean([r["obj_primal"] for r in run_rows if r["status"] in {"OPTIMAL", "TIME_LIMIT"}]),
        "avg_gap": _safe_mean([r["gap"] for r in run_rows if r["status"] in {"OPTIMAL", "TIME_LIMIT"}]),
    }

    report = {
        "batch_id": args.batch_id,
        "experiment": {
            "distributions": list(_DISTRIBUTIONS),
            "groups_per_distribution": args.groups_per_distribution,
            "seed": args.seed,
            "total_demand_scale": args.total_demand_scale,
            "min_demand": args.min_demand,
            "max_demand_ratio_to_capacity": args.max_demand_ratio_to_capacity,
            "base_total_demand": base_total,
            "target_total_demand": target_total,
            "base_instance_id": base.instance_id,
        },
        "source": source_info,
        "summary_overall": summary_overall,
        "summary_by_distribution": summary_by_distribution,
        "runs": run_rows,
    }

    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "demand_experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_csv(
        report_root / "demand_experiment_runs.csv",
        run_rows,
        fieldnames=[
            "distribution",
            "group",
            "instance_id",
            "demand_seed",
            "solve_seed",
            "status",
            "obj_primal",
            "obj_dual",
            "gap",
            "runtime_sec",
            "nodes_processed",
            "global_columns",
            "scenario_path",
            "output_dir",
        ],
    )

    summary_rows = []
    for dist in _DISTRIBUTIONS:
        s = summary_by_distribution[dist]
        summary_rows.append({"distribution": dist, **s})
    _write_csv(
        report_root / "demand_experiment_summary.csv",
        summary_rows,
        fieldnames=[
            "distribution",
            "runs",
            "solved_like",
            "solve_rate",
            "avg_primal",
            "std_primal",
            "avg_gap",
            "std_gap",
            "avg_runtime_sec",
            "std_runtime_sec",
        ],
    )

    print(
        json.dumps(
            {
                "batch_id": args.batch_id,
                "report_path": str(report_path),
                "summary_overall": summary_overall,
                "summary_by_distribution": summary_by_distribution,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
