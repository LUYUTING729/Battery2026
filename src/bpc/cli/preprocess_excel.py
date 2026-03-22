from __future__ import annotations

"""Excel 预处理 CLI：一次解析，多次求解复用。"""

import argparse
import json

from bpc.core.types import ProblemConfig, SolverConfig
from bpc.data.xlsx_loader import load_preprocessed_bundle, preprocess_excel_to_file


def _load_cfg_problem(path: str) -> ProblemConfig:
    cfg = SolverConfig()
    if not path:
        return cfg.problem
    data = json.loads(open(path, "r", encoding="utf-8").read())
    p = data.get("problem", {})
    if isinstance(p, dict):
        for k, v in p.items():
            if hasattr(cfg.problem, k):
                setattr(cfg.problem, k, v)
    return cfg.problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Excel instance into cached JSON bundle")
    parser.add_argument("--config", default="")
    parser.add_argument("--excel-path", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--customer-sheet-index", type=int, default=0)
    parser.add_argument("--depot-sheet-index", type=int, default=1)
    parser.add_argument("--customer-sheet-name", default="")
    parser.add_argument("--depot-sheet-name", default="")
    parser.add_argument("--vehicle-sheet-name", default="")
    parser.add_argument("--override-vehicle-count", type=int, default=0)
    parser.add_argument("--allow-infeasible-vehicle-count", action="store_true")
    parser.add_argument("--override-capacity-u", type=float, default=0.0)
    parser.add_argument("--override-range-q", type=float, default=0.0)
    parser.add_argument("--override-cost-per-km", type=float, default=0.0)
    args = parser.parse_args()
    p = _load_cfg_problem(args.config)

    preprocess_excel_to_file(
        xlsx_path=args.excel_path,
        out_path=args.out_path,
        instance_id=args.instance_id,
        customer_sheet_index=args.customer_sheet_index if args.customer_sheet_index != 0 else p.customer_sheet_index,
        depot_sheet_index=args.depot_sheet_index if args.depot_sheet_index != 1 else p.depot_sheet_index,
        customer_sheet_name=args.customer_sheet_name or p.customer_sheet_name,
        depot_sheet_name=args.depot_sheet_name or p.depot_sheet_name,
        vehicle_sheet_name=args.vehicle_sheet_name or p.vehicle_sheet_name,
        override_vehicle_count=args.override_vehicle_count if args.override_vehicle_count > 0 else p.vehicle_count,
        allow_infeasible_vehicle_count=args.allow_infeasible_vehicle_count or p.allow_infeasible_vehicle_count,
        override_capacity_u=args.override_capacity_u if args.override_capacity_u > 0 else p.capacity_u,
        override_range_q=args.override_range_q if args.override_range_q > 0 else p.range_q,
        override_cost_per_km=args.override_cost_per_km if args.override_cost_per_km > 0 else p.cost_per_km,
    )
    bundle = load_preprocessed_bundle(args.out_path)
    print(
        json.dumps(
            {
                "status": "ok",
                "out_path": args.out_path,
                "instance_id": bundle.instance.instance_id,
                "num_customers": len(bundle.instance.customers),
                "num_depots": len(bundle.instance.depots),
                "num_vehicles": len(bundle.instance.vehicles),
                "num_arcs": len(bundle.instance.arcs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
