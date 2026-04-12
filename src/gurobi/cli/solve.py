from __future__ import annotations

"""单实例直接 Gurobi 求解 CLI。"""

import argparse
import json

from bpc.core.types import SolverConfig
from gurobi.solver import solve_instance


def _load_cfg(path: str) -> SolverConfig:
    cfg = SolverConfig()
    if not path:
        return cfg
    data = json.loads(open(path, "r", encoding="utf-8").read())
    for k, v in data.items():
        if k == "problem" and isinstance(v, dict):
            for pk, pv in v.items():
                if hasattr(cfg.problem, pk):
                    setattr(cfg.problem, pk, pv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve one MDVRP-RL instance with direct Gurobi SCF model")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--db-path", default="instances.db")
    parser.add_argument("--csv-dir", default="")
    parser.add_argument("--excel-path", default="")
    parser.add_argument("--excel-profile-path", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    cfg.output_dir = args.output_dir
    result = solve_instance(
        instance_id=args.instance_id,
        cfg=cfg,
        db_path=args.db_path,
        csv_dir=args.csv_dir,
        excel_path=args.excel_path,
        excel_profile_path=args.excel_profile_path,
    )
    print(json.dumps({"status": result.status, "obj_primal": result.obj_primal, "gap": result.gap}, ensure_ascii=False))


if __name__ == "__main__":
    main()
