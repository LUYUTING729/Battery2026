from __future__ import annotations

"""Excel 批实验 CLI。"""

import argparse
import json

from bpc.search.solver import run_excel_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated experiments from one Excel instance")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--excel-path", default="")
    parser.add_argument("--preprocessed-path", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--instance-prefix", default="excel_inst")
    parser.add_argument("--rmp-solver", default="", choices=["", "auto", "gurobi", "highs"])
    args = parser.parse_args()
    if not args.excel_path and not args.preprocessed_path:
        raise ValueError("either --excel-path or --preprocessed-path is required")

    report = run_excel_experiment(
        batch_id=args.batch_id,
        cfg_path=args.config,
        excel_path=args.excel_path,
        preprocessed_path=args.preprocessed_path,
        repeat=args.repeat,
        instance_prefix=args.instance_prefix,
        rmp_solver_override=args.rmp_solver,
    )
    print(json.dumps(report.summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
