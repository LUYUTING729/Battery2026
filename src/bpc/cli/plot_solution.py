from __future__ import annotations

"""Plot solved MDVRP routes from solution.json + original Excel input."""

import argparse
import json
from pathlib import Path

from bpc.visualization.solution_plot import export_solution_plots, infer_excel_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MDVRP solution routes into PNG files with matplotlib")
    parser.add_argument("--solution-path", required=True)
    parser.add_argument("--excel-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--customer-sheet-index", type=int, default=0)
    parser.add_argument("--depot-sheet-index", type=int, default=1)
    parser.add_argument("--customer-sheet-name", default="")
    parser.add_argument("--depot-sheet-name", default="")
    parser.add_argument("--vehicle-sheet-name", default="")
    args = parser.parse_args()

    solution_path = Path(args.solution_path).resolve()
    if not solution_path.exists():
        raise FileNotFoundError(f"solution file not found: {solution_path}")

    inferred_ctx = infer_excel_context(str(solution_path))

    excel_path = args.excel_path.strip() or str(inferred_ctx.get("xlsx_path", "") or "")
    if not excel_path:
        if not inferred_ctx.get("xlsx_path"):
            raise FileNotFoundError(
                "failed to infer excel path; please provide --excel-path explicitly"
            )

    out_dir = args.output_dir.strip() or str(solution_path.parent)
    manifest = export_solution_plots(
        solution_path=str(solution_path),
        xlsx_path=excel_path,
        out_dir=out_dir,
        customer_sheet_index=args.customer_sheet_index,
        depot_sheet_index=args.depot_sheet_index,
        customer_sheet_name=args.customer_sheet_name or str(inferred_ctx.get("customer_sheet_name", "") or ""),
        depot_sheet_name=args.depot_sheet_name or str(inferred_ctx.get("depot_sheet_name", "") or ""),
        vehicle_sheet_name=args.vehicle_sheet_name or str(inferred_ctx.get("vehicle_sheet_name", "") or ""),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
