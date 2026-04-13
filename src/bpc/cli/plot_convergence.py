from __future__ import annotations

"""Generate convergence and performance plots for a BPC run directory."""

import argparse
import json
from pathlib import Path

from bpc.visualization.convergence_plot import export_convergence_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convergence and runtime diagnostics from an output run directory"
    )
    parser.add_argument("--run-dir", required=True, help="Run output directory, e.g. outputs/data2_case/shanghai")
    parser.add_argument("--output-dir", default="", help="Directory for exported figures and report")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    manifest = export_convergence_plots(str(run_dir), args.output_dir.strip())
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
