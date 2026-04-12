# Today Changes: Accelerate and Plot

Date: 2026-04-12

## Summary

This round focused on two practical goals:

1. Improve BPC solve behavior by adding a restricted-master integer heuristic so the solver can recover feasible incumbents from the current column pool instead of waiting for the LP solution to become integral by itself.
2. Add end-to-end plotting and a direct Gurobi solver path so solved routes can be visualized and compared against the BPC workflow more easily.

## BPC Changes

- Added a restricted master IP heuristic in the RMP layer.
- Added periodic heuristic calls inside the node CG loop.
- Wired heuristic hits into incumbent updates and runtime statistics.
- Exposed heuristic controls through `SolverConfig`.
- Kept the core BPC flow intact: the heuristic improves primal recovery without replacing branch-price-cut logic.

Main files:

- `src/bpc/rmp/master_problem.py`
- `src/bpc/search/solver.py`
- `src/bpc/core/types.py`
- `tests/test_rmp_smoke.py`

## Plotting Support

- Added a plotting CLI that reads `solution.json` and the original Excel workbook.
- Added visualization helpers for route plotting and manifest generation.
- Added a smoke test for plot export.

Main files:

- `src/bpc/cli/plot_solution.py`
- `src/bpc/visualization/__init__.py`
- `src/bpc/visualization/solution_plot.py`
- `tests/test_solution_plot.py`

## Direct Gurobi Solver

- Added a direct SCF-based Gurobi solver under `src/gurobi`.
- Added a CLI to solve directly from Excel/config in the same style as the BPC command.
- Added Gurobi log-file output to the solver output directory.
- Kept direct-solver outputs aligned with the existing project output pattern.

Main files:

- `src/gurobi/__init__.py`
- `src/gurobi/solver.py`
- `src/gurobi/cli/__init__.py`
- `src/gurobi/cli/solve.py`
- `tests/test_gurobi_direct_solve.py`

## Data and Loader Adjustments

- Improved Excel-loader behavior and related tests around vehicle handling and preprocessed bundle expectations.
- Updated config/documentation to reflect the current workflows.

Main files:

- `src/bpc/data/xlsx_loader.py`
- `tests/test_preprocessed_bundle.py`
- `configs/default.json`
- `README.md`

## Validation Performed

- `python3 -m compileall` on the touched source trees
- `python3 -m unittest tests.test_rmp_smoke -v`
- `python3 -m unittest tests.test_pricing_and_cuts -v`
- `python3 -m unittest tests.test_gurobi_direct_solve -v`
- `python3 -m unittest tests.test_solution_plot -v` where environment support exists

## Notes

- Temporary and local runtime artifacts such as `outputs/`, `.DS_Store`, and `.history/` were intentionally excluded from commit scope.
- A later experiment on depot-candidate filtering for initializer/pricing was reverted and is not part of this final state.
