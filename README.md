# MDVRP-RL BCP Solver (Python + Gurobi)

This repository now contains an engineering implementation skeleton for a full Branch-Cut-and-Price (BCP) solver matching the paper setup in `main.tex`.

Detailed architecture and algorithm explanation:
- `docs/ARCHITECTURE_ZH.md`

## Features
- Set-partitioning RMP with incremental column injection.
- RCSPP/ng-route pricing with dominance and 2-cycle elimination.
- Robust cut separators: RCC, Clique, SRI (heuristic separation).
- Arc-flow branching (`x_uv` closest to 0.5), best-bound/depth-first node policy.
- SQLite + CSV instance loader with strict validation.
- Structured outputs: `solution.json`, `routes.csv`, `trace.csv`, `metrics.csv`.
- Batch experiment API and CLI.

## Layout
- `src/bpc/data`: DB/CSV loader and validation.
- `src/bpc/core`: datatypes and route utilities.
- `src/bpc/rmp`: Gurobi RMP model.
- `src/bpc/pricing`: initial columns and ng-route pricing.
- `src/bpc/cuts`: cut definitions and separators.
- `src/bpc/branching`: branching rules.
- `src/bpc/search`: BCP solver orchestration.
- `src/bpc/cli`: command line entrypoints.
- `tests`: unittest suite.
- `configs/default.json`: default solver config.

## DB Schema
The loader expects table `instances`:

```sql
CREATE TABLE instances (
  instance_id TEXT PRIMARY KEY,
  csv_dir TEXT NOT NULL,
  capacity_u REAL NOT NULL,
  range_q REAL NOT NULL
);
```

Batch runner expects table `batch_instances`:

```sql
CREATE TABLE batch_instances (
  batch_id TEXT NOT NULL,
  instance_id TEXT NOT NULL
);
```

## CSV Files per instance folder
- `customers.csv`: `id,demand`
- `depots.csv`: `id,max_vehicles`
- `vehicles.csv`: `id,origin`
- `cost_matrix.csv`: `from,to,value`
- `dist_matrix.csv`: `from,to,value`

## Quick Start
Install Gurobi backend:

```bash
python3 -m pip install -U gurobipy
python3 -m pip install -e .
```

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Solve one instance:

```bash
PYTHONPATH=src .venv/bin/python -m bpc.cli.solve \
  --instance-id instA \
  --db-path instances.db \
  --rmp-solver gurobi \
  --config configs/default.json \
  --output-dir outputs/instA
```

Choose RMP backend in config/CLI:
- config: set `rmp_solver` to `gurobi`
- CLI override: `--rmp-solver gurobi`

Problem parameters can be configured in `configs/*.json` under `problem`, e.g.:

```json
{
  "problem": {
    "capacity_u": 200.0,
    "range_q": 500.0,
    "cost_per_km": 1.0,
    "vehicle_count": 10,
    "customer_sheet_name": "BSS_data",
    "depot_sheet_name": "Charging_data",
    "vehicle_sheet_name": "Vehicle_data"
  }
}
```

Run experiment batch:

```bash
PYTHONPATH=src python3 -m bpc.cli.experiment \
  --batch-id batch_001 \
  --db-path instances.db \
  --config configs/default.json
```

Run repeated experiment from one Excel file:

```bash
PYTHONPATH=src python3 -m bpc.cli.experiment_excel \
  --batch-id excel_batch_001 \
  --excel-path src/bpc/data/stations.xlsx \
  --config configs/default.json \
  --repeat 3 \
  --instance-prefix stations
```

Preprocess Excel once, then solve from cached instance data:

```bash
PYTHONPATH=src .venv/bin/python -m bpc.cli.preprocess_excel \
  --excel-path src/bpc/data/stations.xlsx \
  --instance-id stations_cached \
  --out-path outputs/stations_cached/preprocessed_instance.json
```

For workbooks with named sheets (e.g. `BSS_data/Charging_data/Vehicle_data`):

```bash
PYTHONPATH=src .venv/bin/python -m bpc.cli.preprocess_excel \
  --excel-path src/bpc/data/data2.xlsx \
  --instance-id data2_case \
  --out-path outputs/data2_case/preprocessed_instance.json \
  --customer-sheet-name BSS_data \
  --depot-sheet-name Charging_data \
  --vehicle-sheet-name Vehicle_data
```

```bash
PYTHONPATH=src .venv/bin/python -m bpc.cli.solve \
  --instance-id stations_cached \
  --preprocessed-path outputs/stations_cached/preprocessed_instance.json \
  --config configs/default.json \
  --rmp-solver gurobi \
  --output-dir outputs/stations_cached/solve_from_cache
```

## Minimal Example (SQLite + CSV)
- Example data folder: `examples/minimal/inst_tiny`
- Example DB: `examples/minimal/instances.db`
- One-click run script: `examples/minimal/run_one_click.sh`

Direct command:

```bash
PYTHONPATH=src python3 -m bpc.cli.solve \
  --instance-id inst_tiny \
  --db-path examples/minimal/instances.db \
  --config configs/default.json \
  --output-dir outputs/inst_tiny
```

Or:

```bash
bash examples/minimal/run_one_click.sh
```

Direct Gurobi SCF solve from `main.tex` model:

```bash
PYTHONPATH=src python3 -m gurobi.cli.solve \
  --instance-id inst_tiny \
  --db-path examples/minimal/instances.db \
  --config configs/default.json \
  --output-dir outputs/inst_tiny_gurobi
```

Solve directly from Excel with the same input style as `bpc.cli.solve`:

```bash
PYTHONPATH=src python3 -m gurobi.cli.solve \
  --instance-id data2_case_local \
  --excel-path src/bpc/data/data2.xlsx \
  --config configs/default.json \
  --output-dir outputs/data2_case/gurobi_direct
```
