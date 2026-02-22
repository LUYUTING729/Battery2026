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
Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Solve one instance:

```bash
PYTHONPATH=src python3 -m bpc.cli.solve \
  --instance-id instA \
  --db-path instances.db \
  --config configs/default.json \
  --output-dir outputs/instA
```

Run experiment batch:

```bash
PYTHONPATH=src python3 -m bpc.cli.experiment \
  --batch-id batch_001 \
  --db-path instances.db \
  --config configs/default.json
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
