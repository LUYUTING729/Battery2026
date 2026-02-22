#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH=src python3 -m bpc.cli.solve \
  --instance-id inst_tiny \
  --db-path examples/minimal/instances.db \
  --config configs/default.json \
  --output-dir outputs/inst_tiny
