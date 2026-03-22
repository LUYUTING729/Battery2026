#!/usr/bin/env bash
set -euo pipefail

# 固定使用项目虚拟环境解释器，避免调用系统 python 导致找不到 scipy。
PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Python not found at $PY"
  echo "Please run: bash examples/setup_highs_env.sh"
  exit 1
fi

INSTANCE_ID="${INSTANCE_ID:-stations_001}"
EXCEL_PATH="${EXCEL_PATH:-src/bpc/data/stations.xlsx}"
CONFIG_PATH="${CONFIG_PATH:-configs/default.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${INSTANCE_ID}}"

PYTHONPATH=src "$PY" -m bpc.cli.solve \
  --instance-id "$INSTANCE_ID" \
  --excel-path "$EXCEL_PATH" \
  --config "$CONFIG_PATH" \
  --rmp-solver highs \
  --output-dir "$OUTPUT_DIR"
