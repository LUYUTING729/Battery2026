#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash examples/run_excel_experiment.sh
# 或：
#   BATCH_ID=exp_001 REPEAT=3 bash examples/run_excel_experiment.sh

BATCH_ID="${BATCH_ID:-excel_batch_001}"
EXCEL_PATH="${EXCEL_PATH:-src/bpc/data/stations.xlsx}"
CONFIG_PATH="${CONFIG_PATH:-configs/default.json}"
REPEAT="${REPEAT:-1}"
INSTANCE_PREFIX="${INSTANCE_PREFIX:-stations}"

PYTHONPATH=src python3 -m bpc.cli.experiment_excel \
  --batch-id "${BATCH_ID}" \
  --excel-path "${EXCEL_PATH}" \
  --config "${CONFIG_PATH}" \
  --repeat "${REPEAT}" \
  --instance-prefix "${INSTANCE_PREFIX}"
