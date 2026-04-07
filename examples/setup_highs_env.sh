#!/usr/bin/env bash
set -euo pipefail

# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 升级基础工具
python -m pip install -U pip setuptools wheel

# 安装本项目（Gurobi backend）
python -m pip install -e .

# 快速验证：是否可导入 gurobipy，并打印版本
python - <<'PY'
import gurobipy as gp
print('gurobi version:', gp.gurobi.version())
PY

echo "Use this interpreter to run solver:"
echo "  PYTHONPATH=src .venv/bin/python -m bpc.cli.solve --help"
echo "Gurobi backend environment is ready."
