#!/usr/bin/env bash
set -euo pipefail

# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 升级基础工具
python -m pip install -U pip setuptools wheel

# 安装本项目 + HiGHS(Scipy backend)
python -m pip install -e ".[highs]"

# 快速验证：是否可导入 scipy，并打印版本
python - <<'PY'
import scipy
print('scipy version:', scipy.__version__)
PY

echo "Use this interpreter to run solver:"
echo "  PYTHONPATH=src .venv/bin/python -m bpc.cli.solve --help"
echo "HiGHS backend environment is ready."
