"""MDVRP-RL Branch-Cut-and-Price package.

公共 API:
- solve_instance: 单实例求解
- run_experiment: 批量实验
- run_excel_experiment: Excel 重复实验
"""

from .search.solver import run_excel_experiment, run_experiment, solve_instance

__all__ = ["solve_instance", "run_experiment", "run_excel_experiment"]
