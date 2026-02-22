"""MDVRP-RL Branch-Cut-and-Price package.

公共 API:
- solve_instance: 单实例求解
- run_experiment: 批量实验
"""

from .search.solver import solve_instance, run_experiment

__all__ = ["solve_instance", "run_experiment"]
