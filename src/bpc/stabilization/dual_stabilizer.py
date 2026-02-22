from __future__ import annotations

"""对偶稳定化模块。

目的：缓解列生成中的对偶振荡，减少重复定价与迭代波动。
支持：
- box: 把对偶限制在中心点邻域 [c-w, c+w]
- penalty: 当前对偶与中心点做凸组合
- hybrid: 前期 box，达到阈值后切换 penalty
"""

from typing import Dict

from bpc.core.types import SolverConfig


class DualStabilizer:
    """维护对偶中心并执行稳定化变换。"""

    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self.center: Dict[str, float] = {}

    def stabilize(self, duals: Dict[str, float], iteration: int) -> Dict[str, float]:
        """对输入对偶做稳定化处理。"""
        if not self.cfg.enable_stabilization:
            return duals

        method = self.cfg.stabilization.method
        if method == "hybrid" and iteration >= self.cfg.stabilization.hybrid_switch_iter:
            method = "penalty"

        if method == "box":
            return self._box(duals)
        if method == "penalty":
            return self._penalty(duals)
        return duals

    def _box(self, duals: Dict[str, float]) -> Dict[str, float]:
        """Box stabilization: 区间截断。"""
        width = self.cfg.stabilization.box_width
        stabilized = {}
        for k, v in duals.items():
            c = self.center.get(k, v)
            lo = c - width
            hi = c + width
            stabilized[k] = max(lo, min(hi, v))
        self.center = stabilized.copy()
        return stabilized

    def _penalty(self, duals: Dict[str, float]) -> Dict[str, float]:
        """Penalty stabilization: 与中心点做加权平均。"""
        w = self.cfg.stabilization.penalty_weight
        if w <= 0:
            self.center = duals.copy()
            return duals
        stabilized = {}
        for k, v in duals.items():
            c = self.center.get(k, v)
            stabilized[k] = (1.0 - w) * v + w * c
        self.center = stabilized.copy()
        return stabilized
