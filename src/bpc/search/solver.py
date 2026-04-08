from __future__ import annotations

"""BCP 顶层求解编排模块。

该文件负责把所有子模块连接成可运行的端到端流程：
1) 读取实例并生成初始列；
2) 在节点内执行列生成（RMP + pricing + cuts）；
3) 基于分数解执行分支并维护搜索树；
4) 汇总指标并落盘输出。

对外主入口：
- solve_instance: 单实例求解
- run_experiment: 批量实验
"""

import csv
import heapq
import json
import math
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from bpc.branching.arc_branching import pick_branch_arc, split_node
from bpc.core.types import ExperimentReport, InstanceData, NodeState, SolveResult, SolverConfig
from bpc.cuts.separators import separate_cuts
from bpc.data.loader import load_instance
from bpc.data.xlsx_loader import (
    ExcelInstanceBundle,
    dump_excel_model_profile,
    load_instance_bundle_from_excel,
    load_preprocessed_bundle,
)
from bpc.pricing.initializer import generate_initial_columns
from bpc.pricing.ng_pricing import price_columns
from bpc.rmp.master_problem import GurobiUnavailableError, MasterProblem

TRACE_V1_FIELDS = [
    "node_id",
    "depth",
    "cg_iter",
    "lp_obj",
    "cuts_added",
    "new_columns",
    "active_columns",
    "iter_time_sec",
    "cuts_added_rcc",
    "cuts_added_clique",
    "cuts_added_sri",
    "is_integral",
    "art_violation_post_solve",
]
CG_ITERATION_V1_FIELDS = [
    "instance_id",
    "node_id",
    "depth",
    "cg_iter",
    "lp_obj_before_pricing",
    "cuts_added_this_iter",
    "cuts_added_rcc",
    "cuts_added_clique",
    "cuts_added_sri",
    "new_columns_added",
    "active_columns",
    "is_integral",
    "art_violation",
    "iter_runtime_sec",
]
RMP_COLUMNS_FIELDS = [
    "id",
    "vehicle_id",
    "depot_id",
    "cost",
    "len_path",
    "served_size",
    "served",
    "load",
    "dist",
    "creation_iteration",
    "creation_node_id",
    "pricing_mode",
]
RMP_SELECTED_COLUMNS_FIELDS = [
    "node_id",
    "id",
    "x_value",
    "cost",
    "served_size",
    "served",
    "load",
    "dist",
]
RMP_COVERAGE_DUALS_FIELDS = [
    "node_id",
    "customer",
    "dual_pi",
]
NODE_SUMMARY_V1_FIELDS = [
    "instance_id",
    "node_id",
    "depth",
    "is_root",
    "node_runtime_sec",
    "node_lb",
    "node_ub_if_integer",
    "node_gap",
    "node_is_integer",
    "cg_iters",
    "columns_start",
    "columns_added_total",
    "columns_end",
    "cuts_added_total",
    "cuts_added_rcc",
    "cuts_added_clique",
    "cuts_added_sri",
    "pricing_calls",
    "pricing_negative_found_calls",
    "rmp_solve_calls",
    "cut_sep_calls",
    "rmp_time_sec",
    "cut_sep_time_sec",
    "pricing_time_sec",
    "art_violation_final",
    "prune_reason",
    "fail_stage",
]
BRANCH_DECISION_V1_FIELDS = [
    "instance_id",
    "parent_node_id",
    "depth",
    "frac_arc_count",
    "selected_arc_u",
    "selected_arc_v",
    "selected_arc_flow",
    "selected_arc_dist_to_half",
    "left_child_id",
    "right_child_id",
]
INCUMBENT_UPDATES_V1_FIELDS = [
    "instance_id",
    "node_id",
    "depth",
    "update_reason",
    "old_ub",
    "new_ub",
    "improvement",
    "global_time_sec",
    "route_count",
]
RUN_SUMMARY_V1_FIELDS = [
    "instance_id",
    "status",
    "obj_primal",
    "obj_dual",
    "gap",
    "runtime_sec",
    "nodes_processed",
    "global_columns",
    "global_columns_final",
    "best_lb",
    "root_lb",
    "nodes_pruned_by_bound",
    "nodes_pruned_infeasible",
    "max_open_nodes",
    "max_depth",
    "cg_iters_total",
    "rmp_solve_calls_total",
    "cuts_separation_calls_total",
    "pricing_calls_total",
    "time_rmp_sec",
    "time_cut_sep_sec",
    "time_pricing_sec",
    "time_branch_sec",
    "time_other_sec",
    "time_to_first_feasible_sec",
    "time_to_best_incumbent_sec",
    "solve_seed",
    "fail_after_cuts_nodes",
    "fail_before_cuts_nodes",
    "fail_infeasible_by_columns_nodes",
]
ROOT_BOUND_EVENTS_FIELDS = [
    "time",
    "event_type",
    "bound_before",
    "bound_after",
    "delta_bound",
    "node_id",
]
ROOT_GAP_SERIES_FIELDS = [
    "time",
    "iteration",
    "gap",
]
NODE_DEPTHS_FIELDS = [
    "node_id",
    "depth",
    "status",
]
NODE_TIME_BREAKDOWN_FIELDS = [
    "node_id",
    "time_rmp",
    "time_pricing",
    "time_cut_separation",
    "time_subproblem",
    "total_time",
]


class _RealtimeRunWriter:
    """求解运行期的实时落盘器。"""

    def __init__(self, output_dir: Path, instance_id: str) -> None:
        self.output_dir = output_dir
        self.rmp_dir = output_dir / "rmp"
        self.branch_dir = output_dir / "branch"
        self.analysis_dir = output_dir / "analysis"
        self.rmp_dir.mkdir(parents=True, exist_ok=True)
        self.branch_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.instance_id = instance_id
        self._known_columns: set[str] = set()
        self._init_files()
        self.update_run_summary(
            {
                "instance_id": self.instance_id,
                "status": "RUNNING",
                "obj_primal": float("inf"),
                "obj_dual": float("inf"),
                "gap": float("inf"),
            }
        )

    def _init_csv(self, path: Path, fieldnames: List[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

    def _append_csv(self, path: Path, fieldnames: List[str], row: dict) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._init_csv(path, fieldnames)
        row = {k: row.get(k, "") for k in fieldnames}
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)

    def _init_files(self) -> None:
        self._init_csv(self.output_dir / "trace.csv", TRACE_V1_FIELDS)
        self._init_csv(self.output_dir / "bp_snapshots.csv", CG_ITERATION_V1_FIELDS)
        self._init_csv(self.output_dir / "routes.csv", ["vehicle_id", "depot_id", "customer_seq", "cost", "load", "dist"])
        self._init_csv(self.output_dir / "metrics.csv", ["metric", "value"])
        self._init_csv(self.analysis_dir / "node_summary.csv", NODE_SUMMARY_V1_FIELDS)
        self._init_csv(self.analysis_dir / "incumbent_updates.csv", INCUMBENT_UPDATES_V1_FIELDS)
        self._init_csv(self.rmp_dir / "columns.csv", RMP_COLUMNS_FIELDS)
        self._init_csv(self.rmp_dir / "selected_columns.csv", RMP_SELECTED_COLUMNS_FIELDS)
        self._init_csv(self.rmp_dir / "coverage_duals.csv", RMP_COVERAGE_DUALS_FIELDS)
        self._init_csv(self.branch_dir / "branch_decisions.csv", BRANCH_DECISION_V1_FIELDS)
        self._init_csv(self.analysis_dir / "root_bound_events.csv", ROOT_BOUND_EVENTS_FIELDS)
        self._init_csv(self.analysis_dir / "root_gap_series.csv", ROOT_GAP_SERIES_FIELDS)
        self._init_csv(self.analysis_dir / "node_depths.csv", NODE_DEPTHS_FIELDS)
        self._init_csv(self.analysis_dir / "node_time_breakdown.csv", NODE_TIME_BREAKDOWN_FIELDS)
        (self.output_dir / "solution.json").write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "obj_primal": float("inf"),
                    "obj_dual": float("inf"),
                    "gap": float("inf"),
                    "stats": {},
                    "routes": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.analysis_dir / "run_summary.json").write_text(
            json.dumps({"instance_name": self.instance_id, "status": "RUNNING"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "run_result.json").write_text(
            json.dumps({"instance_id": self.instance_id, "status": "RUNNING"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.rmp_dir / "solution.json").write_text(
            json.dumps({"objective": float("inf"), "picked": [], "duals": {}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_trace(self, row: dict) -> None:
        self._append_csv(self.output_dir / "trace.csv", TRACE_V1_FIELDS, row)

    def append_cg_iteration(self, row: dict) -> None:
        self._append_csv(self.output_dir / "bp_snapshots.csv", CG_ITERATION_V1_FIELDS, row)

    def append_node_summary(self, row: dict) -> None:
        self._append_csv(self.analysis_dir / "node_summary.csv", NODE_SUMMARY_V1_FIELDS, row)

    def append_branch_decision(self, row: dict) -> None:
        self._append_csv(self.branch_dir / "branch_decisions.csv", BRANCH_DECISION_V1_FIELDS, row)

    def append_incumbent_update(self, row: dict) -> None:
        self._append_csv(self.analysis_dir / "incumbent_updates.csv", INCUMBENT_UPDATES_V1_FIELDS, row)

    def update_run_summary(self, row: dict) -> None:
        (self.output_dir / "run_result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_root_bound_event(self, row: dict) -> None:
        self._append_csv(self.analysis_dir / "root_bound_events.csv", ROOT_BOUND_EVENTS_FIELDS, row)

    def append_root_gap_series(self, row: dict) -> None:
        self._append_csv(self.analysis_dir / "root_gap_series.csv", ROOT_GAP_SERIES_FIELDS, row)

    def append_node_depth(self, row: dict) -> None:
        self._append_csv(self.analysis_dir / "node_depths.csv", NODE_DEPTHS_FIELDS, row)

    def append_node_time_breakdown(self, row: dict) -> None:
        self._append_csv(self.analysis_dir / "node_time_breakdown.csv", NODE_TIME_BREAKDOWN_FIELDS, row)

    def write_analysis_run_summary(self, row: dict) -> None:
        path = self.analysis_dir / "run_summary.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_columns(self, rows: List[dict]) -> None:
        for row in rows:
            column_id = str(row.get("id", ""))
            if not column_id or column_id in self._known_columns:
                continue
            self._known_columns.add(column_id)
            self._append_csv(self.rmp_dir / "columns.csv", RMP_COLUMNS_FIELDS, row)

    def write_selected_columns(self, rows: List[dict]) -> None:
        self._init_csv(self.rmp_dir / "selected_columns.csv", RMP_SELECTED_COLUMNS_FIELDS)
        for row in rows:
            self._append_csv(self.rmp_dir / "selected_columns.csv", RMP_SELECTED_COLUMNS_FIELDS, row)

    def write_coverage_duals(self, rows: List[dict]) -> None:
        self._init_csv(self.rmp_dir / "coverage_duals.csv", RMP_COVERAGE_DUALS_FIELDS)
        for row in rows:
            self._append_csv(self.rmp_dir / "coverage_duals.csv", RMP_COVERAGE_DUALS_FIELDS, row)

    def write_rmp_solution(self, payload: dict) -> None:
        (self.rmp_dir / "solution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run_config(output_dir: Path, instance_id: str, cfg: SolverConfig) -> None:
    payload = {
        "instance_id": instance_id,
        "config": asdict(cfg),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_version": "",
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _route_to_column_row(route, creation_iteration: int, creation_node_id: int, pricing_mode: str) -> dict:
    return {
        "id": route.column_id,
        "vehicle_id": route.vehicle_id,
        "depot_id": route.depot_id,
        "cost": route.cost,
        "len_path": len(route.customer_seq),
        "served_size": sum(route.a_i.values()),
        "served": "|".join(route.customer_seq),
        "load": route.load,
        "dist": route.dist,
        "creation_iteration": creation_iteration,
        "creation_node_id": creation_node_id,
        "pricing_mode": pricing_mode,
    }


def _selected_route_rows(rmp: MasterProblem, node_id: int) -> List[dict]:
    if getattr(rmp.model, "SolCount", 0) <= 0:
        return []
    values = rmp._lambda_values or {cid: var.X for cid, var in rmp.vars.items()}
    rows: List[dict] = []
    for cid, val in values.items():
        if val <= 1e-9:
            continue
        route = rmp.columns[cid]
        rows.append(
            {
                "node_id": node_id,
                "id": cid,
                "x_value": val,
                "cost": route.cost,
                "served_size": sum(route.a_i.values()),
                "served": "|".join(route.customer_seq),
                "load": route.load,
                "dist": route.dist,
            }
        )
    return rows


def _coverage_dual_rows(duals, node_id: int) -> List[dict]:
    return [{"node_id": node_id, "customer": customer, "dual_pi": value} for customer, value in duals.cover_pi.items()]


def _rmp_solution_payload(info: float, selected_rows: List[dict], dual_rows: List[dict]) -> dict:
    return {
        "objective": info,
        "picked": [
            {
                "id": row["id"],
                "x": row["x_value"],
                "cost": row["cost"],
                "served": row["served"].split("|") if row["served"] else [],
                "load": row["load"],
                "dist": row["dist"],
            }
            for row in selected_rows
        ],
        "duals": {row["customer"]: row["dual_pi"] for row in dual_rows},
    }


def _emit_rmp_state(realtime_writer: _RealtimeRunWriter | None, rmp: MasterProblem, node_id: int, obj_value: float) -> None:
    """每次 RMP 求解后把当前解与对偶实时覆盖写出。"""
    if realtime_writer is None or not math.isfinite(obj_value) or getattr(rmp.model, "SolCount", 0) <= 0:
        return
    duals = rmp.extract_duals()
    selected_rows = _selected_route_rows(rmp, node_id)
    dual_rows = _coverage_dual_rows(duals, node_id)
    realtime_writer.write_selected_columns(selected_rows)
    realtime_writer.write_coverage_duals(dual_rows)
    realtime_writer.write_rmp_solution(_rmp_solution_payload(obj_value, selected_rows, dual_rows))


def _safe_gap(ub: float, lb: float) -> float:
    """计算相对 gap。

    公式: gap = (UB - LB) / |UB|。
    若 UB 无穷或接近 0，返回 inf，避免数值异常。
    """
    if math.isinf(ub) or abs(ub) < 1e-9:
        return float("inf")
    return max(0.0, (ub - lb) / abs(ub))


def _mkdir(path: str) -> Path:
    """创建输出目录（不存在则递归创建）。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _node_priority(strategy: str, lb: float, depth: int) -> Tuple[float, int]:
    """节点优先级编码。

    - best_bound: 按下界最小优先；
    - depth_first: 深度优先（depth 越大优先级越高）。
    """
    if strategy == "depth_first":
        return (float(-depth), depth)
    return (lb, depth)


def _rounds(cfg: SolverConfig, at_root: bool) -> int:
    """返回当前节点的 cut 分离轮数。"""
    return cfg.cut_rounds_root if at_root else cfg.cut_rounds_node


def _enforce_node_feasible_columns(columns, node: NodeState):
    """过滤与分支约束冲突的列（当前仅处理 banned arcs）。"""
    out = []
    for c in columns:
        if any(a in node.branch_rule.banned_arcs for a in c.arc_flags):
            continue
        out.append(c)
    return out


def _write_trace(output_dir: Path, trace_rows: List[dict]) -> None:
    """写迭代轨迹，用于复现实验与性能诊断。"""
    path = output_dir / "trace.csv"
    if not trace_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = TRACE_V1_FIELDS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace_rows)


def _write_monitoring_csv(output_dir: Path, filename: str, rows: List[dict], fieldnames: List[str] | None = None) -> None:
    """写 monitoring 子目录 CSV。"""
    mon_dir = output_dir / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)
    path = mon_dir / filename
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_routes(output_dir: Path, result: SolveResult) -> None:
    """写最终路线解，便于人工检查和后处理。"""
    path = output_dir / "routes.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["vehicle_id", "depot_id", "customer_seq", "cost", "load", "dist"],
        )
        writer.writeheader()
        for r in result.routes:
            writer.writerow(
                {
                    "vehicle_id": r.vehicle_id,
                    "depot_id": r.depot_id,
                    "customer_seq": "-".join(r.customer_seq),
                    "cost": r.cost,
                    "load": r.load,
                    "dist": r.dist,
                }
            )


def _write_metrics(output_dir: Path, result: SolveResult) -> None:
    """写核心指标（运行时间、节点数、全局列数等）。"""
    path = output_dir / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in result.stats.items():
            writer.writerow({"metric": k, "value": v})


def _write_solution_json(output_dir: Path, result: SolveResult) -> None:
    """写结构化 JSON 解文件，便于程序消费。"""
    path = output_dir / "solution.json"
    payload = {
        "status": result.status,
        "obj_primal": result.obj_primal,
        "obj_dual": result.obj_dual,
        "gap": result.gap,
        "stats": result.stats,
        "routes": [
            {
                "vehicle_id": r.vehicle_id,
                "depot_id": r.depot_id,
                "customer_seq": list(r.customer_seq),
                "cost": r.cost,
                "load": r.load,
                "dist": r.dist,
            }
            for r in result.routes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _solve_node(
    instance: InstanceData,
    node: NodeState,
    cfg: SolverConfig,
    global_columns,
    at_root: bool,
    trace_rows: List[dict],
    cg_rows: List[dict],
    realtime_writer: _RealtimeRunWriter | None = None,
    run_start_time: float | None = None,
    incumbent_value: float = float("inf"),
) -> Tuple[float, bool, List, Dict]:
    """求解单个 BCP 树节点。

    调用逻辑：
    1) 建立该节点 RMP，并注入可行初始列；
    2) 循环执行:
       - 解 RMP LP；
       - 分离并加入 cuts；
       - 提取对偶并运行 pricing 生成负化简成本列；
    3) 无新列后收敛，返回节点下界、整数性、候选路线与辅助信息。
    """
    t_node0 = time.time()
    rmp = MasterProblem(instance, node, relax=True, solver_backend=cfg.rmp_solver)

    seed_cols = _enforce_node_feasible_columns(global_columns, node)
    rmp.add_columns(seed_cols)
    if realtime_writer is not None:
        realtime_writer.append_columns([_route_to_column_row(route, 0, node.node_id, "seed") for route in seed_cols])
    columns_start = len(rmp.vars)

    cuts_added_total = 0
    cuts_added_rcc = 0
    cuts_added_clique = 0
    cuts_added_sri = 0
    columns_added_total = 0
    cg_iter = 0
    rmp_solve_calls = 0
    cut_sep_calls = 0
    pricing_calls = 0
    pricing_negative_found_calls = 0
    rmp_time_sec = 0.0
    cut_sep_time_sec = 0.0
    pricing_time_sec = 0.0
    best_root_lp_seen = float("inf")
    while cg_iter < cfg.max_cg_iters:
        t_iter0 = time.time()
        cg_iter += 1
        # Step A: 先解当前 RMP，得到对偶和分数解。
        t_solve = time.time()
        info = rmp.solve()
        rmp_time_sec += time.time() - t_solve
        rmp_solve_calls += 1
        _emit_rmp_state(realtime_writer, rmp, node.node_id, info.obj_val)
        if not math.isfinite(info.obj_val):
            return float("inf"), False, [], {
                "cg_iters": cg_iter,
                "cuts_added": cuts_added_total,
                "new_cols": 0,
                "active_columns": len(rmp.vars),
                "fail_stage": "solve_before_cuts",
                "columns_start": columns_start,
                "columns_added_total": columns_added_total,
                "cuts_added_rcc": cuts_added_rcc,
                "cuts_added_clique": cuts_added_clique,
                "cuts_added_sri": cuts_added_sri,
                "rmp_solve_calls": rmp_solve_calls,
                "cut_sep_calls": cut_sep_calls,
                "pricing_calls": pricing_calls,
                "pricing_negative_found_calls": pricing_negative_found_calls,
                "rmp_time_sec": rmp_time_sec,
                "cut_sep_time_sec": cut_sep_time_sec,
                "pricing_time_sec": pricing_time_sec,
                "node_runtime_sec": time.time() - t_node0,
                "is_integral_final": 0,
                "art_violation_final": float("inf"),
            }

        rounds = _rounds(cfg, at_root)
        cuts_added_this_iter = 0
        iter_cuts_rcc = 0
        iter_cuts_clique = 0
        iter_cuts_sri = 0
        for cut_round in range(rounds):
            # Step B: 用当前分数解做分离，若发现 violated cuts 则加入并重解。
            t_cut = time.time()
            active = rmp.active_routes()
            cuts = separate_cuts(instance, active, cfg, at_root=at_root, round_id=(cg_iter * 10 + cut_round))
            cut_sep_time_sec += time.time() - t_cut
            cut_sep_calls += 1
            if not cuts:
                continue
            cuts_added = rmp.add_cuts(cuts)
            cuts_added_total += cuts_added
            cuts_added_this_iter += cuts_added
            for cut in cuts:
                if cut.cut_type == "rcc":
                    cuts_added_rcc += 1
                    iter_cuts_rcc += 1
                elif cut.cut_type == "clique":
                    cuts_added_clique += 1
                    iter_cuts_clique += 1
                elif cut.cut_type == "sri":
                    cuts_added_sri += 1
                    iter_cuts_sri += 1
            if cuts_added:
                t_resolve = time.time()
                info = rmp.solve()
                rmp_time_sec += time.time() - t_resolve
                rmp_solve_calls += 1
                _emit_rmp_state(realtime_writer, rmp, node.node_id, info.obj_val)
                if not math.isfinite(info.obj_val):
                    trace_row = {
                        "node_id": node.node_id,
                        "depth": node.depth,
                        "cg_iter": cg_iter,
                        "lp_obj": info.obj_val,
                        "cuts_added": cuts_added_this_iter,
                        "new_columns": 0,
                        "active_columns": len(rmp.vars),
                        "iter_time_sec": time.time() - t_iter0,
                        "art_violation_post_solve": rmp.artificial_violation(),
                    }
                    trace_rows.append(trace_row)
                    if realtime_writer is not None:
                        realtime_writer.append_trace(trace_row)
                    cg_row = {
                        "instance_id": instance.instance_id,
                        "node_id": node.node_id,
                        "depth": node.depth,
                        "cg_iter": cg_iter,
                        "lp_obj_before_pricing": info.obj_val,
                        "cuts_added_this_iter": cuts_added_this_iter,
                        "cuts_added_rcc": iter_cuts_rcc,
                        "cuts_added_clique": iter_cuts_clique,
                        "cuts_added_sri": iter_cuts_sri,
                        "new_columns_added": 0,
                        "active_columns": len(rmp.vars),
                        "is_integral": int(rmp.is_integral()),
                        "art_violation": rmp.artificial_violation(),
                        "iter_runtime_sec": time.time() - t_iter0,
                    }
                    cg_rows.append(cg_row)
                    if realtime_writer is not None:
                        realtime_writer.append_cg_iteration(cg_row)
                        if at_root and run_start_time is not None:
                            if info.obj_val < best_root_lp_seen:
                                before = best_root_lp_seen
                                best_root_lp_seen = info.obj_val
                                if math.isfinite(before):
                                    realtime_writer.append_root_bound_event(
                                        {
                                            "time": time.time() - run_start_time,
                                            "event_type": "cg_lp_improve",
                                            "bound_before": before,
                                            "bound_after": info.obj_val,
                                            "delta_bound": info.obj_val - before,
                                            "node_id": node.node_id,
                                        }
                                    )
                            realtime_writer.append_root_gap_series(
                                {
                                    "time": time.time() - run_start_time,
                                    "iteration": cg_iter,
                                    "gap": _safe_gap(incumbent_value, info.obj_val),
                                }
                            )
                    return float("inf"), False, [], {
                        "cg_iters": cg_iter,
                        "cuts_added": cuts_added_total,
                        "new_cols": 0,
                        "active_columns": len(rmp.vars),
                        "fail_stage": "after_cuts",
                        "columns_start": columns_start,
                        "columns_added_total": columns_added_total,
                        "cuts_added_rcc": cuts_added_rcc,
                        "cuts_added_clique": cuts_added_clique,
                        "cuts_added_sri": cuts_added_sri,
                        "rmp_solve_calls": rmp_solve_calls,
                        "cut_sep_calls": cut_sep_calls,
                        "pricing_calls": pricing_calls,
                        "pricing_negative_found_calls": pricing_negative_found_calls,
                        "rmp_time_sec": rmp_time_sec,
                        "cut_sep_time_sec": cut_sep_time_sec,
                        "pricing_time_sec": pricing_time_sec,
                        "node_runtime_sec": time.time() - t_node0,
                        "is_integral_final": 0,
                        "art_violation_final": float("inf"),
                    }

        # Step C: pricing 子问题基于对偶寻找负化简成本列，增强 RMP。
        duals = rmp.extract_duals()
        t_pricing = time.time()
        new_columns = price_columns(instance, node, duals, cfg)
        pricing_time_sec += time.time() - t_pricing
        pricing_calls += 1
        if new_columns:
            pricing_negative_found_calls += 1
        new_columns = [c for c in new_columns if c.column_id not in rmp.vars]
        if len(new_columns) > cfg.max_new_columns_per_iter:
            new_columns = new_columns[: cfg.max_new_columns_per_iter]
        new_added = rmp.add_columns(new_columns)
        columns_added_total += new_added
        global_columns.extend([c for c in new_columns if c.column_id not in {x.column_id for x in global_columns}])
        if realtime_writer is not None:
            realtime_writer.append_columns(
                [_route_to_column_row(route, cg_iter, node.node_id, "pricing") for route in new_columns[:new_added]]
            )

        iter_art_violation = rmp.artificial_violation()
        iter_integral = int(rmp.is_integral())
        iter_time_sec = time.time() - t_iter0

        trace_row = {
            "node_id": node.node_id,
            "depth": node.depth,
            "cg_iter": cg_iter,
            "lp_obj": info.obj_val,
            "cuts_added": cuts_added_this_iter,
            "new_columns": new_added,
            "active_columns": len(rmp.vars),
            "iter_time_sec": iter_time_sec,
            "cuts_added_rcc": iter_cuts_rcc,
            "cuts_added_clique": iter_cuts_clique,
            "cuts_added_sri": iter_cuts_sri,
            "is_integral": iter_integral,
            "art_violation_post_solve": iter_art_violation,
        }
        trace_rows.append(trace_row)
        if realtime_writer is not None:
            realtime_writer.append_trace(trace_row)
        cg_row = {
            "instance_id": instance.instance_id,
            "node_id": node.node_id,
            "depth": node.depth,
            "cg_iter": cg_iter,
            "lp_obj_before_pricing": info.obj_val,
            "cuts_added_this_iter": cuts_added_this_iter,
            "cuts_added_rcc": iter_cuts_rcc,
            "cuts_added_clique": iter_cuts_clique,
            "cuts_added_sri": iter_cuts_sri,
            "new_columns_added": new_added,
            "active_columns": len(rmp.vars),
            "is_integral": iter_integral,
            "art_violation": iter_art_violation,
            "iter_runtime_sec": iter_time_sec,
        }
        cg_rows.append(cg_row)
        if realtime_writer is not None:
            realtime_writer.append_cg_iteration(cg_row)
            if at_root and run_start_time is not None:
                if info.obj_val < best_root_lp_seen:
                    before = best_root_lp_seen
                    best_root_lp_seen = info.obj_val
                    if math.isfinite(before):
                        realtime_writer.append_root_bound_event(
                            {
                                "time": time.time() - run_start_time,
                                "event_type": "cg_lp_improve",
                                "bound_before": before,
                                "bound_after": info.obj_val,
                                "delta_bound": info.obj_val - before,
                                "node_id": node.node_id,
                            }
                        )
                realtime_writer.append_root_gap_series(
                    {
                        "time": time.time() - run_start_time,
                        "iteration": cg_iter,
                        "gap": _safe_gap(incumbent_value, info.obj_val),
                    }
                )

        if new_added == 0:
            # 无新列 -> 节点 LP 在当前列空间内稳定。
            break

    # 节点终态：用最终 RMP 判断整数性并提取候选路线。
    t_final = time.time()
    final = rmp.solve()
    rmp_time_sec += time.time() - t_final
    rmp_solve_calls += 1
    _emit_rmp_state(realtime_writer, rmp, node.node_id, final.obj_val)
    lp_obj = final.obj_val
    is_integer = rmp.is_integral()
    selected = rmp.selected_routes()
    arc_flow = rmp.aggregated_arc_flow()
    art_violation_final = rmp.artificial_violation()
    node_stats = {
        "cg_iters": cg_iter,
        "cuts_added": cuts_added_total,
        "active_columns": len(rmp.vars),
        "fail_stage": "",
        "columns_start": columns_start,
        "columns_added_total": columns_added_total,
        "cuts_added_rcc": cuts_added_rcc,
        "cuts_added_clique": cuts_added_clique,
        "cuts_added_sri": cuts_added_sri,
        "rmp_solve_calls": rmp_solve_calls,
        "cut_sep_calls": cut_sep_calls,
        "pricing_calls": pricing_calls,
        "pricing_negative_found_calls": pricing_negative_found_calls,
        "rmp_time_sec": rmp_time_sec,
        "cut_sep_time_sec": cut_sep_time_sec,
        "pricing_time_sec": pricing_time_sec,
        "node_runtime_sec": time.time() - t_node0,
        "is_integral_final": int(is_integer),
        "art_violation_final": art_violation_final,
    }
    return lp_obj, is_integer, selected, {"arc_flow": arc_flow, **node_stats}


def solve_instance(
    instance_id: str,
    cfg: SolverConfig,
    db_path: str = "instances.db",
    csv_dir: str = "",
    excel_path: str = "",
    preprocessed_path: str = "",
    excel_profile_path: str = "",
) -> SolveResult:
    """单实例求解入口。

    该函数实现完整 BCP 主循环：
    - 初始化根节点；
    - 反复弹出节点求解；
    - 根据分数弧流执行二叉分支；
    - 用上下界做剪枝；
    - 达到停止条件后输出最优/当前最优。
    """
    t0 = time.time()
    bundle: ExcelInstanceBundle | None = None
    if preprocessed_path:
        bundle = load_preprocessed_bundle(preprocessed_path=preprocessed_path)
        instance = bundle.instance
    elif excel_path:
        bundle = load_instance_bundle_from_excel(
            xlsx_path=excel_path,
            instance_id=instance_id,
            customer_sheet_index=cfg.problem.customer_sheet_index,
            depot_sheet_index=cfg.problem.depot_sheet_index,
            customer_sheet_name=cfg.problem.customer_sheet_name,
            depot_sheet_name=cfg.problem.depot_sheet_name,
            vehicle_sheet_name=cfg.problem.vehicle_sheet_name,
            override_vehicle_count=cfg.problem.vehicle_count,
            allow_infeasible_vehicle_count=cfg.problem.allow_infeasible_vehicle_count,
            override_capacity_u=cfg.problem.capacity_u,
            override_range_q=cfg.problem.range_q,
            override_cost_per_km=cfg.problem.cost_per_km,
        )
        instance = bundle.instance
    else:
        instance = load_instance(instance_id=instance_id, db_path=db_path, csv_dir=csv_dir)

    outdir = _mkdir(cfg.output_dir)
    realtime_writer = _RealtimeRunWriter(output_dir=outdir, instance_id=instance.instance_id)
    _write_run_config(outdir, instance.instance_id, cfg)

    global_columns = generate_initial_columns(instance)

    root = NodeState(node_id=0, depth=0)
    pq = []
    next_node_id = 1

    incumbent = float("inf")
    incumbent_routes = []
    best_lb = float("inf")

    trace_rows: List[dict] = []
    cg_rows: List[dict] = []
    node_summary_rows: List[dict] = []
    branch_rows: List[dict] = []
    incumbent_rows: List[dict] = []
    heapq.heappush(pq, (_node_priority(cfg.branch_strategy, 0.0, 0), root.node_id, root))

    nodes_processed = 0
    nodes_pruned_by_bound = 0
    nodes_pruned_infeasible = 0
    max_open_nodes = 1
    max_depth = 0
    root_lb = float("inf")
    cg_iters_total = 0
    rmp_solve_calls_total = 0
    cut_sep_calls_total = 0
    pricing_calls_total = 0
    time_rmp_sec_total = 0.0
    time_cut_sep_sec_total = 0.0
    time_pricing_sec_total = 0.0
    time_branch_sec_total = 0.0
    first_feasible_ts = float("inf")
    best_incumbent_ts = float("inf")
    fail_stage_counts: Dict[str, int] = {}

    def _emit_running_summary() -> None:
        cur_gap = _safe_gap(incumbent, best_lb) if incumbent < float("inf") and math.isfinite(best_lb) else float("inf")
        realtime_writer.update_run_summary(
            {
                "instance_id": instance.instance_id,
                "status": "RUNNING",
                "obj_primal": incumbent if incumbent < float("inf") else float("inf"),
                "obj_dual": best_lb if math.isfinite(best_lb) else float("inf"),
                "gap": cur_gap,
                "runtime_sec": time.time() - t0,
                "nodes_processed": float(nodes_processed),
                "global_columns": float(len(global_columns)),
                "global_columns_final": float(len(global_columns)),
                "best_lb": best_lb if math.isfinite(best_lb) else float("inf"),
                "root_lb": root_lb if math.isfinite(root_lb) else float("inf"),
                "nodes_pruned_by_bound": float(nodes_pruned_by_bound),
                "nodes_pruned_infeasible": float(nodes_pruned_infeasible),
                "max_open_nodes": float(max_open_nodes),
                "max_depth": float(max_depth),
                "cg_iters_total": float(cg_iters_total),
                "rmp_solve_calls_total": float(rmp_solve_calls_total),
                "cuts_separation_calls_total": float(cut_sep_calls_total),
                "pricing_calls_total": float(pricing_calls_total),
                "time_rmp_sec": time_rmp_sec_total,
                "time_cut_sep_sec": time_cut_sep_sec_total,
                "time_pricing_sec": time_pricing_sec_total,
                "time_branch_sec": time_branch_sec_total,
                "time_to_first_feasible_sec": first_feasible_ts if math.isfinite(first_feasible_ts) else float("inf"),
                "time_to_best_incumbent_sec": best_incumbent_ts if math.isfinite(best_incumbent_ts) else float("inf"),
                "solve_seed": float(cfg.random_seed),
                "fail_after_cuts_nodes": float(fail_stage_counts.get("after_cuts", 0)),
                "fail_before_cuts_nodes": float(fail_stage_counts.get("solve_before_cuts", 0)),
                "fail_infeasible_by_columns_nodes": float(fail_stage_counts.get("infeasible_by_columns", 0)),
            }
        )
    while pq and nodes_processed < cfg.max_nodes:
        max_open_nodes = max(max_open_nodes, len(pq))
        if time.time() - t0 > cfg.time_limit_s:
            break

        # 取下一个待处理节点（best-bound 或 depth-first）。
        _, _, node = heapq.heappop(pq)
        at_root = node.node_id == 0
        max_depth = max(max_depth, int(node.depth))
        node_start = time.time()

        # 节点内部执行 “CG + cuts + pricing”。
        lb, is_integer, routes, node_aux = _solve_node(
            instance,
            node,
            cfg,
            global_columns,
            at_root,
            trace_rows,
            cg_rows,
            realtime_writer=realtime_writer,
            run_start_time=t0,
            incumbent_value=incumbent,
        )
        nodes_processed += 1
        node_runtime_sec = time.time() - node_start
        cg_iters_total += int(node_aux.get("cg_iters", 0))
        rmp_solve_calls_total += int(node_aux.get("rmp_solve_calls", 0))
        cut_sep_calls_total += int(node_aux.get("cut_sep_calls", 0))
        pricing_calls_total += int(node_aux.get("pricing_calls", 0))
        time_rmp_sec_total += float(node_aux.get("rmp_time_sec", 0.0))
        time_cut_sep_sec_total += float(node_aux.get("cut_sep_time_sec", 0.0))
        time_pricing_sec_total += float(node_aux.get("pricing_time_sec", 0.0))

        prune_reason = ""
        if not math.isfinite(lb):
            nodes_pruned_infeasible += 1
            prune_reason = "infeasible"
            fs = str(node_aux.get("fail_stage", "")).strip()
            if fs:
                fail_stage_counts[fs] = fail_stage_counts.get(fs, 0) + 1
            if node_aux.get("fail_stage"):
                trace_row = {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "cg_iter": -1,
                    "lp_obj": lb,
                    "cuts_added": node_aux.get("cuts_added", 0),
                    "new_columns": node_aux.get("new_cols", 0),
                    "active_columns": node_aux.get("active_columns", 0),
                }
                trace_rows.append(trace_row)
                realtime_writer.append_trace(trace_row)
            node_row = {
                "instance_id": instance.instance_id,
                "node_id": node.node_id,
                "depth": node.depth,
                "is_root": int(at_root),
                "node_runtime_sec": node_runtime_sec,
                "node_lb": lb,
                "node_ub_if_integer": float("inf"),
                "node_gap": float("inf"),
                "node_is_integer": 0,
                "cg_iters": node_aux.get("cg_iters", 0),
                "columns_start": node_aux.get("columns_start", 0),
                "columns_added_total": node_aux.get("columns_added_total", 0),
                "columns_end": node_aux.get("active_columns", 0),
                "cuts_added_total": node_aux.get("cuts_added", 0),
                "cuts_added_rcc": node_aux.get("cuts_added_rcc", 0),
                "cuts_added_clique": node_aux.get("cuts_added_clique", 0),
                "cuts_added_sri": node_aux.get("cuts_added_sri", 0),
                "pricing_calls": node_aux.get("pricing_calls", 0),
                "pricing_negative_found_calls": node_aux.get("pricing_negative_found_calls", 0),
                "rmp_solve_calls": node_aux.get("rmp_solve_calls", 0),
                "cut_sep_calls": node_aux.get("cut_sep_calls", 0),
                "rmp_time_sec": node_aux.get("rmp_time_sec", 0.0),
                "cut_sep_time_sec": node_aux.get("cut_sep_time_sec", 0.0),
                "pricing_time_sec": node_aux.get("pricing_time_sec", 0.0),
                "art_violation_final": node_aux.get("art_violation_final", float("inf")),
                "prune_reason": prune_reason,
                "fail_stage": fs,
            }
            node_summary_rows.append(node_row)
            realtime_writer.append_node_summary(node_row)
            realtime_writer.append_node_depth({"node_id": node.node_id, "depth": node.depth, "status": prune_reason})
            realtime_writer.append_node_time_breakdown(
                {
                    "node_id": node.node_id,
                    "time_rmp": node_aux.get("rmp_time_sec", 0.0),
                    "time_pricing": node_aux.get("pricing_time_sec", 0.0),
                    "time_cut_separation": node_aux.get("cut_sep_time_sec", 0.0),
                    "time_subproblem": 0.0,
                    "total_time": node_runtime_sec,
                }
            )
            _emit_running_summary()
            continue

        if lb < best_lb:
            best_lb = lb
        if at_root:
            root_lb = lb

        if is_integer:
            # 节点 LP 已整数，可形成可行解并尝试更新 incumbent。
            obj = sum(r.cost for r in routes)
            if obj < incumbent:
                old_ub = incumbent
                incumbent = obj
                incumbent_routes = routes
                elapsed = time.time() - t0
                if first_feasible_ts == float("inf"):
                    first_feasible_ts = elapsed
                best_incumbent_ts = elapsed
                incumbent_row = {
                    "instance_id": instance.instance_id,
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "update_reason": "integer_node",
                    "old_ub": old_ub,
                    "new_ub": incumbent,
                    "improvement": (old_ub - incumbent) if math.isfinite(old_ub) else float("inf"),
                    "global_time_sec": elapsed,
                    "route_count": len(routes),
                }
                incumbent_rows.append(incumbent_row)
                realtime_writer.append_incumbent_update(incumbent_row)
            node_row = {
                "instance_id": instance.instance_id,
                "node_id": node.node_id,
                "depth": node.depth,
                "is_root": int(at_root),
                "node_runtime_sec": node_runtime_sec,
                "node_lb": lb,
                "node_ub_if_integer": obj,
                "node_gap": _safe_gap(obj, lb),
                "node_is_integer": 1,
                "cg_iters": node_aux.get("cg_iters", 0),
                "columns_start": node_aux.get("columns_start", 0),
                "columns_added_total": node_aux.get("columns_added_total", 0),
                "columns_end": node_aux.get("active_columns", 0),
                "cuts_added_total": node_aux.get("cuts_added", 0),
                "cuts_added_rcc": node_aux.get("cuts_added_rcc", 0),
                "cuts_added_clique": node_aux.get("cuts_added_clique", 0),
                "cuts_added_sri": node_aux.get("cuts_added_sri", 0),
                "pricing_calls": node_aux.get("pricing_calls", 0),
                "pricing_negative_found_calls": node_aux.get("pricing_negative_found_calls", 0),
                "rmp_solve_calls": node_aux.get("rmp_solve_calls", 0),
                "cut_sep_calls": node_aux.get("cut_sep_calls", 0),
                "rmp_time_sec": node_aux.get("rmp_time_sec", 0.0),
                "cut_sep_time_sec": node_aux.get("cut_sep_time_sec", 0.0),
                "pricing_time_sec": node_aux.get("pricing_time_sec", 0.0),
                "art_violation_final": node_aux.get("art_violation_final", float("inf")),
                "prune_reason": "integral",
                "fail_stage": node_aux.get("fail_stage", ""),
            }
            node_summary_rows.append(node_row)
            realtime_writer.append_node_summary(node_row)
            realtime_writer.append_node_depth({"node_id": node.node_id, "depth": node.depth, "status": "integral"})
            realtime_writer.append_node_time_breakdown(
                {
                    "node_id": node.node_id,
                    "time_rmp": node_aux.get("rmp_time_sec", 0.0),
                    "time_pricing": node_aux.get("pricing_time_sec", 0.0),
                    "time_cut_separation": node_aux.get("cut_sep_time_sec", 0.0),
                    "time_subproblem": 0.0,
                    "total_time": node_runtime_sec,
                }
            )
            _emit_running_summary()
            continue

        if incumbent < float("inf") and lb >= incumbent - 1e-9:
            # Bound 剪枝：该节点下界不优于当前最优上界。
            nodes_pruned_by_bound += 1
            prune_reason = "bound"
            node_row = {
                "instance_id": instance.instance_id,
                "node_id": node.node_id,
                "depth": node.depth,
                "is_root": int(at_root),
                "node_runtime_sec": node_runtime_sec,
                "node_lb": lb,
                "node_ub_if_integer": float("inf"),
                "node_gap": _safe_gap(incumbent, lb),
                "node_is_integer": 0,
                "cg_iters": node_aux.get("cg_iters", 0),
                "columns_start": node_aux.get("columns_start", 0),
                "columns_added_total": node_aux.get("columns_added_total", 0),
                "columns_end": node_aux.get("active_columns", 0),
                "cuts_added_total": node_aux.get("cuts_added", 0),
                "cuts_added_rcc": node_aux.get("cuts_added_rcc", 0),
                "cuts_added_clique": node_aux.get("cuts_added_clique", 0),
                "cuts_added_sri": node_aux.get("cuts_added_sri", 0),
                "pricing_calls": node_aux.get("pricing_calls", 0),
                "pricing_negative_found_calls": node_aux.get("pricing_negative_found_calls", 0),
                "rmp_solve_calls": node_aux.get("rmp_solve_calls", 0),
                "cut_sep_calls": node_aux.get("cut_sep_calls", 0),
                "rmp_time_sec": node_aux.get("rmp_time_sec", 0.0),
                "cut_sep_time_sec": node_aux.get("cut_sep_time_sec", 0.0),
                "pricing_time_sec": node_aux.get("pricing_time_sec", 0.0),
                "art_violation_final": node_aux.get("art_violation_final", float("inf")),
                "prune_reason": prune_reason,
                "fail_stage": node_aux.get("fail_stage", ""),
            }
            node_summary_rows.append(node_row)
            realtime_writer.append_node_summary(node_row)
            realtime_writer.append_node_depth({"node_id": node.node_id, "depth": node.depth, "status": prune_reason})
            realtime_writer.append_node_time_breakdown(
                {
                    "node_id": node.node_id,
                    "time_rmp": node_aux.get("rmp_time_sec", 0.0),
                    "time_pricing": node_aux.get("pricing_time_sec", 0.0),
                    "time_cut_separation": node_aux.get("cut_sep_time_sec", 0.0),
                    "time_subproblem": 0.0,
                    "total_time": node_runtime_sec,
                }
            )
            _emit_running_summary()
            continue

        # 选择分数弧并生成左右子节点。
        t_branch = time.time()
        arc = pick_branch_arc(node_aux["arc_flow"])
        time_branch_sec_total += time.time() - t_branch
        if arc is None:
            node_row = {
                "instance_id": instance.instance_id,
                "node_id": node.node_id,
                "depth": node.depth,
                "is_root": int(at_root),
                "node_runtime_sec": node_runtime_sec,
                "node_lb": lb,
                "node_ub_if_integer": float("inf"),
                "node_gap": _safe_gap(incumbent, lb) if incumbent < float("inf") else float("inf"),
                "node_is_integer": 0,
                "cg_iters": node_aux.get("cg_iters", 0),
                "columns_start": node_aux.get("columns_start", 0),
                "columns_added_total": node_aux.get("columns_added_total", 0),
                "columns_end": node_aux.get("active_columns", 0),
                "cuts_added_total": node_aux.get("cuts_added", 0),
                "cuts_added_rcc": node_aux.get("cuts_added_rcc", 0),
                "cuts_added_clique": node_aux.get("cuts_added_clique", 0),
                "cuts_added_sri": node_aux.get("cuts_added_sri", 0),
                "pricing_calls": node_aux.get("pricing_calls", 0),
                "pricing_negative_found_calls": node_aux.get("pricing_negative_found_calls", 0),
                "rmp_solve_calls": node_aux.get("rmp_solve_calls", 0),
                "cut_sep_calls": node_aux.get("cut_sep_calls", 0),
                "rmp_time_sec": node_aux.get("rmp_time_sec", 0.0),
                "cut_sep_time_sec": node_aux.get("cut_sep_time_sec", 0.0),
                "pricing_time_sec": node_aux.get("pricing_time_sec", 0.0),
                "art_violation_final": node_aux.get("art_violation_final", float("inf")),
                "prune_reason": "open",
                "fail_stage": node_aux.get("fail_stage", ""),
            }
            node_summary_rows.append(node_row)
            realtime_writer.append_node_summary(node_row)
            realtime_writer.append_node_depth({"node_id": node.node_id, "depth": node.depth, "status": "open"})
            realtime_writer.append_node_time_breakdown(
                {
                    "node_id": node.node_id,
                    "time_rmp": node_aux.get("rmp_time_sec", 0.0),
                    "time_pricing": node_aux.get("pricing_time_sec", 0.0),
                    "time_cut_separation": node_aux.get("cut_sep_time_sec", 0.0),
                    "time_subproblem": 0.0,
                    "total_time": node_runtime_sec,
                }
            )
            _emit_running_summary()
            continue

        frac_arc_count = 0
        for v in node_aux["arc_flow"].values():
            if abs(v - round(v)) > 1e-6:
                frac_arc_count += 1
        selected_arc_flow = float(node_aux["arc_flow"].get(arc, 0.0))
        left, right = split_node(node, arc, left_id=next_node_id, right_id=next_node_id + 1)
        next_node_id += 2

        heapq.heappush(pq, (_node_priority(cfg.branch_strategy, lb, left.depth), left.node_id, left))
        heapq.heappush(pq, (_node_priority(cfg.branch_strategy, lb, right.depth), right.node_id, right))
        max_open_nodes = max(max_open_nodes, len(pq))

        branch_row = {
            "instance_id": instance.instance_id,
            "parent_node_id": node.node_id,
            "depth": node.depth,
            "frac_arc_count": frac_arc_count,
            "selected_arc_u": arc[0],
            "selected_arc_v": arc[1],
            "selected_arc_flow": selected_arc_flow,
            "selected_arc_dist_to_half": abs(selected_arc_flow - 0.5),
            "left_child_id": left.node_id,
            "right_child_id": right.node_id,
        }
        branch_rows.append(branch_row)
        realtime_writer.append_branch_decision(branch_row)
        node_row = {
            "instance_id": instance.instance_id,
            "node_id": node.node_id,
            "depth": node.depth,
            "is_root": int(at_root),
            "node_runtime_sec": node_runtime_sec,
            "node_lb": lb,
            "node_ub_if_integer": float("inf"),
            "node_gap": _safe_gap(incumbent, lb) if incumbent < float("inf") else float("inf"),
            "node_is_integer": 0,
            "cg_iters": node_aux.get("cg_iters", 0),
            "columns_start": node_aux.get("columns_start", 0),
            "columns_added_total": node_aux.get("columns_added_total", 0),
            "columns_end": node_aux.get("active_columns", 0),
            "cuts_added_total": node_aux.get("cuts_added", 0),
            "cuts_added_rcc": node_aux.get("cuts_added_rcc", 0),
            "cuts_added_clique": node_aux.get("cuts_added_clique", 0),
            "cuts_added_sri": node_aux.get("cuts_added_sri", 0),
            "pricing_calls": node_aux.get("pricing_calls", 0),
            "pricing_negative_found_calls": node_aux.get("pricing_negative_found_calls", 0),
            "rmp_solve_calls": node_aux.get("rmp_solve_calls", 0),
            "cut_sep_calls": node_aux.get("cut_sep_calls", 0),
            "rmp_time_sec": node_aux.get("rmp_time_sec", 0.0),
            "cut_sep_time_sec": node_aux.get("cut_sep_time_sec", 0.0),
            "pricing_time_sec": node_aux.get("pricing_time_sec", 0.0),
            "art_violation_final": node_aux.get("art_violation_final", float("inf")),
            "prune_reason": "branched",
            "fail_stage": node_aux.get("fail_stage", ""),
        }
        node_summary_rows.append(node_row)
        realtime_writer.append_node_summary(node_row)
        realtime_writer.append_node_depth({"node_id": node.node_id, "depth": node.depth, "status": "branched"})
        realtime_writer.append_node_time_breakdown(
            {
                "node_id": node.node_id,
                "time_rmp": node_aux.get("rmp_time_sec", 0.0),
                "time_pricing": node_aux.get("pricing_time_sec", 0.0),
                "time_cut_separation": node_aux.get("cut_sep_time_sec", 0.0),
                "time_subproblem": 0.0,
                "total_time": node_runtime_sec,
            }
        )
        _emit_running_summary()

        if incumbent < float("inf"):
            gap = _safe_gap(incumbent, best_lb)
            if gap <= cfg.mip_gap:
                # 达到目标 gap 提前结束。
                break

    status = "OPTIMAL" if not pq and incumbent < float("inf") else "TIME_LIMIT"
    if incumbent == float("inf"):
        status = "NO_SOLUTION"

    gap = _safe_gap(incumbent, best_lb) if incumbent < float("inf") and math.isfinite(best_lb) else float("inf")
    total_runtime_sec = time.time() - t0
    measured_runtime = time_rmp_sec_total + time_pricing_sec_total + time_cut_sep_sec_total + time_branch_sec_total
    time_other_sec_total = max(0.0, total_runtime_sec - measured_runtime)
    stats = {
        "runtime_sec": total_runtime_sec,
        "nodes_processed": float(nodes_processed),
        "global_columns": float(len(global_columns)),
        "global_columns_final": float(len(global_columns)),
        "best_lb": best_lb if math.isfinite(best_lb) else float("inf"),
        "root_lb": root_lb if math.isfinite(root_lb) else float("inf"),
        "nodes_pruned_by_bound": float(nodes_pruned_by_bound),
        "nodes_pruned_infeasible": float(nodes_pruned_infeasible),
        "max_open_nodes": float(max_open_nodes),
        "max_depth": float(max_depth),
        "cg_iters_total": float(cg_iters_total),
        "rmp_solve_calls_total": float(rmp_solve_calls_total),
        "cuts_separation_calls_total": float(cut_sep_calls_total),
        "pricing_calls_total": float(pricing_calls_total),
        "time_rmp_sec": time_rmp_sec_total,
        "time_cut_sep_sec": time_cut_sep_sec_total,
        "time_pricing_sec": time_pricing_sec_total,
        "time_branch_sec": time_branch_sec_total,
        "time_other_sec": time_other_sec_total,
        "time_to_first_feasible_sec": first_feasible_ts if math.isfinite(first_feasible_ts) else float("inf"),
        "time_to_best_incumbent_sec": best_incumbent_ts if math.isfinite(best_incumbent_ts) else float("inf"),
        "solve_seed": float(cfg.random_seed),
    }
    if fail_stage_counts:
        stats["fail_after_cuts_nodes"] = float(fail_stage_counts.get("after_cuts", 0))
        stats["fail_before_cuts_nodes"] = float(fail_stage_counts.get("solve_before_cuts", 0))
        stats["fail_infeasible_by_columns_nodes"] = float(fail_stage_counts.get("infeasible_by_columns", 0))

    result = SolveResult(
        status=status,
        obj_primal=incumbent if incumbent < float("inf") else float("inf"),
        obj_dual=best_lb if math.isfinite(best_lb) else float("inf"),
        gap=gap,
        routes=incumbent_routes,
        stats=stats,
    )

    if bundle is not None:
        profile_out = excel_profile_path.strip() if excel_profile_path else str(outdir / "excel_model_profile.json")
        dump_excel_model_profile(bundle, profile_out)
    _write_solution_json(outdir, result)
    _write_routes(outdir, result)
    _write_metrics(outdir, result)
    realtime_writer.update_run_summary(
        {
            "instance_id": instance.instance_id,
            "status": result.status,
            "obj_primal": result.obj_primal,
            "obj_dual": result.obj_dual,
            "gap": result.gap,
            **result.stats,
        }
    )
    realtime_writer.write_analysis_run_summary(
        {
            "instance_name": instance.instance_id,
            "variant_name": Path(cfg.output_dir).name,
            "status": result.status,
            "root_gap_end": _safe_gap(result.obj_primal, root_lb) if result.obj_primal < float("inf") and math.isfinite(root_lb) else float("inf"),
            "infeasible_ratio": (nodes_pruned_infeasible / max(1, nodes_processed)),
            "columns_generated": float(len(global_columns)),
        }
    )

    return result


def _load_solver_config(cfg_path: str) -> SolverConfig:
    """读取求解配置（当前实现使用 JSON）。"""
    text = Path(cfg_path).read_text(encoding="utf-8")
    if cfg_path.endswith(".json"):
        data = json.loads(text)
    else:
        # Minimal YAML subset via json fallback by allowing pure JSON-like YAML.
        data = json.loads(text)
    cfg = SolverConfig()
    for k, v in data.items():
        if k == "stabilization" and isinstance(v, dict):
            for sk, sv in v.items():
                setattr(cfg.stabilization, sk, sv)
        elif k == "problem" and isinstance(v, dict):
            for pk, pv in v.items():
                if hasattr(cfg.problem, pk):
                    setattr(cfg.problem, pk, pv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _load_batch_instances(db_path: str, batch_id: str) -> List[str]:
    """按 batch_id 从数据库读取实例列表。"""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT bi.instance_id
            FROM batch_instances bi
            WHERE bi.batch_id = ?
            ORDER BY bi.instance_id
            """,
            (batch_id,),
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def run_experiment(
    batch_id: str,
    cfg_path: str,
    db_path: str = "instances.db",
    csv_dir: str = "",
    rmp_solver_override: str = "",
) -> ExperimentReport:
    """批量实验入口。

    对 batch 中每个实例调用 solve_instance，
    最终汇总平均目标值、平均 gap 并输出 report.json。
    """
    cfg = _load_solver_config(cfg_path)
    if rmp_solver_override:
        cfg.rmp_solver = rmp_solver_override
    instances = _load_batch_instances(db_path, batch_id)
    results: List[SolveResult] = []

    for iid in instances:
        outdir = Path(cfg.output_dir) / batch_id / iid
        cfg_i = SolverConfig(**{**cfg.__dict__})
        cfg_i.stabilization = cfg.stabilization
        cfg_i.output_dir = str(outdir)
        try:
            res = solve_instance(iid, cfg_i, db_path=db_path, csv_dir=csv_dir)
        except GurobiUnavailableError as exc:
            res = SolveResult(
                status="ERROR_GUROBI",
                obj_primal=float("inf"),
                obj_dual=float("inf"),
                gap=float("inf"),
                routes=[],
                stats={"error": str(exc)},
            )
        results.append(res)

    solved = [r for r in results if r.status in {"OPTIMAL", "TIME_LIMIT"}]
    summary = {
        "instances": float(len(results)),
        "solved_like": float(len(solved)),
        "avg_primal": sum(r.obj_primal for r in solved if math.isfinite(r.obj_primal)) / max(1, len([r for r in solved if math.isfinite(r.obj_primal)])),
        "avg_gap": sum(r.gap for r in solved if math.isfinite(r.gap)) / max(1, len([r for r in solved if math.isfinite(r.gap)])),
    }

    report = ExperimentReport(batch_id=batch_id, results=results, summary=summary)
    out = Path(cfg.output_dir) / batch_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(
            {
                "batch_id": report.batch_id,
                "summary": report.summary,
                "results": [
                    {
                        "status": r.status,
                        "obj_primal": r.obj_primal,
                        "obj_dual": r.obj_dual,
                        "gap": r.gap,
                        "stats": r.stats,
                    }
                    for r in report.results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def run_excel_experiment(
    batch_id: str,
    cfg_path: str,
    excel_path: str = "",
    preprocessed_path: str = "",
    repeat: int = 1,
    instance_prefix: str = "excel_inst",
    rmp_solver_override: str = "",
) -> ExperimentReport:
    """Excel 实验入口。

    场景：
    - 仅有一个 xlsx 实例文件；
    - 需要重复运行多次（不同随机种子）做实验统计。
    """
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}")
    if not excel_path and not preprocessed_path:
        raise ValueError("either excel_path or preprocessed_path must be provided")

    cfg = _load_solver_config(cfg_path)
    if rmp_solver_override:
        cfg.rmp_solver = rmp_solver_override
    results: List[SolveResult] = []

    for idx in range(repeat):
        outdir = Path(cfg.output_dir) / batch_id / f"run_{idx + 1:03d}"
        cfg_i = SolverConfig(**{**cfg.__dict__})
        cfg_i.stabilization = cfg.stabilization
        cfg_i.output_dir = str(outdir)
        cfg_i.random_seed = cfg.random_seed + idx
        iid = f"{instance_prefix}_{idx + 1:03d}"
        try:
            res = solve_instance(iid, cfg_i, excel_path=excel_path, preprocessed_path=preprocessed_path)
        except GurobiUnavailableError as exc:
            res = SolveResult(
                status="ERROR_GUROBI",
                obj_primal=float("inf"),
                obj_dual=float("inf"),
                gap=float("inf"),
                routes=[],
                stats={"error": str(exc)},
            )
        results.append(res)

    solved = [r for r in results if r.status in {"OPTIMAL", "TIME_LIMIT"}]
    finite_primal = [r.obj_primal for r in solved if math.isfinite(r.obj_primal)]
    finite_gap = [r.gap for r in solved if math.isfinite(r.gap)]
    summary = {
        "instances": float(len(results)),
        "solved_like": float(len(solved)),
        "avg_primal": sum(finite_primal) / max(1, len(finite_primal)),
        "avg_gap": sum(finite_gap) / max(1, len(finite_gap)),
    }

    report = ExperimentReport(batch_id=batch_id, results=results, summary=summary)
    out = Path(cfg.output_dir) / batch_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(
            {
                "batch_id": report.batch_id,
                "summary": report.summary,
                "excel_path": excel_path,
                "preprocessed_path": preprocessed_path,
                "repeat": repeat,
                "results": [
                    {
                        "status": r.status,
                        "obj_primal": r.obj_primal,
                        "obj_dual": r.obj_dual,
                        "gap": r.gap,
                        "stats": r.stats,
                    }
                    for r in report.results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
