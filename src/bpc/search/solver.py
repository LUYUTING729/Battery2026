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
from bpc.pricing.initializer import generate_initial_columns
from bpc.pricing.ng_pricing import price_columns
from bpc.rmp.master_problem import GurobiUnavailableError, MasterProblem


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)


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
    rmp = MasterProblem(instance, node, relax=True)

    seed_cols = _enforce_node_feasible_columns(global_columns, node)
    rmp.add_columns(seed_cols)

    cuts_added_total = 0
    cg_iter = 0
    while cg_iter < cfg.max_cg_iters:
        cg_iter += 1
        # Step A: 先解当前 RMP，得到对偶和分数解。
        info = rmp.solve()
        if not math.isfinite(info.obj_val):
            return float("inf"), False, [], {"cg_iters": cg_iter, "cuts_added": cuts_added_total, "new_cols": 0}

        rounds = _rounds(cfg, at_root)
        cuts_added_this_iter = 0
        for cut_round in range(rounds):
            # Step B: 用当前分数解做分离，若发现 violated cuts 则加入并重解。
            active = rmp.active_routes()
            cuts = separate_cuts(instance, active, cfg, at_root=at_root, round_id=(cg_iter * 10 + cut_round))
            if not cuts:
                continue
            cuts_added = rmp.add_cuts(cuts)
            cuts_added_total += cuts_added
            cuts_added_this_iter += cuts_added
            if cuts_added:
                info = rmp.solve()

        # Step C: pricing 子问题基于对偶寻找负化简成本列，增强 RMP。
        duals = rmp.extract_duals()
        new_columns = price_columns(instance, node, duals, cfg)
        new_columns = [c for c in new_columns if c.column_id not in rmp.vars]
        if len(new_columns) > cfg.max_new_columns_per_iter:
            new_columns = new_columns[: cfg.max_new_columns_per_iter]
        new_added = rmp.add_columns(new_columns)
        global_columns.extend([c for c in new_columns if c.column_id not in {x.column_id for x in global_columns}])

        trace_rows.append(
            {
                "node_id": node.node_id,
                "depth": node.depth,
                "cg_iter": cg_iter,
                "lp_obj": info.obj_val,
                "cuts_added": cuts_added_this_iter,
                "new_columns": new_added,
                "active_columns": len(rmp.vars),
            }
        )

        if new_added == 0:
            # 无新列 -> 节点 LP 在当前列空间内稳定。
            break

    # 节点终态：用最终 RMP 判断整数性并提取候选路线。
    final = rmp.solve()
    lp_obj = final.obj_val
    is_integer = rmp.is_integral()
    selected = rmp.selected_routes()
    arc_flow = rmp.aggregated_arc_flow()
    node_stats = {
        "cg_iters": cg_iter,
        "cuts_added": cuts_added_total,
        "active_columns": len(rmp.vars),
    }
    return lp_obj, is_integer, selected, {"arc_flow": arc_flow, **node_stats}


def solve_instance(instance_id: str, cfg: SolverConfig, db_path: str = "instances.db", csv_dir: str = "") -> SolveResult:
    """单实例求解入口。

    该函数实现完整 BCP 主循环：
    - 初始化根节点；
    - 反复弹出节点求解；
    - 根据分数弧流执行二叉分支；
    - 用上下界做剪枝；
    - 达到停止条件后输出最优/当前最优。
    """
    t0 = time.time()
    instance = load_instance(instance_id=instance_id, db_path=db_path, csv_dir=csv_dir)

    global_columns = generate_initial_columns(instance)

    root = NodeState(node_id=0, depth=0)
    pq = []
    next_node_id = 1

    incumbent = float("inf")
    incumbent_routes = []
    best_lb = float("inf")

    trace_rows: List[dict] = []
    heapq.heappush(pq, (_node_priority(cfg.branch_strategy, 0.0, 0), root.node_id, root))

    nodes_processed = 0
    while pq and nodes_processed < cfg.max_nodes:
        if time.time() - t0 > cfg.time_limit_s:
            break

        # 取下一个待处理节点（best-bound 或 depth-first）。
        _, _, node = heapq.heappop(pq)
        at_root = node.node_id == 0

        # 节点内部执行 “CG + cuts + pricing”。
        lb, is_integer, routes, node_aux = _solve_node(instance, node, cfg, global_columns, at_root, trace_rows)
        nodes_processed += 1

        if not math.isfinite(lb):
            continue

        if lb < best_lb:
            best_lb = lb

        if is_integer:
            # 节点 LP 已整数，可形成可行解并尝试更新 incumbent。
            obj = sum(r.cost for r in routes)
            if obj < incumbent:
                incumbent = obj
                incumbent_routes = routes
            continue

        if incumbent < float("inf") and lb >= incumbent - 1e-9:
            # Bound 剪枝：该节点下界不优于当前最优上界。
            continue

        # 选择分数弧并生成左右子节点。
        arc = pick_branch_arc(node_aux["arc_flow"])
        if arc is None:
            continue

        left, right = split_node(node, arc, left_id=next_node_id, right_id=next_node_id + 1)
        next_node_id += 2

        heapq.heappush(pq, (_node_priority(cfg.branch_strategy, lb, left.depth), left.node_id, left))
        heapq.heappush(pq, (_node_priority(cfg.branch_strategy, lb, right.depth), right.node_id, right))

        if incumbent < float("inf"):
            gap = _safe_gap(incumbent, best_lb)
            if gap <= cfg.mip_gap:
                # 达到目标 gap 提前结束。
                break

    status = "OPTIMAL" if not pq and incumbent < float("inf") else "TIME_LIMIT"
    if incumbent == float("inf"):
        status = "NO_SOLUTION"

    gap = _safe_gap(incumbent, best_lb) if incumbent < float("inf") and math.isfinite(best_lb) else float("inf")
    stats = {
        "runtime_sec": time.time() - t0,
        "nodes_processed": float(nodes_processed),
        "global_columns": float(len(global_columns)),
        "best_lb": best_lb if math.isfinite(best_lb) else float("inf"),
    }

    result = SolveResult(
        status=status,
        obj_primal=incumbent if incumbent < float("inf") else float("inf"),
        obj_dual=best_lb if math.isfinite(best_lb) else float("inf"),
        gap=gap,
        routes=incumbent_routes,
        stats=stats,
    )

    outdir = _mkdir(cfg.output_dir)
    _write_solution_json(outdir, result)
    _write_routes(outdir, result)
    _write_trace(outdir, trace_rows)
    _write_metrics(outdir, result)

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


def run_experiment(batch_id: str, cfg_path: str, db_path: str = "instances.db", csv_dir: str = "") -> ExperimentReport:
    """批量实验入口。

    对 batch 中每个实例调用 solve_instance，
    最终汇总平均目标值、平均 gap 并输出 report.json。
    """
    cfg = _load_solver_config(cfg_path)
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
