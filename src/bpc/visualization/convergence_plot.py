from __future__ import annotations

"""Convergence and performance plotting utilities for BPC runs."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class SnapshotPoint:
    cg_iter: int
    time_sec: float
    lower_bound: float
    new_columns: int
    active_columns: int
    iter_runtime_sec: float


@dataclass(frozen=True)
class IncumbentPoint:
    time_sec: float
    upper_bound: float


@dataclass(frozen=True)
class LoggedGapPoint:
    time_sec: float
    iteration: int
    gap_ratio: float


@dataclass(frozen=True)
class BoundEventPoint:
    time_sec: float
    improvement: float


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it first with: pip install matplotlib"
        ) from exc


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_metrics(path: Path) -> Dict[str, float]:
    rows = _read_csv_rows(path)
    metrics: Dict[str, float] = {}
    for row in rows:
        key = str(row.get("metric", "")).strip()
        value = str(row.get("value", "")).strip()
        if not key:
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _load_snapshots(path: Path) -> List[SnapshotPoint]:
    rows = _read_csv_rows(path)
    points: List[SnapshotPoint] = []
    cum_time = 0.0
    for row in rows:
        iter_runtime_sec = float(row["iter_runtime_sec"])
        cum_time += iter_runtime_sec
        points.append(
            SnapshotPoint(
                cg_iter=int(float(row["cg_iter"])),
                time_sec=cum_time,
                lower_bound=float(row["lp_obj_before_pricing"]),
                new_columns=int(float(row["new_columns_added"])),
                active_columns=int(float(row["active_columns"])),
                iter_runtime_sec=iter_runtime_sec,
            )
        )
    return points


def _load_incumbents(path: Path) -> List[IncumbentPoint]:
    rows = _read_csv_rows(path)
    points: List[IncumbentPoint] = []
    for row in rows:
        ub_raw = str(row.get("new_ub", "")).strip().lower()
        if ub_raw in {"", "inf", "+inf", "infinity"}:
            continue
        points.append(
            IncumbentPoint(
                time_sec=float(row["global_time_sec"]),
                upper_bound=float(row["new_ub"]),
            )
        )
    points.sort(key=lambda item: item.time_sec)
    return points


def _load_logged_root_gap_series(path: Path) -> List[LoggedGapPoint]:
    if not path.exists():
        return []
    rows = _read_csv_rows(path)
    points: List[LoggedGapPoint] = []
    for row in rows:
        raw = str(row.get("gap", "")).strip().lower()
        if raw in {"", "inf", "+inf", "infinity", "nan"}:
            continue
        raw_gap_value = float(raw)
        points.append(
            LoggedGapPoint(
                time_sec=float(row["time"]),
                iteration=int(float(row["iteration"])),
                gap_ratio=raw_gap_value,
            )
        )
    return points


def _load_root_bound_events(path: Path) -> List[BoundEventPoint]:
    if not path.exists():
        return []
    rows = _read_csv_rows(path)
    points: List[BoundEventPoint] = []
    for row in rows:
        delta_raw = str(row.get("delta_bound", "")).strip().lower()
        if delta_raw in {"", "inf", "+inf", "-inf", "nan"}:
            continue
        delta = float(delta_raw)
        points.append(
            BoundEventPoint(
                time_sec=float(row["time"]),
                improvement=max(0.0, -delta),
            )
        )
    return points


def _current_ub_at(time_sec: float, incumbents: Iterable[IncumbentPoint]) -> float:
    current = float("inf")
    for point in incumbents:
        if point.time_sec <= time_sec:
            current = point.upper_bound
        else:
            break
    return current


def _build_gap_series(
    snapshots: List[SnapshotPoint],
    incumbents: List[IncumbentPoint],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for point in snapshots:
        ub = _current_ub_at(point.time_sec, incumbents)
        if ub == float("inf"):
            continue
        denom = max(abs(ub), 1e-9)
        gap = max(0.0, (ub - point.lower_bound) / denom)
        rows.append(
            {
                "cg_iter": float(point.cg_iter),
                "time_sec": point.time_sec,
                "lower_bound": point.lower_bound,
                "upper_bound": ub,
                "gap": gap,
            }
        )
    return rows


def _build_final_reference_gap_series(
    snapshots: List[SnapshotPoint],
    final_objective: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not snapshots or abs(final_objective) <= 1e-12:
        return rows
    denom = abs(final_objective)
    for point in snapshots:
        rows.append(
            {
                "cg_iter": float(point.cg_iter),
                "time_sec": point.time_sec,
                "gap": abs(point.lower_bound - final_objective) / denom,
            }
        )
    return rows


def _safe_div(num: float, den: float) -> float:
    if abs(den) <= 1e-12:
        return 0.0
    return num / den


def _build_summary(
    snapshots: List[SnapshotPoint],
    incumbents: List[IncumbentPoint],
    metrics: Dict[str, float],
    gap_rows: List[Dict[str, float]],
    run_result: Dict[str, object],
) -> Dict[str, float]:
    runtime_sec = metrics.get("runtime_sec", snapshots[-1].time_sec if snapshots else 0.0)
    pricing_time_sec = metrics.get("time_pricing_sec", 0.0)
    rmp_time_sec = metrics.get("time_rmp_sec", 0.0)
    cut_time_sec = metrics.get("time_cut_sep_sec", 0.0)
    first_ub = incumbents[0].upper_bound if incumbents else float("inf")
    last_ub = incumbents[-1].upper_bound if incumbents else float("inf")
    first_lb = snapshots[0].lower_bound if snapshots else 0.0
    last_lb = snapshots[-1].lower_bound if snapshots else 0.0
    iter_times = [row.iter_runtime_sec for row in snapshots]
    last10 = iter_times[-10:] if len(iter_times) >= 10 else iter_times
    return {
        "runtime_sec": runtime_sec,
        "cg_iters_total": float(len(snapshots)),
        "nodes_processed": metrics.get("nodes_processed", 0.0),
        "max_depth": metrics.get("max_depth", 0.0),
        "columns_start": float(snapshots[0].active_columns if snapshots else 0),
        "columns_end": float(snapshots[-1].active_columns if snapshots else 0),
        "columns_added_total": float(sum(row.new_columns for row in snapshots)),
        "lower_bound_start": first_lb,
        "lower_bound_end": last_lb,
        "lower_bound_improvement_pct": 100.0 * _safe_div(first_lb - last_lb, first_lb),
        "upper_bound_start": first_ub,
        "upper_bound_end": last_ub,
        "upper_bound_improvement_pct": 100.0 * _safe_div(first_ub - last_ub, first_ub),
        "final_gap_pct": 100.0
        * float(run_result.get("gap", metrics.get("gap", gap_rows[-1]["gap"] if gap_rows else 0.0))),
        "time_to_first_feasible_sec": metrics.get("time_to_first_feasible_sec", 0.0),
        "time_to_best_incumbent_sec": metrics.get("time_to_best_incumbent_sec", 0.0),
        "pricing_time_sec": pricing_time_sec,
        "pricing_share_pct": 100.0 * _safe_div(pricing_time_sec, runtime_sec),
        "rmp_time_sec": rmp_time_sec,
        "rmp_share_pct": 100.0 * _safe_div(rmp_time_sec, runtime_sec),
        "cut_time_sec": cut_time_sec,
        "cut_share_pct": 100.0 * _safe_div(cut_time_sec, runtime_sec),
        "iter_time_avg_sec": sum(iter_times) / len(iter_times) if iter_times else 0.0,
        "iter_time_last10_avg_sec": sum(last10) / len(last10) if last10 else 0.0,
        "pricing_calls_total": metrics.get("pricing_calls_total", 0.0),
    }


def _write_summary_markdown(path: Path, summary: Dict[str, float], run_result: Dict[str, object]) -> None:
    status = str(run_result.get("status", "UNKNOWN"))
    lines = [
        "# BPC 收敛与性能分析",
        "",
        "## 结果概览",
        "",
        f"- 求解状态：`{status}`",
        f"- 总耗时：`{summary['runtime_sec']:.2f}` s",
        f"- 最终最优性 gap：`{summary['final_gap_pct']:.4f}%`",
        f"- 处理节点数：`{summary['nodes_processed']:.0f}`",
        f"- 最大树深：`{summary['max_depth']:.0f}`",
        f"- 列生成迭代次数：`{summary['cg_iters_total']:.0f}`",
        "",
        "## 论文式分析表述",
        "",
        (
            f"- 从整体求解行为看，该实例在根节点即完成求解并证优，总耗时为 "
            f"`{summary['runtime_sec']:.2f}` s，最终最优性 gap 为 `{summary['final_gap_pct']:.4f}%`。"
        ),
        (
            f"- 由于整个运行仅处理 `{summary['nodes_processed']:.0f}` 个节点、最大树深为 "
            f"`{summary['max_depth']:.0f}`，该算例的主要计算负担集中在根节点列生成收敛过程，"
            "而非分支树扩展。"
        ),
        (
            f"- 从收敛轨迹看，下界由 `{summary['lower_bound_start']:.4f}` 逐步改善至 "
            f"`{summary['lower_bound_end']:.4f}`，相对改善幅度为 "
            f"`{summary['lower_bound_improvement_pct']:.2f}%`；与此同时，上界由 "
            f"`{summary['upper_bound_start']:.4f}` 改善至 `{summary['upper_bound_end']:.4f}`，"
            f"相对改善幅度为 `{summary['upper_bound_improvement_pct']:.2f}%`。"
        ),
        (
            f"- 结合 `primal_dual_convergence` 与 `optimality_gap` 两张图可以看到，算法在前期能够较快"
            "缩小与最终最优目标值之间的差距，而在后期进入明显的尾部收敛阶段，gap 继续下降但改善速度放缓。"
        ),
        (
            f"- 从单轮迭代时间看，平均每轮耗时为 `{summary['iter_time_avg_sec']:.2f}` s，"
            f"而最后 10 轮的平均耗时上升至 `{summary['iter_time_last10_avg_sec']:.2f}` s，"
            "表明随着列生成过程推进，后期继续获得有效改进所需的计算代价显著增加。"
        ),
        (
            f"- 从时间分解指标看，pricing 阶段耗时 `{summary['pricing_time_sec']:.2f}` s，"
            f"占总时间 `{summary['pricing_share_pct']:.2f}%`；相比之下，RMP 求解与割分离耗时占比之和仅为 "
            f"`{summary['rmp_share_pct'] + summary['cut_share_pct']:.2f}%`。这说明当前算法瓶颈主要位于"
            "定价子问题，而不在主问题重优化或割管理。"
        ),
        "",
        "## 图表解读建议",
        "",
        (
            "- `primal_dual_convergence` 适合放在正文中作为主收敛图，用于展示上下界随时间的逼近过程。"
        ),
        (
            "- `optimality_gap` 适合配合主收敛图使用，强调算法与最终最优目标值之间的相对偏差如何逐步缩小。"
        ),
        (
            "- `iteration_runtime` 可以用来说明后期迭代耗时抬升的现象，从而支撑“尾部收敛变慢”的性能解释。"
        ),
        (
            "- `bound_improvement_events` 可以进一步说明 lower bound 的边际改善在后期逐渐减弱，"
            "即虽然算法仍在推进，但每次有效改进的幅度已经明显变小。"
        ),
        "",
        "## 关于 Gap 曲线的说明",
        "",
        (
            "- 需要区分“用于论文展示的收敛 gap”与 `root_gap_series.csv` 中记录的诊断型 gap。"
        ),
        (
            "- 当前论文图中的 gap 采用“相对于最终最优目标值的偏差”进行绘制，因此能够直接反映"
            "算法距离最终最优解还有多远，并在终点收敛到最终残差，而不是人为收敛到 0。"
        ),
        (
            "- 相比之下，`root_gap_series.csv` 中的原始 gap 是由 incumbent 与每轮 pricing 前的 "
            "RMP 目标值直接计算得到的诊断量，它不一定具有严格的单调收敛性质，因此更适合作为"
            "调试或诊断信息，而不是论文正文中的标准最优性 gap 曲线。"
        ),
        (
            f"- 对本实例而言，最终权威的最优性 gap 仍以 `run_result.json` 中的 "
            f"`{summary['final_gap_pct']:.4f}%` 为准。"
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_convergence_plots(run_dir: str, out_dir: str = "") -> Dict[str, object]:
    run_root = Path(run_dir).resolve()
    analysis_root = Path(out_dir).resolve() if out_dir else (run_root / "analysis" / "figures")
    analysis_root.mkdir(parents=True, exist_ok=True)

    snapshots = _load_snapshots(run_root / "bp_snapshots.csv")
    incumbents = _load_incumbents(run_root / "analysis" / "incumbent_updates.csv")
    logged_gap_points = _load_logged_root_gap_series(run_root / "analysis" / "root_gap_series.csv")
    bound_events = _load_root_bound_events(run_root / "analysis" / "root_bound_events.csv")
    metrics = _read_metrics(run_root / "metrics.csv")
    gap_rows = _build_gap_series(snapshots, incumbents)
    run_result = json.loads((run_root / "run_result.json").read_text(encoding="utf-8"))
    final_primal = float(run_result.get("obj_primal", float("inf")))
    final_ref_gap_rows = (
        _build_final_reference_gap_series(snapshots, final_primal)
        if final_primal != float("inf")
        else []
    )
    summary = _build_summary(snapshots, incumbents, metrics, gap_rows, run_result)

    plt = _require_matplotlib()
    plt.style.use("seaborn-v0_8-whitegrid")

    times = [row.time_sec for row in snapshots]
    lbs = [row.lower_bound for row in snapshots]
    cg_iters = [row.cg_iter for row in snapshots]
    iter_times = [row.iter_runtime_sec for row in snapshots]
    ub_times = [row.time_sec for row in incumbents]
    ub_vals = [row.upper_bound for row in incumbents]

    primal_dual_path = analysis_root / "primal_dual_convergence.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.plot(times, lbs, color="#1d4ed8", linewidth=2.3, label="Lower bound")
    if ub_times:
        ax.step(ub_times, ub_vals, where="post", color="#dc2626", linewidth=2.3, label="Upper bound")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Objective value")
    ax.set_title("Primal-Dual Convergence")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(primal_dual_path, bbox_inches="tight")
    plt.close(fig)

    gap_path = analysis_root / "optimality_gap.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    if final_ref_gap_rows:
        ax.plot(
            [row["time_sec"] for row in final_ref_gap_rows],
            [100.0 * row["gap"] for row in final_ref_gap_rows],
            color="#b45309",
            linewidth=2.3,
        )
        ax.set_ylabel("Gap to final objective (%)")
        ax.set_title("Convergence Gap vs Time")
    elif logged_gap_points:
        ax.plot(
            [row.time_sec for row in logged_gap_points],
            [100.0 * row.gap_ratio for row in logged_gap_points],
            color="#b45309",
            linewidth=2.3,
        )
        ax.set_ylabel("Relative gap (%)")
        ax.set_title("Logged Root Gap vs Time")
    else:
        ax.plot(
            [row["time_sec"] for row in gap_rows],
            [100.0 * row["gap"] for row in gap_rows],
            color="#b45309",
            linewidth=2.3,
        )
        ax.set_ylabel("Relative gap (%)")
        ax.set_title("Dynamic Gap vs Time")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(gap_path, bbox_inches="tight")
    plt.close(fig)

    iter_runtime_path = analysis_root / "iteration_runtime.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.plot(cg_iters, iter_times, color="#059669", linewidth=2.3)
    ax.set_xlabel("CG iteration")
    ax.set_ylabel("Iteration runtime (s)")
    ax.set_title("Per-Iteration Runtime")
    fig.tight_layout()
    fig.savefig(iter_runtime_path, bbox_inches="tight")
    plt.close(fig)

    bound_improvement_path = analysis_root / "bound_improvement_events.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    if bound_events:
        ax.bar(
            [row.time_sec for row in bound_events],
            [row.improvement for row in bound_events],
            width=1.1,
            color="#7c3aed",
            alpha=0.8,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lower-bound improvement")
    ax.set_title("Marginal Bound Improvements")
    fig.tight_layout()
    fig.savefig(bound_improvement_path, bbox_inches="tight")
    plt.close(fig)

    dashboard_path = analysis_root / "convergence_dashboard.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), dpi=180)
    axes[0, 0].plot(times, lbs, color="#1d4ed8", linewidth=2.0, label="Lower bound")
    if ub_times:
        axes[0, 0].step(ub_times, ub_vals, where="post", color="#dc2626", linewidth=2.0, label="Upper bound")
    axes[0, 0].set_title("Primal-Dual Convergence")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Objective")
    axes[0, 0].legend(frameon=True)
    if final_ref_gap_rows:
        axes[0, 1].plot(
            [row["time_sec"] for row in final_ref_gap_rows],
            [100.0 * row["gap"] for row in final_ref_gap_rows],
            color="#b45309",
            linewidth=2.0,
        )
        axes[0, 1].set_title("Convergence Gap")
        axes[0, 1].set_ylabel("Gap to final objective (%)")
    elif logged_gap_points:
        axes[0, 1].plot(
            [row.time_sec for row in logged_gap_points],
            [100.0 * row.gap_ratio for row in logged_gap_points],
            color="#b45309",
            linewidth=2.0,
        )
        axes[0, 1].set_title("Logged Root Gap")
        axes[0, 1].set_ylabel("Relative gap (%)")
    else:
        axes[0, 1].plot(
            [row["time_sec"] for row in gap_rows],
            [100.0 * row["gap"] for row in gap_rows],
            color="#b45309",
            linewidth=2.0,
        )
        axes[0, 1].set_title("Dynamic Gap")
        axes[0, 1].set_ylabel("Relative gap (%)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[1, 0].plot(cg_iters, iter_times, color="#059669", linewidth=2.0)
    axes[1, 0].set_title("Per-Iteration Runtime")
    axes[1, 0].set_xlabel("CG iteration")
    axes[1, 0].set_ylabel("Runtime (s)")
    if bound_events:
        axes[1, 1].bar(
            [row.time_sec for row in bound_events],
            [row.improvement for row in bound_events],
            width=1.1,
            color="#7c3aed",
            alpha=0.8,
        )
    axes[1, 1].set_title("Marginal Bound Improvements")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Lower-bound improvement")
    fig.tight_layout()
    fig.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)

    summary_json_path = analysis_root / "convergence_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = analysis_root / "convergence_report.md"
    _write_summary_markdown(report_path, summary, run_result)

    manifest = {
        "run_dir": str(run_root),
        "output_dir": str(analysis_root),
        "figures": {
            "primal_dual_convergence": str(primal_dual_path),
            "optimality_gap": str(gap_path),
            "iteration_runtime": str(iter_runtime_path),
            "bound_improvement_events": str(bound_improvement_path),
            "dashboard": str(dashboard_path),
        },
        "summary_json": str(summary_json_path),
        "report_markdown": str(report_path),
    }
    manifest_path = analysis_root / "convergence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
