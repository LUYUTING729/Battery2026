from __future__ import annotations

"""需求结构实验结果分析 CLI。"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List


def _safe_mean(vals: List[float]) -> float:
    x = [v for v in vals if math.isfinite(v)]
    return sum(x) / max(1, len(x))


def _safe_std(vals: List[float]) -> float:
    x = [v for v in vals if math.isfinite(v)]
    if len(x) <= 1:
        return 0.0
    return statistics.pstdev(x)


def _safe_median(vals: List[float]) -> float:
    x = [v for v in vals if math.isfinite(v)]
    if not x:
        return float("inf")
    return float(statistics.median(x))


def _summarize_runs(runs: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    distributions = sorted({str(r.get("distribution", "")) for r in runs if r.get("distribution")})
    for dist in distributions:
        sub = [r for r in runs if r.get("distribution") == dist]
        solved = [r for r in sub if r.get("status") in {"OPTIMAL", "TIME_LIMIT"}]

        primal = [float(r.get("obj_primal", float("inf"))) for r in solved]
        gap = [float(r.get("gap", float("inf"))) for r in solved]
        runtime = [float(r.get("runtime_sec", float("inf"))) for r in solved]
        nodes = [float(r.get("nodes_processed", float("inf"))) for r in solved]

        out[dist] = {
            "runs": float(len(sub)),
            "solved_like": float(len(solved)),
            "solve_rate": float(len(solved) / max(1, len(sub))),
            "avg_primal": _safe_mean(primal),
            "median_primal": _safe_median(primal),
            "std_primal": _safe_std(primal),
            "avg_gap": _safe_mean(gap),
            "median_gap": _safe_median(gap),
            "std_gap": _safe_std(gap),
            "avg_runtime_sec": _safe_mean(runtime),
            "median_runtime_sec": _safe_median(runtime),
            "std_runtime_sec": _safe_std(runtime),
            "avg_nodes": _safe_mean(nodes),
        }
    return out


def _relative_to_baseline(summary: Dict[str, dict], baseline: str) -> Dict[str, dict]:
    if baseline not in summary:
        return {}
    b = summary[baseline]
    out: Dict[str, dict] = {}
    for dist, s in summary.items():
        out[dist] = {
            "delta_avg_primal": s["avg_primal"] - b["avg_primal"],
            "delta_avg_gap": s["avg_gap"] - b["avg_gap"],
            "delta_avg_runtime_sec": s["avg_runtime_sec"] - b["avg_runtime_sec"],
            "delta_solve_rate": s["solve_rate"] - b["solve_rate"],
        }
    return out


def _pick_best(summary: Dict[str, dict]) -> Dict[str, str]:
    if not summary:
        return {"best_by_primal": "", "best_by_gap": "", "best_by_runtime": ""}

    candidates = list(summary.keys())
    by_primal = min(candidates, key=lambda k: summary[k]["avg_primal"])
    by_gap = min(candidates, key=lambda k: summary[k]["avg_gap"])
    by_runtime = min(candidates, key=lambda k: summary[k]["avg_runtime_sec"])
    return {
        "best_by_primal": by_primal,
        "best_by_gap": by_gap,
        "best_by_runtime": by_runtime,
    }


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze demand-structure experiment report")
    parser.add_argument("--report-path", required=True, help="path to demand_experiment_report.json")
    parser.add_argument("--baseline", default="uniform", help="baseline distribution for delta comparison")
    parser.add_argument("--out-dir", default="", help="output directory for analysis files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_path = Path(args.report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {report_path}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("report has no runs data")

    summary = _summarize_runs(runs)
    relative = _relative_to_baseline(summary, baseline=args.baseline)
    best = _pick_best(summary)

    out_dir = Path(args.out_dir) if args.out_dir else report_path.parent
    analysis = {
        "batch_id": payload.get("batch_id", ""),
        "source_report": str(report_path),
        "baseline": args.baseline,
        "summary_by_distribution": summary,
        "relative_to_baseline": relative,
        "best": best,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = out_dir / "demand_experiment_analysis.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = [{"distribution": k, **v} for k, v in summary.items()]
    _write_csv(
        out_dir / "demand_experiment_analysis_summary.csv",
        summary_rows,
        fieldnames=[
            "distribution",
            "runs",
            "solved_like",
            "solve_rate",
            "avg_primal",
            "median_primal",
            "std_primal",
            "avg_gap",
            "median_gap",
            "std_gap",
            "avg_runtime_sec",
            "median_runtime_sec",
            "std_runtime_sec",
            "avg_nodes",
        ],
    )

    relative_rows = [{"distribution": k, **v} for k, v in relative.items()]
    if relative_rows:
        _write_csv(
            out_dir / "demand_experiment_analysis_delta.csv",
            relative_rows,
            fieldnames=[
                "distribution",
                "delta_avg_primal",
                "delta_avg_gap",
                "delta_avg_runtime_sec",
                "delta_solve_rate",
            ],
        )

    print(
        json.dumps(
            {
                "analysis_path": str(analysis_path),
                "best": best,
                "summary_by_distribution": summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
