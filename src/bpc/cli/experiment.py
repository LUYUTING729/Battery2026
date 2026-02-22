from __future__ import annotations

"""批量实验 CLI。"""

import argparse
import json

from bpc.search.solver import run_experiment


def main() -> None:
    """读取参数并运行 run_experiment。"""
    parser = argparse.ArgumentParser(description="Run batch experiment for MDVRP-RL BCP")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--db-path", default="instances.db")
    parser.add_argument("--csv-dir", default="")
    args = parser.parse_args()

    report = run_experiment(args.batch_id, args.config, db_path=args.db_path, csv_dir=args.csv_dir)
    print(json.dumps(report.summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
