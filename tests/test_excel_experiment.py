import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bpc.core.types import SolveResult
from bpc.search.solver import run_excel_experiment


class TestExcelExperiment(unittest.TestCase):
    def test_run_excel_experiment_repeat_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg_path = base / "cfg.json"
            out_dir = base / "out"
            cfg_path.write_text(
                json.dumps(
                    {
                        "output_dir": str(out_dir),
                        "time_limit_s": 1.0,
                        "max_nodes": 1,
                    }
                ),
                encoding="utf-8",
            )

            fake_result = SolveResult(
                status="TIME_LIMIT",
                obj_primal=123.0,
                obj_dual=120.0,
                gap=0.02,
                routes=[],
                stats={"runtime_sec": 0.1},
            )

            with patch("bpc.search.solver.solve_instance", return_value=fake_result) as mocked:
                report = run_excel_experiment(
                    batch_id="b1",
                    cfg_path=str(cfg_path),
                    excel_path="src/bpc/data/stations.xlsx",
                    repeat=2,
                    instance_prefix="stations",
                )

            self.assertEqual(len(report.results), 2)
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(report.summary["instances"], 2.0)
            self.assertEqual(report.summary["solved_like"], 2.0)

            report_json = out_dir / "b1" / "report.json"
            self.assertTrue(report_json.exists())
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["repeat"], 2)
            self.assertEqual(payload["excel_path"], "src/bpc/data/stations.xlsx")


if __name__ == "__main__":
    unittest.main()
