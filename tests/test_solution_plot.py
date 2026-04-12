import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from bpc.visualization.solution_plot import export_solution_plots


class TestSolutionPlot(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_export_solution_plots_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            solution_path = tmp_path / "solution.json"
            solution_path.write_text(
                json.dumps(
                    {
                        "status": "OPTIMAL",
                        "routes": [
                            {
                                "vehicle_id": "v1",
                                "depot_id": "d1",
                                "customer_seq": ["c1", "c2"],
                                "cost": 1.0,
                                "load": 2.0,
                                "dist": 1.0,
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            manifest = export_solution_plots(
                solution_path=str(solution_path),
                xlsx_path="src/bpc/data/data2.xlsx",
                out_dir=str(tmp_path),
                customer_sheet_name="BSS_data",
                depot_sheet_name="Charging_data",
                vehicle_sheet_name="Vehicle_data",
            )

            self.assertEqual(manifest["status"], "OPTIMAL")
            self.assertEqual(manifest["route_count"], 1)
            self.assertTrue(Path(manifest["overview_plot"]).exists())
            self.assertNotIn("route_plots", manifest)


if __name__ == "__main__":
    unittest.main()
