import tempfile
import unittest
from pathlib import Path

from bpc.core.types import SolverConfig
from gurobi.solver import GurobiDirectSolverUnavailableError, solve_instance


class TestGurobiDirectSolve(unittest.TestCase):
    def test_tiny_instance_solves(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SolverConfig(time_limit_s=30.0, mip_gap=1e-4, output_dir=td)
            try:
                result = solve_instance(
                    instance_id="inst_tiny",
                    cfg=cfg,
                    db_path="examples/minimal/instances.db",
                )
            except GurobiDirectSolverUnavailableError:
                self.skipTest("gurobipy unavailable")

            self.assertIn(result.status, {"OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"})
            served = [cid for route in result.routes for cid in route.customer_seq]
            self.assertEqual(sorted(served), ["c1", "c2", "c3"])
            self.assertTrue((Path(td) / "solution.json").exists())
            self.assertTrue((Path(td) / "routes.csv").exists())


if __name__ == "__main__":
    unittest.main()
