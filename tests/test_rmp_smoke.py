import unittest

from bpc.core.types import BranchRule, Customer, Depot, InstanceData, NodeState, Vehicle
from bpc.pricing.initializer import generate_initial_columns
from bpc.rmp.master_problem import GurobiUnavailableError, MasterProblem


class TestRMPSmoke(unittest.TestCase):
    def test_rmp_build_or_skip(self):
        instance = InstanceData(
            instance_id="x",
            customers={"c1": Customer("c1", 1.0)},
            depots={"d1": Depot("d1", 1)},
            vehicles={"v1": Vehicle("v1", "o1")},
            demand={"c1": 1.0},
            capacity_u=5,
            range_q=10,
            cost={("o1", "d1"): 1, ("d1", "c1"): 1, ("c1", "d1"): 1},
            dist={("o1", "d1"): 1, ("d1", "c1"): 1, ("c1", "d1"): 1},
            dispatch_cost={("v1", "d1"): 1},
            arcs={("d1", "c1"), ("c1", "d1"), ("o1", "d1")},
        )
        node = NodeState(node_id=0, depth=0, branch_rule=BranchRule())
        try:
            rmp = MasterProblem(instance, node, relax=True)
        except GurobiUnavailableError:
            self.skipTest("gurobipy unavailable")
            return
        cols = generate_initial_columns(instance)
        self.assertGreaterEqual(rmp.add_columns(cols), 1)
        info = rmp.solve()
        self.assertTrue(info.obj_val >= 0)


if __name__ == "__main__":
    unittest.main()
