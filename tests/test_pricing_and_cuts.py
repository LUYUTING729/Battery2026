import unittest

from bpc.core.types import BranchRule, Customer, Depot, DualValues, InstanceData, NodeState, SolverConfig, Vehicle
from bpc.cuts.separators import separate_cuts
from bpc.pricing.initializer import generate_initial_columns
from bpc.pricing.ng_pricing import price_columns


def _instance():
    customers = {
        "c1": Customer("c1", 2.0),
        "c2": Customer("c2", 2.0),
    }
    depots = {"d1": Depot("d1", 2)}
    vehicles = {"v1": Vehicle("v1", "o1")}
    cost = {
        ("o1", "d1"): 1,
        ("d1", "c1"): 1,
        ("c1", "d1"): 1,
        ("d1", "c2"): 1,
        ("c2", "d1"): 1,
        ("c1", "c2"): 1,
        ("c2", "c1"): 1,
    }
    dist = cost.copy()
    arcs = set(cost.keys())
    return InstanceData(
        instance_id="inst",
        customers=customers,
        depots=depots,
        vehicles=vehicles,
        demand={"c1": 2.0, "c2": 2.0},
        capacity_u=5.0,
        range_q=10.0,
        cost=cost,
        dist=dist,
        dispatch_cost={("v1", "d1"): 1},
        arcs=arcs,
    )


class TestPricingAndCuts(unittest.TestCase):
    def test_generate_initial_columns(self):
        ins = _instance()
        cols = generate_initial_columns(ins)
        self.assertTrue(len(cols) >= 2)

    def test_price_columns_negative(self):
        ins = _instance()
        node = NodeState(node_id=0, depth=0, branch_rule=BranchRule())
        duals = DualValues(
            cover_pi={"c1": 3.0, "c2": 3.0},
            vehicle_alpha={"v1": 0.0},
            depot_beta={"d1": 0.0},
            branch_arc_dual={},
            clique_customer_bonus={"c1": 0.0, "c2": 0.0},
        )
        cfg = SolverConfig(max_new_columns_per_iter=5)
        cols = price_columns(ins, node, duals, cfg)
        self.assertTrue(len(cols) >= 1)

    def test_cut_separation_runs(self):
        ins = _instance()
        cols = generate_initial_columns(ins)
        active = [(cols[0], 0.6), (cols[1], 0.6)]
        cfg = SolverConfig(max_cuts_per_round=5)
        cuts = separate_cuts(ins, active, cfg, at_root=True, round_id=1)
        self.assertIsInstance(cuts, list)


if __name__ == "__main__":
    unittest.main()
