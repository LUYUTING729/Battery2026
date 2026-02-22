import unittest

from bpc.core.route_utils import build_ai, route_feasible
from bpc.core.types import Customer, Depot, InstanceData, RouteColumn, Vehicle


class TestRouteUtils(unittest.TestCase):
    def test_route_feasible(self):
        instance = InstanceData(
            instance_id="x",
            customers={"c1": Customer("c1", 2.0)},
            depots={"d1": Depot("d1", 2)},
            vehicles={"v1": Vehicle("v1", "o1")},
            demand={"c1": 2.0},
            capacity_u=10.0,
            range_q=20.0,
            cost={("d1", "c1"): 1, ("c1", "d1"): 1, ("o1", "d1"): 1},
            dist={("d1", "c1"): 1, ("c1", "d1"): 1, ("o1", "d1"): 1},
            dispatch_cost={("v1", "d1"): 1},
            arcs={("d1", "c1"), ("c1", "d1"), ("o1", "d1")},
        )
        route = RouteColumn(
            column_id="r1",
            vehicle_id="v1",
            depot_id="d1",
            customer_seq=("c1",),
            a_i=build_ai(instance.customers.keys(), ("c1",)),
            arc_flags={("d1", "c1"), ("c1", "d1")},
            cost=3,
            load=2,
            dist=2,
        )
        self.assertTrue(route_feasible(route, instance))


if __name__ == "__main__":
    unittest.main()
