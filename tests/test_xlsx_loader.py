import unittest
from pathlib import Path

from bpc.data.xlsx_loader import load_instance_bundle_from_excel


class TestXlsxLoader(unittest.TestCase):
    def test_load_bundle_from_stations_xlsx(self):
        xlsx = Path("src/bpc/data/stations.xlsx")
        bundle = load_instance_bundle_from_excel(str(xlsx), instance_id="xlsx_inst")
        ins = bundle.instance

        self.assertGreater(len(ins.customers), 0)
        self.assertGreater(len(ins.depots), 0)
        self.assertGreater(len(ins.vehicles), 0)

        c1 = "c1"
        d1 = "d1"
        self.assertIn((c1, d1), ins.dist)
        self.assertIn((d1, c1), ins.dist)
        self.assertGreater(ins.dist[(c1, d1)], 0.0)
        self.assertAlmostEqual(ins.dist[(c1, d1)], ins.dist[(d1, c1)], places=9)

        # sheet3 中 c=1，因此 cost 与 dist 数值一致。
        self.assertAlmostEqual(ins.cost[(c1, d1)], ins.dist[(c1, d1)], places=9)

        # depot->depot 弧应被过滤掉。
        depots = list(ins.depots.keys())
        if len(depots) >= 2:
            self.assertNotIn((depots[0], depots[1]), ins.arcs)

        profile = bundle.model_profile
        self.assertEqual(profile["resolved_values"]["capacity_u"], 200.0)
        self.assertEqual(profile["resolved_values"]["range_q"], 500.0)
        self.assertEqual(profile["resolved_values"]["vehicle_count"], 20)
        self.assertTrue(profile["missing_data_filled"]["depot_max_vehicles"])

    def test_load_bundle_from_data2_named_sheets(self):
        xlsx = Path("src/bpc/data/data2.xlsx")
        bundle = load_instance_bundle_from_excel(str(xlsx), instance_id="xlsx_data2")
        ins = bundle.instance

        self.assertGreater(len(ins.customers), 0)
        self.assertGreater(len(ins.depots), 0)
        self.assertGreater(len(ins.vehicles), 0)

        profile = bundle.model_profile
        self.assertEqual(profile["sheet_mapping"]["customers_sheet"], "BSS_data")
        self.assertEqual(profile["sheet_mapping"]["depots_sheet"], "Charging_data")
        self.assertEqual(profile["sheet_mapping"]["vehicles_sheet"], "Vehicle_data")
        self.assertTrue(profile["missing_data_filled"]["vehicle_sheet_used"])


if __name__ == "__main__":
    unittest.main()
