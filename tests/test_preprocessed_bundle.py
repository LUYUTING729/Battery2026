import json
import tempfile
import unittest
import warnings
from pathlib import Path

from bpc.data.xlsx_loader import (
    load_instance_bundle_from_excel,
    load_preprocessed_bundle,
    preprocess_excel_to_file,
)


class TestPreprocessedBundle(unittest.TestCase):
    def test_preprocess_and_reload(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "stations_preprocessed.json"
            preprocess_excel_to_file(
                xlsx_path="src/bpc/data/stations.xlsx",
                out_path=str(out),
                instance_id="stations_cached",
            )
            self.assertTrue(out.exists())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "bpc_preprocessed_instance_v1")

            bundle = load_preprocessed_bundle(str(out))
            ins = bundle.instance
            self.assertEqual(ins.instance_id, "stations_cached")
            self.assertGreater(len(ins.customers), 0)
            self.assertGreater(len(ins.depots), 0)
            self.assertGreater(len(ins.vehicles), 0)
            self.assertIn(("c1", "d1"), ins.cost)
            self.assertIn(("c1", "d1"), ins.dist)

    def test_override_vehicle_count(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "stations_preprocessed_v9.json"
            preprocess_excel_to_file(
                xlsx_path="src/bpc/data/stations.xlsx",
                out_path=str(out),
                instance_id="stations_cached_v9",
                override_vehicle_count=9,
            )
            bundle = load_preprocessed_bundle(str(out))
            self.assertEqual(len(bundle.instance.vehicles), 9)
            self.assertEqual(bundle.model_profile["resolved_values"]["vehicle_count"], 9)

    def test_override_vehicle_count_infeasible_raises(self):
        with self.assertRaises(ValueError):
            load_instance_bundle_from_excel(
                xlsx_path="src/bpc/data/stations.xlsx",
                instance_id="stations_bad_v5",
                override_vehicle_count=5,
            )

    def test_override_vehicle_count_warns_when_vehicle_sheet_rows_differ(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = load_instance_bundle_from_excel(
                xlsx_path="src/bpc/data/data2.xlsx",
                instance_id="data2_v10",
                vehicle_sheet_name="Vehicle_data",
                override_vehicle_count=10,
            )

        self.assertEqual(len(bundle.instance.vehicles), 10)
        self.assertEqual(bundle.model_profile["resolved_values"]["explicit_vehicle_rows"], 5)
        self.assertEqual(bundle.model_profile["missing_data_filled"]["synthetic_vehicles_added"], 5)
        self.assertTrue(bundle.model_profile["warnings"])
        self.assertTrue(any("override_vehicle_count differs from Vehicle_data row count" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
