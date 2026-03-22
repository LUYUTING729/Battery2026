import json
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
