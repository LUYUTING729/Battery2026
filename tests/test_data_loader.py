import sqlite3
import tempfile
import unittest
from pathlib import Path

from bpc.data.loader import load_instance


class TestDataLoader(unittest.TestCase):
    def _write_csv(self, path: Path, header: str, rows):
        content = [header] + rows
        path.write_text("\n".join(content) + "\n", encoding="utf-8")

    def _build_minimal_fixture(self, base: Path):
        inst_dir = base / "instA"
        inst_dir.mkdir(parents=True, exist_ok=True)

        self._write_csv(inst_dir / "customers.csv", "id,demand", ["c1,2", "c2,3"])
        self._write_csv(inst_dir / "depots.csv", "id,max_vehicles", ["d1,2"])
        self._write_csv(inst_dir / "vehicles.csv", "id,origin", ["v1,o1", "v2,o1"])

        matrix_rows = [
            "o1,d1,5",
            "d1,c1,2", "c1,d1,2",
            "d1,c2,3", "c2,d1,3",
            "c1,c2,1", "c2,c1,1",
            "o1,c1,4", "o1,c2,4"
        ]
        self._write_csv(inst_dir / "cost_matrix.csv", "from,to,value", matrix_rows)
        self._write_csv(inst_dir / "dist_matrix.csv", "from,to,value", matrix_rows)

        db_path = base / "instances.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE instances (instance_id TEXT PRIMARY KEY, csv_dir TEXT, capacity_u REAL, range_q REAL)"
        )
        conn.execute(
            "INSERT INTO instances(instance_id,csv_dir,capacity_u,range_q) VALUES (?,?,?,?)",
            ("instA", str(inst_dir), 10.0, 20.0),
        )
        conn.commit()
        conn.close()
        return db_path, inst_dir

    def test_load_instance_success(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path, _ = self._build_minimal_fixture(base)
            ins = load_instance("instA", str(db_path))
            self.assertEqual(ins.instance_id, "instA")
            self.assertEqual(len(ins.customers), 2)
            self.assertEqual(len(ins.depots), 1)
            self.assertEqual(len(ins.vehicles), 2)
            self.assertIn(("d1", "c1"), ins.arcs)
            self.assertNotIn(("d1", "d1"), ins.arcs)

    def test_load_instance_missing_col(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path, inst_dir = self._build_minimal_fixture(base)
            (inst_dir / "customers.csv").write_text("id\nc1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_instance("instA", str(db_path))


if __name__ == "__main__":
    unittest.main()
