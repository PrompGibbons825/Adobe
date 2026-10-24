"""Unit tests for tools/gate/sealed.py, on a throwaway store (never the real one)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.gate import sealed


def _stage(d: Path, n: int = 3) -> Path:
    (d / "items").mkdir(parents=True)
    (d / "assets").mkdir()
    (d / "assets" / "a.png").write_bytes(b"png")
    for i in range(n):
        t = {"task_id": f"i{i:03d}", "target_canvas": {"width": 728, "height": 90},
             "source": {"canvas": {"width": 100, "height": 100},
                        "elements": [{"id": "e0", "type": "image", "asset_id": "a.png"}]}}
        (d / "items" / f"i{i:03d}.json").write_text(json.dumps(t))
    (d / "exclude.json").write_text(json.dumps(["h1", "h2"]))
    return d


class SealedStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = {"CHAMELEON_SEALED_ROOT": str(self.tmp / "store"),
               "CHAMELEON_GATE_MANIFESTS": str(self.tmp / "manifests"),
               "CHAMELEON_GATE_DEV": str(self.tmp / "dev")}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)

    def test_add_moves_in_and_locks_down(self):
        st = _stage(self.tmp / "stage")
        r = sealed.add(st, "toy")
        self.assertEqual(r, {"version": "toy-v1", "items": 3, "assets": 1, "files": 5})
        self.assertFalse(st.exists())
        vd = self.tmp / "store" / "toy" / "v1"
        for p in (self.tmp / "store", self.tmp / "store" / "toy", vd, vd / "items"):
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o700, p)
        self.assertTrue((self.tmp / "manifests" / "toy-v1.sha256").exists())
        self.assertEqual((vd / "manifest.sha256").read_text(), (self.tmp / "manifests" / "toy-v1.sha256").read_text())

    def test_versions_count_up_and_list_without_names(self):
        sealed.add(_stage(self.tmp / "s1"), "toy")
        sealed.add(_stage(self.tmp / "s2", 5), "toy")
        vs = sealed.list_versions()
        self.assertEqual([v["version"] for v in vs], ["toy-v1", "toy-v2"])
        self.assertEqual(vs[1]["items"], 5)
        self.assertNotIn("i000", json.dumps(vs))
        self.assertEqual(sealed.latest("toy"), "toy-v2")

    def test_verify_catches_tampering(self):
        sealed.add(_stage(self.tmp / "s"), "toy")
        self.assertTrue(sealed.verify("toy-v1")["ok"])
        f = self.tmp / "store" / "toy" / "v1" / "items" / "i001.json"
        f.write_text("{}")
        r = sealed.verify("toy-v1")
        self.assertFalse(r["ok"])
        self.assertEqual(r["committed"], "changed 1 missing 0 extra 0")

    def test_aggregate_returns_only_aggregates(self):
        sealed.add(_stage(self.tmp / "s", 4), "toy")

        def metric(task, assets):
            if task["task_id"] == "i003":
                raise RuntimeError("boom")
            self.assertTrue((assets / "a.png").exists())
            return {"w": task["target_canvas"]["width"]}

        r = sealed.aggregate("toy-v1", metric)
        self.assertEqual((r["n"], r["errors"]), (4, 1))
        self.assertEqual(r["metrics"]["w"]["mean"], 728)
        self.assertNotIn("i00", json.dumps(r))

    def test_retire_moves_to_dev_and_never_reuses_number(self):
        sealed.add(_stage(self.tmp / "s"), "toy")
        self.assertEqual(sealed.exclusions("toy"), {"h1", "h2"})
        sealed.retire("toy-v1", "peeked")
        self.assertFalse((self.tmp / "store" / "toy" / "v1").exists())
        self.assertTrue((self.tmp / "dev" / "toy-v1" / "items" / "i000.json").exists())
        self.assertEqual(sealed.sealed_versions("toy"), [])
        self.assertEqual(sealed.exclusions("toy"), {"h1", "h2"})
        self.assertEqual(sealed.add(_stage(self.tmp / "s2"), "toy")["version"], "toy-v2")
        statuses = {v["version"]: v["status"] for v in sealed.list_versions()}
        self.assertEqual(statuses, {"toy-v1": "retired", "toy-v2": "sealed"})

    def test_rejects_bad_staging(self):
        (self.tmp / "empty").mkdir()
        with self.assertRaises(ValueError):
            sealed.add(self.tmp / "empty", "toy")
        st = _stage(self.tmp / "s")
        (st / "assets" / "a.png").unlink()
        with self.assertRaises(ValueError):
            sealed.add(st, "toy")
        self.assertTrue(st.exists())  # a refused staging folder is left alone
        with self.assertRaises(ValueError):
            sealed.add(_stage(self.tmp / "s3"), "bad-name")


class DevSetsTest(unittest.TestCase):
    def test_registry_shapes(self):
        from tools.gate import devsets

        reg = devsets.registry()
        for name in ("samples", "cases", "crello_old", "finicky", "designs"):
            self.assertIn(name, reg)
        ts, assets = devsets.tasks("samples")
        self.assertEqual(len(ts), 3)
        self.assertTrue(assets.is_dir())
        for t in ts + devsets.tasks("finicky")[0][:5]:
            self.assertTrue({"task_id", "source", "target_canvas"} <= set(t))


if __name__ == "__main__":
    unittest.main()
