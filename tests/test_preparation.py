"""Exercise the entire preparation stage using a synthetic offline RData file."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyreadr

from spectradrift.data import SENSOR_COLUMNS
from spectradrift.preprocessing import prepare_normal_data


class PreparationTests(unittest.TestCase):
    def test_preparation_artifacts_keep_disjoint_runs_and_channel_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = pd.DataFrame(np.tile(np.arange(1000, dtype=float)[:, None], (1, 52)), columns=SENSOR_COLUMNS)
            table["faultNumber"] = 0
            table["simulationRun"] = np.repeat([1, 2], 500)
            table["sample"] = np.tile(np.arange(1, 501), 2)
            path = root / "TEP_FaultFree_Training.RData"
            pyreadr.write_rdata(str(path), table, df_name="fault_free_training")
            manifest = json.loads(Path("data/source.json").read_text())
            record = next(item for item in manifest["files"] if item["r_object"] == "fault_free_training")
            record["size_bytes"] = path.stat().st_size
            record["md5"] = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
            manifest_path = root / "source.json"
            manifest_path.write_text(json.dumps(manifest))
            output = root / "prepared"
            report = prepare_normal_data(
                data_dir=root, manifest_path=manifest_path, run_ids=[1, 2],
                calibration_fraction=0.5, window_size=20, output_dir=output,
            )
            self.assertTrue(report["checksum_verified"])
            self.assertEqual(report["partitions"]["fitting"]["eligible_windows"], 481)
            self.assertEqual(report["partitions"]["calibration"]["eligible_windows"], 481)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), report)
            with np.load(output / "fitting.npz", allow_pickle=False) as fit, np.load(output / "calibration.npz", allow_pickle=False) as cal:
                self.assertEqual(fit["features"].shape, (1, 481, 104))
                self.assertEqual(cal["normalized_values"].shape, (1, 500, 52))
                self.assertFalse(set(fit["run_ids"]) & set(cal["run_ids"]))
                np.testing.assert_array_equal(fit["end_samples"][0], np.arange(20, 501))
                np.testing.assert_allclose(fit["normalized_values"].mean(axis=(0, 1)), 0, atol=1e-14)
                np.testing.assert_allclose(fit["normalized_values"].std(axis=(0, 1)), 1, atol=1e-14)
                self.assertGreater(abs(cal["normalized_values"].mean()), 3)
                self.assertTrue(np.isfinite(cal["features"]).all())
            preserved = (output / "manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already exist"):
                prepare_normal_data(output_dir=output)
            self.assertEqual((output / "manifest.json").read_bytes(), preserved)


if __name__ == "__main__":
    unittest.main()
