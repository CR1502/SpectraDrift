"""Offline fixtures test data integrity and simulation identity, not TEP performance."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pandas as pd
import pyreadr

from spectradrift.data import (
    FileSpec, SENSOR_COLUMNS, TEPRun, download_file, load_manifest, load_runs,
    runs_from_frame, verify_file,
)


def frame_for(source="normal-training", run_ids=(1, 2), faults=(0,)):
    length = 500 if source == "normal-training" else 960
    records = []
    for fault in faults:
        for run in run_ids:
            block = pd.DataFrame(np.full((length, 52), run + fault, dtype=float), columns=SENSOR_COLUMNS)
            block["faultNumber"] = fault
            block["simulationRun"] = run
            block["sample"] = np.arange(1, length + 1)
            records.append(block)
    return pd.concat(records, ignore_index=True)


def spec_for(payload, filename="fixture.RData"):
    return FileSpec("normal-training", filename, "fault_free_training", len(payload),
                    hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                    "https://dataverse.harvard.edu/api/access/datafile/fixture")


class DataTests(unittest.TestCase):
    def test_publisher_manifest_has_expected_sources(self):
        specs = load_manifest()
        self.assertEqual(set(specs), {"normal-training", "normal-testing", "faulty-testing"})
        self.assertEqual(specs["faulty-testing"].samples_per_run, 960)
        self.assertEqual(len(specs["faulty-testing"].allowed_faults), 20)

    def test_reorders_rows_and_columns_without_mixing_runs(self):
        frame = frame_for().sample(frac=1, random_state=4)
        frame = frame.loc[:, list(reversed(frame.columns))]
        runs = runs_from_frame(frame, "normal-training", run_ids=[1, 2])
        self.assertEqual([run.key for run in runs], [("normal-training", 0, 1), ("normal-training", 0, 2)])
        np.testing.assert_array_equal(runs[0].samples, np.arange(1, 501))
        np.testing.assert_array_equal(runs[0].values, np.ones((500, 52)))
        self.assertFalse(runs[0].values.flags.writeable)

    def test_source_is_part_of_identity(self):
        train = runs_from_frame(frame_for(run_ids=(1,)), "normal-training", run_ids=[1])[0]
        test = runs_from_frame(frame_for("normal-testing", (1,)), "normal-testing", run_ids=[1])[0]
        self.assertNotEqual(train.key, test.key)

    def test_fault_number_is_a_condition_not_a_per_row_onset_label(self):
        frame = frame_for("faulty-testing", (1,), (1, 2))
        runs = runs_from_frame(frame, "faulty-testing", run_ids=[1], fault_ids=[1, 2])
        self.assertEqual([run.fault_number for run in runs], [1, 2])
        self.assertEqual(len(runs[0].samples), 960)

    def test_missing_requested_simulation_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing requested"):
            runs_from_frame(frame_for(run_ids=(1,)), "normal-training", run_ids=[1, 2])

    def test_incomplete_run_fails(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runs_from_frame(frame_for(run_ids=(1,)).iloc[:-1], "normal-training", run_ids=[1])

    def test_duplicate_sample_and_gap_fail_even_with_correct_row_count(self):
        frame = frame_for(run_ids=(1,))
        frame.loc[0, "sample"] = 2
        with self.assertRaisesRegex(ValueError, "consecutive"):
            runs_from_frame(frame, "normal-training", run_ids=[1])

    def test_fractional_metadata_fails(self):
        frame = frame_for(run_ids=(1,))
        frame["sample"] = frame["sample"].astype(float)
        frame.loc[0, "sample"] = 1.5
        with self.assertRaisesRegex(ValueError, "integers"):
            runs_from_frame(frame, "normal-training", run_ids=[1])

    def test_nonfinite_sensor_fails(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                frame = frame_for(run_ids=(1,))
                frame.loc[0, "xmeas_1"] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    runs_from_frame(frame, "normal-training", run_ids=[1])

    def test_wrong_schema_and_nonnumeric_sensor_fail(self):
        for frame in (frame_for().drop(columns="xmv_11"), frame_for().assign(xmv_11="bad")):
            with self.subTest(columns=len(frame.columns)), self.assertRaises(ValueError):
                runs_from_frame(frame, "normal-training", run_ids=[1, 2])

    def test_bad_selection_fails(self):
        for ids in ([], [0], [501], [1, 1], [1.5], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                runs_from_frame(frame_for(), "normal-training", run_ids=ids)

    def test_faults_cannot_be_in_normal_source(self):
        with self.assertRaisesRegex(ValueError, "unexpected fault"):
            runs_from_frame(frame_for(faults=(1,)), "normal-training", run_ids=[1, 2])

    def test_teprun_rejects_invalid_shapes_and_samples(self):
        with self.assertRaises(ValueError):
            TEPRun("normal-training", 0, 1, np.arange(1, 5), np.ones((4, 51)))
        with self.assertRaises(ValueError):
            TEPRun("normal-training", 0, 1, np.array([1, 3]), np.ones((2, 52)))

    def test_rdata_roundtrip_loads_real_parser_and_verifies_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.RData"
            pyreadr.write_rdata(str(path), frame_for(run_ids=(1,)), df_name="fault_free_training")
            spec = spec_for(path.read_bytes())
            runs = load_runs(path, spec, run_ids=[1])
            self.assertEqual(runs[0].values.shape, (500, 52))
            path.write_bytes(b"not RData")
            with self.assertRaisesRegex(ValueError, "byte size"):
                load_runs(path, spec, run_ids=[1])

    def test_checksum_catches_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.RData"
            path.write_bytes(b"abcd")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_file(path, spec_for(b"wxyz"))

    def test_download_verifies_then_reuses_existing_file(self):
        payload = b"offline transfer fixture"
        with tempfile.TemporaryDirectory() as folder, patch("spectradrift.data.urlopen", return_value=io.BytesIO(payload)) as request:
            spec = spec_for(payload)
            path = download_file(spec, folder)
            self.assertEqual(path.read_bytes(), payload)
            download_file(spec, folder)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_failed_download_cleans_its_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder, patch("spectradrift.data.urlopen", return_value=io.BytesIO(b"bad")):
            with self.assertRaises(ValueError):
                download_file(spec_for(b"good"), folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_invalid_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch("spectradrift.data.urlopen") as request:
            path = Path(folder) / "fixture.RData"
            path.write_bytes(b"user file")
            with self.assertRaises(ValueError):
                download_file(spec_for(b"good"), folder)
            self.assertEqual(path.read_bytes(), b"user file")
            request.assert_not_called()

    def test_transient_network_error_is_retried(self):
        payload = b"retry fixture"
        with tempfile.TemporaryDirectory() as folder, patch("spectradrift.data.time.sleep"), patch(
            "spectradrift.data.urlopen", side_effect=[URLError("temporary"), io.BytesIO(payload)]
        ) as request:
            path = download_file(spec_for(payload), folder)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
