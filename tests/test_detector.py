"""Check leakage guards, calibration semantics, causality, and model persistence."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectradrift.data import TEPRun
from spectradrift.detector import decision_flags, fit_detector, load_detector, save_detector, summarize_scores


def normal_run(run_id, *, source="normal-training", length=100):
    values = np.random.default_rng(run_id).normal(size=(length, 52))
    return TEPRun(source, 0, run_id, np.arange(1, length + 1), values)


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fitting = [normal_run(1), normal_run(2)]
        cls.calibration = [normal_run(3), normal_run(4)]
        cls.detector, cls.report = fit_detector(
            cls.fitting, cls.calibration, window_size=10, reference_size=64,
            variance_retained=0.8, target_flag_rate=0.1,
        )

    def test_or_rule_flags_each_component_and_preserves_threshold_ties(self):
        anomaly, drift, combined = decision_flags([0, 2, 0, 2, 1], [0, 0, 2, 2, 1], 1, 1)
        np.testing.assert_array_equal(anomaly, [False, True, False, True, False])
        np.testing.assert_array_equal(drift, [False, False, True, True, False])
        np.testing.assert_array_equal(combined, [False, True, True, True, False])

    def test_zero_threshold_handles_zero_and_positive_scores(self):
        _, _, flags = decision_flags([0, 1], [0, 0], 0, 0)
        np.testing.assert_array_equal(flags, [False, True])

    def test_misaligned_or_invalid_scores_fail(self):
        for anomaly, drift, a_threshold, d_threshold in (
            ([1], [1, 2], 1, 1), ([np.nan], [0], 1, 1), ([-1], [0], 1, 1),
            ([1], [0], -1, 1), ([1], [0], 1, np.inf),
        ):
            with self.subTest(anomaly=anomaly), self.assertRaises(ValueError):
                decision_flags(anomaly, drift, a_threshold, d_threshold)

    def test_normal_only_and_disjoint_partition_guards(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_detector(self.fitting, self.fitting, window_size=10)
        with self.assertRaisesRegex(ValueError, "normal-training"):
            fit_detector(self.fitting, [normal_run(3, source="normal-testing")])
        faulty = TEPRun("faulty-testing", 1, 3, np.arange(1, 101), self.calibration[0].values)
        with self.assertRaisesRegex(ValueError, "normal-training"):
            fit_detector([faulty], self.calibration)

    def test_calibration_allowance_limits_empirical_or_flags(self):
        summary = self.report["calibration"]
        self.assertLessEqual(summary["combined_flag_rate"], 0.1)
        self.assertEqual(summary["eligible_windows"], 182)
        self.assertEqual(self.report["component_quantile"], 0.95)
        self.assertEqual(self.report["quantile_method"], "higher")

    def test_calibration_shift_cannot_change_fitted_pca_or_ks_reference(self):
        shifted = [TEPRun(run.source, 0, run.simulation_run, run.samples, run.values + 100) for run in self.calibration]
        detector, _ = fit_detector(
            self.fitting, shifted, window_size=10, reference_size=64,
            variance_retained=0.8, target_flag_rate=0.1,
        )
        np.testing.assert_array_equal(detector.anomaly.components, self.detector.anomaly.components)
        np.testing.assert_array_equal(detector.drift.sorted_reference, self.detector.drift.sorted_reference)
        np.testing.assert_array_equal(detector.normalizer.mean, self.detector.normalizer.mean)
        self.assertGreater(detector.anomaly_threshold, self.detector.anomaly_threshold)

    def test_score_timestamps_are_causal_and_parameters_stay_frozen(self):
        run = normal_run(5, source="normal-testing")
        altered = run.values.copy()
        altered[50:] += 100
        future = TEPRun(run.source, 0, run.simulation_run, run.samples, altered)
        before = self.detector.score_run(run)
        after = self.detector.score_run(future)
        eligible = before.end_samples <= 50
        np.testing.assert_array_equal(before.anomaly_scores[eligible], after.anomaly_scores[eligible])
        np.testing.assert_array_equal(before.drift_scores[eligible], after.drift_scores[eligible])
        np.testing.assert_array_equal(before.flags[eligible], after.flags[eligible])
        self.assertTrue(after.flags[-20:].all())
        np.testing.assert_array_equal(before.end_samples, np.arange(10, 101))

    def test_summary_uses_eligible_windows_and_counts_or_once(self):
        batches = [self.detector.score_run(run) for run in self.calibration]
        summary = summarize_scores(batches)
        self.assertEqual(summary["eligible_windows"], 182)
        self.assertEqual(summary["combined_flagged_windows"], sum(int(batch.flags.sum()) for batch in batches))
        self.assertEqual(summary["combined_flag_rate"], summary["combined_flagged_windows"] / 182)

    def test_npz_and_json_roundtrip_preserves_both_scores_and_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            record = save_detector(self.detector, folder, self.report)
            restored = load_detector(folder)
            run = normal_run(5, source="normal-testing")
            original = self.detector.score_run(run)
            replay = restored.score_run(run)
            np.testing.assert_array_equal(original.anomaly_scores, replay.anomaly_scores)
            np.testing.assert_array_equal(original.drift_scores, replay.drift_scores)
            np.testing.assert_array_equal(original.flags, replay.flags)
            self.assertEqual(record["decision_rule"], "anomaly_flag OR drift_flag")
            with np.load(Path(folder) / "model.npz", allow_pickle=False) as arrays:
                self.assertTrue(all(array.dtype.kind != "O" for array in arrays.values()))
            with self.assertRaisesRegex(ValueError, "already exist"):
                save_detector(self.detector, folder, self.report)

    def test_corrupt_model_and_reordered_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            save_detector(self.detector, folder, self.report)
            path = Path(folder) / "manifest.json"
            record = json.loads(path.read_text())
            record["feature_names"].reverse()
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "order"):
                load_detector(folder)
            record["feature_names"].reverse()
            path.write_text(json.dumps(record))
            (Path(folder) / "model.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_detector(folder)


if __name__ == "__main__":
    unittest.main()
