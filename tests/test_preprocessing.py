"""Leakage, causality, and boundary tests use explicit synthetic fixtures."""

import numpy as np
import unittest

from spectradrift.data import TEPRun
from spectradrift.preprocessing import ChannelNormalizer, FEATURE_NAMES, split_normal_runs, window_features


def make_run(run_id=1, source="normal-training", fault=0, values=None):
    if values is None:
        values = np.tile(np.arange(8, dtype=float)[:, None] + run_id, (1, 52))
    return TEPRun(source, fault, run_id, np.arange(1, len(values) + 1), values)


class PreprocessingTests(unittest.TestCase):
    def test_split_uses_disjoint_whole_runs_and_is_order_independent(self):
        runs = [make_run(index) for index in range(1, 11)]
        fitting, calibration = split_normal_runs(runs, seed=42)
        reverse_fit, reverse_cal = split_normal_runs(list(reversed(runs)), seed=42)
        self.assertEqual(len(fitting), 8)
        self.assertEqual(len(calibration), 2)
        self.assertFalse({run.key for run in fitting} & {run.key for run in calibration})
        self.assertEqual([run.key for run in fitting], [run.key for run in reverse_fit])
        self.assertEqual([run.key for run in calibration], [run.key for run in reverse_cal])

    def test_fitting_and_split_reject_evaluation_and_fault_sources(self):
        for run in (make_run(source="normal-testing"), make_run(source="faulty-testing", fault=1)):
            with self.subTest(source=run.source):
                with self.assertRaisesRegex(ValueError, "normal-training"):
                    ChannelNormalizer.fit([run])
                with self.assertRaisesRegex(ValueError, "normal-training"):
                    split_normal_runs([make_run(), run])

    def test_duplicate_and_empty_fit_runs_fail(self):
        for runs in ([], [make_run(), make_run()]):
            with self.subTest(count=len(runs)), self.assertRaises(ValueError):
                ChannelNormalizer.fit(runs)

    def test_bad_split_fraction_or_too_few_runs_fail(self):
        for fraction in (0, 1, -0.1, np.nan, 0.99):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_normal_runs([make_run(1), make_run(2)], calibration_fraction=fraction)
        with self.assertRaises(ValueError):
            split_normal_runs([make_run()])

    def test_normalization_matches_population_moments(self):
        runs = [make_run(1), make_run(2)]
        raw = np.concatenate([run.values for run in runs])
        normalizer = ChannelNormalizer.fit(runs)
        np.testing.assert_allclose(normalizer.mean, raw.mean(axis=0))
        np.testing.assert_allclose(normalizer.scale, raw.std(axis=0, ddof=0))
        normalized = np.concatenate([normalizer.transform(run) for run in runs])
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-14)

    def test_calibration_shift_does_not_change_fitting_statistics(self):
        fitting = make_run(1)
        calibration = make_run(2, values=fitting.values + 1000)
        normalizer = ChannelNormalizer.fit([fitting])
        before_mean = normalizer.mean.copy()
        before_scale = normalizer.scale.copy()
        transformed = normalizer.transform(calibration)
        self.assertGreater(transformed.mean(), 100)
        np.testing.assert_array_equal(normalizer.mean, before_mean)
        np.testing.assert_array_equal(normalizer.scale, before_scale)

    def test_saved_normalizer_reuses_statistics_and_rejects_sensor_reordering(self):
        normalizer = ChannelNormalizer.fit([make_run()])
        restored = ChannelNormalizer.from_dict(normalizer.as_dict())
        test_run = make_run(source="normal-testing", values=make_run().values + 10)
        np.testing.assert_array_equal(normalizer.transform(test_run), restored.transform(test_run))
        self.assertFalse(restored.mean.flags.writeable)
        record = normalizer.as_dict()
        record["sensor_columns"].reverse()
        with self.assertRaisesRegex(ValueError, "sensor order"):
            ChannelNormalizer.from_dict(record)

    def test_constant_channels_produce_finite_zeros(self):
        run = make_run(values=np.full((8, 52), 7.0))
        normalizer = ChannelNormalizer.fit([run])
        np.testing.assert_array_equal(normalizer.scale, np.ones(52))
        batch = window_features(run, normalizer, window_size=3)
        np.testing.assert_array_equal(batch.features, np.zeros((6, 104)))

    def test_window_features_and_endpoints_have_explicit_expected_values(self):
        values = np.tile(np.array([0., 2., 4., 6.])[:, None], (1, 52))
        run = make_run(values=values)
        normalizer = ChannelNormalizer.fit([run])
        batch = window_features(run, normalizer, window_size=2)
        np.testing.assert_array_equal(batch.end_samples, [2, 3, 4])
        self.assertEqual(batch.sensor_windows.shape, (3, 2, 52))
        np.testing.assert_allclose(batch.features[:, 0], [-2 / np.sqrt(5), 0, 2 / np.sqrt(5)])
        np.testing.assert_allclose(batch.features[:, 52], np.full(3, 1 / np.sqrt(5)))
        self.assertEqual(len(FEATURE_NAMES), 104)
        self.assertEqual(FEATURE_NAMES[0], "mean:xmeas_1")
        self.assertEqual(FEATURE_NAMES[52], "std:xmeas_1")

    def test_future_changes_cannot_modify_earlier_features(self):
        run = make_run()
        normalizer = ChannelNormalizer.fit([run])
        changed = run.values.copy()
        changed[5:] += 1e6
        altered = make_run(values=changed)
        original_batch = window_features(run, normalizer, window_size=3)
        altered_batch = window_features(altered, normalizer, window_size=3)
        np.testing.assert_array_equal(original_batch.features[:3], altered_batch.features[:3])
        self.assertFalse(np.array_equal(original_batch.features[3:], altered_batch.features[3:]))

    def test_windows_never_cross_simulation_boundaries(self):
        left = make_run(1, values=np.zeros((8, 52)))
        right = make_run(2, values=np.full((8, 52), 100.0))
        normalizer = ChannelNormalizer.fit([left, right])
        left_batch = window_features(left, normalizer, window_size=3)
        right_batch = window_features(right, normalizer, window_size=3)
        self.assertEqual(len(left_batch.end_samples) + len(right_batch.end_samples), 12)
        np.testing.assert_array_equal(left_batch.features[:, 52:], np.zeros((6, 52)))
        np.testing.assert_array_equal(right_batch.features[:, 52:], np.zeros((6, 52)))
        self.assertNotEqual(left_batch.run_key, right_batch.run_key)

    def test_stride_keeps_original_sample_timestamps(self):
        run = make_run()
        batch = window_features(run, ChannelNormalizer.fit([run]), window_size=3, stride=2)
        np.testing.assert_array_equal(batch.end_samples, [3, 5, 7])
        self.assertEqual(batch.features.shape, (3, 104))

    def test_invalid_window_and_stride_fail(self):
        run = make_run()
        normalizer = ChannelNormalizer.fit([run])
        for size in (0, 1, 9, True, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                window_features(run, normalizer, window_size=size)
        for stride in (0, -1, True, 1.5):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                window_features(run, normalizer, stride=stride, window_size=3)


if __name__ == "__main__":
    unittest.main()
