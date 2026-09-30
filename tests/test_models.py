"""Independent numerical checks of the reconstruction and KS statistics."""

import unittest

import numpy as np
from scipy.stats import ks_2samp

from spectradrift.models import KSDriftScorer, PCAReconstructionScorer


class PCATests(unittest.TestCase):
    def test_reconstructs_rank_one_data_and_scores_orthogonal_deviation(self):
        x = np.linspace(-3, 3, 101)
        features = np.column_stack([x, 2 * x, -x])
        scorer = PCAReconstructionScorer.fit(features, variance_retained=0.95)
        self.assertEqual(scorer.components.shape, (1, 3))
        np.testing.assert_allclose(scorer.score(features), 0, atol=1e-28)
        normal = np.array([[0., 0., 0.]])
        shifted = np.array([[0., 10., 0.]])
        self.assertGreater(scorer.score(shifted)[0], scorer.score(normal)[0] + 1)

    def test_feature_standardization_does_not_refit_at_inference(self):
        rng = np.random.default_rng(2)
        training = rng.normal(size=(200, 4)) * [1, 10, 100, 1000]
        scorer = PCAReconstructionScorer.fit(training, variance_retained=0.6)
        mean = scorer.feature_mean.copy()
        components = scorer.components.copy()
        scorer.score(training + 10000)
        np.testing.assert_array_equal(scorer.feature_mean, mean)
        np.testing.assert_array_equal(scorer.components, components)
        self.assertFalse(scorer.components.flags.writeable)

    def test_constant_feature_is_safe_and_new_shift_scores_positive(self):
        x = np.arange(30, dtype=float)
        training = np.column_stack([x, x, np.full(30, 5.)])
        scorer = PCAReconstructionScorer.fit(training)
        self.assertEqual(scorer.feature_scale[2], 1)
        changed = training[:1].copy()
        changed[:, 2] += 10
        self.assertGreater(scorer.score(changed)[0], 20)

    def test_bad_training_and_inference_inputs_fail(self):
        for values in (np.ones((20, 3)), np.ones((1, 3)), np.empty((0, 3)), np.full((4, 3), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                PCAReconstructionScorer.fit(values)
        training = np.arange(40).reshape(20, 2)
        for retention in (0, 1, -1, np.nan):
            with self.subTest(retention=retention), self.assertRaises(ValueError):
                PCAReconstructionScorer.fit(training, variance_retained=retention)
        scorer = PCAReconstructionScorer.fit(training)
        with self.assertRaises(ValueError):
            scorer.score(np.ones((4, 3)))


class KSTests(unittest.TestCase):
    def test_continuous_statistics_match_scipy_in_every_channel(self):
        rng = np.random.default_rng(9)
        reference = rng.normal(size=(37, 52))
        windows = rng.normal(size=(5, 13, 52))
        scorer = KSDriftScorer(np.sort(reference, axis=0))
        actual = scorer.statistics(windows)
        expected = np.array([
            [ks_2samp(reference[:, channel], window[:, channel], method="asymp").statistic
             for channel in range(52)] for window in windows
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(scorer.score(windows), actual.max(axis=1))

    def test_tied_discrete_statistics_match_scipy(self):
        rng = np.random.default_rng(8)
        reference = rng.integers(0, 4, size=(21, 52))
        windows = rng.integers(0, 4, size=(5, 9, 52))
        actual = KSDriftScorer(np.sort(reference, axis=0)).statistics(windows)
        expected = np.array([
            [ks_2samp(reference[:, channel], window[:, channel], method="asymp").statistic
             for channel in range(52)] for window in windows
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)

    def test_identical_constant_distributions_are_zero_and_disjoint_are_one(self):
        scorer = KSDriftScorer(np.zeros((10, 52)))
        np.testing.assert_array_equal(scorer.score(np.zeros((3, 4, 52))), np.zeros(3))
        np.testing.assert_array_equal(scorer.score(np.ones((3, 4, 52))), np.ones(3))

    def test_one_sensor_shift_is_detected_independently(self):
        reference = np.tile(np.arange(20.)[:, None], (1, 52))
        scorer = KSDriftScorer(reference)
        window = reference[None].copy()
        window[:, :, 7] += 100
        statistics = scorer.statistics(window)
        self.assertEqual(statistics[0, 7], 1)
        np.testing.assert_array_equal(np.delete(statistics, 7, axis=1), np.zeros((1, 51)))
        self.assertEqual(scorer.score(window)[0], 1)

    def test_reference_sampling_is_reproducible_and_frozen(self):
        candidates = np.random.default_rng(2).normal(size=(100, 52))
        left = KSDriftScorer.fit(candidates, reference_size=20, seed=5)
        right = KSDriftScorer.fit(candidates, reference_size=20, seed=5)
        np.testing.assert_array_equal(left.sorted_reference, right.sorted_reference)
        before = left.sorted_reference.copy()
        left.score(np.full((2, 10, 52), 1e6))
        np.testing.assert_array_equal(left.sorted_reference, before)
        self.assertFalse(left.sorted_reference.flags.writeable)

    def test_reference_size_is_capped_at_available_normal_samples(self):
        scorer = KSDriftScorer.fit(np.ones((10, 52)), reference_size=512)
        self.assertEqual(scorer.sorted_reference.shape, (10, 52))

    def test_bad_reference_or_sensor_windows_fail(self):
        with self.assertRaises(ValueError):
            KSDriftScorer(np.tile([1., 0.], (52, 1)).T)
        with self.assertRaises(ValueError):
            KSDriftScorer.fit(np.ones((3, 51)))
        scorer = KSDriftScorer(np.zeros((3, 52)))
        for windows in (np.ones((3, 52)), np.ones((1, 1, 52)), np.full((2, 3, 52), np.nan)):
            with self.subTest(shape=windows.shape), self.assertRaises(ValueError):
                scorer.score(windows)


if __name__ == "__main__":
    unittest.main()
