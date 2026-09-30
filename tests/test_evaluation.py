"""Hand-counted synthetic flags test metric semantics; no TEP performance claims."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectradrift.detector import ScoreBatch
from spectradrift.evaluation import evaluate_batches, fault_run_record, results_markdown, _save_scores


def batch(*, source="faulty-testing", fault=1, run=1, ends=tuple(range(3, 11)), anomaly=(), drift=()):
    ends = np.array(ends)
    a = np.array([sample in anomaly for sample in ends], dtype=bool)
    d = np.array([sample in drift for sample in ends], dtype=bool)
    return ScoreBatch((source, fault, run), ends, a.astype(float), d.astype(float),
                      np.zeros((len(ends), 52)), a, d, a | d)


def normal(run=1, **kwargs):
    return batch(source="normal-testing", fault=0, run=run, **kwargs)


class EvaluationTests(unittest.TestCase):
    def test_pre_onset_flags_do_not_detect_fault_and_miss_is_null(self):
        record = fault_run_record(batch(anomaly=(3, 4)), onset_sample=5)
        result = record["signals"]["combined"]
        self.assertFalse(result["detected"])
        self.assertIsNone(result["latency_steps"])
        self.assertIsNone(result["first_flag_sample"])
        self.assertEqual(record["observation_horizon_steps"], 5)
        self.assertEqual(record["eligible_post_onset_windows"], 6)

    def test_flag_at_onset_has_zero_latency_and_continuation_is_explicit(self):
        record = fault_run_record(batch(anomaly=(4, 5, 6)), onset_sample=5)
        self.assertEqual(record["signals"]["combined"]["latency_steps"], 0)
        self.assertTrue(record["signals"]["combined"]["continues_pre_onset_flag_episode"])

    def test_latency_uses_sample_coordinates_not_window_array_index(self):
        record = fault_run_record(batch(ends=(3, 6, 9), drift=(9,)), onset_sample=5)
        self.assertEqual(record["first_eligible_post_onset_sample"], 6)
        self.assertEqual(record["signals"]["combined"]["latency_steps"], 4)
        self.assertEqual(record["signals"]["combined"]["latency_minutes"], 12)

    def test_warmup_can_delay_first_eligible_endpoint(self):
        record = fault_run_record(batch(ends=(8, 9, 10), anomaly=(8,)), onset_sample=5)
        self.assertEqual(record["signals"]["combined"]["latency_steps"], 3)

    def test_components_and_or_have_separate_first_flags(self):
        record = fault_run_record(batch(anomaly=(7,), drift=(6,)), onset_sample=5)
        self.assertEqual(record["signals"]["anomaly"]["latency_steps"], 2)
        self.assertEqual(record["signals"]["drift"]["latency_steps"], 1)
        self.assertEqual(record["signals"]["combined"]["latency_steps"], 1)

    def test_mean_median_misses_and_fault_macro_weighting_are_hand_counted(self):
        faults = [batch(run=1, anomaly=(5,)), batch(run=2, anomaly=(7,)), batch(run=3),
                  batch(fault=2, run=1, drift=(9,))]
        report = evaluate_batches([normal()], faults, onset_sample=5, expected_faults=[1, 2])
        overall = report["overall"]["combined"]
        self.assertEqual(overall["detected_simulations"], 3)
        self.assertEqual(overall["undetected_simulations"], 1)
        self.assertEqual(overall["detection_rate"], 0.75)
        self.assertEqual(overall["mean_latency_steps_detected"], 2)
        self.assertEqual(overall["median_latency_steps_detected"], 2)
        self.assertEqual(overall["mean_latency_minutes_detected"], 6)
        macro = report["macro_across_fault_types"]["combined"]
        self.assertEqual(macro["mean_of_fault_mean_latency_steps_detected"], 2.5)
        self.assertEqual(macro["median_of_fault_median_latency_steps_detected"], 2.5)
        self.assertEqual(len(report["per_fault"]["1"]["runs"]), 3)
        self.assertIsNone(report["per_fault"]["1"]["runs"][2]["signals"]["combined"]["latency_steps"])

    def test_false_positive_denominator_uses_eligible_windows_and_or_once(self):
        normals = [normal(anomaly=(3, 4, 10), drift=(4, 7)), normal(run=2)]
        report = evaluate_batches(normals, [batch(anomaly=(5,))], onset_sample=5, expected_faults=[1])
        summary = report["normal_operation"]
        combined = summary["signals"]["combined"]
        self.assertEqual(summary["eligible_windows"], 16)
        self.assertEqual(combined["flagged_windows"], 4)
        self.assertEqual(combined["false_positive_rate"], 0.25)
        self.assertEqual(combined["flag_episodes"], 3)
        self.assertEqual(combined["runs_with_any_flag"], 1)
        self.assertEqual(combined["fraction_runs_with_any_flag"], 0.5)

    def test_prefix_rate_excludes_true_onset_and_all_post_onset_windows(self):
        report = evaluate_batches([normal()], [batch(anomaly=(3, 5, 6))], onset_sample=5, expected_faults=[1])
        prefix = report["fault_normal_prefix"]
        self.assertEqual(prefix["eligible_windows"], 2)
        self.assertEqual(prefix["signals"]["combined"]["flagged_windows"], 1)
        self.assertEqual(prefix["signals"]["combined"]["false_positive_rate"], 0.5)

    def test_all_missed_fault_type_remains_in_results_with_no_fabricated_latency(self):
        report = evaluate_batches([normal()], [batch(), batch(fault=2)], onset_sample=5, expected_faults=[1, 2])
        for entry in report["per_fault"].values():
            self.assertEqual(entry["signals"]["combined"]["detected_simulations"], 0)
            self.assertIsNone(entry["signals"]["combined"]["mean_latency_steps_detected"])
        self.assertEqual(report["macro_across_fault_types"]["combined"]["fault_types_with_any_detection"], 0)
        serialized = json.dumps(report, allow_nan=False)
        self.assertIn('"latency_steps": null', serialized)

    def test_no_prefix_windows_reports_null_rate_not_zero(self):
        report = evaluate_batches([normal()], [batch(ends=(8, 9, 10), anomaly=(8,))], onset_sample=5, expected_faults=[1])
        self.assertEqual(report["fault_normal_prefix"]["eligible_windows"], 0)
        self.assertIsNone(report["fault_normal_prefix"]["signals"]["combined"]["false_positive_rate"])

    def test_unrequested_missing_or_duplicate_fault_sets_fail(self):
        for expected in ([1, 2], [], [1, 1], [0]):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                evaluate_batches([normal()], [batch()], onset_sample=5, expected_faults=expected)

    def test_duplicate_simulations_and_training_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_batches([normal(), normal()], [batch()], expected_faults=[1])
        with self.assertRaisesRegex(ValueError, "held-out"):
            evaluate_batches([batch(source="normal-training", fault=0)], [batch()], expected_faults=[1])

    def test_bad_endpoints_or_nonboolean_flags_fail(self):
        for ends in ((3, 3, 4), (3, 2, 4), (3, 4.5, 5), (3, 4, 961)):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                fault_run_record(batch(ends=ends), onset_sample=5)
        broken = batch()
        broken = ScoreBatch(broken.run_key, broken.end_samples, broken.anomaly_scores,
                            broken.drift_scores, broken.sensor_ks, broken.anomaly_flags,
                            broken.drift_flags, broken.flags.astype(int))
        with self.assertRaisesRegex(ValueError, "Boolean"):
            fault_run_record(broken)

    def test_no_post_onset_observations_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "No eligible"):
            fault_run_record(batch(), onset_sample=161)

    def test_markdown_keeps_undetected_rows_and_references_json(self):
        metrics = evaluate_batches([normal()], [batch()], onset_sample=5, expected_faults=[1])
        report = {"generated_at_utc": "offline-fixture", "metrics": metrics,
                  "model": {"model_sha256": "offline-fixture"}}
        rendered = results_markdown(report)
        self.assertIn("0/1 simulations", rendered)
        self.assertIn("| 1 | 0/1 | undetected | undetected |", rendered)
        self.assertIn("[results.json](results.json)", rendered)

    def test_score_archives_roundtrip_boolean_flags_and_original_coordinates(self):
        batches = [normal(anomaly=(3, 7)), normal(run=2, drift=(10,))]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.npz"
            _save_scores(path, batches)
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(arrays["flags"].shape, (2, 8))
                self.assertEqual(arrays["flags"].dtype.kind, "b")
                np.testing.assert_array_equal(arrays["end_samples"][0], range(3, 11))
                np.testing.assert_array_equal(arrays["simulation_runs"], [1, 2])
                self.assertTrue(all(array.dtype.kind != "O" for array in arrays.values()))


if __name__ == "__main__":
    unittest.main()
