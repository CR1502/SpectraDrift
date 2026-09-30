"""Synthetic plotting fixtures test validation, not TEP performance."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py"
spec = importlib.util.spec_from_file_location("spectradrift_plot_results", SCRIPT)
plot_results = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plot_results
spec.loader.exec_module(plot_results)


class PlotResultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.report = {"model": {"anomaly_threshold": 1.0, "drift_threshold": 0.5},
                       "metrics": {"onset_sample": 161}, "score_archives": {}}
        self.arrays = {}
        for kind, source, faults in (("normal", "normal-testing", [0]), ("faulty", "faulty-testing", [1, 9])):
            n = len(faults)
            anomaly = np.tile([0.5, 1.0, 1.5], (n, 1))
            drift = np.tile([0.2, 0.5, 0.3], (n, 1))
            self.arrays[kind] = {
                "sources": np.array([source] * n), "fault_numbers": np.array(faults),
                "simulation_runs": np.ones(n, dtype=int), "end_samples": np.tile([160, 161, 162], (n, 1)),
                "anomaly_scores": anomaly, "drift_scores": drift,
                "anomaly_flags": anomaly > 1, "drift_flags": drift > 0.5,
                "flags": (anomaly > 1) | (drift > 0.5),
            }
            self.report["score_archives"][kind] = {"path": str(self.root / f"{kind}_scores.npz")}
        self.results = self.root / "results.json"
        self.save()

    def save(self):
        for kind, arrays in self.arrays.items():
            record = self.report["score_archives"][kind]
            path = Path(record["path"])
            np.savez_compressed(path, **arrays)
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.results.write_text(json.dumps(self.report))

    def test_examples_preserve_identity_ratios_strict_thresholds_and_latencies(self):
        examples, onset = plot_results.load_examples(self.results)
        self.assertEqual(onset, 161)
        self.assertEqual([(run.source, run.fault, run.run) for run in examples],
                         [("normal-testing", 0, 1), ("faulty-testing", 1, 1), ("faulty-testing", 9, 1)])
        np.testing.assert_array_equal(examples[0].anomaly_ratio, [0.5, 1, 1.5])
        np.testing.assert_array_equal(examples[0].drift_ratio, [0.4, 1, 0.6])
        np.testing.assert_array_equal(examples[0].flags, [False, False, True])
        self.assertIn("1/3 windows flagged", examples[0].title(onset))
        self.assertIn("latency: 1 steps", examples[1].title(onset))

    def test_changed_archive_hash_is_rejected(self):
        Path(self.report["score_archives"]["normal"]["path"]).write_bytes(b"modified fixture")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            plot_results.load_examples(self.results)

    def test_missing_archive_has_regeneration_hint(self):
        self.report["score_archives"]["normal"]["path"] = str(self.root / "missing.npz")
        self.results.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(FileNotFoundError, "regenerate scores"):
            plot_results.load_examples(self.results)

    def test_relocated_archive_directory_still_requires_matching_hashes(self):
        self.report["score_archives"]["normal"]["path"] = "old/normal_scores.npz"
        self.report["score_archives"]["faulty"]["path"] = "old/faulty_scores.npz"
        self.results.write_text(json.dumps(self.report))
        examples, _ = plot_results.load_examples(self.results, self.root)
        self.assertEqual(len(examples), 3)

    def test_nonpositive_or_nonfinite_thresholds_fail(self):
        for threshold in (0, -1, float("nan"), float("inf")):
            self.report["model"]["anomaly_threshold"] = threshold
            self.results.write_text(json.dumps(self.report))
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "thresholds"):
                plot_results.load_examples(self.results)

    def test_requested_example_must_be_present_exactly_once(self):
        self.arrays["faulty"]["fault_numbers"][:] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            plot_results.load_examples(self.results)

    def test_flags_inconsistent_with_frozen_rule_fail(self):
        self.arrays["normal"]["anomaly_flags"][0, 1] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "strict-threshold OR"):
            plot_results.load_examples(self.results)

    def test_invalid_endpoints_or_scores_fail(self):
        self.arrays["normal"]["end_samples"][0, 1] = 160
        self.save()
        with self.assertRaisesRegex(ValueError, "endpoints"):
            plot_results.load_examples(self.results)
        self.arrays["normal"]["end_samples"][0, 1] = 161
        self.arrays["normal"]["drift_scores"][0, 0] = 1.1
        self.save()
        with self.assertRaisesRegex(ValueError, "scores"):
            plot_results.load_examples(self.results)

    def test_no_post_onset_flags_reports_undetected(self):
        self.arrays["faulty"]["anomaly_scores"][:] = 0.5
        self.arrays["faulty"]["anomaly_flags"][:] = False
        self.arrays["faulty"]["flags"][:] = False
        self.save()
        examples, onset = plot_results.load_examples(self.results)
        self.assertIn("undetected", examples[1].title(onset))

    def test_renderer_creates_png_without_mutating_archives_and_never_overwrites(self):
        output = self.root / "figures" / "test.png"
        hashes = {kind: record["sha256"] for kind, record in self.report["score_archives"].items()}
        plot_results.render_plot(self.results, output)
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for kind, record in self.report["score_archives"].items():
            self.assertEqual(plot_results._sha256(Path(record["path"])), hashes[kind])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            plot_results.render_plot(self.results, output)


if __name__ == "__main__":
    unittest.main()
