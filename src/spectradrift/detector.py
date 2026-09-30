"""Normal-only fitting/calibration, the OR decision rule, and safe model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Sequence

import numpy as np

from spectradrift.data import SENSOR_COLUMNS, TEPRun, load_manifest, load_runs
from spectradrift.models import KSDriftScorer, PCAReconstructionScorer
from spectradrift.preprocessing import (
    ChannelNormalizer, FEATURE_NAMES, _normal_training_only, split_normal_runs, window_features,
)


def decision_flags(
    anomaly_scores: np.ndarray, drift_scores: np.ndarray,
    anomaly_threshold: float, drift_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag strictly above either threshold; equality is not a flag."""
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    drift = np.asarray(drift_scores, dtype=np.float64)
    if anomaly.ndim != 1 or anomaly.shape != drift.shape or not len(anomaly):
        raise ValueError("Both score vectors must be nonempty and have the same shape")
    if not np.isfinite(anomaly).all() or not np.isfinite(drift).all() or (anomaly < 0).any() or (drift < 0).any():
        raise ValueError("Scores must be finite and nonnegative")
    if not np.isfinite([anomaly_threshold, drift_threshold]).all() or min(anomaly_threshold, drift_threshold) < 0:
        raise ValueError("Thresholds must be finite and nonnegative")
    anomaly_flags = anomaly > anomaly_threshold
    drift_flags = drift > drift_threshold
    return anomaly_flags, drift_flags, anomaly_flags | drift_flags


@dataclass(frozen=True)
class ScoreBatch:
    run_key: tuple[str, int, int]
    end_samples: np.ndarray
    anomaly_scores: np.ndarray
    drift_scores: np.ndarray
    sensor_ks: np.ndarray
    anomaly_flags: np.ndarray
    drift_flags: np.ndarray
    flags: np.ndarray


@dataclass(frozen=True)
class TwoSignalDetector:
    normalizer: ChannelNormalizer
    anomaly: PCAReconstructionScorer
    drift: KSDriftScorer
    anomaly_threshold: float
    drift_threshold: float
    window_size: int = 20
    stride: int = 1

    def __post_init__(self) -> None:
        if len(self.anomaly.feature_mean) != len(FEATURE_NAMES):
            raise ValueError("Detector PCA must use the 104 TEP window features")
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool) or self.window_size < 2:
            raise ValueError("Invalid detector window size")
        if not isinstance(self.stride, int) or isinstance(self.stride, bool) or self.stride < 1:
            raise ValueError("Invalid detector stride")
        decision_flags(np.zeros(1), np.zeros(1), self.anomaly_threshold, self.drift_threshold)

    def score_run(self, run: TEPRun) -> ScoreBatch:
        windows = window_features(run, self.normalizer, window_size=self.window_size, stride=self.stride)
        anomaly_scores = self.anomaly.score(windows.features)
        statistics = self.drift.statistics(windows.sensor_windows)
        drift_scores = statistics.max(axis=1)
        anomaly_flags, drift_flags, flags = decision_flags(
            anomaly_scores, drift_scores, self.anomaly_threshold, self.drift_threshold,
        )
        return ScoreBatch(run.key, windows.end_samples, anomaly_scores, drift_scores,
                          statistics, anomaly_flags, drift_flags, flags)


def summarize_scores(batches: Sequence[ScoreBatch]) -> dict:
    if not batches:
        raise ValueError("At least one scored run is required")
    windows = sum(len(batch.flags) for batch in batches)
    report = {"simulations": len(batches), "eligible_windows": windows}
    for name, attribute in (("anomaly", "anomaly_flags"), ("drift", "drift_flags"), ("combined", "flags")):
        flagged = sum(int(getattr(batch, attribute).sum()) for batch in batches)
        report[f"{name}_flagged_windows"] = flagged
        report[f"{name}_flag_rate"] = flagged / windows
    report["per_run"] = [
        {"key": list(batch.run_key), "eligible_windows": len(batch.flags),
         "anomaly_flagged_windows": int(batch.anomaly_flags.sum()),
         "drift_flagged_windows": int(batch.drift_flags.sum()),
         "combined_flagged_windows": int(batch.flags.sum())}
        for batch in batches
    ]
    return report


def fit_detector(
    fitting_runs: Sequence[TEPRun], calibration_runs: Sequence[TEPRun], *,
    window_size: int = 20, stride: int = 1, variance_retained: float = 0.95,
    reference_size: int = 512, target_flag_rate: float = 0.01, seed: int = 42,
) -> tuple[TwoSignalDetector, dict]:
    """Fit both components on normal fitting runs; calibrate on other normal runs.

    Split the target calibration flag allowance equally across components. The
    finite-sample higher quantile and strict > comparison limit calibration flags;
    this does not guarantee the same rate on unseen, autocorrelated process runs.
    """
    _normal_training_only(fitting_runs)
    _normal_training_only(calibration_runs)
    if {run.key for run in fitting_runs} & {run.key for run in calibration_runs}:
        raise ValueError("Fitting and calibration simulations must be disjoint")
    if not 0 < target_flag_rate < 1:
        raise ValueError("target_flag_rate must be strictly between 0 and 1")
    normalizer = ChannelNormalizer.fit(fitting_runs)
    fitting_features = np.concatenate([
        window_features(run, normalizer, window_size=window_size, stride=stride).features
        for run in fitting_runs
    ])
    anomaly = PCAReconstructionScorer.fit(fitting_features, variance_retained=variance_retained)
    drift = KSDriftScorer.fit(
        np.concatenate([normalizer.transform(run) for run in fitting_runs]),
        reference_size=reference_size, seed=seed,
    )
    calibration_anomaly = []
    calibration_drift = []
    for run in calibration_runs:
        batch = window_features(run, normalizer, window_size=window_size, stride=stride)
        calibration_anomaly.append(anomaly.score(batch.features))
        calibration_drift.append(drift.score(batch.sensor_windows))
    anomaly_scores = np.concatenate(calibration_anomaly)
    drift_scores = np.concatenate(calibration_drift)
    quantile = 1 - target_flag_rate / 2
    anomaly_threshold = float(np.quantile(anomaly_scores, quantile, method="higher"))
    drift_threshold = float(np.quantile(drift_scores, quantile, method="higher"))
    detector = TwoSignalDetector(normalizer, anomaly, drift, anomaly_threshold, drift_threshold, window_size, stride)
    report = {
        "fitting_run_keys": [list(run.key) for run in fitting_runs],
        "calibration_run_keys": [list(run.key) for run in calibration_runs],
        "fitting_windows": len(fitting_features),
        "variance_retained_requested": variance_retained,
        "variance_retained_actual": float(anomaly.explained_variance_ratio.sum()),
        "pca_components": len(anomaly.components),
        "reference_size_requested": reference_size,
        "reference_size_actual": len(drift.sorted_reference),
        "reference_sampling_seed": seed,
        "target_calibration_flag_rate": target_flag_rate,
        "component_quantile": quantile,
        "quantile_method": "higher",
        "comparison": "strictly greater than threshold",
        "decision_rule": "anomaly_flag OR drift_flag",
        "calibration": summarize_scores([detector.score_run(run) for run in calibration_runs]),
    }
    return detector, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_detector(detector: TwoSignalDetector, output_dir: Path | str, training_report: dict) -> dict:
    directory = Path(output_dir)
    if any((directory / name).exists() for name in ("model.npz", "manifest.json")):
        raise ValueError("Model artifacts already exist; choose a new --output directory")
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.npz"
    np.savez_compressed(
        model_path, feature_mean=detector.anomaly.feature_mean,
        feature_scale=detector.anomaly.feature_scale, pca_mean=detector.anomaly.pca_mean,
        components=detector.anomaly.components,
        explained_variance_ratio=detector.anomaly.explained_variance_ratio,
        ks_reference=detector.drift.sorted_reference,
    )
    record = {
        "artifact_version": 1,
        "model_sha256": _sha256(model_path),
        "sensor_columns": list(SENSOR_COLUMNS),
        "feature_names": list(FEATURE_NAMES),
        "normalizer": detector.normalizer.as_dict(),
        "window_size": detector.window_size,
        "stride": detector.stride,
        "anomaly_threshold": detector.anomaly_threshold,
        "drift_threshold": detector.drift_threshold,
        "variance_retained": detector.anomaly.variance_retained,
        "decision_rule": "anomaly_flag OR drift_flag",
        "software": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ("spectradrift", "numpy", "scikit-learn", "scipy", "pyreadr")
        }},
        "training": training_report,
    }
    (directory / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def load_detector(model_dir: Path | str) -> TwoSignalDetector:
    directory = Path(model_dir)
    record = json.loads((directory / "manifest.json").read_text())
    if record["artifact_version"] != 1 or record["decision_rule"] != "anomaly_flag OR drift_flag":
        raise ValueError("Unsupported detector artifact version or decision rule")
    if record["sensor_columns"] != list(SENSOR_COLUMNS) or record["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Saved model sensor/feature order does not match TEP")
    model_path = directory / "model.npz"
    if _sha256(model_path) != record["model_sha256"]:
        raise ValueError("Model artifact checksum mismatch")
    with np.load(model_path, allow_pickle=False) as arrays:
        anomaly = PCAReconstructionScorer(
            arrays["feature_mean"], arrays["feature_scale"], arrays["pca_mean"],
            arrays["components"], arrays["explained_variance_ratio"], record["variance_retained"],
        )
        drift = KSDriftScorer(arrays["ks_reference"])
    return TwoSignalDetector(
        ChannelNormalizer.from_dict(record["normalizer"]), anomaly, drift,
        record["anomaly_threshold"], record["drift_threshold"], record["window_size"], record["stride"],
    )


def train_from_data(
    *, data_dir: Path | str = "data", manifest_path: Path | str = "data/source.json",
    run_ids: Sequence[int] = tuple(range(1, 11)), calibration_fraction: float = 0.2,
    seed: int = 42, window_size: int = 20, stride: int = 1, variance_retained: float = 0.95,
    reference_size: int = 512, target_flag_rate: float = 0.01,
    output_dir: Path | str = "artifacts/detector",
) -> dict:
    directory = Path(output_dir)
    if any((directory / name).exists() for name in ("model.npz", "manifest.json")):
        raise ValueError("Model artifacts already exist; choose a new --output directory")
    spec = load_manifest(manifest_path)["normal-training"]
    runs = load_runs(Path(data_dir) / spec.filename, spec, run_ids=run_ids)
    fitting, calibration = split_normal_runs(runs, calibration_fraction=calibration_fraction, seed=seed)
    detector, report = fit_detector(
        fitting, calibration, seed=seed, window_size=window_size, stride=stride,
        variance_retained=variance_retained, reference_size=reference_size, target_flag_rate=target_flag_rate,
    )
    report.update({"dataset_doi": "10.7910/DVN/6C3JR1", "dataset_version": "1.0",
                   "source_file": spec.filename, "publisher_md5": spec.md5,
                   "source_checksum_verified": True, "calibration_fraction": calibration_fraction,
                   "split_seed": seed})
    return save_detector(detector, directory, report)


def score_dataset(
    *, model_dir: Path | str = "artifacts/detector", data_dir: Path | str = "data",
    manifest_path: Path | str = "data/source.json", source: str = "normal-testing",
    run_ids: Sequence[int] = (1, 2, 3), fault_ids: Sequence[int] | None = None,
    output_dir: Path | str = "artifacts/normal-check",
) -> dict:
    directory = Path(output_dir)
    if any((directory / name).exists() for name in ("scores.npz", "summary.json")):
        raise ValueError("Score artifacts already exist; choose a new --output directory")
    if source not in {"normal-testing", "faulty-testing"}:
        raise ValueError("score-data requires a held-out testing source")
    detector = load_detector(model_dir)
    spec = load_manifest(manifest_path)[source]
    runs = load_runs(Path(data_dir) / spec.filename, spec, run_ids=run_ids, fault_ids=fault_ids)
    batches = [detector.score_run(run) for run in runs]
    report = {
        "source": source, "publisher_md5": spec.md5, "source_checksum_verified": True,
        "model_sha256": _sha256(Path(model_dir) / "model.npz"),
        "window_size": detector.window_size, "stride": detector.stride,
        "anomaly_threshold": detector.anomaly_threshold, "drift_threshold": detector.drift_threshold,
        "summary": summarize_scores(batches),
        "interpretation": ("Held-out normal window-level false-positive rate" if source == "normal-testing"
                           else "Fault-condition scores only; onset-aware latency is not evaluated here"),
        "benchmark_status": "Full fault latency benchmark pending",
    }
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "scores.npz", sources=np.array([batch.run_key[0] for batch in batches]),
        fault_numbers=np.array([batch.run_key[1] for batch in batches]),
        simulation_runs=np.array([batch.run_key[2] for batch in batches]),
        **{name: np.stack([getattr(batch, name) for batch in batches])
           for name in ("end_samples", "anomaly_scores", "drift_scores", "sensor_ks", "anomaly_flags", "drift_flags", "flags")},
    )
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
