"""Normal-only channel normalization and causal per-simulation window features."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from spectradrift.data import SENSOR_COLUMNS, TEPRun, load_manifest, load_runs


FEATURE_NAMES = tuple(f"mean:{sensor}" for sensor in SENSOR_COLUMNS) + tuple(
    f"std:{sensor}" for sensor in SENSOR_COLUMNS
)


def _normal_training_only(runs: Sequence[TEPRun]) -> None:
    if not runs:
        raise ValueError("At least one normal training run is required")
    if any(run.source != "normal-training" or run.fault_number != 0 for run in runs):
        raise ValueError("Fitting and calibration require normal-training runs only")
    if len({run.key for run in runs}) != len(runs):
        raise ValueError("Duplicate simulation identities are not allowed")


def split_normal_runs(
    runs: Sequence[TEPRun], *, calibration_fraction: float = 0.2, seed: int = 42
) -> tuple[list[TEPRun], list[TEPRun]]:
    """Split entire normal simulations, never overlapping windows from one run."""
    _normal_training_only(runs)
    if len(runs) < 2 or not 0 < calibration_fraction < 1:
        raise ValueError("Need at least two runs and a calibration fraction strictly between 0 and 1")
    ordered = sorted(runs, key=lambda run: run.key)
    count = int(np.ceil(len(ordered) * calibration_fraction))
    if count >= len(ordered):
        raise ValueError("The calibration fraction leaves no fitting simulations")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    calibration_indices = set(permutation[:count].tolist())
    fitting = [run for index, run in enumerate(ordered) if index not in calibration_indices]
    calibration = [run for index, run in enumerate(ordered) if index in calibration_indices]
    return fitting, calibration


@dataclass(frozen=True)
class ChannelNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    variance: np.ndarray
    fitting_run_keys: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        for name in ("mean", "scale", "variance"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.shape != (len(SENSOR_COLUMNS),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid normalizer {name}: expected 52 finite channel values")
            if name == "scale" and not (values > 0).all():
                raise ValueError("Normalizer scales must be positive")
            if name == "variance" and not (values >= 0).all():
                raise ValueError("Normalizer variances must be nonnegative")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if not self.fitting_run_keys or len(set(self.fitting_run_keys)) != len(self.fitting_run_keys):
            raise ValueError("Normalizer must record unique fitting simulation identities")
        for source, fault, run in self.fitting_run_keys:
            if source != "normal-training" or fault != 0 or not isinstance(run, int) or not 1 <= run <= 500:
                raise ValueError("Normalizer must come from normal-training simulations")

    @classmethod
    def fit(cls, runs: Sequence[TEPRun]) -> ChannelNormalizer:
        """Estimate population channel moments using normal fitting data only."""
        _normal_training_only(runs)
        scaler = StandardScaler()
        for run in runs:
            scaler.partial_fit(run.values)
        return cls(
            mean=scaler.mean_.copy(),
            scale=scaler.scale_.copy(),
            variance=scaler.var_.copy(),
            fitting_run_keys=tuple(run.key for run in runs),
        )

    def transform(self, run: TEPRun) -> np.ndarray:
        """Apply frozen fitting statistics; this never adapts to evaluation data."""
        return (run.values - self.mean) / self.scale

    def as_dict(self) -> dict:
        return {
            "sensor_columns": list(SENSOR_COLUMNS),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "variance": self.variance.tolist(),
            "fitting_run_keys": [list(key) for key in self.fitting_run_keys],
            "constant_channel_rule": "StandardScaler uses scale=1 for constant channels",
        }

    @classmethod
    def from_dict(cls, record: dict) -> ChannelNormalizer:
        """Reload frozen statistics while checking the saved sensor order."""
        if record["sensor_columns"] != list(SENSOR_COLUMNS):
            raise ValueError("Saved normalizer sensor order does not match TEP channels")
        return cls(
            mean=np.asarray(record["mean"]),
            scale=np.asarray(record["scale"]),
            variance=np.asarray(record["variance"]),
            fitting_run_keys=tuple(tuple(key) for key in record["fitting_run_keys"]),
        )


@dataclass(frozen=True)
class WindowBatch:
    run_key: tuple[str, int, int]
    end_samples: np.ndarray
    features: np.ndarray
    sensor_windows: np.ndarray


def window_features(
    run: TEPRun,
    normalizer: ChannelNormalizer,
    *,
    window_size: int = 20,
    stride: int = 1,
) -> WindowBatch:
    """Mean and population std for each trailing window of normalized sensors.

    The tensor is (eligible windows, time within window, 52 channels). Feature
    order is all 52 means followed by all 52 stds. A window uses no future samples.
    """
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 2:
        raise ValueError("window_size must be an integer of at least 2")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if window_size > len(run.samples):
        raise ValueError("window_size exceeds the simulation length")
    normalized = normalizer.transform(run)
    windows = np.moveaxis(sliding_window_view(normalized, window_size, axis=0), -1, 1)[::stride]
    means = windows.mean(axis=1)
    deviations = windows.std(axis=1, ddof=0)
    features = np.concatenate([means, deviations], axis=1)
    ends = run.samples[window_size - 1::stride].copy()
    features.setflags(write=False)
    ends.setflags(write=False)
    return WindowBatch(run.key, ends, features, windows)


def prepare_normal_data(
    *,
    data_dir: Path | str = "data",
    manifest_path: Path | str = "data/source.json",
    run_ids: Sequence[int] = tuple(range(1, 11)),
    window_size: int = 20,
    stride: int = 1,
    calibration_fraction: float = 0.2,
    seed: int = 42,
    output_dir: Path | str = "artifacts/preprocessing",
) -> dict:
    """Prepare reproducible normal fitting/calibration arrays for the next stage.

    NPZs preserve the run axis and retain normalized sensor samples so that the
    independent statistical detector can consume raw windows as well as features.
    """
    output_dir = Path(output_dir)
    targets = [output_dir / name for name in ("fitting.npz", "calibration.npz", "manifest.json")]
    if any(path.exists() for path in targets):
        raise ValueError("Output artifacts already exist; choose a new --output directory")
    spec = load_manifest(manifest_path)["normal-training"]
    path = Path(data_dir) / spec.filename
    runs = load_runs(path, spec, run_ids=run_ids)
    fitting, calibration = split_normal_runs(runs, calibration_fraction=calibration_fraction, seed=seed)
    normalizer = ChannelNormalizer.fit(fitting)
    batches = {
        "fitting": [window_features(run, normalizer, window_size=window_size, stride=stride) for run in fitting],
        "calibration": [window_features(run, normalizer, window_size=window_size, stride=stride) for run in calibration],
    }
    report = {
        "artifact_version": 1,
        "stage": "normal-data-preprocessing",
        "dataset_doi": "10.7910/DVN/6C3JR1",
        "dataset_version": "1.0",
        "source": str(path.resolve()),
        "publisher_md5": spec.md5,
        "checksum_verified": True,
        "window_size": window_size,
        "stride": stride,
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "software": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("spectradrift", "numpy", "pandas", "scikit-learn", "pyreadr")},
        },
        "feature_names": list(FEATURE_NAMES),
        "normalizer": normalizer.as_dict(),
        "partitions": {},
        "benchmark_status": "not yet implemented; no detection metrics",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in (("fitting", fitting), ("calibration", calibration)):
        subset = batches[name]
        np.savez_compressed(
            output_dir / f"{name}.npz",
            source=np.array("normal-training"),
            run_ids=np.array([run.simulation_run for run in partition]),
            normalized_values=np.stack([normalizer.transform(run) for run in partition]),
            features=np.stack([batch.features for batch in subset]),
            end_samples=np.stack([batch.end_samples for batch in subset]),
        )
        report["partitions"][name] = {
            "run_ids": [run.simulation_run for run in partition],
            "simulations": len(partition),
            "raw_samples": sum(len(run.samples) for run in partition),
            "eligible_windows": sum(len(batch.end_samples) for batch in subset),
            "feature_dimension": len(FEATURE_NAMES),
        }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
