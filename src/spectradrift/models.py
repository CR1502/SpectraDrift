"""Independent PCA reconstruction and empirical distribution-shift scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spectradrift.data import SENSOR_COLUMNS


def _matrix(values: np.ndarray, name: str, columns: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if columns is not None and values.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values


def _frozen(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Model parameters must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PCAReconstructionScorer:
    """Reconstruction MSE in standardized window-feature space; no labels used."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    variance_retained: float

    def __post_init__(self) -> None:
        for name in ("feature_mean", "feature_scale", "pca_mean", "components", "explained_variance_ratio"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        columns = self.feature_mean.size
        if self.feature_mean.shape != (columns,) or columns < 1:
            raise ValueError("Invalid feature mean vector")
        if self.feature_scale.shape != (columns,) or not (self.feature_scale > 0).all():
            raise ValueError("Feature scales must be positive and match the feature dimension")
        if self.pca_mean.shape != (columns,):
            raise ValueError("PCA center must match the feature dimension")
        if self.components.ndim != 2 or self.components.shape[1] != columns:
            raise ValueError("Invalid PCA component matrix")
        rank = len(self.components)
        if not 1 <= rank <= columns or self.explained_variance_ratio.shape != (rank,):
            raise ValueError("Invalid PCA rank or explained variance vector")
        if not (self.explained_variance_ratio >= 0).all() or not 0 < self.variance_retained < 1:
            raise ValueError("Invalid explained variance configuration")
        if not np.allclose(self.components @ self.components.T, np.eye(rank), atol=1e-8):
            raise ValueError("PCA components must be orthonormal")

    @classmethod
    def fit(cls, normal_features: np.ndarray, *, variance_retained: float = 0.95) -> PCAReconstructionScorer:
        features = _matrix(normal_features, "Normal fitting features")
        if len(features) < 2 or not 0 < variance_retained < 1:
            raise ValueError("PCA needs at least two windows and variance_retained in (0, 1)")
        scaler = StandardScaler().fit(features)
        standardized = scaler.transform(features)
        if not np.any(scaler.var_ > 0):
            raise ValueError("PCA cannot learn from entirely constant fitting features")
        pca = PCA(n_components=variance_retained, svd_solver="full").fit(standardized)
        return cls(scaler.mean_, scaler.scale_, pca.mean_, pca.components_,
                   pca.explained_variance_ratio_, variance_retained)

    def score(self, features: np.ndarray) -> np.ndarray:
        features = _matrix(features, "Inference features", len(self.feature_mean))
        centered = (features - self.feature_mean) / self.feature_scale - self.pca_mean
        reconstruction = (centered @ self.components.T) @ self.components
        return np.mean((centered - reconstruction) ** 2, axis=1)


@dataclass(frozen=True)
class KSDriftScorer:
    """Exact two-sample KS statistics against a frozen normal reference.

    This computes the empirical CDF distance, not a nominal p-value. Rolling,
    autocorrelated process samples are calibrated empirically on other normal runs.
    """

    sorted_reference: np.ndarray

    def __post_init__(self) -> None:
        reference = _matrix(self.sorted_reference, "KS reference", len(SENSOR_COLUMNS))
        if len(reference) < 2 or not (np.diff(reference, axis=0) >= 0).all():
            raise ValueError("KS reference needs at least two sorted readings per channel")
        object.__setattr__(self, "sorted_reference", _frozen(reference))

    @classmethod
    def fit(cls, normal_values: np.ndarray, *, reference_size: int = 512, seed: int = 42) -> KSDriftScorer:
        values = _matrix(normal_values, "Normal reference candidates", len(SENSOR_COLUMNS))
        if isinstance(reference_size, bool) or not isinstance(reference_size, int) or reference_size < 2:
            raise ValueError("reference_size must be an integer of at least 2")
        count = min(reference_size, len(values))
        if count < 2:
            raise ValueError("At least two normal reference samples are required")
        indices = np.random.default_rng(seed).choice(len(values), size=count, replace=False)
        return cls(np.sort(values[indices], axis=0))

    def statistics(self, sensor_windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(sensor_windows, dtype=np.float64)
        if windows.ndim != 3 or not windows.shape[0] or windows.shape[1] < 2 or windows.shape[2] != len(SENSOR_COLUMNS):
            raise ValueError("Expected nonempty (windows, time>=2, 52) sensor windows")
        if not np.isfinite(windows).all():
            raise ValueError("Sensor windows must be finite")
        ordered = np.sort(windows, axis=1)
        width = windows.shape[1]
        before = np.arange(width) / width
        after = np.arange(1, width + 1) / width
        result = np.empty((len(windows), len(SENSOR_COLUMNS)))
        for channel in range(len(SENSOR_COLUMNS)):
            reference = self.sorted_reference[:, channel]
            observations = ordered[:, :, channel]
            ref_before = np.searchsorted(reference, observations, side="left") / len(reference)
            ref_after = np.searchsorted(reference, observations, side="right") / len(reference)
            # For ties, the first/last occurrence gives each empirical jump's
            # endpoints. Between current observations the reference CDF is monotone,
            # so these endpoints suffice for the exact two-sided KS supremum.
            positive = np.max(after - ref_after, axis=1)
            negative = np.max(ref_before - before, axis=1)
            result[:, channel] = np.maximum(positive, negative)
        return result

    def score(self, sensor_windows: np.ndarray) -> np.ndarray:
        return self.statistics(sensor_windows).max(axis=1)
