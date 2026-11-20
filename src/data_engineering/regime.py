"""
data_engineering.regime
~~~~~~~~~~~~~~~~~~~~~~~
Operational-regime detection via K-Means clustering on the three
flight-condition settings (setting1, setting2, setting3).

After fitting, each row in a DataFrame is labelled with its regime id
(0-5). Regime labels are used downstream by RegimeNormalizer to apply
within-regime Z-score standardisation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans as _SklearnKMeans
    from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from config import N_REGIMES, REGIME_RANDOM_STATE, SETTING_COLS, MODELS_DIR

logger = logging.getLogger(__name__)

_CLUSTERER_FNAME = "regime_clusterer.pkl"


class _NumpyStandardScaler:
    """Pure NumPy standard scaler."""

    def __init__(self) -> None:
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        return (X - self.mean_) / self.scale_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X * self.scale_) + self.mean_


class _NumpyKMeans:
    """Pure NumPy K-Means implementation (k=6)."""

    def __init__(self, n_clusters: int = 6, random_state: int = 42, max_iter: int = 100) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.inertia_ = 0.0

    def fit(self, X: np.ndarray) -> "_NumpyKMeans":
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        if n_samples <= self.n_clusters:
            self.cluster_centers_ = X.copy()
            return self

        # KMeans++ initialization
        centers = [X[rng.integers(n_samples)]]
        for _ in range(1, self.n_clusters):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / (np.sum(dists) + 1e-12)
            next_idx = rng.choice(n_samples, p=probs)
            centers.append(X[next_idx])
        centers = np.array(centers)

        for _ in range(self.max_iter):
            # Assign
            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                cluster_pts = X[labels == k]
                if len(cluster_pts) > 0:
                    new_centers[k] = cluster_pts.mean(axis=0)
                else:
                    new_centers[k] = centers[k]

            if np.allclose(centers, new_centers, atol=1e-5):
                break
            centers = new_centers

        self.cluster_centers_ = centers
        dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        self.inertia_ = float(np.sum(dists))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


class RegimeClusterer:
    """K-Means regime detector on operational settings.

    Parameters
    ----------
    n_regimes : int
        Number of clusters (default 6, matching NASA C-MAPSS regimes).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = N_REGIMES,
        random_state: int = REGIME_RANDOM_STATE,
    ) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._scaler = _SklearnStandardScaler() if HAS_SKLEARN else _NumpyStandardScaler()
        self._kmeans = None

    def fit(self, df: pd.DataFrame) -> "RegimeClusterer":
        """Fit K-Means on the operational setting columns of *df*."""
        X = df[SETTING_COLS].values.astype(float)
        X_scaled = self._scaler.fit_transform(X)

        if HAS_SKLEARN:
            self._kmeans = _SklearnKMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=10,
            )
        else:
            self._kmeans = _NumpyKMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
            )

        self._kmeans.fit(X_scaled)
        logger.info(
            "RegimeClusterer fitted — %d regimes, inertia=%.2f",
            self.n_regimes,
            self._kmeans.inertia_,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Assign regime label (0 … k-1) to each row."""
        self._check_fitted()
        X = df[SETTING_COLS].values.astype(float)
        X_scaled = self._scaler.transform(X)
        return self._kmeans.predict(X_scaled)

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.predict(df)

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = Path(path) if path else MODELS_DIR / _CLUSTERER_FNAME
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("RegimeClusterer saved → %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RegimeClusterer":
        load_path = Path(path) if path else MODELS_DIR / _CLUSTERER_FNAME
        with open(load_path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("RegimeClusterer loaded ← %s", load_path)
        return obj

    def _check_fitted(self) -> None:
        if self._kmeans is None:
            raise RuntimeError("RegimeClusterer is not fitted yet. Call .fit() first.")

    @property
    def cluster_centers_(self) -> np.ndarray:
        self._check_fitted()
        return self._scaler.inverse_transform(self._kmeans.cluster_centers_)
