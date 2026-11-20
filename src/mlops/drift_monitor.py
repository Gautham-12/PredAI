"""
mlops.drift_monitor
~~~~~~~~~~~~~~~~~~~
Two-sample Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI)
for real-time streaming telemetry drift detection.

Mathematically aligns with AURA-Fleet Section 5 (Page 4):
    PSI = Σ (P_actual(b) - P_expected(b)) * ln(P_actual(b) / P_expected(b))
    If PSI > 0.20 OR p-value < 0.05 → Data Drift Alert dispatched.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp as _scipy_ks_2samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from config import (
    KS_PVALUE_THRESHOLD,
    MODELS_DIR,
    PSI_ALERT_THRESHOLD,
    PSI_BINS,
)

logger = logging.getLogger(__name__)

_DRIFT_FNAME = "drift_baseline.pkl"


def _numpy_ks_2samp(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """Pure NumPy implementation of two-sample Kolmogorov-Smirnov test."""
    d1 = np.sort(data1)
    d2 = np.sort(data2)
    n1, n2 = len(d1), len(d2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    all_vals = np.concatenate([d1, d2])
    cdf1 = np.searchsorted(d1, all_vals, side="right") / float(n1)
    cdf2 = np.searchsorted(d2, all_vals, side="right") / float(n2)
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic p-value approximation
    en = np.sqrt((n1 * n2) / float(n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    # Kolmogorov distribution approximation: 2 * exp(-2 * lambda^2)
    p_val = float(np.clip(2.0 * np.exp(-2.0 * (lambda_val ** 2)), 0.0, 1.0))
    return d_stat, p_val


def ks_2samp(data1: np.ndarray, data2: np.ndarray):
    """Wrapper that calls scipy.stats.ks_2samp or fallback NumPy ks_2samp."""
    if HAS_SCIPY:
        res = _scipy_ks_2samp(data1, data2)
        return float(res.statistic), float(res.pvalue)
    else:
        return _numpy_ks_2samp(data1, data2)


@dataclass
class FeatureDriftResult:
    """Per-feature drift analysis output."""

    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    psi_drift: bool
    ks_drift: bool
    drift_detected: bool


@dataclass
class DriftReport:
    """Aggregate fleet telemetry drift report."""

    timestamp: str
    sample_size: int
    overall_drift: bool
    drift_features: List[str] = field(default_factory=list)
    feature_metrics: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    mean_psi: float = 0.0
    max_psi: float = 0.0

    def summary(self) -> str:
        status = "⚠️ DRIFT ALERT" if self.overall_drift else "✅ NOMINAL STABILITY"
        return (
            f"[{status}] n_samples={self.sample_size} | "
            f"Drifting features: {len(self.drift_features)}/{len(self.feature_metrics)} | "
            f"Mean PSI: {self.mean_psi:.4f} | Max PSI: {self.max_psi:.4f}"
        )


class DriftMonitor:
    """Monitors live telemetry against reference training distribution.

    Parameters
    ----------
    psi_threshold : float
        PSI threshold above which drift is flagged (default 0.20).
    ks_threshold : float
        p-value threshold below which KS rejects null hypothesis of same distribution (default 0.05).
    n_bins : int
        Number of quantile bins for PSI computation (default 10).
    eps : float
        Epsilon to avoid division by zero or ln(0).
    """

    def __init__(
        self,
        psi_threshold: float = PSI_ALERT_THRESHOLD,
        ks_threshold: float = KS_PVALUE_THRESHOLD,
        n_bins: int = PSI_BINS,
        eps: float = 1e-4,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.n_bins = n_bins
        self.eps = eps

        self._baseline_features: List[str] = []
        self._baseline_data: Dict[str, np.ndarray] = {}
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._expected_probs: Dict[str, np.ndarray] = {}

    def fit_baseline(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> "DriftMonitor":
        """Record baseline empirical distributions from training data."""
        cols = feature_cols or list(df.select_dtypes(include=[np.number]).columns)
        self._baseline_features = cols

        for col in cols:
            vals = df[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue

            self._baseline_data[col] = vals

            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(vals, percentiles)
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([vals.min() - 1e-5, vals.max() + 1e-5])
            else:
                edges[0] -= 1e-5
                edges[-1] += 1e-5

            self._bin_edges[col] = edges

            counts, _ = np.histogram(vals, bins=edges)
            probs = counts / (len(vals) + 1e-12)
            probs = np.where(probs == 0, self.eps, probs)
            probs = probs / probs.sum()
            self._expected_probs[col] = probs

        logger.info(
            "DriftMonitor fitted on %d baseline features (%d samples).",
            len(self._baseline_features),
            len(df),
        )
        return self

    def calculate_psi(self, actual: np.ndarray, col: str) -> float:
        """Calculate Population Stability Index for a single feature."""
        if col not in self._bin_edges or col not in self._expected_probs:
            return 0.0

        edges = self._bin_edges[col]
        expected = self._expected_probs[col]

        counts, _ = np.histogram(actual, bins=edges)
        actual_probs = counts / (len(actual) + 1e-12)
        actual_probs = np.where(actual_probs == 0, self.eps, actual_probs)
        actual_probs = actual_probs / actual_probs.sum()

        psi_value = np.sum((actual_probs - expected) * np.log(actual_probs / expected))
        return float(max(0.0, psi_value))

    def evaluate_drift(
        self,
        new_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> DriftReport:
        """Evaluate a batch of streaming telemetry against baseline."""
        self._check_fitted()
        cols = feature_cols or self._baseline_features
        valid_cols = [c for c in cols if c in new_df.columns and c in self._baseline_data]

        feature_metrics: Dict[str, FeatureDriftResult] = {}
        drifting_features: List[str] = []
        psi_scores: List[float] = []

        for col in valid_cols:
            actual = new_df[col].dropna().values.astype(float)
            if len(actual) < 5:
                continue

            ref = self._baseline_data[col]

            # 1. PSI
            psi = self.calculate_psi(actual, col)
            psi_drift = psi > self.psi_threshold
            psi_scores.append(psi)

            # 2. KS-test
            ks_stat, ks_pval = ks_2samp(actual, ref)
            ks_drift = ks_pval < self.ks_threshold

            is_drifting = psi_drift or ks_drift
            if is_drifting:
                drifting_features.append(col)

            feature_metrics[col] = FeatureDriftResult(
                feature_name=col,
                psi_score=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                psi_drift=psi_drift,
                ks_drift=ks_drift,
                drift_detected=is_drifting,
            )

        overall = len(drifting_features) > 0
        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).isoformat()

        report = DriftReport(
            timestamp=now_str,
            sample_size=len(new_df),
            overall_drift=overall,
            drift_features=drifting_features,
            feature_metrics=feature_metrics,
            mean_psi=float(np.mean(psi_scores)) if psi_scores else 0.0,
            max_psi=float(np.max(psi_scores)) if psi_scores else 0.0,
        )

        return report

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = Path(path) if path else MODELS_DIR / _DRIFT_FNAME
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info("DriftMonitor saved → %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DriftMonitor":
        load_path = Path(path) if path else MODELS_DIR / _DRIFT_FNAME
        with open(load_path, "rb") as f:
            obj = pickle.load(f)
        logger.info("DriftMonitor loaded ← %s", load_path)
        return obj

    def _check_fitted(self) -> None:
        if not self._baseline_features:
            raise RuntimeError("DriftMonitor is not fitted. Call fit_baseline() first.")
