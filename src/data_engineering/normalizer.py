"""
data_engineering.normalizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Within-regime Z-score standardisation of sensor readings.

After RegimeClusterer assigns a regime label to each row, this module
computes per-(regime, sensor) mean and standard deviation on the training
set and applies the same transformation to test data.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ACTIVE_SENSORS, MODELS_DIR

logger = logging.getLogger(__name__)

_NORMALIZER_FNAME = "regime_normalizer.pkl"


class RegimeNormalizer:
    """Z-score normaliser that operates within each operational regime.

    Parameters
    ----------
    sensor_cols : list of str, optional
        Columns to normalise. Defaults to ``config.ACTIVE_SENSORS``.
    eps : float
        Small constant added to std to avoid division by zero.
    """

    def __init__(
        self,
        sensor_cols: Optional[List[str]] = None,
        eps: float = 1e-8,
    ) -> None:
        self.sensor_cols = sensor_cols or list(ACTIVE_SENSORS)
        self.eps = eps
        # Mapping: regime_id → {col: (mean, std)}
        self._stats: Dict[int, Dict[str, Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Fit / transform interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, regime_col: str = "regime") -> "RegimeNormalizer":
        """Compute per-(regime, sensor) mean and std on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``regime_col`` and all ``self.sensor_cols``.
        regime_col : str
            Name of the column holding integer regime labels.
        """
        present_sensors = [c for c in self.sensor_cols if c in df.columns]
        if not present_sensors:
            raise ValueError("No sensor columns found in DataFrame.")

        self._stats = {}
        for regime_id, grp in df.groupby(regime_col):
            self._stats[int(regime_id)] = {}
            for col in present_sensors:
                mu = float(grp[col].mean())
                sigma = float(grp[col].std(ddof=1))
                self._stats[int(regime_id)][col] = (mu, max(sigma, self.eps))

        logger.info(
            "RegimeNormalizer fitted — %d regimes, %d sensors.",
            len(self._stats),
            len(present_sensors),
        )
        return self

    def transform(self, df: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
        """Apply within-regime Z-score normalisation.

        Unknown regimes (not seen during fit) are normalised using the
        global mean/std across all regimes for each sensor as a fallback.

        Parameters
        ----------
        df : pd.DataFrame
        regime_col : str

        Returns
        -------
        pd.DataFrame — copy with sensor columns standardised in-place.
        """
        self._check_fitted()
        out = df.copy()

        present_sensors = [c for c in self.sensor_cols if c in out.columns]
        for col in present_sensors:
            if not np.issubdtype(out[col].dtype, np.floating):
                out[col] = out[col].astype(np.float64)

        # Build fallback global stats
        global_stats: Dict[str, Tuple[float, float]] = {}
        if self._stats:
            for col in present_sensors:
                all_means = [v[col][0] for v in self._stats.values() if col in v]
                all_stds = [v[col][1] for v in self._stats.values() if col in v]
                global_stats[col] = (
                    float(np.mean(all_means)) if all_means else 0.0,
                    float(np.mean(all_stds)) if all_stds else 1.0,
                )

        for regime_id, grp_idx in out.groupby(regime_col).groups.items():
            rid = int(regime_id)
            stats = self._stats.get(rid, None)
            for col in present_sensors:
                if stats and col in stats:
                    mu, sigma = stats[col]
                else:
                    mu, sigma = global_stats.get(col, (0.0, 1.0))
                    logger.warning(
                        "Regime %d unseen during fit — using global stats for %s.", rid, col
                    )
                out.loc[grp_idx, col] = (out.loc[grp_idx, col] - mu) / sigma

        return out

    def fit_transform(self, df: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
        """Fit then transform (convenience wrapper)."""
        return self.fit(df, regime_col).transform(df, regime_col)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = Path(path) if path else MODELS_DIR / _NORMALIZER_FNAME
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("RegimeNormalizer saved → %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RegimeNormalizer":
        load_path = Path(path) if path else MODELS_DIR / _NORMALIZER_FNAME
        with open(load_path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("RegimeNormalizer loaded ← %s", load_path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._stats:
            raise RuntimeError("RegimeNormalizer is not fitted yet. Call .fit() first.")
