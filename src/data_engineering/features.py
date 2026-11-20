"""
data_engineering.features
~~~~~~~~~~~~~~~~~~~~~~~~~
Rolling feature engineering for the AURA-Fleet pipeline.

For each active sensor and each rolling window (5, 15, 30 cycles) this
module computes:
  1. OLS slope     — degradation velocity / trend
  2. Rolling std   — volatility
  3. EWMA          — exponential smoothing (α = 2 / (window + 1))

All computations are performed per engine unit to prevent data leakage
across trajectories.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import ACTIVE_SENSORS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def _ols_slope(series: pd.Series) -> float:
    """Compute OLS slope of *series* against its integer index positions."""
    y = series.values
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    # Closed-form OLS: β = cov(x,y) / var(x)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


class FeatureEngineer:
    """Generate rolling statistical features from normalised sensor data.

    Parameters
    ----------
    sensor_cols : list of str, optional
        Sensors to process. Defaults to ``config.ACTIVE_SENSORS``.
    windows : list of int, optional
        Rolling window lengths. Defaults to ``config.ROLLING_WINDOWS``.
    unit_col : str
        Column that identifies individual engine trajectories.
    """

    def __init__(
        self,
        sensor_cols: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        unit_col: str = "unit_id",
    ) -> None:
        self.sensor_cols = sensor_cols or list(ACTIVE_SENSORS)
        self.windows = windows or list(ROLLING_WINDOWS)
        self.unit_col = unit_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append rolling features to *df* (preserves original columns).

        Parameters
        ----------
        df : pd.DataFrame
            Normalised sensor DataFrame, sorted by (unit_id, cycle).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional feature columns appended.
            NaN values introduced by the rolling windows are filled with 0.
        """
        present_sensors = [c for c in self.sensor_cols if c in df.columns]
        if not present_sensors:
            raise ValueError("No active sensor columns found in DataFrame.")

        out = df.copy()
        new_cols: dict[str, pd.Series] = {}

        for sensor in present_sensors:
            for w in self.windows:
                alpha = 2.0 / (w + 1)
                grouped = out.groupby(self.unit_col, sort=False)[sensor]

                # 1. OLS slope (rolling trend)
                slope_col = f"{sensor}_slope_{w}"
                new_cols[slope_col] = grouped.transform(
                    lambda s, win=w: s.rolling(win, min_periods=2).apply(
                        _ols_slope, raw=False
                    )
                )

                # 2. Rolling std (volatility)
                std_col = f"{sensor}_std_{w}"
                new_cols[std_col] = grouped.transform(
                    lambda s, win=w: s.rolling(win, min_periods=2).std(ddof=1)
                )

                # 3. EWMA (exponential moving average)
                ewma_col = f"{sensor}_ewma_{w}"
                new_cols[ewma_col] = grouped.transform(
                    lambda s, a=alpha: s.ewm(alpha=a, adjust=False).mean()
                )

        new_df = pd.DataFrame(new_cols, index=out.index)
        out = pd.concat([out, new_df], axis=1)

        # Fill NaN introduced by insufficient window history with 0
        feature_cols = list(new_cols.keys())
        out[feature_cols] = out[feature_cols].fillna(0.0)

        logger.info(
            "FeatureEngineer: added %d rolling feature columns (%d sensors × %d windows × 3).",
            len(feature_cols),
            len(present_sensors),
            len(self.windows),
        )
        return out

    @property
    def feature_names(self) -> List[str]:
        """Names of all generated feature columns (before transform is called)."""
        names = []
        for sensor in self.sensor_cols:
            for w in self.windows:
                names.extend([
                    f"{sensor}_slope_{w}",
                    f"{sensor}_std_{w}",
                    f"{sensor}_ewma_{w}",
                ])
        return names


def compute_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    sensor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience functional wrapper around FeatureEngineer."""
    fe = FeatureEngineer(windows=windows, sensor_cols=sensor_cols)
    return fe.transform(df)

