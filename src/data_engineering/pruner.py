"""
data_engineering.pruner
~~~~~~~~~~~~~~~~~~~~~~~
Removes zero-variance (constant) sensor columns from C-MAPSS DataFrames.

The seven sensors listed in ``PRUNED_SENSORS`` carry no discriminative
information across all four sub-datasets and are dropped before any
further feature engineering.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from config import PRUNED_SENSORS, ACTIVE_SENSORS, SETTING_COLS

logger = logging.getLogger(__name__)


def prune_sensors(
    df: pd.DataFrame,
    extra_drop: List[str] | None = None,
) -> pd.DataFrame:
    """Drop zero-variance sensor columns from a C-MAPSS DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially processed C-MAPSS DataFrame.
    extra_drop : list of str, optional
        Additional column names to drop beyond the standard pruned set.

    Returns
    -------
    pd.DataFrame
        DataFrame with pruned sensor columns removed.
    """
    to_drop = list(PRUNED_SENSORS)
    if extra_drop:
        to_drop.extend(extra_drop)

    # Only drop columns that actually exist
    present = [c for c in to_drop if c in df.columns]
    missing = [c for c in to_drop if c not in df.columns]
    if missing:
        logger.debug("Columns already absent (skipping): %s", missing)

    pruned = df.drop(columns=present)
    logger.info(
        "Pruner: dropped %d sensor columns %s — %d columns remain.",
        len(present),
        present,
        pruned.shape[1],
    )
    return pruned


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return ordered list of active sensor + setting columns present in df."""
    candidates = SETTING_COLS + ACTIVE_SENSORS
    return [c for c in candidates if c in df.columns]
