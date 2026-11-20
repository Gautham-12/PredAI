"""
ml_engine.rul_target
~~~~~~~~~~~~~~~~~~~~~
Computes the piecewise-linear capped RUL target label.

RUL(t) = min(cycles_remaining, RUL_CAP)

For the training set the true RUL at each cycle is derived from the
maximum observed cycle per engine (i.e. the engine was run to failure).
For the test set, the ground-truth terminal RUL is provided separately
(``RUL_FDxxx.txt``) and must be added to the last cycle of each test
engine trajectory.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import RUL_CAP

logger = logging.getLogger(__name__)


def add_rul_target(
    df: pd.DataFrame,
    rul_cap: int = RUL_CAP,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    target_col: str = "RUL",
) -> pd.DataFrame:
    """Add a capped RUL column to a *training* DataFrame.

    The function assumes the training data contains **run-to-failure**
    trajectories, so the maximum cycle per engine equals the failure cycle.
    True RUL at cycle t is ``(max_cycle - t)``, capped at ``rul_cap``.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame with ``unit_col`` and ``cycle_col``.
    rul_cap : int
        Maximum allowable RUL label (piecewise-linear cap). Default 125.
    unit_col : str
        Column identifying individual engine units.
    cycle_col : str
        Column with the integer cycle (time step) index.
    target_col : str
        Name of the new RUL column to append.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional ``target_col`` column.
    """
    out = df.copy()
    max_cycles = out.groupby(unit_col)[cycle_col].transform("max")
    true_rul = max_cycles - out[cycle_col]
    out[target_col] = true_rul.clip(upper=rul_cap)
    logger.info(
        "RUL target added — cap=%d, range=[%d, %d].",
        rul_cap,
        int(out[target_col].min()),
        int(out[target_col].max()),
    )
    return out


def add_test_rul_target(
    test_df: pd.DataFrame,
    rul_series: pd.Series,
    rul_cap: int = RUL_CAP,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
    target_col: str = "RUL",
) -> pd.DataFrame:
    """Add a capped RUL column to a *test* DataFrame using ground-truth labels.

    For the test set, only the last cycle of each engine has a known RUL
    (from ``RUL_FDxxx.txt``). This function back-calculates RUL for every
    cycle in the test trajectory and caps it.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test DataFrame, sorted by (unit_id, cycle).
    rul_series : pd.Series
        True RUL at the *last observed* cycle, one value per engine,
        0-indexed (matches engine order in the test file).
    rul_cap : int
        RUL cap (default 125).
    unit_col : str
    cycle_col : str
    target_col : str

    Returns
    -------
    pd.DataFrame
        Copy of *test_df* with ``target_col`` appended.
    """
    out = test_df.copy()
    max_cycles = out.groupby(unit_col)[cycle_col].transform("max")

    # Build lookup: unit_id → terminal RUL (from ground truth)
    units = sorted(out[unit_col].unique())
    terminal_rul_map = {uid: int(rul_series.iloc[i]) for i, uid in enumerate(units)}

    terminal_rul_per_row = out[unit_col].map(terminal_rul_map)
    cycles_from_end = max_cycles - out[cycle_col]
    true_rul = terminal_rul_per_row + cycles_from_end
    out[target_col] = true_rul.clip(upper=rul_cap)

    logger.info(
        "Test RUL target added — cap=%d, range=[%d, %d].",
        rul_cap,
        int(out[target_col].min()),
        int(out[target_col].max()),
    )
    return out


# Backward-compatible alias
compute_rul = add_rul_target

