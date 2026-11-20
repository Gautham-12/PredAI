"""
ml_engine.nasa_loss
~~~~~~~~~~~~~~~~~~~~
NASA CMAPSS asymmetric scoring function.

The NASA score penalises late predictions (under-estimation of RUL) far
more harshly than early predictions (over-estimation). This asymmetry
reflects the real-world cost of an unexpected failure.

Scoring formula:
    d_i = RUL_predicted_i - RUL_true_i
    s_i = exp(-d_i / 13) - 1   if d_i < 0   (late  → heavy penalty)
    s_i = exp( d_i / 10) - 1   if d_i >= 0  (early → light penalty)
    S   = Σ s_i
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _score_single(d: float) -> float:
    """Score contribution for a single error d = ŷ - y."""
    if d < 0:
        return float(np.exp(-d / 13.0) - 1.0)
    return float(np.exp(d / 10.0) - 1.0)


_score_vec = np.vectorize(_score_single)


def nasa_score(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the total NASA CMAPSS score over all predictions.

    Parameters
    ----------
    y_true : array-like
        True RUL values.
    y_pred : array-like
        Predicted RUL values (p50 / median recommended).

    Returns
    -------
    float
        Aggregate NASA score S = Σ s_i. Lower is better (0 is perfect).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    d = y_pred - y_true
    scores = _score_vec(d)
    total = float(scores.sum())
    logger.debug("NASA score computed — n=%d, S=%.2f", len(d), total)
    return total


def nasa_score_per_engine(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    engine_ids: Union[np.ndarray, pd.Series],
) -> pd.Series:
    """Compute per-engine NASA score (useful for diagnosis).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    engine_ids : array-like
        Integer or string engine identifiers aligned with y_true/y_pred.

    Returns
    -------
    pd.Series
        NASA score indexed by engine id, sorted descending (worst first).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ids = np.asarray(engine_ids)
    d = y_pred - y_true
    scores = _score_vec(d)

    per_engine = pd.Series(scores, index=ids).groupby(level=0).sum()
    return per_engine.sort_values(ascending=False)


def pinball_loss(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    tau: float,
) -> float:
    """Pinball (quantile) loss for a single quantile level *tau*.

    L_τ(y, ŷ) = mean( max(τ*(y - ŷ), (τ-1)*(y - ŷ)) )

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    tau : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        Mean pinball loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    loss = np.maximum(tau * err, (tau - 1.0) * err)
    return float(loss.mean())
