"""
ml_engine.quantile_model
~~~~~~~~~~~~~~~~~~~~~~~~
Quantile regression for probabilistic RUL estimation.
Uses LightGBM when available, with a resilient NumPy gradient boosting quantile model fallback.

Three separate quantile regressors are trained — one per quantile τ ∈
{0.10, 0.50, 0.90} — using the pinball (quantile) loss objective:
    Loss_τ(y, ŷ) = max(τ * (y - ŷ), (τ - 1) * (y - ŷ))

Predictions are packaged as a DataFrame with columns:
    rul_p10, rul_p50, rul_p90
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import LGBM_BASE_PARAMS, MODELS_DIR, QUANTILES

logger = logging.getLogger(__name__)

_QM_FNAME = "quantile_rul_model.pkl"

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.info("LightGBM not found; using resilient NumPy Quantile Regressor implementation.")


class _NumpyQuantileRegressor:
    """Pure NumPy Quantile Regressor optimizing pinball loss."""

    def __init__(self, tau: float = 0.5, n_iter: int = 250, lr: float = 0.05) -> None:
        self.tau = tau
        self.n_iter = n_iter
        self.lr = lr
        self.weights = None
        self.bias = 0.0
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyQuantileRegressor":
        n, d = X.shape
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        X_norm = (X - self.mean_) / self.std_

        self.weights = np.zeros(d, dtype=float)
        # Initialize bias to empirical quantile
        self.bias = float(np.percentile(y, self.tau * 100))

        # Iterative subgradient descent on pinball loss
        for epoch in range(1, self.n_iter + 1):
            preds = X_norm @ self.weights + self.bias
            diff = y - preds
            # Gradient: -tau if diff > 0 else (1 - tau)
            grad = np.where(diff > 0, -self.tau, (1.0 - self.tau))

            # Weight update with decay
            step = self.lr / np.sqrt(epoch)
            dw = X_norm.T @ grad / float(n)
            db = np.mean(grad)

            self.weights -= step * dw
            self.bias -= step * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self.mean_) / self.std_
        return X_norm @ self.weights + self.bias


class QuantileRULModel:
    """Probabilistic RUL predictor via quantile regression.

    Parameters
    ----------
    quantiles : list of float
        Quantile levels. Default ``[0.10, 0.50, 0.90]``.
    base_params : dict
        LightGBM hyper-parameters.
    early_stopping_rounds : int
    val_fraction : float
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        base_params: Optional[dict] = None,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.1,
    ) -> None:
        self.quantiles = quantiles or list(QUANTILES)
        self.base_params = base_params or dict(LGBM_BASE_PARAMS)
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self._models: Dict[float, Any] = {}
        self._feature_cols: List[str] = []
        self._is_lightgbm: bool = HAS_LIGHTGBM

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
    ) -> "QuantileRULModel":
        """Train one model per quantile."""
        self._feature_cols = feature_cols if feature_cols else list(X.columns)
        X_use = X[self._feature_cols].values.astype(np.float32)
        y_use = y.values.astype(np.float32)

        # Train/val split
        n = len(X_use)
        val_size = int(n * self.val_fraction)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        val_idx = perm[:val_size]
        tr_idx = perm[val_size:]

        X_tr, y_tr = X_use[tr_idx], y_use[tr_idx]
        X_val, y_val = X_use[val_idx], y_use[val_idx]

        for tau in self.quantiles:
            if HAS_LIGHTGBM:
                try:
                    params = {**self.base_params, "objective": "quantile", "alpha": tau}
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric="quantile",
                        callbacks=[
                            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        ],
                    )
                    self._models[tau] = model
                    self._is_lightgbm = True
                except Exception as e:
                    logger.warning("LightGBM fitting failed (%s); using NumPy quantile regressor.", e)
                    np_model = _NumpyQuantileRegressor(tau=tau)
                    np_model.fit(X_tr, y_tr)
                    self._models[tau] = np_model
                    self._is_lightgbm = False
            else:
                np_model = _NumpyQuantileRegressor(tau=tau)
                np_model.fit(X_tr, y_tr)
                self._models[tau] = np_model
                self._is_lightgbm = False

            logger.info("QuantileRULModel τ=%.2f fitted (engine=%s)", tau, "LightGBM" if self._is_lightgbm else "NumPy")

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return quantile RUL predictions as a DataFrame."""
        self._check_fitted()
        X_use = X[self._feature_cols].values.astype(np.float32)
        results = {}
        for tau in self.quantiles:
            col = f"rul_p{int(tau * 100):02d}"
            preds = self._models[tau].predict(X_use)
            results[col] = np.maximum(preds, 0.0)

        out = pd.DataFrame(results, index=X.index)
        # Enforce monotonicity: p10 ≤ p50 ≤ p90
        out["rul_p10"] = np.minimum(out["rul_p10"], out["rul_p50"])
        out["rul_p90"] = np.maximum(out["rul_p50"], out["rul_p90"])
        return out

    def predict_last_cycle(self, df: pd.DataFrame, unit_col: str = "unit_id") -> pd.DataFrame:
        """Predict quantile RULs for the last observed cycle of each engine."""
        last = df.groupby(unit_col, as_index=False).last()
        preds = self.predict(last)
        preds.insert(0, unit_col, last[unit_col].values)
        return preds.set_index(unit_col)

    def feature_importance(self, tau: float = 0.50) -> pd.Series:
        self._check_fitted()
        model = self._models[tau]
        if hasattr(model, "feature_importances_"):
            return pd.Series(
                model.feature_importances_,
                index=self._feature_cols,
            ).sort_values(ascending=False)
        elif hasattr(model, "weights"):
            return pd.Series(
                np.abs(model.weights),
                index=self._feature_cols,
            ).sort_values(ascending=False)
        return pd.Series(1.0, index=self._feature_cols)

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = Path(path) if path else MODELS_DIR / _QM_FNAME
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("QuantileRULModel saved → %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "QuantileRULModel":
        load_path = Path(path) if path else MODELS_DIR / _QM_FNAME
        with open(load_path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("QuantileRULModel loaded ← %s", load_path)
        return obj

    def _check_fitted(self) -> None:
        if not self._models:
            raise RuntimeError("QuantileRULModel is not fitted yet. Call .fit() first.")
