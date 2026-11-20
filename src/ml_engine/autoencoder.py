"""
ml_engine.autoencoder
~~~~~~~~~~~~~~~~~~~~~
Sparse undercomplete autoencoder for anomaly detection / health-index
computation in the AURA-Fleet pipeline.

Architecture (D = number of active feature dimensions):
    Encoder: D → 32 → 16 → 8
    Decoder: 8 → 16 → 32 → D

Training strategy:
    - Fit ONLY on "pristine" cycles (cycle ≤ AE_PRISTINE_CYCLES=30)
    - Baseline MSE statistics (µ, σ) computed on the same pristine window
    - Health Index: HI(t) = 1 - MSE(t) / MSE_max
    - FTTP trigger: MSE(t) > µ_baseline + k*σ_baseline for k consecutive cycles

Resilience:
    Uses PyTorch when available, with a built-in NumPy MLP autoencoder fallback.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    AE_FTTP_CONSEC,
    AE_FTTP_SIGMA,
    AE_PRISTINE_CYCLES,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)

_AE_FNAME = "health_autoencoder.pkl"

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not found; using resilient NumPy Autoencoder implementation.")


if HAS_TORCH:
    class _AETorchNetwork(nn.Module):
        """PyTorch undercomplete autoencoder: D→32→16→8→16→32→D."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)


class _AENumpyNetwork:
    """NumPy-based symmetric autoencoder D→32→16→8→16→32→D."""

    def __init__(self, input_dim: int, seed: int = 42) -> None:
        self.input_dim = input_dim
        rng = np.random.default_rng(seed)
        # He initialization
        self.W1 = rng.standard_normal((input_dim, 32)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(32)
        self.W2 = rng.standard_normal((32, 16)) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros(16)
        self.W3 = rng.standard_normal((16, 8)) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros(8)
        self.W4 = rng.standard_normal((8, 16)) * np.sqrt(2.0 / 8)
        self.b4 = np.zeros(16)
        self.W5 = rng.standard_normal((16, 32)) * np.sqrt(2.0 / 16)
        self.b5 = np.zeros(32)
        self.W6 = rng.standard_normal((32, input_dim)) * np.sqrt(2.0 / 32)
        self.b6 = np.zeros(input_dim)

    def encode(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0.0, X @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        h3 = np.maximum(0.0, h2 @ self.W3 + self.b3)
        return h3

    def forward(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0.0, X @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        h3 = np.maximum(0.0, h2 @ self.W3 + self.b3)
        h4 = np.maximum(0.0, h3 @ self.W4 + self.b4)
        h5 = np.maximum(0.0, h4 @ self.W5 + self.b5)
        out = h5 @ self.W6 + self.b6
        return out

    def fit(self, X: np.ndarray, epochs: int = 60, lr: float = 1e-3, batch_size: int = 128) -> None:
        """Adam optimizer implementation in NumPy."""
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                  self.W4, self.b4, self.W5, self.b5, self.W6, self.b6]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        n = len(X)

        for _ in range(epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                xb = X[idx]
                if len(xb) == 0:
                    continue

                # Forward pass with cache
                h1_raw = xb @ self.W1 + self.b1
                h1 = np.maximum(0.0, h1_raw)
                h2_raw = h1 @ self.W2 + self.b2
                h2 = np.maximum(0.0, h2_raw)
                h3_raw = h2 @ self.W3 + self.b3
                h3 = np.maximum(0.0, h3_raw)
                h4_raw = h3 @ self.W4 + self.b4
                h4 = np.maximum(0.0, h4_raw)
                h5_raw = h4 @ self.W5 + self.b5
                h5 = np.maximum(0.0, h5_raw)
                out = h5 @ self.W6 + self.b6

                # MSE gradient: dL/dout = 2 * (out - xb) / (B * D)
                B, D = xb.shape
                dout = 2.0 * (out - xb) / (B * D)

                dW6 = h5.T @ dout
                db6 = np.sum(dout, axis=0)

                dh5 = dout @ self.W6.T
                dh5_raw = dh5 * (h5_raw > 0)
                dW5 = h4.T @ dh5_raw
                db5 = np.sum(dh5_raw, axis=0)

                dh4 = dh5_raw @ self.W5.T
                dh4_raw = dh4 * (h4_raw > 0)
                dW4 = h3.T @ dh4_raw
                db4 = np.sum(dh4_raw, axis=0)

                dh3 = dh4_raw @ self.W4.T
                dh3_raw = dh3 * (h3_raw > 0)
                dW3 = h2.T @ dh3_raw
                db3 = np.sum(dh3_raw, axis=0)

                dh2 = dh3_raw @ self.W3.T
                dh2_raw = dh2 * (h2_raw > 0)
                dW2 = h1.T @ dh2_raw
                db2 = np.sum(dh2_raw, axis=0)

                dh1 = dh2_raw @ self.W2.T
                dh1_raw = dh1 * (h1_raw > 0)
                dW1 = xb.T @ dh1_raw
                db1 = np.sum(dh1_raw, axis=0)

                grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, dW6, db6]
                t += 1
                for i in range(len(params)):
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i]
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (grads[i] ** 2)
                    m_hat = m[i] / (1.0 - beta1 ** t)
                    v_hat = v[i] / (1.0 - beta2 ** t)
                    params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class HealthAutoencoder:
    """Train and apply a reconstruction-error autoencoder for health monitoring.

    Parameters
    ----------
    input_dim : int
        Number of input features (D).
    pristine_cycles : int
        Cycles ≤ this threshold are considered "pristine" (no degradation).
    fttp_sigma : float
        FTTP threshold expressed as standard deviations above baseline mean.
    fttp_consec : int
        Number of consecutive cycles above threshold to trigger FTTP.
    lr : float
        Adam learning rate.
    epochs : int
        Training epochs.
    batch_size : int
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        input_dim: int,
        pristine_cycles: int = AE_PRISTINE_CYCLES,
        fttp_sigma: float = AE_FTTP_SIGMA,
        fttp_consec: int = AE_FTTP_CONSEC,
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.pristine_cycles = pristine_cycles
        self.fttp_sigma = fttp_sigma
        self.fttp_consec = fttp_consec
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device_str = device

        self._net: Optional[Any] = None
        self._is_torch: bool = HAS_TORCH
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._mse_max: float = 1.0

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        cycle_col: str = "cycle",
    ) -> "HealthAutoencoder":
        """Train on pristine cycles only."""
        pristine = df[df[cycle_col] <= self.pristine_cycles]
        if len(pristine) == 0:
            raise ValueError(
                f"No pristine cycles found (cycle ≤ {self.pristine_cycles})."
            )

        X = pristine[feature_cols].values.astype(np.float32)
        self.input_dim = X.shape[1]

        if HAS_TORCH:
            try:
                device = torch.device(self.device_str)
                self._net = _AETorchNetwork(self.input_dim).to(device)
                optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
                criterion = nn.MSELoss()

                tensor_ds = TensorDataset(torch.from_numpy(X))
                loader = DataLoader(tensor_ds, batch_size=self.batch_size, shuffle=True)

                self._net.train()
                for epoch in range(self.epochs):
                    for (batch,) in loader:
                        batch = batch.to(device)
                        optimiser.zero_grad()
                        recon = self._net(batch)
                        loss = criterion(recon, batch)
                        loss.backward()
                        optimiser.step()

                self._is_torch = True
            except Exception as e:
                logger.warning("Torch training failed (%s); falling back to NumPy AE.", e)
                self._is_torch = False
                self._net = _AENumpyNetwork(self.input_dim)
                self._net.fit(X, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size)
        else:
            self._is_torch = False
            self._net = _AENumpyNetwork(self.input_dim)
            self._net.fit(X, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size)

        pristine_mse = self._reconstruction_mse(X)
        self._baseline_mean = float(np.mean(pristine_mse))
        self._baseline_std = float(max(1e-6, np.std(pristine_mse, ddof=1)))
        self._mse_max = float(np.max(pristine_mse)) + 1e-4

        logger.info(
            "HealthAutoencoder fitted — µ=%.4f, σ=%.4f, mse_max=%.4f (engine=%s)",
            self._baseline_mean,
            self._baseline_std,
            self._mse_max,
            "PyTorch" if self._is_torch else "NumPy",
        )
        return self

    def _reconstruction_mse(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        if self._is_torch and HAS_TORCH and hasattr(self._net, "eval"):
            self._net.eval()
            with torch.no_grad():
                t = torch.from_numpy(X_arr).to(torch.device(self.device_str))
                recon = self._net(t).cpu().numpy()
            return np.mean((X_arr - recon) ** 2, axis=1)
        else:
            recon = self._net.forward(X_arr)
            return np.mean((X_arr - recon) ** 2, axis=1)

    def reconstruction_mse(self, X: np.ndarray) -> np.ndarray:
        return self._reconstruction_mse(X)

    def health_index(self, X: np.ndarray) -> np.ndarray:
        """HI(t) = 1 - MSE(t) / MSE_max, clipped to [0, 1]."""
        mse = self._reconstruction_mse(X)
        hi = 1.0 - (mse / self._mse_max)
        return np.clip(hi, 0.0, 1.0)

    def fttp_flags(self, mse_sequence: np.ndarray) -> np.ndarray:
        """First Time To Predict flags along a single trajectory."""
        threshold = self._baseline_mean + self.fttp_sigma * self._baseline_std
        above = mse_sequence > threshold
        flags = np.zeros(len(above), dtype=bool)
        consec = 0
        for i, a in enumerate(above):
            consec = consec + 1 if a else 0
            if consec >= self.fttp_consec:
                flags[i] = True
        return flags

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        if self._is_torch and HAS_TORCH and hasattr(self._net, "encode"):
            self._net.eval()
            with torch.no_grad():
                t = torch.from_numpy(X_arr).to(torch.device(self.device_str))
                return self._net.encode(t).cpu().numpy()
        else:
            return self._net.encode(X_arr)

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = Path(path) if path else MODELS_DIR / _AE_FNAME
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info("HealthAutoencoder saved → %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HealthAutoencoder":
        load_path = Path(path) if path else MODELS_DIR / _AE_FNAME
        with open(load_path, "rb") as f:
            obj = pickle.load(f)
        logger.info("HealthAutoencoder loaded ← %s", load_path)
        return obj

    def _check_fitted(self) -> None:
        if self._net is None:
            raise RuntimeError("HealthAutoencoder is not fitted yet. Call .fit() first.")

    @property
    def threshold(self) -> float:
        return self._baseline_mean + self.fttp_sigma * self._baseline_std
