"""
pipelines.train_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end training pipeline for the AURA-Fleet platform.

Orchestrates all 4 architectural layers:
    1. Data Ingestion & Pruning
    2. K-Means (k=6) Operational Regime Normalization
    3. Multi-Scale Kinematic Feature Extraction (w ∈ {5, 15, 30})
    4. Piecewise RUL Target Formulation (cap = 125)
    5. Stage-1 Autoencoder Health Index Training (t ≤ 30)
    6. Stage-2 Quantile Gradient Regressor (p10, p50, p90)
    7. NASA Asymmetric Score & RMSE Evaluation
    8. Telemetry Drift Monitor Baseline Fitting
    9. Artifact Persistence & Champion-Challenger Registration
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure src and root are on sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    MODELS_DIR,
    RUL_CAP,
    ACTIVE_SENSORS,
    SETTING_COLS,
)
from data_engineering.ingestion import load_dataset
from data_engineering.pruner import prune_sensors, get_feature_columns
from data_engineering.regime import RegimeClusterer
from data_engineering.normalizer import RegimeNormalizer
from data_engineering.features import FeatureEngineer
from ml_engine.rul_target import add_rul_target, add_test_rul_target
from ml_engine.autoencoder import HealthAutoencoder
from ml_engine.quantile_model import QuantileRULModel
from ml_engine.nasa_loss import nasa_score
from mlops.drift_monitor import DriftMonitor

logger = logging.getLogger(__name__)


def generate_synthetic_cmapss(
    dataset_id: str = "FD001",
    n_train_engines: int = 25,
    n_test_engines: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate realistic synthetic C-MAPSS data for benchmarking and testing."""
    rng = np.random.default_rng(42)
    from config import RAW_COLUMNS

    def _make_engine_data(unit_id: int, total_cycles: int, is_test: bool = False, terminal_rul: int = 0):
        records = []
        for t in range(1, total_cycles + 1):
            deg = max(0.0, (t - 40) / float(total_cycles - 40)) if t > 40 else 0.0

            # 3 settings
            if dataset_id in ("FD002", "FD004"):
                regime_idx = rng.integers(0, 6)
                alt = [0.0, 10.0, 20.0, 25.0, 35.0, 42.0][regime_idx]
                mach = [0.0, 0.25, 0.50, 0.62, 0.70, 0.84][regime_idx]
                tra = [100.0, 60.0, 80.0, 100.0, 60.0, 100.0][regime_idx]
            else:
                alt = float(rng.normal(0.0, 0.001))
                mach = float(rng.normal(0.0, 0.001))
                tra = float(rng.normal(100.0, 0.001))

            row = [unit_id, t, alt, mach, tra]

            # 21 sensors
            # Pruned sensors s1, s5, s6, s10, s16, s18, s19 have zero variance
            pruned_map = {1: 518.67, 5: 14.62, 6: 21.61, 10: 1.30, 16: 0.03, 18: 2388.0, 19: 100.0}
            for s_idx in range(1, 22):
                if s_idx in pruned_map:
                    val = pruned_map[s_idx]
                else:
                    # Degrading sensors
                    base = 500.0 + s_idx * 20.0
                    trend = (15.0 if s_idx in (2, 3, 4, 8, 9, 11, 13, 14) else -15.0) * deg
                    noise = rng.normal(0.0, 1.0)
                    val = base + trend + noise
                row.append(val)
            records.append(row)
        return records

    train_rows = []
    for u in range(1, n_train_engines + 1):
        total_len = rng.integers(140, 220)
        train_rows.extend(_make_engine_data(u, total_len))

    test_rows = []
    true_terminal_ruls = []
    for u in range(1, n_test_engines + 1):
        total_len = rng.integers(80, 150)
        term_rul = rng.integers(10, 110)
        true_terminal_ruls.append(term_rul)
        test_rows.extend(_make_engine_data(u, total_len, is_test=True, terminal_rul=term_rul))

    train_df = pd.DataFrame(train_rows, columns=RAW_COLUMNS)
    test_df = pd.DataFrame(test_rows, columns=RAW_COLUMNS)
    rul_series = pd.Series(true_terminal_ruls, name="rul")

    return train_df, test_df, rul_series


class AURATrainPipeline:
    """End-to-end training and artifact persistence manager.

    Parameters
    ----------
    models_dir : Path, optional
        Target directory to store fitted models and baselines.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        dataset_id: str = "FD001",
        data_dir: Optional[Path] = None,
        use_synthetic_if_missing: bool = True,
    ) -> Dict[str, Any]:
        """Execute full training pipeline.

        Returns
        -------
        dict with performance metrics, model references, and artifact paths.
        """
        logger.info("=== Starting AURA-Fleet Training Pipeline [%s] ===", dataset_id)

        # 1. Load Data
        try:
            train_df, test_df, rul_series = load_dataset(dataset_id=dataset_id, data_dir=data_dir)
        except Exception as e:
            if use_synthetic_if_missing:
                logger.warning("NASA files not found (%s). Using synthetic C-MAPSS dataset.", e)
                train_df, test_df, rul_series = generate_synthetic_cmapss(dataset_id=dataset_id)
            else:
                raise

        # 2. Prune Zero-Variance Sensors
        train_pruned = prune_sensors(train_df)
        test_pruned = prune_sensors(test_df)

        # 3. Fit K-Means Regime Clusterer (k=6)
        clusterer = RegimeClusterer()
        train_pruned["regime"] = clusterer.fit_predict(train_pruned)
        test_pruned["regime"] = clusterer.predict(test_pruned)

        # 4. Within-Regime Z-Score Normalization
        active_in_df = [s for s in ACTIVE_SENSORS if s in train_pruned.columns]
        normalizer = RegimeNormalizer(sensor_cols=active_in_df)
        train_norm = normalizer.fit_transform(train_pruned)
        test_norm = normalizer.transform(test_pruned)

        # 5. Extract Multi-Scale Kinematic Features (windows 5, 15, 30)
        feat_eng = FeatureEngineer(sensor_cols=active_in_df)
        train_feat = feat_eng.transform(train_norm)
        test_feat = feat_eng.transform(test_norm)

        # 6. Piecewise Linear Target RUL (cap=125)
        train_feat = add_rul_target(train_feat, rul_cap=RUL_CAP)
        test_feat = add_test_rul_target(test_feat, rul_series=rul_series, rul_cap=RUL_CAP)

        # 7. Stage-1 Autoencoder Health Index Training (pristine cycles t <= 30)
        model_input_cols = [c for c in active_in_df if c in train_feat.columns]
        ae = HealthAutoencoder(input_dim=len(model_input_cols), pristine_cycles=30)
        ae.fit(train_feat, feature_cols=model_input_cols)

        # Compute Health Index feature
        train_feat["health_index"] = ae.health_index(train_feat[model_input_cols].values)
        test_feat["health_index"] = ae.health_index(test_feat[model_input_cols].values)

        # 8. Stage-2 Quantile Gradient Boosting (p10, p50, p90)
        regressor_features = (
            model_input_cols
            + feat_eng.feature_names
            + ["health_index"]
        )
        regressor_features = [c for c in regressor_features if c in train_feat.columns]

        quantile_model = QuantileRULModel()
        quantile_model.fit(
            X=train_feat[regressor_features],
            y=train_feat["RUL"],
            feature_cols=regressor_features,
        )

        # 9. Evaluate on Terminal Cycles of Test Fleet
        test_terminal = test_feat.groupby("unit_id", as_index=False).last()
        preds = quantile_model.predict(test_terminal[regressor_features])

        y_true_terminal = test_terminal["RUL"].values
        y_pred_p50 = preds["rul_p50"].values
        y_pred_p10 = preds["rul_p10"].values

        score_p50 = float(nasa_score(y_true_terminal, y_pred_p50))
        rmse_p50 = float(np.sqrt(np.mean((y_pred_p50 - y_true_terminal) ** 2)))
        rmse_p10 = float(np.sqrt(np.mean((y_pred_p10 - y_true_terminal) ** 2)))

        logger.info(
            "Evaluation on %d test engines: NASA Score=%.2f, RMSE (p50)=%.2f, RMSE (p10)=%.2f",
            len(test_terminal),
            score_p50,
            rmse_p50,
            rmse_p10,
        )

        # 10. Fit Drift Monitor Baseline on Clean Training Features
        drift_monitor = DriftMonitor()
        drift_monitor.fit_baseline(train_feat, feature_cols=model_input_cols[:6])

        # 11. Save Artifacts to Models Directory
        clusterer_path = clusterer.save(self.models_dir / f"regime_clusterer_{dataset_id}.pkl")
        normalizer_path = normalizer.save(self.models_dir / f"regime_normalizer_{dataset_id}.pkl")
        ae_path = ae.save(self.models_dir / f"health_autoencoder_{dataset_id}.pkl")
        qm_path = quantile_model.save(self.models_dir / f"quantile_rul_{dataset_id}.pkl")
        drift_path = drift_monitor.save(self.models_dir / f"drift_baseline_{dataset_id}.pkl")

        # Also save default non-suffixed versions for immediate inference
        clusterer.save(self.models_dir / "regime_clusterer.pkl")
        normalizer.save(self.models_dir / "regime_normalizer.pkl")
        ae.save(self.models_dir / "health_autoencoder.pkl")
        quantile_model.save(self.models_dir / "quantile_rul_model.pkl")
        drift_monitor.save(self.models_dir / "drift_baseline.pkl")

        artifact_paths = {
            "regime_clusterer": str(clusterer_path),
            "regime_normalizer": str(normalizer_path),
            "health_autoencoder": str(ae_path),
            "quantile_model": str(qm_path),
            "drift_baseline": str(drift_path),
        }

        output = {
            "dataset_id": dataset_id,
            "test_nasa_score": score_p50,
            "test_rmse": rmse_p50,
            "test_rmse_p10": rmse_p10,
            "n_test_engines": len(test_terminal),
            "artifact_paths": artifact_paths,
            "models": {
                "clusterer": clusterer,
                "normalizer": normalizer,
                "feature_engineer": feat_eng,
                "autoencoder": ae,
                "quantile_model": quantile_model,
                "drift_monitor": drift_monitor,
            },
        }

        logger.info("=== AURA-Fleet Training Completed Successfully ===")
        return output
