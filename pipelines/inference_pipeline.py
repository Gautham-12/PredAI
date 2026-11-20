"""
pipelines.inference_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real-time streaming inference orchestrator for the AURA-Fleet platform.

Ingests streaming raw multi-sensor telemetry, performs regime normalization,
computes Health Index and Quantile RUL horizons (p10, p50, p90), evaluates
data drift, and outputs ERP-ready JSON work orders via the deterministic Triage Agent.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure src and root are on sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from config import ACTIVE_SENSORS, MODELS_DIR
from data_engineering.features import FeatureEngineer
from data_engineering.normalizer import RegimeNormalizer
from data_engineering.pruner import prune_sensors
from data_engineering.regime import RegimeClusterer
from ml_engine.autoencoder import HealthAutoencoder
from ml_engine.quantile_model import QuantileRULModel
from mlops.drift_monitor import DriftMonitor, DriftReport
from triage_agent.schemas import WorkOrderSchema
from triage_agent.work_order import WorkOrderGenerator

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Complete diagnostic and operational output for an engine evaluation."""

    asset_id: str
    current_cycle: int
    rul_p10: float
    rul_p50: float
    rul_p90: float
    health_index: float
    state: str
    priority: str
    drift_alert: bool
    drift_report: Optional[DriftReport]
    work_order: WorkOrderSchema

    def to_json(self, indent: int = 2) -> str:
        return self.work_order.to_json(indent=indent)


class AURAInferencePipeline:
    """Production inference pipeline executing with deterministic sub-millisecond triage.

    Parameters
    ----------
    models_dir : Path, optional
        Directory where production model artifacts are stored.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.clusterer: Optional[RegimeClusterer] = None
        self.normalizer: Optional[RegimeNormalizer] = None
        self.autoencoder: Optional[HealthAutoencoder] = None
        self.quantile_model: Optional[QuantileRULModel] = None
        self.drift_monitor: Optional[DriftMonitor] = None
        self.work_order_gen = WorkOrderGenerator()
        self.feature_eng = FeatureEngineer()

        self._load_production_models()

    def _load_production_models(self) -> None:
        """Load active production champion models from disk."""
        try:
            self.clusterer = RegimeClusterer.load(self.models_dir / "regime_clusterer.pkl")
            self.normalizer = RegimeNormalizer.load(self.models_dir / "regime_normalizer.pkl")
            self.autoencoder = HealthAutoencoder.load(self.models_dir / "health_autoencoder.pkl")
            self.quantile_model = QuantileRULModel.load(self.models_dir / "quantile_rul_model.pkl")
            self.drift_monitor = DriftMonitor.load(self.models_dir / "drift_baseline.pkl")
            logger.info("AURA-Fleet inference models loaded successfully.")
        except Exception as e:
            logger.warning(
                "Could not load all production models (%s). Call load_from_dict() or train first.",
                e,
            )

    def load_from_dict(self, models_dict: Dict[str, Any]) -> None:
        """In-memory model attachment (e.g. immediately after training)."""
        self.clusterer = models_dict["clusterer"]
        self.normalizer = models_dict["normalizer"]
        self.autoencoder = models_dict["autoencoder"]
        self.quantile_model = models_dict["quantile_model"]
        self.drift_monitor = models_dict["drift_monitor"]
        if "feature_engineer" in models_dict:
            self.feature_eng = models_dict["feature_engineer"]

    def process_engine_trajectory(
        self,
        engine_df: pd.DataFrame,
        dataset_id: str = "FD001",
        check_drift: bool = True,
    ) -> InferenceResult:
        """Process a full or partial engine trajectory and evaluate its terminal state.

        Parameters
        ----------
        engine_df : pd.DataFrame
            Telemetry rows for ONE engine, sorted chronologically by cycle.
        dataset_id : str
        check_drift : bool

        Returns
        -------
        InferenceResult with work order and diagnostic telemetry.
        """
        if len(engine_df) == 0:
            raise ValueError("Input trajectory is empty.")

        # 1. Prune Zero-Variance Sensors
        pruned = prune_sensors(engine_df)

        # 2. Regime Assignment
        pruned["regime"] = self.clusterer.predict(pruned)

        # 3. Within-Regime Z-Score Normalization
        norm = self.normalizer.transform(pruned)

        # 4. Multi-Scale Kinematic Features
        feat = self.feature_eng.transform(norm).copy()

        # 5. Stage-1 Health Index
        active_in_df = [s for s in ACTIVE_SENSORS if s in feat.columns]
        feat["health_index"] = self.autoencoder.health_index(feat[active_in_df].values)

        # 6. Quantile RUL Model Prediction
        regressor_features = (
            active_in_df
            + self.feature_eng.feature_names
            + ["health_index"]
        )
        regressor_features = [c for c in regressor_features if c in feat.columns]
        preds = self.quantile_model.predict(feat[regressor_features])

        # Terminal row evaluation
        last_idx = feat.index[-1]
        last_row = feat.loc[last_idx]
        current_cycle = int(last_row.get("cycle", len(feat)))
        unit_id = last_row.get("unit_id", 1)

        p10 = float(preds.loc[last_idx, "rul_p10"])
        p50 = float(preds.loc[last_idx, "rul_p50"])
        p90 = float(preds.loc[last_idx, "rul_p90"])
        hi = float(last_row["health_index"])

        # 7. Drift Detection Check
        drift_report = None
        drift_alert = False
        if check_drift and self.drift_monitor is not None:
            drift_cols = [c for c in active_in_df[:6] if c in feat.columns]
            drift_report = self.drift_monitor.evaluate_drift(feat, feature_cols=drift_cols)
            drift_alert = drift_report.overall_drift

        # 8. Synthesize ERP Work Order
        raw_last = engine_df.iloc[-1]
        work_order = self.work_order_gen.create_work_order(
            asset_id=unit_id,
            dataset_id=dataset_id,
            current_cycle=current_cycle,
            rul_p10=p10,
            rul_p50=p50,
            rul_p90=p90,
            health_index=hi,
            latest_sensors=raw_last,
            sensor_trend_conflict=False,
        )

        return InferenceResult(
            asset_id=f"TURBOFAN_ENGINE_{int(unit_id):03d}",
            current_cycle=current_cycle,
            rul_p10=p10,
            rul_p50=p50,
            rul_p90=p90,
            health_index=hi,
            state=work_order.health_metrics.state,
            priority=work_order.triage_decision.priority,
            drift_alert=drift_alert,
            drift_report=drift_report,
            work_order=work_order,
        )


def main():
    import argparse
    from data_engineering.ingestion import load_dataset

    parser = argparse.ArgumentParser(description="AURA-Fleet Real-time Inference Pipeline")
    parser.add_argument("--dataset", type=str, default="FD001", help="C-MAPSS dataset ID (FD001-FD004)")
    parser.add_argument("--engine", type=int, default=1, help="Unit ID of engine to evaluate")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to models directory")
    args = parser.parse_args()

    pipeline = AURAInferencePipeline(models_dir=args.models_dir)
    _, test_df, _ = load_dataset(dataset_id=args.dataset)
    engine_df = test_df[test_df["unit_id"] == args.engine]
    if len(engine_df) == 0:
        print(f"Engine unit_id {args.engine} not found in test dataset {args.dataset}.")
        return

    result = pipeline.process_engine_trajectory(engine_df, dataset_id=args.dataset)
    print(result.to_json())


if __name__ == "__main__":
    main()

# Backward and forward compatibility aliases
PredAIInferencePipeline = AURAInferencePipeline


