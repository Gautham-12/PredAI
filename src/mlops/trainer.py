"""
mlops.trainer
~~~~~~~~~~~~~
Automated closed-loop retraining pipeline triggered by data drift or new cycle ingestion.

Follows AURA-Fleet Section 5 (MLOps Closed-Loop Architecture, Page 4):
    Retraining is executed automatically whenever drift is flagged OR when completed
    run-to-failure cycle data is ingested. Newly trained candidate models are registered
    and promoted via champion-challenger gating (>3% NASA score improvement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, MODELS_DIR
from ml_engine.nasa_loss import nasa_score
from mlops.drift_monitor import DriftReport
from mlops.model_registry import ChampionChallengerRegistry

logger = logging.getLogger(__name__)


@dataclass
class RetrainingResult:
    """Outcome of an automated retraining trigger."""

    timestamp: str
    trigger_reason: str
    dataset_id: str
    candidate_version: Optional[str]
    nasa_score: float
    rmse: float
    promoted: bool
    status_message: str


class AutoRetrainer:
    """Orchestrates closed-loop model retraining and champion-challenger gating.

    Parameters
    ----------
    registry : ChampionChallengerRegistry, optional
    models_dir : Path, optional
    """

    def __init__(
        self,
        registry: Optional[ChampionChallengerRegistry] = None,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.registry = registry or ChampionChallengerRegistry(models_dir=self.models_dir)

    def on_drift_detected(
        self,
        drift_report: DriftReport,
        dataset_id: str = "FD001",
    ) -> RetrainingResult:
        """Trigger retraining when data drift is detected."""
        if not drift_report.overall_drift:
            return RetrainingResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger_reason="NO_DRIFT",
                dataset_id=dataset_id,
                candidate_version=None,
                nasa_score=0.0,
                rmse=0.0,
                promoted=False,
                status_message="Telemetry drift check passed. No retraining required.",
            )

        reason = (
            f"DATA_DRIFT_ALERT: {len(drift_report.drift_features)} features drifted "
            f"(mean PSI={drift_report.mean_psi:.3f})."
        )
        logger.warning("Initiating self-healing retraining: %s", reason)
        return self.retrain(dataset_id=dataset_id, trigger_reason=reason)

    def retrain(
        self,
        dataset_id: str = "FD001",
        trigger_reason: str = "MANUAL_OR_DEPOT_INGESTION",
    ) -> RetrainingResult:
        """Execute full training pipeline, evaluate candidate, and gate promotion."""
        from pipelines.train_pipeline import AURATrainPipeline

        logger.info("Executing AURA-Fleet retraining pipeline for %s...", dataset_id)
        pipeline = AURATrainPipeline(models_dir=self.models_dir)
        train_out = pipeline.run(dataset_id=dataset_id)

        candidate_score = train_out["test_nasa_score"]
        candidate_rmse = train_out["test_rmse"]
        artifacts = train_out["artifact_paths"]

        # Register candidate in Champion-Challenger registry
        candidate_version = self.registry.register_candidate(
            dataset_id=dataset_id,
            nasa_score=candidate_score,
            rmse=candidate_rmse,
            artifacts=artifacts,
            notes=f"Retrained via {trigger_reason}",
        )

        # Gate candidate
        promoted, gate_msg = self.registry.evaluate_and_gate(candidate_version)

        res = RetrainingResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger_reason=trigger_reason,
            dataset_id=dataset_id,
            candidate_version=candidate_version,
            nasa_score=candidate_score,
            rmse=candidate_rmse,
            promoted=promoted,
            status_message=gate_msg,
        )
        return res


# Alias for backward compatibility
AURATrainer = AutoRetrainer
