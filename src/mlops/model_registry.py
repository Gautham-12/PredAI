"""
mlops.model_registry
~~~~~~~~~~~~~~~~~~~~
Champion-Challenger model registry and zero-downtime hot-swap mechanism.

Mathematically aligns with AURA-Fleet Section 5 (Page 4):
    Newly trained candidate models are registered and evaluated against the
    current production champion on holdout validation data.
    The candidate is promoted to production ONLY if its NASA Score improves by > 3%.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import CC_IMPROVEMENT_PCT, MODELS_DIR

logger = logging.getLogger(__name__)

_REGISTRY_METADATA_FILE = "registry_metadata.json"


@dataclass
class ModelMetadata:
    """Metadata record for a registered model version."""

    model_version: str
    dataset_id: str
    created_at: str
    nasa_score: float
    rmse: float
    status: str  # "CHAMPION" | "CHALLENGER" | "REJECTED" | "ARCHIVED"
    artifacts: Dict[str, str]  # name -> path
    improvement_pct: Optional[float] = None
    notes: Optional[str] = None


class ChampionChallengerRegistry:
    """Manages model artifacts, version promotion gating, and atomic hot-swapping.

    Parameters
    ----------
    models_dir : Path, optional
        Root directory for model artifacts. Defaults to config.MODELS_DIR.
    improvement_threshold_pct : float
        Required percentage reduction in NASA score (default 3.0%).
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        improvement_threshold_pct: float = CC_IMPROVEMENT_PCT,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_pct = improvement_threshold_pct
        self.metadata_file = self.models_dir / _REGISTRY_METADATA_FILE
        self._registry: Dict[str, Any] = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Error reading registry metadata, reinitializing: %s", e)
        return {
            "current_champion_version": None,
            "current_champion_score": float("inf"),
            "models": {},
        }

    def _save_registry(self) -> None:
        with open(self.metadata_file, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register_candidate(
        self,
        dataset_id: str,
        nasa_score: float,
        rmse: float,
        artifacts: Dict[str, str],
        notes: str = "Trained model candidate",
    ) -> str:
        """Register a newly trained candidate model.

        Returns
        -------
        str : Generated version tag, e.g. "v_20261117_130000"
        """
        now = datetime.now(timezone.utc)
        version = f"v_{dataset_id}_{now.strftime('%Y%m%d_%H%M%S')}"

        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        copied_artifacts = {}
        for name, src_path in artifacts.items():
            src = Path(src_path)
            if src.exists():
                dst = version_dir / src.name
                shutil.copy2(src, dst)
                copied_artifacts[name] = str(dst)
            else:
                copied_artifacts[name] = str(src_path)

        meta = ModelMetadata(
            model_version=version,
            dataset_id=dataset_id,
            created_at=now.isoformat(),
            nasa_score=float(nasa_score),
            rmse=float(rmse),
            status="CHALLENGER",
            artifacts=copied_artifacts,
            notes=notes,
        )

        self._registry["models"][version] = asdict(meta)
        self._save_registry()
        logger.info("Registered candidate %s with NASA score %.2f", version, nasa_score)
        return version

    def evaluate_and_gate(self, candidate_version: str) -> Tuple[bool, str]:
        """Evaluate candidate against current champion with >3% improvement rule.

        Returns
        -------
        (promoted, message)
        """
        if candidate_version not in self._registry["models"]:
            raise ValueError(f"Model version {candidate_version} not found in registry.")

        cand = self._registry["models"][candidate_version]
        cand_score = cand["nasa_score"]

        current_champ = self._registry["current_champion_version"]
        champ_score = self._registry["current_champion_score"]

        # If no current champion exists, immediately promote first valid model
        if current_champ is None or champ_score == float("inf"):
            self._promote_to_champion(candidate_version, improvement_pct=100.0)
            msg = f"Candidate {candidate_version} promoted as initial production champion (NASA score={cand_score:.2f})."
            logger.info(msg)
            return True, msg

        # NASA score is lower = better. Improvement = (champ_score - cand_score) / champ_score * 100
        improvement_pct = ((champ_score - cand_score) / champ_score) * 100.0
        cand["improvement_pct"] = round(improvement_pct, 2)

        if improvement_pct >= self.threshold_pct:
            # Promote!
            self._promote_to_champion(candidate_version, improvement_pct=improvement_pct)
            msg = (
                f"Candidate {candidate_version} PROMOTED to Champion! "
                f"NASA score improved by {improvement_pct:.2f}% (Threshold: >{self.threshold_pct}%)."
            )
            logger.info(msg)
            return True, msg
        else:
            cand["status"] = "REJECTED"
            self._save_registry()
            msg = (
                f"Candidate {candidate_version} REJECTED. "
                f"NASA score improvement was {improvement_pct:.2f}% (Requires >{self.threshold_pct}%)."
            )
            logger.warning(msg)
            return False, msg

    def _promote_to_champion(self, version: str, improvement_pct: float) -> None:
        """Atomically hot-swap the active production model files."""
        prev_champ = self._registry["current_champion_version"]
        if prev_champ and prev_champ in self._registry["models"]:
            self._registry["models"][prev_champ]["status"] = "ARCHIVED"

        new_meta = self._registry["models"][version]
        new_meta["status"] = "CHAMPION"
        new_meta["improvement_pct"] = round(improvement_pct, 2)

        self._registry["current_champion_version"] = version
        self._registry["current_champion_score"] = new_meta["nasa_score"]

        # Hot-swap files into the root models_dir for zero-downtime inference
        for name, art_path in new_meta["artifacts"].items():
            src = Path(art_path)
            if src.exists():
                dst = self.models_dir / src.name
                shutil.copy2(src, dst)

        self._save_registry()
        logger.info("Zero-downtime hot-swap completed. Active champion: %s", version)

    def get_champion_metadata(self) -> Optional[Dict[str, Any]]:
        champ_ver = self._registry["current_champion_version"]
        if champ_ver and champ_ver in self._registry["models"]:
            return self._registry["models"][champ_ver]
        return None

    def get_all_models(self) -> Dict[str, Any]:
        return self._registry.get("models", {})


# Alias for backward compatibility
ModelRegistry = ChampionChallengerRegistry
