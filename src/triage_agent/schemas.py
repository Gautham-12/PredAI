"""
triage_agent.schemas
~~~~~~~~~~~~~~~~~~~~~
Data models for the AURA-Fleet Work Order system.
Uses Pydantic when available, with a built-in standard library fallback.

These schemas define the canonical JSON structure emitted by the
Triage FSM and consumed by maintenance dispatching systems.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel as _PydanticBaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


if HAS_PYDANTIC:
    class HealthMetrics(_PydanticBaseModel):
        current_cycle: int = Field(..., description="Latest observed cycle number.")
        predicted_rul_p50: float = Field(..., description="Median (p50) RUL prediction.")
        conservative_rul_p10: float = Field(..., description="Conservative (p10) RUL prediction.")
        prediction_interval: str = Field(..., description='Human-readable interval string, e.g. "14 - 24 cycles".')
        health_index: float = Field(..., ge=0.0, le=1.0, description="Health Index in [0, 1]; 1 = pristine.")
        state: str = Field(..., description="Qualitative state label: NOMINAL | DEGRADED | CRITICAL.")

    class TriageDecision(_PydanticBaseModel):
        priority: str = Field(..., description="Priority tier: P0 CRITICAL | P1 URGENT | P2 ADVISORY | P3 NORMAL.")
        action_required: str = Field(..., description="Human-readable maintenance action.")
        deadline_cycles: float = Field(..., description="Maximum cycles until mandatory maintenance.")
        flagged_sensors: List[str] = Field(default_factory=list, description="Sensors contributing to the alert.")
        required_parts: List[str] = Field(default_factory=list, description="Part numbers required.")
        estimated_downtime_hours: float = Field(default=0.0, description="Estimated downtime in hours.")

    class Governance(_PydanticBaseModel):
        human_override_required: bool = Field(default=False, description="True when uncertainty warrants HITL.")
        auto_dispatched: bool = Field(default=False, description="True when automatically dispatched.")
        override_reason: Optional[str] = Field(default=None, description="Reason for override.")

    class WorkOrderSchema(_PydanticBaseModel):
        work_order_id: str = Field(..., description="Unique work-order identifier.")
        timestamp: datetime = Field(..., description="ISO 8601 timestamp.")
        asset_id: str = Field(..., description="Turbofan engine identifier.")
        dataset_id: str = Field(default="FD001", description="C-MAPSS sub-dataset.")
        health_metrics: HealthMetrics
        triage_decision: TriageDecision
        governance: Governance

        def to_json(self, indent: int = 2) -> str:
            d = self.model_dump()
            if isinstance(d.get("timestamp"), datetime):
                d["timestamp"] = d["timestamp"].isoformat()
            return json.dumps(d, indent=indent, default=str)

else:
    class _BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if isinstance(v, _BaseModel):
                    res[k] = v.model_dump()
                elif isinstance(v, datetime):
                    res[k] = v.isoformat()
                else:
                    res[k] = v
            return res

        def model_dump_json(self, indent: int = 2) -> str:
            return json.dumps(self.model_dump(), indent=indent, default=str)

        def to_json(self, indent: int = 2) -> str:
            return self.model_dump_json(indent=indent)

    class HealthMetrics(_BaseModel):
        def __init__(
            self,
            current_cycle: int,
            predicted_rul_p50: float,
            conservative_rul_p10: float,
            prediction_interval: str,
            health_index: float,
            state: str,
        ) -> None:
            super().__init__(
                current_cycle=current_cycle,
                predicted_rul_p50=predicted_rul_p50,
                conservative_rul_p10=conservative_rul_p10,
                prediction_interval=prediction_interval,
                health_index=health_index,
                state=state,
            )

    class TriageDecision(_BaseModel):
        def __init__(
            self,
            priority: str,
            action_required: str,
            deadline_cycles: float,
            flagged_sensors: Optional[List[str]] = None,
            required_parts: Optional[List[str]] = None,
            estimated_downtime_hours: float = 0.0,
        ) -> None:
            super().__init__(
                priority=priority,
                action_required=action_required,
                deadline_cycles=deadline_cycles,
                flagged_sensors=flagged_sensors or [],
                required_parts=required_parts or [],
                estimated_downtime_hours=estimated_downtime_hours,
            )

    class Governance(_BaseModel):
        def __init__(
            self,
            human_override_required: bool = False,
            auto_dispatched: bool = False,
            override_reason: Optional[str] = None,
        ) -> None:
            super().__init__(
                human_override_required=human_override_required,
                auto_dispatched=auto_dispatched,
                override_reason=override_reason,
            )

    class WorkOrderSchema(_BaseModel):
        def __init__(
            self,
            work_order_id: str,
            timestamp: datetime,
            asset_id: str,
            health_metrics: HealthMetrics,
            triage_decision: TriageDecision,
            governance: Governance,
            dataset_id: str = "FD001",
        ) -> None:
            super().__init__(
                work_order_id=work_order_id,
                timestamp=timestamp,
                asset_id=asset_id,
                dataset_id=dataset_id,
                health_metrics=health_metrics,
                triage_decision=triage_decision,
                governance=governance,
            )
