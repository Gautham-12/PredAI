"""
triage_agent.work_order
~~~~~~~~~~~~~~~~~~~~~~~
Generates ERP-ready JSON work orders from model outputs and Triage FSM decisions.

Complies strictly with the AURA-Fleet Work Order Schema (PDF Page 4).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from triage_agent.fsm import TriageFSM, TriageResult, P0, P1, P2, P3, HITL
from triage_agent.schemas import (
    HealthMetrics,
    TriageDecision,
    Governance,
    WorkOrderSchema,
)

logger = logging.getLogger(__name__)

# Human-readable sensor descriptions
SENSOR_DESCRIPTIONS = {
    "s1": "Fan Inlet Temp",
    "s2": "LPC Outlet Temp",
    "s3": "HPC Outlet Temp",
    "s4": "LPT Outlet Temp",
    "s5": "LPC Outlet Pressure",
    "s6": "HPC Inlet Pressure",
    "s7": "HPC Outlet Pressure",
    "s8": "Physical Fan Speed",
    "s9": "Physical Core Speed",
    "s10": "Bypass Duct Pressure",
    "s11": "Static HPT Pressure",
    "s12": "Fuel Flow Ratio",
    "s13": "Corrected Fan Speed",
    "s14": "HPC Outlet Temp",
    "s15": "HPT Coolant Bleed",
    "s16": "Bypass Ratio",
    "s17": "Bleed Enthalpy",
    "s18": "Demanded Fan Speed",
    "s19": "Demanded Corrected Fan Speed",
    "s20": "HPT Coolant Bleed",
    "s21": "LPT Coolant Bleed",
}

# Bill of Materials (BOM) per priority tier & fault mode
PARTS_CATALOG = {
    P0: [
        "Part# HPC-BLADE-KIT-V2",
        "Part# HPT-ROTOR-ASSY-01",
        "Part# SEAL-RING-HPT-09",
        "Part# BEARING-MAIN-4B",
    ],
    P1: [
        "Part# HPC-BLADE-KIT-V2",
        "Part# SEAL-RING-HPT-09",
    ],
    P2: [
        "Part# BORESCOPE-PROBE-KIT",
        "Part# GASKET-SET-HPC",
    ],
    P3: [],
    HITL: [
        "Part# DIAGNOSTIC-HARNESS-KIT",
    ],
}

DOWNTIME_HOURS = {
    P0: 12.0,
    P1: 6.5,
    P2: 2.0,
    P3: 0.0,
    HITL: 4.0,
}

ACTIONS = {
    P0: "Immediate emergency grounding and hangar rapid overhaul",
    P1: "Depot overhaul of High Pressure Compressor (HPC)",
    P2: "Borescopic inspection during next scheduled turn",
    P3: "Nominal operation logging",
    HITL: "Safety quarantine for reliability engineering verification",
}


class WorkOrderGenerator:
    """Synthesizes structured enterprise work orders from telemetry evaluations.

    Parameters
    ----------
    fsm : TriageFSM, optional
        Deterministic finite state machine instance.
    """

    def __init__(self, fsm: Optional[TriageFSM] = None) -> None:
        self.fsm = fsm or TriageFSM()

    def identify_flagged_sensors(
        self,
        latest_sensor_readings: Union[pd.Series, Dict[str, float]],
        top_k: int = 2,
    ) -> List[str]:
        """Identify sensors showing largest deviation from baseline or extreme readings."""
        if isinstance(latest_sensor_readings, pd.Series):
            readings = latest_sensor_readings.to_dict()
        else:
            readings = dict(latest_sensor_readings)

        # Filter to sensor columns (s1..s21)
        sensor_items = [
            (k, abs(float(v)))
            for k, v in readings.items()
            if (k.startswith("s") and k[1:].isdigit()) or (k.startswith("sensor_") and k[7:].isdigit())
        ]

        if not sensor_items:
            # Fallback to key critical turbofan sensors
            return ["sensor_11 (Static HPT Pressure)", "sensor_14 (HPC Outlet Temp)"]

        # Sort by magnitude of reading / deviation
        sensor_items.sort(key=lambda x: x[1], reverse=True)
        flagged = []
        for name, _ in sensor_items[:top_k]:
            clean_name = name.lower()
            if clean_name.startswith("sensor_"):
                s_key = "s" + clean_name.split("_")[1]
                s_idx = clean_name.split("_")[1]
            elif clean_name.startswith("s"):
                s_key = clean_name
                s_idx = clean_name[1:]
            else:
                s_key = clean_name
                s_idx = clean_name

            desc = SENSOR_DESCRIPTIONS.get(s_key, "Telemetry Sensor")
            flagged.append(f"sensor_{s_idx} ({desc})")

        return flagged

    def create_work_order(
        self,
        asset_id: Union[int, str],
        dataset_id: str,
        current_cycle: int,
        rul_p10: float,
        rul_p50: float,
        rul_p90: float,
        health_index: float,
        latest_sensors: Optional[Union[pd.Series, Dict[str, float]]] = None,
        sensor_trend_conflict: bool = False,
        timestamp: Optional[datetime] = None,
    ) -> WorkOrderSchema:
        """Create a complete WorkOrderSchema instance."""
        # Clean formatting for asset_id
        try:
            int_id = int(float(asset_id))
            asset_str = f"TURBOFAN_ENGINE_{int_id:03d}"
            asset_code = f"ENG{int_id:03d}"
        except (ValueError, TypeError):
            asset_str = str(asset_id)
            asset_code = str(asset_id).replace("TURBOFAN_ENGINE_", "ENG")

        # Flagged sensors
        sensors_input = latest_sensors if latest_sensors is not None else {}
        flagged = self.identify_flagged_sensors(sensors_input)

        # Run FSM
        triage_res: TriageResult = self.fsm.evaluate(
            rul_p10=rul_p10,
            rul_p50=rul_p50,
            rul_p90=rul_p90,
            health_index=health_index,
            sensor_trend_conflict=sensor_trend_conflict,
            flagged_sensors=flagged,
        )

        # Derive state label
        if triage_res.priority == P0 or health_index < 0.20 or rul_p10 <= 10:
            state = "CRITICAL"
        elif triage_res.priority in (P1, P2) or health_index < 0.70:
            state = "DEGRADED"
        else:
            state = "NOMINAL"

        p10_int = max(0, int(round(rul_p10)))
        p50_int = max(0, int(round(rul_p50)))
        p90_int = max(0, int(round(rul_p90)))
        interval_str = f"{p10_int} - {p90_int} cycles"

        health_metrics = HealthMetrics(
            current_cycle=int(current_cycle),
            predicted_rul_p50=float(p50_int),
            conservative_rul_p10=float(p10_int),
            prediction_interval=interval_str,
            health_index=round(float(health_index), 2),
            state=state,
        )

        priority = triage_res.priority
        action_req = ACTIONS.get(priority, triage_res.action_required)
        parts = PARTS_CATALOG.get(priority, [])
        downtime = DOWNTIME_HOURS.get(priority, 0.0)
        deadline = p10_int if p10_int > 0 else 1.0

        triage_decision = TriageDecision(
            priority=priority,
            action_required=action_req,
            deadline_cycles=float(deadline),
            flagged_sensors=flagged,
            required_parts=parts,
            estimated_downtime_hours=float(downtime),
        )

        now = timestamp or datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        work_order_id = f"WO-CMAPSS-{dataset_id}-{asset_code}-{date_str}"

        governance = Governance(
            human_override_required=triage_res.hitl_required,
            auto_dispatched=triage_res.auto_dispatchable and priority != P3,
            override_reason=triage_res.hitl_reason,
        )

        return WorkOrderSchema(
            work_order_id=work_order_id,
            timestamp=now,
            asset_id=asset_str,
            dataset_id=dataset_id,
            health_metrics=health_metrics,
            triage_decision=triage_decision,
            governance=governance,
        )
