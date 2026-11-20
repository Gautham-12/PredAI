"""
triage_agent.fsm
~~~~~~~~~~~~~~~~~
Finite State Machine for maintenance triage prioritisation.

Priority tiers (based on conservative p10 RUL and Health Index):
    P0 CRITICAL  : RUL_p10 ≤ 10 OR HI < 0.20
    P1 URGENT    : 10 < RUL_p10 ≤ 25
    P2 ADVISORY  : 25 < RUL_p10 ≤ 45
    P3 NORMAL    : RUL_p10 > 45
    HITL EXCEPT  : (RUL_p90 - RUL_p10) > 35 OR conflicting sensor trends

The FSM is stateless — each call to `evaluate()` returns a fresh decision
based solely on the supplied inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    TRIAGE_HITL_INTERVAL_THRESHOLD,
    TRIAGE_P0_HI_THRESHOLD,
    TRIAGE_P0_RUL_THRESHOLD,
    TRIAGE_P1_RUL_THRESHOLD,
    TRIAGE_P2_RUL_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Priority strings
P0 = "P0 - CRITICAL"
P1 = "P1 - URGENT"
P2 = "P2 - ADVISORY"
P3 = "P3 - NORMAL"
HITL = "HITL - EXCEPTION"


@dataclass
class TriageResult:
    """Output of a single FSM evaluation pass."""

    priority: str
    action_required: str
    deadline_cycles: float
    hitl_required: bool = False
    hitl_reason: Optional[str] = None
    flagged_sensors: List[str] = field(default_factory=list)

    @property
    def auto_dispatchable(self) -> bool:
        """Work orders may be auto-dispatched unless HITL is required."""
        return not self.hitl_required


# ── Action library ────────────────────────────────────────────────────────────

_ACTIONS = {
    P0: "Emergency shutdown and immediate depot overhaul.",
    P1: "Depot overhaul of High Pressure Compressor (HPC). Schedule within deadline.",
    P2: "Borescope inspection and predictive part pre-ordering. Monitor closely.",
    P3: "Continue normal operation. Next scheduled maintenance applies.",
    HITL: "Escalate to human engineering team for uncertainty resolution.",
}

_DEADLINES = {P0: 5.0, P1: 15.0, P2: 40.0, P3: 999.0, HITL: 20.0}


class TriageFSM:
    """Stateless triage finite state machine.

    Parameters
    ----------
    p0_rul_threshold : int
    p0_hi_threshold : float
    p1_rul_threshold : int
    p2_rul_threshold : int
    hitl_interval_threshold : int
        Width of (p90 - p10) interval above which HITL override is required.
    """

    def __init__(
        self,
        p0_rul_threshold: int = TRIAGE_P0_RUL_THRESHOLD,
        p0_hi_threshold: float = TRIAGE_P0_HI_THRESHOLD,
        p1_rul_threshold: int = TRIAGE_P1_RUL_THRESHOLD,
        p2_rul_threshold: int = TRIAGE_P2_RUL_THRESHOLD,
        hitl_interval_threshold: int = TRIAGE_HITL_INTERVAL_THRESHOLD,
    ) -> None:
        self.p0_rul = p0_rul_threshold
        self.p0_hi = p0_hi_threshold
        self.p1_rul = p1_rul_threshold
        self.p2_rul = p2_rul_threshold
        self.hitl_interval = hitl_interval_threshold

    def evaluate(
        self,
        rul_p10: float,
        rul_p50: float,
        rul_p90: float,
        health_index: float,
        sensor_trend_conflict: bool = False,
        flagged_sensors: Optional[List[str]] = None,
    ) -> TriageResult:
        """Run one triage evaluation and return a TriageResult.

        Parameters
        ----------
        rul_p10 : float
            Conservative (10th-percentile) RUL.
        rul_p50 : float
            Median RUL.
        rul_p90 : float
            Optimistic (90th-percentile) RUL.
        health_index : float
            Current Health Index in [0, 1].
        sensor_trend_conflict : bool
            True when sensors show conflicting degradation trends.
        flagged_sensors : list of str, optional
            Sensor names contributing to the alert.

        Returns
        -------
        TriageResult
        """
        sensors = flagged_sensors or []
        interval_width = rul_p90 - rul_p10

        # ── HITL check ────────────────────────────────────────────────────────
        hitl = False
        hitl_reason: Optional[str] = None
        if interval_width > self.hitl_interval:
            hitl = True
            hitl_reason = (
                f"Wide prediction interval ({interval_width:.1f} cycles > "
                f"{self.hitl_interval} threshold)."
            )
        elif sensor_trend_conflict:
            hitl = True
            hitl_reason = "Conflicting sensor degradation trends detected."

        # ── Priority assignment (using p10 — conservative) ────────────────────
        if rul_p10 <= self.p0_rul or health_index < self.p0_hi:
            priority = P0
        elif rul_p10 <= self.p1_rul:
            priority = P1
        elif rul_p10 <= self.p2_rul:
            priority = P2
        else:
            priority = P3

        # Override priority string if HITL and not already critical
        if hitl and priority not in (P0, P1):
            effective_priority = HITL
        else:
            effective_priority = priority

        deadline = _DEADLINES[effective_priority]
        # For HITL, also enforce the underlying priority deadline
        if hitl and priority in (P0, P1):
            deadline = min(deadline, _DEADLINES[priority])

        result = TriageResult(
            priority=effective_priority,
            action_required=_ACTIONS[effective_priority],
            deadline_cycles=deadline,
            hitl_required=hitl,
            hitl_reason=hitl_reason,
            flagged_sensors=sensors,
        )

        logger.info(
            "FSM: p10=%.1f, p50=%.1f, p90=%.1f, HI=%.3f → %s (HITL=%s)",
            rul_p10,
            rul_p50,
            rul_p90,
            health_index,
            result.priority,
            hitl,
        )
        return result
