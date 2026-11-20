"""
Tests for AURA-Fleet Triage Agent (Layer 3).
Covers FSM priority classification, HITL escalation, and work order generation.
"""
import sys, os, unittest, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.triage_agent.fsm import TriageFSM, TriageResult, P0, P1, P2, P3, HITL
from src.triage_agent.work_order import WorkOrderGenerator


class TestTriageFSM(unittest.TestCase):
    """Tests for the deterministic Finite State Machine triage."""

    def setUp(self):
        self.fsm = TriageFSM()

    def test_p0_critical_rul_below_10(self):
        """P0 CRITICAL: RUL_p10 ≤ 10."""
        result = self.fsm.evaluate(rul_p10=5, rul_p50=20, rul_p90=40, health_index=0.5)
        self.assertEqual(result.priority, P0,
                         f"RUL_p10=5 should be {P0}, got {result.priority}")

    def test_p0_critical_rul_exactly_10(self):
        """P0 CRITICAL: RUL_p10 = 10 (boundary)."""
        result = self.fsm.evaluate(rul_p10=10, rul_p50=20, rul_p90=40, health_index=0.5)
        self.assertEqual(result.priority, P0,
                         f"RUL_p10=10 should be {P0}, got {result.priority}")

    def test_p0_critical_hi_below_020(self):
        """P0 CRITICAL: HI < 0.20 regardless of RUL."""
        result = self.fsm.evaluate(rul_p10=100, rul_p50=110, rul_p90=120, health_index=0.15)
        self.assertEqual(result.priority, P0,
                         f"HI=0.15 should be {P0} regardless of RUL, got {result.priority}")

    def test_p1_urgent_rul_range(self):
        """P1 URGENT: 10 < RUL_p10 ≤ 25."""
        result = self.fsm.evaluate(rul_p10=20, rul_p50=25, rul_p90=30, health_index=0.5)
        self.assertEqual(result.priority, P1,
                         f"RUL_p10=20 should be {P1}, got {result.priority}")

    def test_p1_boundary_11(self):
        """P1 URGENT: RUL_p10 = 11 (just above P0 boundary)."""
        result = self.fsm.evaluate(rul_p10=11, rul_p50=15, rul_p90=20, health_index=0.5)
        self.assertEqual(result.priority, P1,
                         f"RUL_p10=11 should be {P1}, got {result.priority}")

    def test_p1_boundary_25(self):
        """P1 URGENT: RUL_p10 = 25 (upper boundary)."""
        result = self.fsm.evaluate(rul_p10=25, rul_p50=30, rul_p90=34, health_index=0.5)
        self.assertEqual(result.priority, P1,
                         f"RUL_p10=25 should be {P1}, got {result.priority}")

    def test_p2_advisory_rul_range(self):
        """P2 ADVISORY: 25 < RUL_p10 ≤ 45."""
        result = self.fsm.evaluate(rul_p10=35, rul_p50=40, rul_p90=45, health_index=0.7)
        self.assertEqual(result.priority, P2,
                         f"RUL_p10=35 should be {P2}, got {result.priority}")

    def test_p3_normal_rul_above_45(self):
        """P3 NORMAL: RUL_p10 > 45."""
        result = self.fsm.evaluate(rul_p10=60, rul_p50=70, rul_p90=80, health_index=0.9)
        self.assertEqual(result.priority, P3,
                         f"RUL_p10=60 should be {P3}, got {result.priority}")

    def test_hitl_high_uncertainty_overrides_low_priority(self):
        """HITL escalation: interval (p90-p10) > 35 overrides P2/P3."""
        # P2 base priority with wide interval → HITL
        result = self.fsm.evaluate(rul_p10=30, rul_p50=50, rul_p90=70, health_index=0.6)
        interval = 70 - 30  # = 40 > 35
        self.assertEqual(result.priority, HITL,
                         f"Wide interval=40 on P2 should be {HITL}, got {result.priority}")
        self.assertTrue(result.hitl_required)

    def test_hitl_does_not_override_p0(self):
        """P0 is never overridden by HITL — HITL flag set but priority stays P0."""
        result = self.fsm.evaluate(rul_p10=5, rul_p50=25, rul_p90=55, health_index=0.5)
        self.assertEqual(result.priority, P0,
                         f"P0 should not be overridden by HITL, got {result.priority}")
        self.assertTrue(result.hitl_required)

    def test_hitl_does_not_override_p1(self):
        """P1 is never overridden by HITL — HITL flag set but priority stays P1."""
        result = self.fsm.evaluate(rul_p10=15, rul_p50=35, rul_p90=55, health_index=0.5)
        self.assertEqual(result.priority, P1,
                         f"P1 should not be overridden by HITL, got {result.priority}")
        self.assertTrue(result.hitl_required)

    def test_narrow_interval_no_hitl(self):
        """No HITL for narrow interval ≤ 35."""
        result = self.fsm.evaluate(rul_p10=30, rul_p50=40, rul_p90=50, health_index=0.7)
        self.assertFalse(result.hitl_required,
                         "Interval=20 should NOT trigger HITL")

    def test_sensor_conflict_triggers_hitl(self):
        """Conflicting sensor trends trigger HITL."""
        result = self.fsm.evaluate(
            rul_p10=35, rul_p50=40, rul_p90=45, health_index=0.7,
            sensor_trend_conflict=True,
        )
        self.assertTrue(result.hitl_required)
        self.assertIn("Conflicting", result.hitl_reason)


class TestTriageFSMEdgeCases(unittest.TestCase):
    """Edge case tests for the FSM."""

    def setUp(self):
        self.fsm = TriageFSM()

    def test_zero_rul(self):
        """RUL=0 should be P0."""
        result = self.fsm.evaluate(rul_p10=0, rul_p50=5, rul_p90=10, health_index=0.1)
        self.assertEqual(result.priority, P0)

    def test_max_rul(self):
        """RUL=125 should be P3."""
        result = self.fsm.evaluate(rul_p10=125, rul_p50=125, rul_p90=125, health_index=0.95)
        self.assertEqual(result.priority, P3)

    def test_p0_hi_overrides_high_rul(self):
        """Very low HI forces P0 even with high RUL."""
        result = self.fsm.evaluate(rul_p10=120, rul_p50=125, rul_p90=125, health_index=0.10)
        self.assertEqual(result.priority, P0,
                         "Low HI=0.10 should override high RUL to P0")

    def test_triage_result_auto_dispatchable(self):
        """Auto-dispatchable when no HITL."""
        result = self.fsm.evaluate(rul_p10=60, rul_p50=70, rul_p90=80, health_index=0.9)
        self.assertTrue(result.auto_dispatchable)

    def test_triage_result_not_auto_dispatchable_on_hitl(self):
        """Not auto-dispatchable when HITL flagged."""
        result = self.fsm.evaluate(rul_p10=30, rul_p50=50, rul_p90=70, health_index=0.6)
        self.assertFalse(result.auto_dispatchable)

    def test_deadline_p0(self):
        """P0 deadline should be short (5 cycles)."""
        result = self.fsm.evaluate(rul_p10=5, rul_p50=10, rul_p90=15, health_index=0.1)
        self.assertEqual(result.deadline_cycles, 5.0)


class TestWorkOrderGenerator(unittest.TestCase):
    """Tests for Work Order generation."""

    def setUp(self):
        self.generator = WorkOrderGenerator()

    def test_work_order_has_required_fields(self):
        """Work order should have all required schema fields."""
        wo = self.generator.create_work_order(
            asset_id=1,
            dataset_id="FD001",
            current_cycle=150,
            rul_p10=5.0,
            rul_p50=15.0,
            rul_p90=25.0,
            health_index=0.18,
        )

        # WorkOrderSchema dataclass/dict should have these
        self.assertTrue(hasattr(wo, 'work_order_id'))
        self.assertTrue(hasattr(wo, 'asset_id'))
        self.assertTrue(hasattr(wo, 'health_metrics'))
        self.assertTrue(hasattr(wo, 'triage_decision'))
        self.assertTrue(hasattr(wo, 'governance'))

    def test_work_order_priority_matches_p0(self):
        """P0-level inputs should produce P0 priority in the work order."""
        wo = self.generator.create_work_order(
            asset_id=42,
            dataset_id="FD001",
            current_cycle=200,
            rul_p10=3.0,
            rul_p50=10.0,
            rul_p90=20.0,
            health_index=0.12,
        )
        self.assertEqual(wo.triage_decision.priority, P0)

    def test_work_order_has_parts_for_critical(self):
        """P0 work orders should suggest parts."""
        wo = self.generator.create_work_order(
            asset_id=1,
            dataset_id="FD001",
            current_cycle=180,
            rul_p10=3.0,
            rul_p50=10.0,
            rul_p90=20.0,
            health_index=0.12,
        )
        self.assertGreater(len(wo.triage_decision.required_parts), 0,
                           "P0 work order should have required parts")

    def test_work_order_id_format(self):
        """Work order ID should contain dataset and asset info."""
        wo = self.generator.create_work_order(
            asset_id=7,
            dataset_id="FD002",
            current_cycle=100,
            rul_p10=50.0,
            rul_p50=60.0,
            rul_p90=70.0,
            health_index=0.85,
        )
        self.assertIn("FD002", wo.work_order_id)
        self.assertIn("ENG007", wo.work_order_id)

    def test_work_order_asset_id_formatting(self):
        """Asset ID should be formatted as TURBOFAN_ENGINE_NNN."""
        wo = self.generator.create_work_order(
            asset_id=1,
            dataset_id="FD001",
            current_cycle=100,
            rul_p10=50.0,
            rul_p50=60.0,
            rul_p90=70.0,
            health_index=0.85,
        )
        self.assertEqual(wo.asset_id, "TURBOFAN_ENGINE_001")

    def test_work_order_state_critical(self):
        """Low HI / low RUL should set state to CRITICAL."""
        wo = self.generator.create_work_order(
            asset_id=1,
            dataset_id="FD001",
            current_cycle=200,
            rul_p10=3.0,
            rul_p50=10.0,
            rul_p90=20.0,
            health_index=0.10,
        )
        self.assertEqual(wo.health_metrics.state, "CRITICAL")

    def test_work_order_governance_hitl(self):
        """Wide interval should flag human override required."""
        wo = self.generator.create_work_order(
            asset_id=1,
            dataset_id="FD001",
            current_cycle=100,
            rul_p10=30.0,
            rul_p50=50.0,
            rul_p90=70.0,  # interval = 40 > 35
            health_index=0.6,
        )
        self.assertTrue(wo.governance.human_override_required)


if __name__ == '__main__':
    unittest.main()
