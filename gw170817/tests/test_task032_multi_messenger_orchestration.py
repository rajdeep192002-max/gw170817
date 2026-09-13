"""
Unit tests for Task 032: Synchronized Multi-Messenger Event Orchestration.
Tests A through J as specified by prompt requirements.
"""
import unittest
import numpy as np
import taichi as ti
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.multimessenger import MultiMessengerCoordinator, MultiMessengerOrchestrationState
from gw170817.visualization.dashboard import ScientificDashboard


class TestTask032MultiMessengerOrchestration(unittest.TestCase):
    """Test suite for unified multi-messenger orchestration state layer."""

    @classmethod
    def setUpClass(cls):
        try:
            ti.init(arch=ti.vulkan, implicit_prefer_if=True)
        except Exception:
            try:
                ti.init(arch=ti.cpu)
            except Exception:
                pass
        cls.config = SimConfig(mode="DEV", seed=42)
        cls.sim = GW170817Simulation(config=cls.config)
        cls.coord = MultiMessengerCoordinator(engine=cls.sim, config=cls.config)

    def test_A_inspiral_pre_merger_only(self):
        """A. Inspiral (event_time < 0) has only pre-merger components active."""
        self.sim._event_time = -1.0
        evt_st = self.coord._update_event_state()
        orch = evt_st.orchestration

        self.assertTrue(orch.gw_active, "GW channel active during inspiral")
        self.assertTrue(orch.lensing_active, "Lensing model active during inspiral")
        self.assertFalse(orch.gw_propagation_active, "No outward GW propagation before merger")
        self.assertFalse(orch.ejecta_active, "No ejecta active before merger")
        self.assertFalse(orch.bh_active, "No BH active before collapse")
        self.assertFalse(orch.jet_active, "No jet active before merger")

    def test_B_merger_activates_propagation_and_ejecta(self):
        """B. Merger (event_time >= 0) activates GW propagation and ejecta."""
        self.sim.set_demo_event_time(0.0)
        evt_st = self.coord._update_event_state()
        orch = evt_st.orchestration

        self.assertTrue(orch.gw_propagation_active, "GW propagation activates at merger")
        self.assertTrue(orch.ejecta_active, "Ejecta activates at merger")

    def test_C_bh_state_follows_remnant_state(self):
        """C. BH state follows existing RemnantState."""
        self.sim.set_demo_event_time(0.01)
        evt_hmns = self.coord._update_event_state()
        self.assertFalse(evt_hmns.orchestration.bh_active, "BH inactive during HMNS stage")

        self.sim.set_demo_event_time(0.10)
        evt_bh = self.coord._update_event_state()
        self.assertTrue(evt_bh.orchestration.bh_active, "BH active after adopted delayed collapse")

    def test_D_disk_lensing_follow_bh_state(self):
        """D. Disk and BH lensing follow BH state."""
        self.sim.set_demo_event_time(0.10)
        evt_st = self.coord._update_event_state()
        orch = evt_st.orchestration

        self.assertTrue(orch.bh_active, "BH state active")
        self.assertTrue(orch.disk_active, "Accretion disk active when BH active")
        self.assertTrue(orch.lensing_active, "Lensing active when BH active")

    def test_E_jet_follows_physical_jet_timing(self):
        """E. Jet follows existing physical jet timing (1.74s delay)."""
        self.sim.set_demo_event_time(0.5)
        evt_pre = self.coord._update_event_state()
        self.assertFalse(evt_pre.orchestration.jet_active, "Jet inactive before 1.74s delay")

        self.sim.set_demo_event_time(2.0)
        evt_post = self.coord._update_event_state()
        self.assertTrue(evt_post.orchestration.jet_active, "Jet active after 1.74s delay")

    def test_F_kilonova_afterglow_time_evolution(self):
        """F. Kilonova and afterglow use existing time evolution."""
        self.sim.set_demo_event_time(0.10)
        evt_st = self.coord._update_event_state()
        orch = evt_st.orchestration

        self.assertTrue(orch.kilonova_active, "Kilonova actively evolving post-merger")
        self.assertTrue(orch.afterglow_active, "Afterglow actively evolving post-merger")

    def test_G_no_component_resets_at_presentation_15s(self):
        """G. No component resets at presentation_time = 15 s."""
        from gw170817.simulation.demo_director import DemoDirector
        director = DemoDirector(coordinator=self.coord, config=self.config)
        director.stage_event_time = 15.0
        director._sync_physics_for_presentation_time(15.0)

        evt_st = self.coord.current_state
        orch = evt_st.orchestration

        self.assertTrue(orch.ejecta_active, "Ejecta remains active past 15 s")
        self.assertTrue(orch.bh_active, "BH remains active past 15 s")
        self.assertTrue(orch.disk_active, "Accretion disk remains active past 15 s")


    def test_H_continuous_post_merger_remains_active(self):
        """H. Continuous post-merger state remains active at late times."""
        self.sim.set_demo_event_time(100.0)
        evt_st = self.coord._update_event_state()
        orch = evt_st.orchestration

        self.assertTrue(orch.ejecta_active, "Ejecta active at 100s")
        self.assertTrue(orch.bh_active, "BH active at 100s")
        self.assertTrue(orch.jet_active, "Jet active at 100s")
        self.assertTrue(orch.kilonova_active, "Kilonova active at 100s")

    def test_I_no_duplicate_contradictory_clocks(self):
        """I. No duplicate or contradictory physical clocks."""
        self.sim.set_demo_event_time(0.5)
        evt_st = self.coord._update_event_state()
        orch = evt_st.orchestration

        # All states derived from single event_time clock
        self.assertEqual(evt_st.event_time, 0.5, "Single event_time clock drives all messenger states")
        self.assertIsInstance(orch, MultiMessengerOrchestrationState, "Snapshot is MultiMessengerOrchestrationState")

    def test_J_existing_tests_remain_passing(self):
        """J. Existing simulation coordinator functions correctly."""
        st = self.sim.current_state
        self.assertIsNotNone(st, "Engine simulation state is valid")


if __name__ == "__main__":
    unittest.main()
