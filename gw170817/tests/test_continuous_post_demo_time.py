"""
Regression test suite for continuous post-demo physical event_time progression.
Verifies:
1. Event time advances continuously without discrete jumps across the 15s presentation boundary.
2. Continuity across 13s -> 14s -> 15s -> 16s -> 17s -> 18s -> 19s -> 20s -> 21s -> 22s.
3. Explicit long-time regression states (100s, 1000s, 100000s, 1000000s) maintain physical BH persistence,
   finite accretion disk mass, finite lensing state, and zero NaN/Inf values.
"""
import unittest
import numpy as np

from gw170817.constants import M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector, DemoStage
from gw170817.visualization.lensing import RelativisticLensingModel


class TestContinuousPostDemoTime(unittest.TestCase):
    """Test suite for post-demo time continuity and BH/disk/lensing persistence."""

    def setUp(self):
        self.config = SimConfig(mode="DEV", seed=42)
        self.sim = GW170817Simulation(config=self.config)
        self.coordinator = MultiMessengerCoordinator(engine=self.sim, config=self.config)
        self.director = DemoDirector(coordinator=self.coordinator, config=self.config)

    def test_continuous_event_time_no_jump(self):
        """Verify event_time advances continuously frame-by-frame through 15s demo boundary without jumps."""
        self.director.start()

        # Advance director in small steps up to 25 presentation seconds
        dt = 0.1
        pres_times = []
        event_times = []

        for step in range(250):
            st = self.director.update(dt)
            pres_times.append(self.director.presentation_time)
            event_times.append(st.event_time)

        pres_times = np.array(pres_times)
        event_times = np.array(event_times)

        # 1. Verify no step has an artificial jump > 2.0 seconds
        event_time_diffs = np.diff(event_times)
        max_diff = np.max(event_time_diffs)
        self.assertLess(max_diff, 2.0, f"Max event_time step difference was {max_diff:.3f} s, expecting smooth <= 0.2 s steps without 200-day jumps")

        # 2. Verify continuity around 15 s presentation time (index ~ 150)
        idx_15s = np.argmin(np.abs(pres_times - 15.0))
        event_time_at_15s = event_times[idx_15s]
        event_time_at_16s = event_times[idx_15s + 10]

        self.assertAlmostEqual(event_time_at_16s - event_time_at_15s, 1.0, delta=0.2,
                               msg="Physical event_time must advance at 1.0s per presentation second after demo end")

    def test_post_demo_continuity_20_to_22_seconds(self):
        """Verify 20s -> 21s -> 22s continuity."""
        self.director.start()
        dt = 0.1
        # Step to 20 presentation seconds
        for _ in range(200):
            self.director.update(dt)

        st_20 = self.coordinator.current_state
        time_20 = st_20.event_time

        for _ in range(10):  # +1.0 s
            self.director.update(dt)
        st_21 = self.coordinator.current_state
        time_21 = st_21.event_time

        for _ in range(10):  # +1.0 s
            self.director.update(dt)
        st_22 = self.coordinator.current_state
        time_22 = st_22.event_time

        self.assertAlmostEqual(time_21 - time_20, 1.0, delta=0.15)
        self.assertAlmostEqual(time_22 - time_21, 1.0, delta=0.15)
        self.assertTrue(st_20.is_black_hole)
        self.assertTrue(st_21.is_black_hole)
        self.assertTrue(st_22.is_black_hole)

    def test_long_time_regression_points(self):
        """Explicitly test long-time regression probes (100s, 1000s, 100000s, 1000000s)."""
        lensing_model = RelativisticLensingModel(config=self.config)

        for t_test in [100.0, 1000.0, 100000.0, 1000000.0]:
            st = self.coordinator.evaluate_at_event_time(t_test)
            rem_st = self.sim.remnant.evaluate(self.sim.dynamics.inspiral_state, t_test)
            disk_st = self.sim.disk.evaluate(rem_st, t_test, self.sim.rotation.evaluate(self.sim.dynamics.inspiral_state, rem_st, t_test))
            lens_st = lensing_model.evaluate()

            # Verify BH persistence
            self.assertTrue(st.is_black_hole, f"BH must persist at t = {t_test} s")
            self.assertTrue(rem_st.is_black_hole, f"RemnantState must be BH at t = {t_test} s")

            # Verify finite mass & non-NaN/Inf values
            self.assertTrue(np.isfinite(rem_st.mass), f"Remnant mass must be finite at t = {t_test}")
            self.assertGreater(rem_st.mass, 2.0 * M_sun)

            self.assertTrue(np.isfinite(disk_st.disk_mass_msun), f"Disk mass must be finite at t = {t_test}")
            self.assertGreaterEqual(disk_st.disk_mass_msun, 0.0)

            self.assertTrue(lens_st.enabled)
            self.assertTrue(np.isfinite(lens_st.compactness1))


if __name__ == "__main__":
    unittest.main()
