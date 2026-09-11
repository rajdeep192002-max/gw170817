"""
Unit and integration test suite for GW170817 Visual Pass (Task 022).
"""
import unittest
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.demo_director import DemoDirector, DemoStage, PlaybackMode
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.lensing import RelativisticLensingModel


class TestVisualPass(unittest.TestCase):
    """Test suite for Task 022 visual comprehensibility pass."""

    def setUp(self):
        self.config = SimConfig(mode="DEV", seed=42)
        self.engine = GW170817Simulation(config=self.config)
        self.director = DemoDirector(config=self.config)
        self.renderer = ParticleRenderer(self.engine.psys)
        self.lensing = RelativisticLensingModel(config=self.config)

    def test_default_reset_to_minus_five_seconds(self):
        """Test default reset starts at -5.0 s event_time."""
        st = self.director.reset()
        self.assertAlmostEqual(st.event_time, -5.0, delta=0.5)
        self.assertTrue(70.0 <= st.gw_frequency <= 76.0, f"Expected f_gw ~ 72.4 Hz, got {st.gw_frequency}")

    def test_presentation_progress_variables(self):
        """Test presentation time progress variables map smoothly in [0.0, 1.0]."""
        self.director.reset()
        self.director.start()

        for _ in range(10):
            self.director.update(0.1)
            self.assertTrue(0.0 <= self.director.presentation_time <= 5.0)
            self.assertTrue(0.0 <= self.director.inspiral_progress <= 1.0)
            self.assertTrue(0.0 <= self.director.merger_progress <= 1.0)
            self.assertTrue(0.0 <= self.director.ejecta_progress <= 1.0)
            self.assertTrue(0.0 <= self.director.disk_progress <= 1.0)
            self.assertTrue(0.0 <= self.director.b_winding_progress <= 1.0)
            self.assertTrue(0.0 <= self.director.jet_progress <= 1.0)
            self.assertTrue(0.0 <= self.director.kilonova_progress <= 1.0)
            self.assertTrue(0.0 <= self.director.afterglow_progress <= 1.0)

    def test_slow_motion_scaling(self):
        """Test SLOW MOTION (0.10x) mode scales presentation time advance directly."""
        self.director.reset()
        self.director.start()
        t0 = self.director.presentation_time

        self.director.set_slow_motion()
        self.director.update(1.0)
        t_slow = self.director.presentation_time - t0
        self.assertAlmostEqual(t_slow, 0.10, delta=0.01)

        self.director.set_real_time()
        self.director.update(1.0)
        t_real = self.director.presentation_time - (t0 + t_slow)
        self.assertAlmostEqual(t_real, 1.00, delta=0.01)

    def test_starfield_initialization_and_lensing_deflection(self):
        """Test 2,000 GPU background stars and lensing deflection difference (OFF vs ON)."""
        self.assertEqual(self.renderer.n_stars, 2000)
        star_orig = self.renderer.star_pos.to_numpy()
        self.assertEqual(star_orig.shape, (2000, 3))
        self.assertFalse(np.isnan(star_orig).any())

        r1, r2 = self.engine.inspiral.orbital_positions(
            self.engine.dynamics.inspiral_state,
            self.config.m1, self.config.m2
        )

        # 1. Lensing OFF: deflected position must equal original position
        self.renderer.update_star_lensing_deflection_kernel(
            float(r1[0]), float(r1[1]), float(r1[2]),
            float(r2[0]), float(r2[1]), float(r2[2]),
            float(self.config.m1), float(self.config.m2),
            0, 1.0
        )
        star_off = self.renderer.star_deflected_pos.to_numpy()
        np.testing.assert_array_almost_equal(star_orig, star_off)

        # 2. Lensing ON: deflected position MUST differ from original position (stars warp!)
        self.renderer.update_star_lensing_deflection_kernel(
            float(r1[0]), float(r1[1]), float(r1[2]),
            float(r2[0]), float(r2[1]), float(r2[2]),
            float(self.config.m1), float(self.config.m2),
            1, 1.0
        )
        star_on = self.renderer.star_deflected_pos.to_numpy()
        deflection_mag = np.linalg.norm(star_on - star_orig, axis=1)
        self.assertTrue(np.max(deflection_mag) > 0.0, "Lensing ON must cause observable star deflection")

    def test_dynamic_jet_lines(self):
        """Test dynamic jet line geometry update using physical event_time propagation."""
        # Use event_time after jet_delay (1.7s) so jet has non-zero propagation
        self.renderer.update_jet_geometry(
            is_active=True,
            opening_angle_rad=0.08,
            intensity=1.0,
            event_time=5.0,   # 5s after merger → 3.3s after jet launch
            jet_progress=0.8,
            jet_delay=1.7,
            beta_jet=0.9999
        )
        jet_verts = self.jet_vertices_np = self.renderer.jet_vertices.to_numpy()
        self.assertEqual(jet_verts.shape, (128, 3))
        self.assertFalse(np.isnan(jet_verts).any())

    def test_thousand_frame_stability(self):
        """Test 1,000-frame stability with zero NaNs or Infs."""
        self.director.reset()
        self.director.start()

        for _ in range(1000):
            st = self.director.update(0.005)
            self.assertFalse(np.isnan(st.event_time))
            self.assertFalse(np.isinf(st.event_time))
            self.assertFalse(np.isnan(st.gw_frequency))
            self.assertFalse(np.isinf(st.gw_frequency))


if __name__ == "__main__":
    unittest.main()
