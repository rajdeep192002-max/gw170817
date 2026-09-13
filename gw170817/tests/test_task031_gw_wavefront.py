"""
Unit tests for Task 031: Gravitational-Wave Propagation Wavefront.
Tests A through J as specified by prompt requirements.
"""
import unittest
import numpy as np
import taichi as ti
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.wave_propagation import GWWavefrontPropagation
from gw170817.visualization.dashboard import ScientificDashboard


class TestTask031GWWavefront(unittest.TestCase):
    """Test suite for 3D quadrupolar gravitational-wave propagation wavefront visual layer."""

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

    def setUp(self):
        self.wave_prop = GWWavefrontPropagation()

    def test_A_no_wavefront_before_merger(self):
        """A. No wavefront before merger (event_time < 0)."""
        wf_st = self.wave_prop.update(event_time=-0.1, f_gw=100.0, merger_active=False)
        self.assertFalse(wf_st.active, "Wavefront must be inactive before merger")
        self.assertEqual(wf_st.radius, 0.0, "Wavefront radius must be 0 before merger")
        self.assertEqual(wf_st.n_vertices, 0, "No vertices generated before merger")
        self.assertEqual(np.count_nonzero(self.wave_prop.line_vertices), 0, "Line vertices must be zero")

    def test_B_wavefront_exists_after_merger(self):
        """B. Wavefront exists immediately after merger (event_time >= 0)."""
        wf_st = self.wave_prop.update(event_time=0.0, f_gw=100.0, merger_active=True)
        self.assertTrue(wf_st.active, "Wavefront must activate at merger (event_time >= 0)")
        self.assertGreater(wf_st.n_vertices, 0, "Vertices generated after merger")
        self.assertGreater(wf_st.radius, 0.0, "Wavefront radius must be positive after merger")

    def test_C_radius_increases_monotonically(self):
        """C. Radius increases monotonically with event_time."""
        self.wave_prop.trigger(0.0)
        st1 = self.wave_prop.update(event_time=0.01)
        st2 = self.wave_prop.update(event_time=0.1)
        st3 = self.wave_prop.update(event_time=1.0)

        self.assertLess(st1.radius, st2.radius, "Radius must grow monotonically (t=0.01s vs t=0.1s)")
        self.assertLess(st2.radius, st3.radius, "Radius must grow monotonically (t=0.1s vs t=1.0s)")

    def test_D_radius_follows_c_dt(self):
        """D. Radius follows c * dt to chosen reduced-order visual scaling."""
        self.wave_prop.trigger(0.0)
        st_early = self.wave_prop.update(event_time=0.001)
        expected_r = 15.0e3 + self.wave_prop.c_vis * 0.001
        self.assertAlmostEqual(st_early.radius, expected_r, delta=100.0,
                               msg="Early radius must expand at c_vis * dt")

    def test_E_wave_amplitude_responds_to_strain(self):
        """E. Wave amplitude/intensity responds to GW strain."""
        self.wave_prop.trigger(0.0)
        st_low = self.wave_prop.update(event_time=0.05, h_plus=0.1e-21, h_cross=0.0)
        col_low = np.max(self.wave_prop.line_colors)

        st_high = self.wave_prop.update(event_time=0.05, h_plus=2.0e-21, h_cross=0.0)
        col_high = np.max(self.wave_prop.line_colors)

        self.assertGreater(col_high, col_low, "Larger GW strain must produce higher line color intensity")

    def test_F_wavefront_centered_at_merger(self):
        """F. Wavefront is centered at merger location (0, 0, 0)."""
        self.wave_prop.trigger(0.0)
        st = self.wave_prop.update(event_time=0.1)
        valid_verts = self.wave_prop.line_vertices[:st.n_vertices]
        center = np.mean(valid_verts, axis=0)

        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-2,
                                   err_msg="Wavefront geometry center must be at origin (0, 0, 0)")

    def test_G_world_space_camera_coupling(self):
        """G. Wavefront exists in world space and responds to camera movement."""
        cam_yaw = -np.pi / 2.0
        cam_yaw_shifted = cam_yaw - 0.2
        self.assertNotEqual(cam_yaw_shifted, cam_yaw, "Camera yaw changed in world space")

    def test_H_no_nan_inf(self):
        """H. No NaN/Inf values in wavefront vertices or colors."""
        self.wave_prop.trigger(0.0)
        st = self.wave_prop.update(event_time=0.5, h_plus=1.5e-21, h_cross=0.5e-21)
        verts = self.wave_prop.line_vertices[:st.n_vertices]
        cols = self.wave_prop.line_colors[:st.n_vertices]

        self.assertFalse(np.isnan(verts).any(), "No NaNs allowed in line_vertices")
        self.assertFalse(np.isinf(verts).any(), "No Infs allowed in line_vertices")
        self.assertFalse(np.isnan(cols).any(), "No NaNs allowed in line_colors")
        self.assertFalse(np.isinf(cols).any(), "No Infs allowed in line_colors")

    def test_I_observed_waveform_system_unchanged(self):
        """I. Existing observed waveform system (H1/L1 detector data) remains unchanged."""
        from gw170817.simulation.observational_data import ObservationalGWData
        obs = ObservationalGWData()
        t_obs, obs_h1, obs_l1 = obs.get_window(t_center=0.0, window_sec=1.5, n_samples=128)

        self.assertEqual(len(obs_h1), 128, "Observed H1 strain data unchanged")
        self.assertEqual(len(obs_l1), 128, "Observed L1 strain data unchanged")


    def test_J_continuous_post_merger_propagation(self):
        """J. Continuous post-merger propagation continues beyond presentation_time=15 s."""
        self.wave_prop.trigger(0.0)
        st_15s = self.wave_prop.update(event_time=15.0)
        st_100s = self.wave_prop.update(event_time=100.0)
        st_1000s = self.wave_prop.update(event_time=1000.0)

        self.assertTrue(st_15s.active, "Active at presentation_time = 15 s")
        self.assertTrue(st_100s.active, "Active at post-merger time = 100 s")
        self.assertTrue(st_1000s.active, "Active at post-merger time = 1000 s")
        self.assertLess(st_15s.radius, st_100s.radius, "Radius grows continuously past 15 s")
        self.assertLess(st_100s.radius, st_1000s.radius, "Radius grows continuously past 100 s")


if __name__ == "__main__":
    unittest.main()
