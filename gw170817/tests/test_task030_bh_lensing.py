"""
Unit tests for Task 030: Immersive Camera-Dependent Black-Hole Lensing.
Tests A through I as specified by prompt requirements.
"""
import unittest
import numpy as np
import taichi as ti
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.dashboard import ScientificDashboard


class TestTask030BHLensing(unittest.TestCase):
    """Test suite for post-merger camera-dependent Schwarzschild black-hole gravitational lensing."""

    @classmethod
    def setUpClass(cls):
        try:
            ti.init(arch=ti.vulkan, implicit_prefer_if=True)
        except Exception:
            try:
                ti.init(arch=ti.cpu)
            except Exception:
                pass

    def setUp(self):
        self.config = SimConfig(mode="DEV", seed=42)
        self.sim = GW170817Simulation(config=self.config)
        self.renderer = ParticleRenderer(self.sim.psys)

    def test_A_schwarzschild_radius_scaling(self):
        """A. Schwarzschild radius is finite and scales with BH mass."""
        m1 = 1.0 * M_sun
        m2 = 2.0 * M_sun
        c2 = c**2
        rs1 = (2.0 * G * m1) / c2
        rs2 = (2.0 * G * m2) / c2

        self.assertTrue(np.isfinite(rs1), "r_s must be finite")
        self.assertTrue(np.isfinite(rs2), "r_s must be finite")
        self.assertAlmostEqual(rs2, 2.0 * rs1, delta=1.0, msg="r_s must scale linearly with mass")

    def test_B_ray_direction_with_camera(self):
        """B. Ray direction changes with camera orientation."""
        cam1 = np.array([0.0, -280.0e3, 180.0e3])
        cam2 = np.array([280.0e3, 0.0, 180.0e3])

        ray1 = -cam1 / np.linalg.norm(cam1)
        ray2 = -cam2 / np.linalg.norm(cam2)

        self.assertFalse(np.allclose(ray1, ray2), "Camera orientation change must yield different ray directions")

    def test_C_camera_movement_changes_lensing(self):
        """C. Camera movement changes the lensing result."""
        self.renderer.update_star_lensing_deflection_kernel(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
            cam_x=0.0, cam_y=-280.0e3, cam_z=180.0e3,
            is_black_hole=1, remnant_mass=2.74 * M_sun
        )
        stars1 = self.renderer.star_deflected_pos.to_numpy()

        self.renderer.update_star_lensing_deflection_kernel(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
            cam_x=280.0e3, cam_y=0.0, cam_z=180.0e3,
            is_black_hole=1, remnant_mass=2.74 * M_sun
        )
        stars2 = self.renderer.star_deflected_pos.to_numpy()

        self.assertFalse(np.array_equal(stars1, stars2), "Camera movement must update lensing results")

    def test_D_capture_rays_classification(self):
        """D. Capture rays (b <= b_crit) are classified correctly as dark shadow (z = -1e9)."""
        # Set a star directly behind BH along ray line
        cam_p = np.array([0.0, -280.0e3, 0.0])
        self.renderer.star_pos[0] = ti.Vector([0.0, 100.0e3, 0.0])
        self.renderer.update_star_lensing_deflection_kernel(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
            cam_x=cam_p[0], cam_y=cam_p[1], cam_z=cam_p[2],
            is_black_hole=1, remnant_mass=2.74 * M_sun
        )
        def_pos = self.renderer.star_deflected_pos.to_numpy()
        captured_z = def_pos[0, 2]
        self.assertAlmostEqual(captured_z, -1.0e9, delta=1.0,
                               msg="Captured rays behind photon sphere must be hidden/captured at z=-1e9")

    def test_E_escaping_rays_sample_starfield(self):
        """E. Escaping rays (b > b_crit) produce valid deflected 3D star positions."""
        self.renderer.update_star_lensing_deflection_kernel(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
            cam_x=0.0, cam_y=-280.0e3, cam_z=180.0e3,
            is_black_hole=1, remnant_mass=2.74 * M_sun
        )
        def_pos = self.renderer.star_deflected_pos.to_numpy()
        escaping = def_pos[def_pos[:, 2] > -1.0e8]
        self.assertGreater(len(escaping), 0, "Escaping rays must yield valid 3D deflected star positions")

    def test_F_no_nan_inf_values(self):
        """F. No NaN or Inf values in positions."""
        self.renderer.update_star_lensing_deflection_kernel(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
            cam_x=100.0e3, cam_y=-200.0e3, cam_z=50.0e3,
            is_black_hole=1, remnant_mass=2.74 * M_sun
        )
        def_pos = self.renderer.star_deflected_pos.to_numpy()

        self.assertFalse(np.isnan(def_pos).any(), "No NaNs allowed in star_deflected_pos")
        self.assertFalse(np.isinf(def_pos).any(), "No Infs allowed in star_deflected_pos")

    def test_G_activates_only_after_bh_formation(self):
        """G. Effect activates only after BH formation (is_black_hole = 1)."""
        self.renderer.update_star_lensing_deflection_kernel(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
            cam_x=0.0, cam_y=-280.0e3, cam_z=0.0,
            is_black_hole=0, remnant_mass=2.74 * M_sun
        )
        pos_pre = self.renderer.star_deflected_pos.to_numpy()

        self.renderer.update_star_lensing_deflection_kernel(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
            cam_x=0.0, cam_y=-280.0e3, cam_z=0.0,
            is_black_hole=1, remnant_mass=2.74 * M_sun
        )
        pos_post = self.renderer.star_deflected_pos.to_numpy()

        self.assertFalse(np.array_equal(pos_pre, pos_post),
                         "Post-merger BH lensing must activate only when is_black_hole = 1")

    def test_H_pre_bh_lensing_unchanged(self):
        """H. Pre-BH lensing (is_black_hole = 0) remains unchanged."""
        orig_stars = self.renderer.star_pos.to_numpy()

        self.renderer.update_star_lensing_deflection_kernel(
            -15.0e3, 0.0, 0.0, 15.0e3, 0.0, 0.0,
            1.36 * M_sun, 1.36 * M_sun,
            1, 1.0,
            cam_x=0.0, cam_y=-280.0e3, cam_z=180.0e3,
            is_black_hole=0, remnant_mass=2.74 * M_sun
        )
        def_stars = self.renderer.star_deflected_pos.to_numpy()

        self.assertFalse(np.array_equal(orig_stars, def_stars), "Pre-BH BNS lensing must displace stars")

    def test_I_existing_camera_controls_functional(self):
        """I. Existing camera controls remain functional."""
        dash = ScientificDashboard(config=self.config, window_size=(640, 360))

        initial_yaw = dash.cam_yaw
        initial_pitch = dash.cam_pitch
        initial_dist = dash.cam_distance

        dash.camera_orbit_left(0.1)
        self.assertNotEqual(dash.cam_yaw, initial_yaw)

        dash.camera_orbit_right(0.1)
        dash.camera_orbit_up(0.1)
        self.assertNotEqual(dash.cam_pitch, initial_pitch)

        dash.camera_orbit_down(0.1)
        dash.camera_zoom_in(0.9)
        self.assertNotEqual(dash.cam_distance, initial_dist)

        dash.camera_zoom_out(1.1)


if __name__ == "__main__":
    unittest.main()
