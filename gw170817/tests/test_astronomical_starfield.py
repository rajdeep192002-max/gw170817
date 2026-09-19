"""
Unit test suite for Task 026D.7 — 3D Astronomical Starfield for Gravitational Lensing.

Verifies:
1. Deterministic star generation from StellarCatalog.
2. Correct visual star counts for DEV (2000), NORMAL (3500), and HIGH (7000) quality modes.
3. Physically-motivated magnitude distribution (rarer bright stars, abundant faint stars).
4. Genuine 3D spatial depth distribution (continuous radial range, not a 2D plane or single shell).
5. Zero lens mass produces zero deflection (unchanged star positions).
6. Larger lens mass produces larger deflection magnitude.
7. Smaller impact parameter produces larger deflection.
8. Post-merger BH uses RemnantState mass at origin [0,0,0] as the lens source.
9. Starfield positions remain stable across frames with unchanged simulation state.
"""
import unittest
import numpy as np

from gw170817.constants import M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.background import StellarCatalog, BackgroundStarfield
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.dashboard import ScientificDashboard


class TestAstronomicalStarfield(unittest.TestCase):
    """Test suite for 3D Astronomical Starfield and Lensing Integration."""

    def test_deterministic_star_generation(self):
        """Test StellarCatalog produces identical stars for equal seeds."""
        cat1 = StellarCatalog(n_stars=500, seed=42)
        cat2 = StellarCatalog(n_stars=500, seed=42)
        cat_other = StellarCatalog(n_stars=500, seed=99)

        pos1 = np.array([s.position_3d for s in cat1.stars])
        pos2 = np.array([s.position_3d for s in cat2.stars])
        pos_other = np.array([s.position_3d for s in cat_other.stars])

        mags1 = np.array([s.magnitude for s in cat1.stars])
        mags2 = np.array([s.magnitude for s in cat2.stars])

        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(mags1, mags2)
        self.assertFalse(np.array_equal(pos1, pos_other))

    def test_star_counts_by_mode(self):
        """Test visual star budgets across DEV, NORMAL, and HIGH quality modes."""
        config_dev = SimConfig(mode="DEV", seed=42)
        sim_dev = GW170817Simulation(config=config_dev)
        renderer_dev = ParticleRenderer(sim_dev.psys)
        self.assertEqual(renderer_dev.n_stars, 2000, "DEV mode must allocate 2,000 visual stars")

        config_normal = SimConfig(mode="NORMAL", seed=42)
        sim_normal = GW170817Simulation(config=config_normal)
        renderer_normal = ParticleRenderer(sim_normal.psys)
        self.assertEqual(renderer_normal.n_stars, 3500, "NORMAL mode must allocate 3,500 visual stars")

        config_high = SimConfig(mode="HIGH", seed=42)
        sim_high = GW170817Simulation(config=config_high)
        renderer_high = ParticleRenderer(sim_high.psys)
        self.assertEqual(renderer_high.n_stars, 7000, "HIGH mode must allocate 7,000 visual stars")

    def test_star_magnitude_distribution(self):
        """Test star magnitude distribution covers range and is non-uniform (exponential growth)."""
        cat = StellarCatalog(n_stars=2000, seed=42)
        mags = np.array([s.magnitude for s in cat.stars])

        self.assertGreaterEqual(np.min(mags), 2.0)
        self.assertLessEqual(np.max(mags), 11.0)

        bright_count = np.sum(mags < 4.5)
        faint_count = np.sum(mags > 7.5)

        self.assertGreater(faint_count, bright_count, "Fainter stars must be more numerous than bright stars")

    def test_star_3d_depth_distribution(self):
        """Test star positions span genuine 3D depth and are not confined to a single plane or sphere shell."""
        cat = StellarCatalog(n_stars=2000, seed=42)
        positions = np.array([s.position_3d for s in cat.stars])
        distances = np.linalg.norm(positions, axis=1)

        min_d = np.min(distances)
        max_d = np.max(distances)
        std_d = np.std(distances)

        self.assertGreaterEqual(min_d, 0.75e6)
        self.assertLessEqual(max_d, 3.1e6)
        self.assertGreater(std_d, 0.25e6, "Star distances must have broad radial dispersion (3D depth)")

        # Verify not 2D plane (z coordinate non-zero)
        z_abs = np.abs(positions[:, 2])
        self.assertGreater(np.mean(z_abs), 0.1e6)

    def test_zero_mass_no_deflection(self):
        """Test zero lens mass produces unchanged star positions."""
        config = SimConfig(mode="DEV", seed=42)
        sim = GW170817Simulation(config=config)
        renderer = ParticleRenderer(sim.psys)

        star_orig = renderer.star_pos.to_numpy()

        renderer.update_star_lensing_deflection_kernel(
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            1, 1.0
        )
        star_zero_mass = renderer.star_deflected_pos.to_numpy()

        np.testing.assert_array_almost_equal(star_orig, star_zero_mass, decimal=0)

    def test_larger_mass_larger_deflection(self):
        """Test larger compact lens mass produces larger deflection magnitude."""
        config = SimConfig(mode="DEV", seed=42)
        sim = GW170817Simulation(config=config)
        renderer = ParticleRenderer(sim.psys)

        star_orig = renderer.star_pos.to_numpy()

        # Small lens mass M = 1.0 M_sun
        renderer.update_star_lensing_deflection_kernel(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0 * M_sun, 0.0,
            1, 1.0
        )
        star_m1 = renderer.star_deflected_pos.to_numpy()
        deflect_m1 = np.max(np.linalg.norm(star_m1 - star_orig, axis=1))

        # Large lens mass M = 3.0 M_sun
        renderer.update_star_lensing_deflection_kernel(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            3.0 * M_sun, 0.0,
            1, 1.0
        )
        star_m3 = renderer.star_deflected_pos.to_numpy()
        deflect_m3 = np.max(np.linalg.norm(star_m3 - star_orig, axis=1))

        self.assertGreater(deflect_m3, deflect_m1, "Larger lens mass must produce larger maximum deflection")

    def test_impact_parameter_deflection_scaling(self):
        """Test smaller impact parameter produces larger deflection magnitude using camera-ray geometry."""
        config = SimConfig(mode="DEV", seed=42)
        sim = GW170817Simulation(config=config)
        renderer = ParticleRenderer(sim.psys)

        star_orig = renderer.star_pos.to_numpy()

        # Camera and lens geometry matching the kernel's default pre-merger setup
        cam_pos = np.array([0.0, -280.0e3, 180.0e3])
        lens_pos = np.array([0.0, 0.0, 0.0])

        # Compute actual geometric impact parameters b using camera-ray projection
        ray_dirs = star_orig - cam_pos
        ray_dists = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
        ray_hats = ray_dirs / ray_dists

        v_cam = lens_pos - cam_pos
        t = np.sum(v_cam * ray_hats, axis=1)
        perps = v_cam - t[:, None] * ray_hats
        b = np.linalg.norm(perps, axis=1)

        # Select stars behind the lens (t > 0) with unclamped b (>= 15 km > 12 km clamp)
        # and within the sensitive deflection regime (<= 120 km attenuation scale)
        valid_mask = (t > 0.0) & (b >= 15.0e3) & (b <= 120.0e3)
        valid_indices = np.where(valid_mask)[0]

        self.assertGreater(len(valid_indices), 1, "Must have valid stars in sensitive deflection regime")

        # Sort valid stars by impact parameter b
        sorted_indices = valid_indices[np.argsort(b[valid_mask])]
        idx_near = sorted_indices[0]
        idx_far = sorted_indices[-1]

        b_near = b[idx_near]
        b_far = b[idx_far]

        self.assertLess(b_near, b_far)
        self.assertGreater(b_near, 12.0e3, "b_near must exceed the 12 km minimum clamping threshold")

        renderer.update_star_lensing_deflection_kernel(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.7 * M_sun, 0.0,
            1, 1.0,
            cam_x=float(cam_pos[0]),
            cam_y=float(cam_pos[1]),
            cam_z=float(cam_pos[2])
        )
        star_deflected = renderer.star_deflected_pos.to_numpy()
        displacements = np.linalg.norm(star_deflected - star_orig, axis=1)

        disp_near = displacements[idx_near]
        disp_far = displacements[idx_far]

        self.assertGreater(disp_near, disp_far,
                           f"Smaller impact parameter (b_near={b_near/1e3:.1f} km, disp={disp_near:.1f} m) "
                           f"must produce larger deflection than larger impact parameter (b_far={b_far/1e3:.1f} km, disp={disp_far:.1f} m)")

    def test_post_merger_remnant_lens_source(self):
        """Test post-merger delayed collapse BH lens source uses RemnantState mass at origin."""
        config = SimConfig(mode="DEV", seed=42)
        sim = GW170817Simulation(config=config)
        dashboard = ScientificDashboard.__new__(ScientificDashboard)
        dashboard.engine = sim
        dashboard.config = config

        # Set post-merger collapse state (t_event = 0.10s)
        sim.current_state.event_time = 0.10
        sim.current_state.merger_contact_fraction = 1.0

        p1, p2, m1, m2 = dashboard.get_lensing_sources()
        rem_st = sim.remnant.evaluate(sim.dynamics.inspiral_state, 0.10)

        self.assertEqual(p1, (0.0, 0.0, 0.0))
        self.assertEqual(p2, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(m1, rem_st.mass)
        self.assertEqual(m2, 0.0)

    def test_starfield_frame_stability(self):
        """Test starfield position updates remain 100% identical across frames for unchanged state."""
        config = SimConfig(mode="DEV", seed=42)
        sim = GW170817Simulation(config=config)
        renderer = ParticleRenderer(sim.psys)

        renderer.update_star_lensing_deflection_kernel(
            10.0e3, 0.0, 0.0, -10.0e3, 0.0, 0.0,
            1.36 * M_sun, 1.36 * M_sun,
            1, 1.0
        )
        star_frame1 = renderer.star_deflected_pos.to_numpy()

        for _ in range(5):
            renderer.update_star_lensing_deflection_kernel(
                10.0e3, 0.0, 0.0, -10.0e3, 0.0, 0.0,
                1.36 * M_sun, 1.36 * M_sun,
                1, 1.0
            )
            star_frame_n = renderer.star_deflected_pos.to_numpy()
            np.testing.assert_array_equal(star_frame1, star_frame_n, err_msg="Starfield positions must be frame-stable")


if __name__ == "__main__":
    unittest.main()
