"""
Unit tests for Relativistic Gravitational Lensing Model.
"""
import unittest
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.visualization.lensing import RelativisticLensingModel, LensingState


class TestRelativisticLensingModel(unittest.TestCase):
    """Test cases for RelativisticLensingModel."""

    def setUp(self):
        self.config = SimConfig()
        self.lensing = RelativisticLensingModel(config=self.config)

    def test_compactness_calculation(self):
        """Test physical compactness calculation u = 2GM/(c^2 R)."""
        u1 = self.lensing.u1
        u2 = self.lensing.u2

        # For NS masses (~1.35-1.39 Msun) and radius (~11 km), u should be ~0.35
        self.assertTrue(0.25 <= u1 <= 0.45, f"Expected compactness u1 in [0.25, 0.45], got {u1}")
        self.assertTrue(0.25 <= u2 <= 0.45, f"Expected compactness u2 in [0.25, 0.45], got {u2}")

    def test_beloborodov_bending(self):
        """Test Beloborodov light deflection formula cos(alpha) = u + (1-u)*cos(psi)."""
        cos_alpha_front = self.lensing.compute_beloborodov_bending(cos_psi=1.0)
        self.assertAlmostEqual(cos_alpha_front, 1.0, places=5)

        cos_alpha_perp = self.lensing.compute_beloborodov_bending(cos_psi=0.0)
        self.assertAlmostEqual(cos_alpha_perp, self.lensing.u1, places=5)

        self.assertFalse(np.isnan(cos_alpha_perp))
        self.assertFalse(np.isinf(cos_alpha_perp))

    def test_max_visible_surface_angle(self):
        """Test maximum visible polar surface angle psi_max on far side."""
        psi_max_rad = self.lensing.max_visible_surface_angle()
        psi_max_deg = np.rad2deg(psi_max_rad)

        # Under Schwarzschild light bending, psi_max > 90 deg (exposure of far side)
        self.assertGreater(psi_max_deg, 90.0)
        self.assertLessEqual(psi_max_deg, 180.0)

    def test_toggles(self):
        """Test lensing toggle methods."""
        self.assertTrue(self.lensing.enabled)
        en = self.lensing.toggle_enabled()
        self.assertFalse(en)
        self.assertFalse(self.lensing.enabled)

        self.assertFalse(self.lensing.enhanced_mode)
        enh = self.lensing.toggle_enhanced()
        self.assertTrue(enh)
        self.assertTrue(self.lensing.enhanced_mode)

    def test_evaluate(self):
        """Test evaluate returns a valid LensingState."""
        state = self.lensing.evaluate()
        self.assertIsInstance(state, LensingState)
        self.assertTrue(np.isfinite(state.compactness1))
        self.assertTrue(np.isfinite(state.max_visible_angle_deg))


if __name__ == "__main__":
    unittest.main()
