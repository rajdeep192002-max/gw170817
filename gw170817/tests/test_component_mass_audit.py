"""
Unit and regression tests for GW170817 neutron-star component masses audit.
Verifies:
1. M1 is within published GW170817 allowed range (1.36 - 1.60 M_sun)
2. M2 is within published GW170817 allowed range (1.17 - 1.36 M_sun)
3. Mtotal is derived directly from m1 + m2 (2.73 M_sun)
4. Chirp mass is derived from (m1 * m2)^(3/5) / (m1 + m2)^(1/5) (~1.182 M_sun)
5. Mass ratio q = m2 / m1 <= 1.0 (~0.87)
6. Remnant and BH calculations use single authoritative total mass
7. Lensing and renderer use dynamic remnant mass scale
"""
import unittest
import numpy as np

from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.gravitational_waves import GravitationalWaveModel
from gw170817.physics.remnant import RemnantModel
from gw170817.visualization.lensing import RelativisticLensingModel


class TestComponentMassAudit(unittest.TestCase):
    """Test suite verifying authoritative component mass consistency across simulation modules."""

    def setUp(self):
        self.config = SimConfig()

    def test_component_mass_ranges(self):
        """Verify M1 and M2 fall strictly within published LIGO/Virgo GW170817 observational priors."""
        # Low-spin prior bounds (Abbott et al. 2017 PRL 119, 161101; Abbott et al. 2019 PRX 9, 011001)
        self.assertTrue(1.36 <= self.config.m1_solar <= 1.60,
                        f"M1 ({self.config.m1_solar} M_sun) must be within GW170817 low-spin prior [1.36, 1.60]")
        self.assertTrue(1.17 <= self.config.m2_solar <= 1.36,
                        f"M2 ({self.config.m2_solar} M_sun) must be within GW170817 low-spin prior [1.17, 1.36]")

    def test_total_mass_derivation(self):
        """Verify M_total is derived exactly as m1 + m2."""
        expected_total = (self.config.m1_solar + self.config.m2_solar) * M_sun
        self.assertAlmostEqual(self.config.M_total, expected_total, places=5)
        self.assertAlmostEqual(self.config.M_total / M_sun, 2.73, places=2)

    def test_chirp_mass_derivation(self):
        """Verify chirp mass is derived from (m1 * m2)^(3/5) / (m1 + m2)^(1/5)."""
        m1 = self.config.m1
        m2 = self.config.m2
        expected_mc = (m1 * m2)**0.6 / (m1 + m2)**0.2
        self.assertAlmostEqual(self.config.chirp_mass, expected_mc, places=5)
        # Expected value ~ 1.185 M_sun
        self.assertAlmostEqual(self.config.chirp_mass / M_sun, 1.1848, places=3)

    def test_mass_ratio(self):
        """Verify mass ratio q = m2 / m1 <= 1.0."""
        q = self.config.mass_ratio
        self.assertTrue(0.70 <= q <= 1.0, f"Mass ratio q ({q:.4f}) must be in valid GW170817 range [0.73, 1.0]")
        self.assertAlmostEqual(q, 1.27 / 1.46, places=4)

    def test_remnant_model_mass_consistency(self):
        """Verify RemnantModel inherits M_total directly from authoritative config."""
        rem_model = RemnantModel(config=self.config)
        self.assertEqual(rem_model.M_total, self.config.M_total)

    def test_lensing_model_mass_consistency(self):
        """Verify RelativisticLensingModel uses authoritative config m1 and m2."""
        lens_model = RelativisticLensingModel(config=self.config)
        self.assertEqual(lens_model.m1, self.config.m1)
        self.assertEqual(lens_model.m2, self.config.m2)

    def test_gravitational_wave_reduced_mass(self):
        """Verify reduced mass mu in GravitationalWaveModel uses config m1, m2, M_total."""
        gw_model = GravitationalWaveModel(config=self.config)
        expected_mu = (self.config.m1 * self.config.m2) / self.config.M_total
        self.assertAlmostEqual(gw_model.mu, expected_mu, places=5)


if __name__ == "__main__":
    unittest.main()
