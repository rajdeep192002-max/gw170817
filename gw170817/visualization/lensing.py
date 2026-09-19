"""
Relativistic Gravitational Lensing Model for GW170817 Compact Objects.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC DISCLAIMER:
Relativistic light-bending visualization layer using the Schwarzschild compact-star
exterior metric approximation.
Uses the Beloborodov (2002) analytical relation cos(alpha) approx u + (1-u)cos(psi)
where u = 2GM/(c^2 R) is stellar compactness, exposing far-side surface light emission.
This is NOT a full dynamical-spacetime ray trace through binary neutron-star GR geometry.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig


@dataclass
class LensingState:
    """Instantaneous state of the relativistic gravitational lensing visualization."""
    enabled: bool                  # True if lensing visualization active
    enhanced_mode: bool           # True if enhanced mode active
    compactness1: float            # Compactness u1 = 2GM1/(c^2 R1)
    compactness2: float            # Compactness u2 = 2GM2/(c^2 R2)
    max_visible_angle_deg: float   # Max visible surface polar angle psi_max [deg]
    deflection_scale: float        # Visual deflection scaling factor


class RelativisticLensingModel:
    """
    Schwarzschild Compact-Star Relativistic Gravitational Lensing Model.
    Computes analytical light bending parameters for compact objects and background ray deflection.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        if config is None:
            config = SimConfig()
        self.config = config

        self.enabled = True
        self.enhanced_mode = False

        # Physical compactness calculation u = 2GM / (c^2 R)
        # GW170817 reference configuration: m1 = 1.46 M_sun, m2 = 1.27 M_sun, R1 = 11.5 km, R2 = 11.5 km
        self.m1 = getattr(config, "m1", 1.46 * M_sun)
        self.m2 = getattr(config, "m2", 1.27 * M_sun)
        self.r1 = getattr(config, "R_NS1", 12.0e3)
        self.r2 = getattr(config, "R_NS2", 12.0e3)

        self.u1 = (2.0 * G * self.m1) / (c**2 * self.r1)
        self.u2 = (2.0 * G * self.m2) / (c**2 * self.r2)

    def toggle_enabled(self) -> bool:
        """Toggle lensing ON / OFF."""
        self.enabled = not self.enabled
        return self.enabled

    def toggle_enhanced(self) -> bool:
        """Toggle enhanced lensing mode."""
        self.enhanced_mode = not self.enhanced_mode
        return self.enhanced_mode

    def compute_beloborodov_bending(self, cos_psi: float, compactness: Optional[float] = None) -> float:
        """
        Compute surface emission direction angle alpha from surface polar angle psi relative to observer line of sight.
        Beloborodov (2002) relation:
            cos(alpha) = u + (1 - u) * cos(psi)
        Returns cos(alpha), clamped to [-1.0, 1.0].
        """
        if compactness is None:
            compactness = self.u1
        u = float(np.clip(compactness, 0.0, 0.45))
        cos_alpha = u + (1.0 - u) * float(cos_psi)
        return float(np.clip(cos_alpha, -1.0, 1.0))

    def max_visible_surface_angle(self, compactness: Optional[float] = None) -> float:
        """
        Compute maximum visible polar surface angle psi_max [rad] on the far side of the star.
        In flat space, psi_max = pi/2 (90 deg).
        Under Schwarzschild lensing, cos(psi_max) = -u / (1 - u).
        """
        if compactness is None:
            compactness = self.u1
        u = float(np.clip(compactness, 0.0, 0.45))
        cos_psi_max = -u / (1.0 - u)
        return float(np.arccos(np.clip(cos_psi_max, -1.0, 0.0)))

    def screen_space_deflection(self, impact_param_m: float, total_mass_kg: float) -> float:
        """
        Compute gravitational deflection angle alpha_def [rad] for background light rays passing at impact parameter b [m].
        Einstein deflection formula: alpha_def = 4GM / (c^2 b).
        Visual distortion proxy for background star field.
        """
        b = max(1.0e3, float(impact_param_m))
        M = max(1.0 * M_sun, float(total_mass_kg))
        alpha_def = (4.0 * G * M) / (c**2 * b)
        scale = 2.5 if self.enhanced_mode else 1.0
        return float(alpha_def * scale)

    def evaluate(self) -> LensingState:
        """Return instantaneous LensingState."""
        psi_max_rad = self.max_visible_surface_angle(self.u1)
        deflection_scale = 2.5 if self.enhanced_mode else 1.0
        return LensingState(
            enabled=self.enabled,
            enhanced_mode=self.enhanced_mode,
            compactness1=self.u1,
            compactness2=self.u2,
            max_visible_angle_deg=float(np.rad2deg(psi_max_rad)),
            deflection_scale=deflection_scale
        )
