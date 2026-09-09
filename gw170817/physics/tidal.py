"""
Reduced-order tidal-deformability model for GW170817.

REDUCED-ORDER APPROXIMATION:
This is a reduced-order tidal-deformability model for GW170817,
not a full numerical-relativity tidal calculation.

It computes static quadrupolar tidal deformabilities (Lambda),
compactness (C), effective binary tidal deformability (Lambda_tilde),
and instantaneous tidal interaction strength driven by InspiralState.
"""
from dataclasses import dataclass
import numpy as np
from gw170817.constants import G, c
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralState


@dataclass
class TidalStar:
    """Tidal parameters for an individual neutron star."""
    mass: float                # Mass [kg]
    radius: float              # Radius [m]
    k2: float                  # Quadrupolar Love number [dimensionless]
    compactness: float         # Compactness C = G M / (c^2 R) [dimensionless]
    lambda_dimensionless: float # Dimensionless tidal deformability Lambda


@dataclass
class TidalState:
    """Instantaneous tidal state of the binary system."""
    separation: float           # Orbital separation a [m]
    star1: TidalStar            # Primary NS tidal parameters
    star2: TidalStar            # Secondary NS tidal parameters
    tidal_distortion_1: float   # Instantaneous tidal deformation factor star 1
    tidal_distortion_2: float   # Instantaneous tidal deformation factor star 2
    lambda_tilde: float         # Effective binary tidal deformability Lambda_tilde


class TidalModel:
    """
    Physics-informed reduced-order tidal deformability model.
    Evolves tidal responses driven by InspiralState.
    """

    def __init__(
        self,
        config: SimConfig = None,
        r1_ns: float = 12.0e3,
        r2_ns: float = 12.0e3,
        k2_1: float = 0.08,
        k2_2: float = 0.08,
    ):
        if config is None:
            config = SimConfig()
        self.config = config

        # Validate inputs
        if config.m1 <= 0.0 or config.m2 <= 0.0:
            raise ValueError(f"Stellar masses must be positive, got m1={config.m1}, m2={config.m2}")
        if r1_ns <= 0.0 or r2_ns <= 0.0:
            raise ValueError(f"Stellar radii must be positive, got r1={r1_ns}, r2={r2_ns}")
        if k2_1 <= 0.0 or k2_2 <= 0.0:
            raise ValueError(f"Love numbers must be positive, got k2_1={k2_1}, k2_2={k2_2}")

        # Compute static parameters for Star 1
        c1 = (G * config.m1) / (c**2 * r1_ns)
        lam1 = (2.0 / 3.0) * k2_1 * (1.0 / (c1**5))
        self.star1 = TidalStar(
            mass=config.m1,
            radius=r1_ns,
            k2=k2_1,
            compactness=c1,
            lambda_dimensionless=lam1
        )

        # Compute static parameters for Star 2
        c2 = (G * config.m2) / (c**2 * r2_ns)
        lam2 = (2.0 / 3.0) * k2_2 * (1.0 / (c2**5))
        self.star2 = TidalStar(
            mass=config.m2,
            radius=r2_ns,
            k2=k2_2,
            compactness=c2,
            lambda_dimensionless=lam2
        )

        # Safety checks on Lambda
        if not (np.isfinite(lam1) and lam1 > 0.0) or not (np.isfinite(lam2) and lam2 > 0.0):
            raise ValueError(f"Non-finite or non-positive Lambda computed: Lambda1={lam1}, Lambda2={lam2}")

        # Pre-compute Effective Binary Tidal Deformability (Lambda_tilde)
        m1, m2 = config.m1, config.m2
        M_tot = m1 + m2
        numerator = (m1 + 12.0 * m2) * (m1**4) * lam1 + (m2 + 12.0 * m1) * (m2**4) * lam2
        self.lambda_tilde = (16.0 / 13.0) * (numerator / (M_tot**5))

        if not (np.isfinite(self.lambda_tilde) and self.lambda_tilde > 0.0):
            raise ValueError(f"Invalid Lambda_tilde computed: {self.lambda_tilde}")

    def evaluate(self, inspiral_state: InspiralState) -> TidalState:
        """
        Evaluate instantaneous tidal state from InspiralState.
        Tidal distortion scales as (R / separation)^3.
        """
        a = inspiral_state.separation
        if a <= 0.0 or not np.isfinite(a):
            raise ValueError(f"Invalid separation for tidal evaluation: {a}")

        # Dimensionless quadrupolar tidal interaction scaling: (R / a)^3
        dist1 = (self.star1.radius / a)**3
        dist2 = (self.star2.radius / a)**3

        return TidalState(
            separation=a,
            star1=self.star1,
            star2=self.star2,
            tidal_distortion_1=dist1,
            tidal_distortion_2=dist2,
            lambda_tilde=self.lambda_tilde
        )

    def initial_state(self) -> TidalState:
        """Construct initial TidalState from config initial separation."""
        a0 = self.config.initial_separation
        dist1 = (self.star1.radius / a0)**3
        dist2 = (self.star2.radius / a0)**3

        return TidalState(
            separation=a0,
            star1=self.star1,
            star2=self.star2,
            tidal_distortion_1=dist1,
            tidal_distortion_2=dist2,
            lambda_tilde=self.lambda_tilde
        )
