"""
Reduced-order binary neutron star (BNS) merger model for GW170817.

REDUCED-ORDER APPROXIMATION:
Reduced-order BNS merger model; not a full numerical-relativity simulation.

1. The contact condition is strictly based on the physical stellar radii sum:
   a_contact = R1 + R2
2. The orbital energy and angular momentum are Newtonian diagnostic proxies.
3. Full BNS merger dynamics require numerical relativity, general relativistic hydrodynamics (GRHD),
   magnetohydrodynamics (MHD), and neutrino transport, which are outside this laptop-scale model.
"""
from dataclasses import dataclass
import numpy as np
from gw170817.constants import G
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralState
from gw170817.physics.tidal import TidalModel


@dataclass
class MergerState:
    """
    State representation of the binary system during contact/merger transition.
    """
    time: float                    # Physical simulation time [s]
    separation: float              # Orbital separation a [m]
    orbital_frequency: float       # Orbital frequency f_orb [Hz]
    gw_frequency: float            # GW frequency f_gw [Hz]
    orbital_phase: float           # Orbital phase phi [rad]

    contact_separation: float      # First contact separation threshold a_contact = R1 + R2 [m]
    contact_fraction: float        # Smooth contact progress parameter in [0.0, 1.0]

    in_contact: bool               # True if separation <= contact_separation
    merger_started: bool           # True if contact_fraction >= 0.5
    merger_complete: bool          # True if separation <= 0.8 * contact_separation

    orbital_radius: float          # Orbital separation radius a [m]
    orbital_speed: float           # Relative orbital velocity v = omega * a [m/s]

    angular_momentum_proxy: float  # Newtonian orbital angular momentum L_orb [kg m^2/s]
    orbital_energy_proxy: float    # Newtonian orbital energy E_orb [J]
    merger_timescale: float        # Characteristic dynamical timescale t_dyn [s]


class MergerModel:
    """
    Physics-informed reduced-order merger transition model for GW170817.
    Evaluates system state and contact progression driven by InspiralState.
    """

    def __init__(self, config: SimConfig = None, tidal_model: TidalModel = None):
        if config is None:
            config = SimConfig()
        self.config = config

        if tidal_model is None:
            tidal_model = TidalModel(config)
        self.tidal_model = tidal_model

        # Physical contact separation: a_contact = R1 + R2
        self.r1 = tidal_model.star1.radius
        self.r2 = tidal_model.star2.radius
        self.contact_separation = self.r1 + self.r2

        if self.contact_separation <= 0.0:
            raise ValueError(f"Contact separation must be positive, got {self.contact_separation}")

        # Mass parameters
        self.m1 = config.m1
        self.m2 = config.m2
        self.M_total = self.m1 + self.m2
        self.mu = (self.m1 * self.m2) / self.M_total

        # Characteristic dynamical timescale: t_dyn ~ sqrt(a_contact^3 / (G * M_total))
        self.merger_timescale = float(np.sqrt((self.contact_separation**3) / (G * self.M_total)))

        if not (np.isfinite(self.merger_timescale) and self.merger_timescale > 0.0):
            raise ValueError(f"Invalid merger timescale computed: {self.merger_timescale}")

    def _compute_contact_fraction(self, separation: float) -> float:
        """
        Compute smooth contact progress fraction in [0.0, 1.0].
        - Well outside contact (a >= 2 * a_contact): 0.0
        - Transition regime (a_contact < a < 2 * a_contact): smoothstep S(x)
        - Contact regime (a <= a_contact): 1.0
        """
        a_outer = 2.0 * self.contact_separation
        a_inner = self.contact_separation

        if separation >= a_outer:
            return 0.0
        if separation <= a_inner:
            return 1.0

        # Normalized linear parameter x in [0, 1] as separation decreases from a_outer to a_inner
        x = (a_outer - separation) / (a_outer - a_inner)
        # Smoothstep interpolation: 3x^2 - 2x^3
        smooth_frac = 3.0 * (x**2) - 2.0 * (x**3)
        return float(np.clip(smooth_frac, 0.0, 1.0))

    def evaluate(self, inspiral_state: InspiralState) -> MergerState:
        """
        Evaluate instantaneous merger state from InspiralState.
        """
        a = inspiral_state.separation
        if a <= 0.0 or not np.isfinite(a):
            raise ValueError(f"Invalid separation for merger evaluation: {a}")

        frac = self._compute_contact_fraction(a)

        in_contact = bool(a <= self.contact_separation)
        merger_started = bool(frac >= 0.5)
        merger_complete = bool(a <= 0.8 * self.contact_separation)

        # Orbital kinematics
        omega = inspiral_state.omega_orb
        v_orb = omega * a

        # Diagnostic Newtonian proxies
        E_orb = -(G * self.m1 * self.m2) / (2.0 * a)
        L_orb = self.mu * np.sqrt(G * self.M_total * a)

        # Numerical safety checks
        if not np.isfinite(E_orb) or not np.isfinite(L_orb) or not np.isfinite(v_orb):
            raise ValueError("Calculated merger diagnostic proxies contain NaN or Inf")

        return MergerState(
            time=inspiral_state.time,
            separation=a,
            orbital_frequency=inspiral_state.orbital_frequency,
            gw_frequency=inspiral_state.f_gw,
            orbital_phase=inspiral_state.orbital_phase,
            contact_separation=self.contact_separation,
            contact_fraction=frac,
            in_contact=in_contact,
            merger_started=merger_started,
            merger_complete=merger_complete,
            orbital_radius=a,
            orbital_speed=v_orb,
            angular_momentum_proxy=L_orb,
            orbital_energy_proxy=E_orb,
            merger_timescale=self.merger_timescale
        )

    def initial_state(self) -> MergerState:
        """Construct initial MergerState from config initial separation."""
        a0 = self.config.initial_separation
        omega0 = np.pi * self.config.f_gw_start
        f_orb0 = self.config.f_gw_start / 2.0

        frac0 = self._compute_contact_fraction(a0)

        in_contact = bool(a0 <= self.contact_separation)
        merger_started = bool(frac0 >= 0.5)
        merger_complete = bool(a0 <= 0.8 * self.contact_separation)

        v_orb0 = omega0 * a0
        E_orb0 = -(G * self.m1 * self.m2) / (2.0 * a0)
        L_orb0 = self.mu * np.sqrt(G * self.M_total * a0)

        return MergerState(
            time=0.0,
            separation=a0,
            orbital_frequency=f_orb0,
            gw_frequency=self.config.f_gw_start,
            orbital_phase=0.0,
            contact_separation=self.contact_separation,
            contact_fraction=frac0,
            in_contact=in_contact,
            merger_started=merger_started,
            merger_complete=merger_complete,
            orbital_radius=a0,
            orbital_speed=v_orb0,
            angular_momentum_proxy=L_orb0,
            orbital_energy_proxy=E_orb0,
            merger_timescale=self.merger_timescale
        )
