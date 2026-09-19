"""
Compact Remnant Evolution Physics Model for BNS Mergers.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Models compact remnant evolution post-contact/merger (HMNS vs prompt BH collapse).
Configurable remnant scenarios:
  - "GW170817_LIKE" (Default: hypermassive neutron star HMNS with configurable collapse)
  - "PROMPT_BH_REFERENCE" (Hayashi et al. 2025 prompt BH reference pathway)
  - "HMNS_REFERENCE" (Long-lived hypermassive neutron star reference pathway)

Does NOT solve full GR Einstein equations or MHD equations dynamically.
References:
  Hayashi et al., Phys. Rev. Lett. 134, 211407 (2025)
  Shibata & Taniguchi, Living Rev. Relativ. 14, 6 (2011)
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralState


@dataclass
class RemnantState:
    """Instantaneous state of the compact remnant."""
    time: float                  # Event time relative to merger [s]
    scenario: str                # Active scenario ("GW170817_LIKE", "PROMPT_BH_REFERENCE", "HMNS_REFERENCE")
    remnant_type: str            # Current physical state ("INSPIRAL", "HMNS", "BH")
    mass: float                  # Remnant mass [kg]
    radius: float                # Remnant physical radius / horizon radius [m]
    spin_parameter: float        # Dimensionless Kerr spin a* = c J / (G M^2)
    omega_rot: float             # Angular rotation frequency [rad/s]
    compactness: float           # Compactness u = 2 G M / (c^2 R)
    is_black_hole: bool          # True if remnant has collapsed to BH
    collapse_time: float         # Time of collapse post-merger [s] (inf if no collapse)


class RemnantModel:
    """
    Reduced-order compact remnant model for binary neutron star mergers.
    """

    def __init__(self, config: Optional[SimConfig] = None, scenario: str = "GW170817_LIKE"):
        if config is None:
            config = SimConfig()
        self.config = config
        self.scenario = scenario if scenario != "GW170817_LIKE" else getattr(config, "REMNANT_MODEL", "GW170817_LIKE")

        # Baseline parameters
        self.M_total = config.M_total
        self.R_NS = getattr(config, "R_NS1", 11.5e3)

        # Scenario configurations
        if self.scenario == "PROMPT_BH_REFERENCE":
            # Hayashi et al. 2025 reference pathway (prompt BH formation)
            self.t_collapse_delay = 0.001   # Prompt collapse (~1 ms post-contact)
            self.spin_final = 0.68
        elif self.scenario == "HMNS_REFERENCE":
            # Long-lived HMNS reference pathway
            self.t_collapse_delay = 100.0   # Long-lived HMNS (>100 s)
            self.spin_final = 0.72
        else:
            # "GW170817_LIKE" (Default: HMNS with delayed collapse ~0.05-0.1 s)
            self.t_collapse_delay = getattr(config, "HMNS_LIFETIME", 0.08)
            self.spin_final = 0.70

        self._has_collapsed: bool = False

    def reset(self):
        """Reset remnant model collapse state."""
        self._has_collapsed = False

    def evaluate(self, inspiral_state: InspiralState, event_time: float) -> RemnantState:
        """
        Evaluate remnant state at physical event time t [s] relative to merger.
        """
        if event_time < 0.0:
            # Pre-merger inspiral phase
            r_eff = self.R_NS
            m_rem = self.M_total
            u = (2.0 * G * m_rem) / (c**2 * r_eff)
            return RemnantState(
                time=event_time,
                scenario=self.scenario,
                remnant_type="INSPIRAL",
                mass=m_rem,
                radius=r_eff,
                spin_parameter=0.0,
                omega_rot=inspiral_state.omega_orb,
                compactness=u,
                is_black_hole=False,
                collapse_time=self.t_collapse_delay
            )

        # Post-merger phase (t >= 0.0 s): Monotonic delayed collapse to persistent Black Hole
        if event_time >= self.t_collapse_delay:
            self._has_collapsed = True

        is_bh = bool(self._has_collapsed or event_time >= self.t_collapse_delay)
        remnant_type = "BH" if is_bh else "HMNS"

        # Mass loss to ejecta and GWs during merger (~3-5% total mass)
        mass_loss_frac = min(0.05, 0.01 + 0.02 * (1.0 - np.exp(-event_time / 0.02)))
        m_rem = self.M_total * (1.0 - mass_loss_frac)

        # Spin parameter evolution
        a_star = float(min(self.spin_final, self.spin_final * (1.0 - np.exp(-event_time / 0.01))))

        if is_bh:
            # Black Hole horizon radius (Schwarzschild / Kerr proxy: R_H = G M / c^2 (1 + sqrt(1 - a*^2)))
            r_horizon = (G * m_rem / c**2) * (1.0 + np.sqrt(max(0.0, 1.0 - a_star**2)))
            r_eff = float(r_horizon)
            u = (2.0 * G * m_rem) / (c**2 * r_eff)
            omega_rot = (c**3 / (G * m_rem)) * (a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2))))
        else:
            # HMNS radius (diffuse differential rotation, R ~ 13-16 km)
            r_hmns = 14.0e3 * (1.0 + 0.15 * np.exp(-event_time / 0.02))
            r_eff = float(r_hmns)
            u = (2.0 * G * m_rem) / (c**2 * r_eff)
            omega_rot = float(2.0 * np.pi * 1800.0 * np.exp(-event_time / 0.05))

        return RemnantState(
            time=event_time,
            scenario=self.scenario,
            remnant_type=remnant_type,
            mass=m_rem,
            radius=r_eff,
            spin_parameter=a_star,
            omega_rot=float(omega_rot),
            compactness=float(u),
            is_black_hole=is_bh,
            collapse_time=self.t_collapse_delay
        )
