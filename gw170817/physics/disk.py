"""
Accretion Disk Surrogate Model for BNS Mergers.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Models post-merger accretion disk formation, Keplerian differential rotation profile
Omega_K(r) = sqrt(G M / r^3), viscous accretion, disk scale height (H/R),
specific angular momentum l(r), accretion rate M_dot onto central remnant, and thermal state.

References:
  Hayashi et al., Phys. Rev. Lett. 134, 211407 (2025)
  Metzger, Living Rev. Relativ. 22, 1 (2019)
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.physics.remnant import RemnantState
from gw170817.physics.rotation import RotationalState


@dataclass
class DiskState:
    """Instantaneous state of the accretion disk."""
    time: float                  # Event time relative to merger [s]
    disk_mass: float             # Disk mass [kg]
    disk_mass_msun: float        # Disk mass [M_sun]
    accretion_rate: float        # Mass accretion rate M_dot [kg/s]
    scale_height: float          # Disk aspect ratio H/R (dimensionless)
    inner_radius: float          # Inner disk radius [m]
    outer_radius: float          # Outer disk radius [m]
    omega_disk: float            # Keplerian angular velocity at inner radius [rad/s]
    v_phi: float                 # Azimuthal velocity at inner radius [m/s]
    specific_angular_momentum: float # Specific angular momentum l = v_phi * r [m^2/s]
    density_proxy: float         # Peak disk density proxy [kg/m^3]
    temperature: float           # Peak disk temperature [K]
    is_active: bool              # True if disk is formed post-merger


class DiskModel:
    """
    Reduced-order accretion disk evolution model coupled to rotational dynamics.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        if config is None:
            config = SimConfig()
        self.config = config

        # Initial disk mass ~ 0.06 M_sun for GW170817 binary mass ratio
        self.M_disk_0 = getattr(config, "INITIAL_DISK_MASS", 0.06 * M_sun)
        self.tau_acc = getattr(config, "DISK_ACCRETION_TIMESCALE", 0.30)  # [s]

    def omega_keplerian(self, mass: float, radius: float) -> float:
        """Keplerian angular velocity Omega_K(r) = sqrt(G M / r^3)."""
        r_clamped = max(1.0e3, radius)
        m_clamped = max(0.1 * M_sun, mass)
        return float(np.sqrt(G * m_clamped / (r_clamped**3)))

    def evaluate(
        self,
        remnant_state: RemnantState,
        event_time: float,
        rotational_state: Optional[RotationalState] = None
    ) -> DiskState:
        """
        Evaluate disk state at event time t relative to merger [s].
        """
        if event_time < 0.0 or not (remnant_state.remnant_type in ["HMNS", "BH"]):
            return DiskState(
                time=event_time,
                disk_mass=0.0,
                disk_mass_msun=0.0,
                accretion_rate=0.0,
                scale_height=0.0,
                inner_radius=remnant_state.radius,
                outer_radius=remnant_state.radius,
                omega_disk=0.0,
                v_phi=0.0,
                specific_angular_momentum=0.0,
                density_proxy=0.0,
                temperature=0.0,
                is_active=False
            )

        # Accretion timescale responds to rotational angular-momentum transport
        tau_visc = self.tau_acc
        if rotational_state is not None:
            tau_visc = max(0.05, rotational_state.transport_timescale * 10.0)

        # Post-merger disk growth -> viscous decay with self-similar late-time power-law accretion tail (Metzger 2019, Hayashi et al. 2025)
        growth_factor = 1.0 - np.exp(-event_time / 0.01)
        decay_factor = float(np.exp(-min(event_time / tau_visc, 15.0)) + 2.0e-3 / ((1.0 + event_time / tau_visc) ** 0.5))
        m_disk = self.M_disk_0 * growth_factor * decay_factor

        # Accretion rate M_dot = M_disk / tau_visc (continuous and positive for all post-merger times)
        mdot = (m_disk / tau_visc)

        # Aspect ratio H/R ~ 0.2 - 0.25 (thick advection-dominated disk)
        h_over_r = 0.22 * (1.0 + 0.1 * np.exp(-event_time / 0.1))

        # Disk Radii
        r_in = max(remnant_state.radius, 1.2 * remnant_state.radius)
        r_out = r_in + 120.0e3 * (1.0 + 0.5 * (1.0 - np.exp(-event_time / 0.2)))

        # Keplerian rotation profile Omega_K(r_in)
        omega_k = self.omega_keplerian(remnant_state.mass, r_in)
        v_phi = omega_k * r_in
        spec_l = v_phi * r_in

        # Density proxy rho_disk ~ M_disk / (2 pi R_out^2 H)
        h_abs = r_in * h_over_r
        vol_proxy = np.pi * (r_out**2 - r_in**2) * h_abs
        rho_proxy = m_disk / max(1.0, vol_proxy)

        # Peak thermal temperature T ~ 10^10 K
        t_peak = 3.5e10 * (mdot / (0.1 * M_sun))**0.25 * (20.0e3 / r_in)**0.75

        return DiskState(
            time=event_time,
            disk_mass=float(m_disk),
            disk_mass_msun=float(m_disk / M_sun),
            accretion_rate=float(mdot),
            scale_height=float(h_over_r),
            inner_radius=float(r_in),
            outer_radius=float(r_out),
            omega_disk=float(omega_k),
            v_phi=float(v_phi),
            specific_angular_momentum=float(spec_l),
            density_proxy=float(rho_proxy),
            temperature=float(t_peak),
            is_active=True
        )
