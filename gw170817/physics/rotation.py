"""
Rotational Dynamics & Angular Momentum Physics Module for GW170817.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Models orbital angular momentum tracking during inspiral and post-merger transfer
into the remnant, accretion disk, ejecta, GW radiation loss, and other losses.
Evolves differential rotation, shear, rotational energy, T/|W| diagnostic,
and gravitational redshift.

Provenance Labels:
  [OBS] GW inspiral orbital dynamics & separation
  [REF] Differential rotation profiles inspired by published BNS MHD simulations
  [MOD] Reduced-order angular momentum partitioning & transport model
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralState
from gw170817.physics.remnant import RemnantState


@dataclass
class RotationalState:
    """Instantaneous state of rotational dynamics and angular momentum budget."""
    time: float                     # Event time relative to merger [s]
    angular_momentum: float        # Total system angular momentum [kg m^2 / s]
    l_orb: float                   # Orbital angular momentum [kg m^2 / s]
    j_remnant: float               # Remnant angular momentum [kg m^2 / s]
    j_disk: float                  # Accretion disk angular momentum [kg m^2 / s]
    j_ejecta: float                # Dynamically unbound ejecta angular momentum [kg m^2 / s]
    j_gw_loss: float               # Cumulative GW radiation loss [kg m^2 / s]
    j_other_loss: float            # Other outflow/neutrino losses [kg m^2 / s]
    omega_core: float              # Central core angular velocity [rad/s]
    omega_outer: float             # Outer envelope angular velocity [rad/s]
    omega_mean: float              # Mass-weighted mean angular velocity [rad/s]
    differential_rotation: float   # Delta Omega = Omega_core - Omega_outer [rad/s]
    rotational_energy: float       # Rotational kinetic energy T_rot [J]
    binding_energy_proxy: float    # Gravitational binding energy proxy |W| [J]
    T_over_W: float                # T / |W| ratio (dimensionless diagnostic)
    shear: float                   # Shearing rate |r dOmega/dr| [s^-1]
    angular_momentum_transfer: float # Transport rate dJ/dt [J / s]
    transport_timescale: float     # Angular momentum transport timescale tau_transport [s]
    compactness: float             # Compactness u = 2 G M / (c^2 R)
    gravitational_redshift: float  # Redshift z = 1 / sqrt(1 - u) - 1


class RotationalModel:
    """
    Reduced-order rotational dynamics and explicit angular momentum budget model.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        if config is None:
            config = SimConfig()
        self.config = config

        self.m1 = config.m1
        self.m2 = config.m2
        self.M_total = config.M_total
        self.mu = (self.m1 * self.m2) / self.M_total

        # Configurable reduced-order loss fractions [MOD]
        self.f_gw_loss = getattr(config, "J_LOSS_FRAC_GW", 0.04)       # ~4% radiated in GWs
        self.f_ejecta_loss = getattr(config, "J_LOSS_FRAC_EJECTA", 0.02) # ~2% in ejecta
        self.f_disk_frac = getattr(config, "J_FRAC_DISK", 0.15)         # ~15% in accretion disk
        self.f_other_loss = getattr(config, "J_LOSS_FRAC_OTHER", 0.01)  # ~1% in other outflows

        # Transport & profile parameters
        self.tau_transport_init = getattr(config, "J_TRANSPORT_TIMESCALE", 0.025) # [s]
        self.alpha_diff_0 = 0.8  # Initial differential rotation fraction

    def compute_orbital_l(self, inspiral_state: InspiralState) -> float:
        """
        Calculate orbital angular momentum L_orb = mu * omega_orb * a^2 = mu * sqrt(G M a).
        """
        a = max(1.0, inspiral_state.separation)
        return float(self.mu * np.sqrt(G * self.M_total * a))

    def evaluate(
        self,
        inspiral_state: InspiralState,
        remnant_state: RemnantState,
        event_time: float,
        magnetic_stress: float = 0.0
    ) -> RotationalState:
        """
        Evaluate complete rotational dynamics and angular momentum budget at event_time [s].
        """
        # Pre-merger inspiral phase
        if event_time < 0.0:
            l_orb = self.compute_orbital_l(inspiral_state)
            u = (2.0 * G * self.M_total) / (c**2 * max(1.0, inspiral_state.separation))
            z = 1.0 / np.sqrt(max(1.0e-6, 1.0 - u)) - 1.0

            return RotationalState(
                time=event_time,
                angular_momentum=l_orb,
                l_orb=l_orb,
                j_remnant=0.0,
                j_disk=0.0,
                j_ejecta=0.0,
                j_gw_loss=0.0,
                j_other_loss=0.0,
                omega_core=inspiral_state.omega_orb,
                omega_outer=inspiral_state.omega_orb,
                omega_mean=inspiral_state.omega_orb,
                differential_rotation=0.0,
                rotational_energy=0.5 * 0.35 * self.M_total * (11.5e3)**2 * inspiral_state.omega_orb**2,
                binding_energy_proxy=(0.6 * G * self.M_total**2) / (11.5e3),
                T_over_W=0.01,
                shear=0.0,
                angular_momentum_transfer=0.0,
                transport_timescale=self.tau_transport_init,
                compactness=float(u),
                gravitational_redshift=float(z)
            )

        # Contact / Merger orbital angular momentum anchor
        # Evaluated at contact separation (a_contact ~ 3 * (R1 + R2))
        a_contact = 3.0 * (self.config.R_NS1 + self.config.R_NS2)
        l_contact = self.mu * np.sqrt(G * self.M_total * a_contact)

        # Time-dependent angular momentum partitioning [MOD]
        t = max(0.0, event_time)
        gw_decay = 1.0 - np.exp(-t / 0.005)
        ej_decay = 1.0 - np.exp(-t / 0.01)
        disk_decay = (1.0 - np.exp(-t / 0.01)) * np.exp(-t / 0.30)
        oth_decay = 1.0 - np.exp(-t / 0.05)

        j_gw = l_contact * self.f_gw_loss * gw_decay
        j_ej = l_contact * self.f_ejecta_loss * ej_decay
        j_dk = l_contact * self.f_disk_frac * disk_decay
        j_oth = l_contact * self.f_other_loss * oth_decay

        j_rem = l_contact - (j_gw + j_ej + j_dk + j_oth)
        l_total = j_rem + j_dk + j_ej + j_gw + j_oth  # Strictly conserved sum!

        # Transport timescale accelerated by magnetic stress
        tau_trans = self.tau_transport_init / (1.0 + 10.0 * min(1.0, magnetic_stress / 1.0e14))
        tau_trans = max(0.002, float(tau_trans))

        # Remnant geometry & moment of inertia
        r_rem = max(1.0e3, remnant_state.radius)
        m_rem = max(0.1 * M_sun, remnant_state.mass)
        i_rem = 0.35 * m_rem * (r_rem**2)

        omega_mean = j_rem / i_rem

        # Differential rotation profile Omega(r) = Omega_core / (1 + (r/A)^2)
        diff_decay = np.exp(-t / tau_trans)
        alpha_diff = self.alpha_diff_0 * diff_decay
        omega_core = omega_mean * (1.0 + alpha_diff)
        omega_outer = omega_mean * (1.0 - 0.5 * alpha_diff)
        delta_omega = max(0.0, omega_core - omega_outer)

        # Shear |r dOmega/dr| ~ Delta Omega / 0.5
        scale_A = 0.5 * r_rem
        shear = (omega_core * r_rem * scale_A) / ((scale_A**2 + (0.5 * r_rem)**2)**1.5)

        # Energetics & Diagnostics
        t_rot = 0.5 * i_rem * (omega_mean**2)
        w_proxy = (0.6 * G * (m_rem**2)) / r_rem
        t_over_w = t_rot / max(1.0, w_proxy)

        # Gravitational compactness & redshift diagnostic
        u = (2.0 * G * m_rem) / (c**2 * r_rem)
        u_clamped = min(0.95, max(1.0e-5, u))
        z = 1.0 / np.sqrt(1.0 - u_clamped) - 1.0

        # Rate of angular momentum transfer onto disk/outflows
        dj_dt = (l_contact * self.f_disk_frac / 0.30) * np.exp(-t / 0.30)

        return RotationalState(
            time=event_time,
            angular_momentum=float(l_total),
            l_orb=float(l_contact),
            j_remnant=float(j_rem),
            j_disk=float(j_dk),
            j_ejecta=float(j_ej),
            j_gw_loss=float(j_gw),
            j_other_loss=float(j_oth),
            omega_core=float(omega_core),
            omega_outer=float(omega_outer),
            omega_mean=float(omega_mean),
            differential_rotation=float(delta_omega),
            rotational_energy=float(t_rot),
            binding_energy_proxy=float(w_proxy),
            T_over_W=float(t_over_w),
            shear=float(shear),
            angular_momentum_transfer=float(dj_dt),
            transport_timescale=float(tau_trans),
            compactness=float(u),
            gravitational_redshift=float(z)
        )
