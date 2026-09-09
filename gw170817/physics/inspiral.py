"""
Post-Newtonian circular binary inspiral model for GW170817.

REDUCED-ORDER APPROXIMATION:
Uses quadrupole formula radiation reaction (2.5PN equivalent leading order)
for circular compact binary inspiral.
Does not include spin-orbit coupling, eccentricity, or full GR tidal back-reaction.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig


@dataclass
class InspiralState:
    """Current state of binary inspiral."""
    time: float              # Physical time since start [s]
    f_gw: float              # Instantaneous GW frequency [Hz]
    orbital_frequency: float # Instantaneous orbital frequency f_orb [Hz] (f_gw / 2)
    omega_orb: float         # Instantaneous orbital angular frequency [rad/s] (pi * f_gw)
    separation: float        # Orbital separation a [m]
    orbital_phase: float     # Orbital phase phi [rad]
    df_dt: float             # Instantaneous GW frequency derivative [Hz/s]
    chirp_mass: float        # Binary chirp mass M_c [kg]


class InspiralModel:
    """
    PN-inspired reduced-order inspiral model.
    Evolves gravitational-wave frequency f_gw and orbital phase phi.
    """

    def __init__(self, config: SimConfig = None, f_max: float = 1500.0):
        if config is None:
            config = SimConfig()
        self.config = config
        self.f_max = f_max
        
        # Pre-compute constant factor for df/dt:
        # df/dt = (96/5) * pi^(8/3) * (G * M_c / c^3)^(5/3) * f^(11/3)
        m_c = config.chirp_mass
        g_mc_over_c3 = (G * m_c) / (c**3)
        self._df_coeff = (96.0 / 5.0) * (np.pi ** (8.0 / 3.0)) * (g_mc_over_c3 ** (5.0 / 3.0))

    def _compute_df_dt(self, f_gw: float) -> float:
        """Calculate df_gw/dt from quadrupole formula."""
        if f_gw <= 0.0:
            return 0.0
        return self._df_coeff * (f_gw ** (11.0 / 3.0))

    def _compute_separation(self, f_gw: float) -> float:
        """
        Calculate orbital separation a [m] from f_gw:
        a = (G * M_total / (pi * f_gw)^2)^(1/3)
        """
        f_gw_clamped = max(f_gw, 1.0e-3)
        return (G * self.config.M_total / (np.pi * f_gw_clamped)**2)**(1.0 / 3.0)

    def initial_state(self, config: SimConfig = None) -> InspiralState:
        """Construct initial InspiralState from configuration."""
        if config is not None:
            self.config = config
            m_c = config.chirp_mass
            g_mc_over_c3 = (G * m_c) / (c**3)
            self._df_coeff = (96.0 / 5.0) * (np.pi ** (8.0 / 3.0)) * (g_mc_over_c3 ** (5.0 / 3.0))

        f_gw0 = self.config.f_gw_start
        f_orb0 = f_gw0 / 2.0
        omega0 = np.pi * f_gw0
        a0 = self._compute_separation(f_gw0)
        df_dt0 = self._compute_df_dt(f_gw0)

        return InspiralState(
            time=0.0,
            f_gw=f_gw0,
            orbital_frequency=f_orb0,
            omega_orb=omega0,
            separation=a0,
            orbital_phase=0.0,
            df_dt=df_dt0,
            chirp_mass=self.config.chirp_mass
        )

    def derivative(self, f_gw: float, phi: float) -> Tuple[float, float]:
        """
        Compute state derivatives (df_gw/dt, dphi/dt).
        """
        f_clamped = max(f_gw, 1.0e-3)
        df_dt = self._compute_df_dt(f_clamped)
        dphi_dt = np.pi * f_clamped
        return df_dt, dphi_dt

    def step(self, state: InspiralState, dt: float) -> InspiralState:
        """
        Advance state by timestep dt using 4th-order Runge-Kutta (RK4).
        """
        f0 = state.f_gw
        phi0 = state.orbital_phase

        if f0 >= self.f_max:
            # Terminated / Merged state
            return InspiralState(
                time=state.time + dt,
                f_gw=self.f_max,
                orbital_frequency=self.f_max / 2.0,
                omega_orb=np.pi * self.f_max,
                separation=self._compute_separation(self.f_max),
                orbital_phase=phi0 + np.pi * self.f_max * dt,
                df_dt=0.0,
                chirp_mass=state.chirp_mass
            )

        # RK4 Integration step for [f_gw, phi]
        k1_f, k1_phi = self.derivative(f0, phi0)
        
        f_k2 = max(f0 + 0.5 * dt * k1_f, 1.0e-3)
        phi_k2 = phi0 + 0.5 * dt * k1_phi
        k2_f, k2_phi = self.derivative(f_k2, phi_k2)

        f_k3 = max(f0 + 0.5 * dt * k2_f, 1.0e-3)
        phi_k3 = phi0 + 0.5 * dt * k2_phi
        k3_f, k3_phi = self.derivative(f_k3, phi_k3)

        f_k4 = max(f0 + dt * k3_f, 1.0e-3)
        phi_k4 = phi0 + dt * k3_phi
        k4_f, k4_phi = self.derivative(f_k4, phi_k4)

        f_next = f0 + (dt / 6.0) * (k1_f + 2.0 * k2_f + 2.0 * k3_f + k4_f)
        phi_next = phi0 + (dt / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi)

        # Numerical safety clamps
        f_next = min(max(f_next, 1.0e-3), self.f_max)
        
        # Derived quantities for next state
        a_next = self._compute_separation(f_next)
        omega_next = np.pi * f_next
        f_orb_next = f_next / 2.0
        df_dt_next = self._compute_df_dt(f_next)

        return InspiralState(
            time=state.time + dt,
            f_gw=f_next,
            orbital_frequency=f_orb_next,
            omega_orb=omega_next,
            separation=a_next,
            orbital_phase=phi_next,
            df_dt=df_dt_next,
            chirp_mass=state.chirp_mass
        )

    def orbital_positions(
        self, state: InspiralState, m1: float = None, m2: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate center-of-mass positions of both neutron stars.
        Returns r1, r2 as 3D numpy vectors [x, y, 0].
        """
        if m1 is None:
            m1 = self.config.m1
        if m2 is None:
            m2 = self.config.m2

        M_total = m1 + m2
        a = state.separation
        phi = state.orbital_phase

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        r1 = -a * (m2 / M_total) * np.array([cos_p, sin_p, 0.0], dtype=np.float64)
        r2 =  a * (m1 / M_total) * np.array([cos_p, sin_p, 0.0], dtype=np.float64)

        return r1, r2

    def orbital_velocities(
        self, state: InspiralState, m1: float = None, m2: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate tangential orbital velocities of both neutron stars.
        Returns v1, v2 as 3D numpy vectors [vx, vy, 0].
        """
        if m1 is None:
            m1 = self.config.m1
        if m2 is None:
            m2 = self.config.m2

        M_total = m1 + m2
        a = state.separation
        phi = state.orbital_phase
        omega = state.omega_orb

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        v1 = -a * (m2 / M_total) * omega * np.array([-sin_p, cos_p, 0.0], dtype=np.float64)
        v2 =  a * (m1 / M_total) * omega * np.array([-sin_p, cos_p, 0.0], dtype=np.float64)

        return v1, v2
