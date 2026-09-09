"""
Reduced-order two-component kilonova model for GW170817.

REDUCED-ORDER APPROXIMATION:
This is a reduced-order two-component kilonova model.
It is not a full radiation-transfer calculation.
It does not model detailed atomic opacities, wavelength-dependent spectra,
Monte Carlo transport, or neutrino-radiation hydrodynamics.
"""
from dataclasses import dataclass
from typing import Dict, Any, Union
import numpy as np
from gw170817.constants import c, sigma_SB, day, M_sun
from gw170817.config import SimConfig
from gw170817.physics.ejecta import EjectaState


@dataclass
class KilonovaState:
    """Instantaneous state of the two-component kilonova emission."""
    time: float          # Time since merger [s]
    L_blue: float        # Blue component luminosity [W]
    L_red: float         # Red component luminosity [W]
    L_total: float       # Total kilonova luminosity [W]
    T_blue: float        # Blue component effective temperature [K]
    T_red: float         # Red component effective temperature [K]
    R_blue: float        # Blue component photospheric radius [m]
    R_red: float         # Red component photospheric radius [m]
    t_diff_blue: float   # Blue component diffusion timescale [s]
    t_diff_red: float    # Red component diffusion timescale [s]
    M_blue: float        # Blue component ejecta mass [kg]
    M_red: float         # Red component ejecta mass [kg]
    kappa_blue: float    # Blue component opacity [m^2/kg]
    kappa_red: float     # Red component opacity [m^2/kg]


class KilonovaModel:
    """
    Phenomenological two-component (Blue + Red) kilonova light curve model.
    Driven directly by EjectaState.
    """

    def __init__(
        self,
        config: SimConfig = None,
        kappa_blue_cgs: float = 1.0,     # cm^2/g
        kappa_red_cgs: float = 10.0,     # cm^2/g
        epsilon0_cgs: float = 2.0e10,    # erg/g/s at 1 day
        beta: float = 13.8,              # Geometric factor
        v_blue_ratio: float = 1.2,       # Velocity ratio for blue component relative to mean
        v_red_ratio: float = 0.8         # Velocity ratio for red component relative to mean
    ):
        if config is None:
            config = SimConfig()
        self.config = config

        # Convert opacities: 1 cm^2/g = 0.1 m^2/kg
        self.kappa_blue = float(kappa_blue_cgs * 0.1)
        self.kappa_red = float(kappa_red_cgs * 0.1)

        if self.kappa_blue <= 0.0 or self.kappa_red <= 0.0:
            raise ValueError(f"Opacities must be positive, got blue={self.kappa_blue}, red={self.kappa_red}")

        # Heating rate normalization: 1 erg/g/s = 1e-4 W/kg -> 2e10 erg/g/s = 2.0e6 W/kg?
        # Wait: 1 erg = 1e-7 J, 1 g = 1e-3 kg => 1 erg/g = 1e-4 J/kg => 2e10 erg/g/s = 2.0e6 W/kg
        # Let's verify: 2.0e10 * 1e-4 = 2.0e6 W/kg (2000 kW/kg).
        self.epsilon0 = float(epsilon0_cgs * 1.0e-4)   # W/kg at 1 day

        self.beta = float(beta)
        self.v_blue_ratio = float(v_blue_ratio)
        self.v_red_ratio = float(v_red_ratio)

    def _heating_rate(self, t_seconds: float) -> float:
        """
        r-process radioactive heating rate per unit mass [W/kg]:
        eps(t) = eps_0 * ((t + t0) / 1 day)^(-1.3)
        """
        t0 = 10.0  # softening time [s]
        t_days = max(t_seconds + t0, t0) / day
        return self.epsilon0 * (t_days ** (-1.3))

    def _compute_diffusion_time(self, mass: float, velocity: float, kappa: float) -> float:
        """
        Diffusion timescale t_diff = sqrt(2 * kappa * M / (beta * c * v))
        """
        if mass <= 0.0 or velocity <= 0.0 or kappa <= 0.0:
            return 0.0
        val = (2.0 * kappa * mass) / (self.beta * c * velocity)
        return float(np.sqrt(val))

    def evaluate(self, ejecta_state: EjectaState, t_seconds: float) -> KilonovaState:
        """
        Evaluate instant kilonova state (luminosities, temperatures, radii) at t_seconds after merger.
        """
        t = float(t_seconds)

        if ejecta_state.ejecta_mass <= 0.0 or t <= 0.0:
            return KilonovaState(
                time=max(0.0, t),
                L_blue=0.0,
                L_red=0.0,
                L_total=0.0,
                T_blue=0.0,
                T_red=0.0,
                R_blue=0.0,
                R_red=0.0,
                t_diff_blue=0.0,
                t_diff_red=0.0,
                M_blue=0.0,
                M_red=0.0,
                kappa_blue=self.kappa_blue,
                kappa_red=self.kappa_red
            )

        # Mass split
        m_ej = ejecta_state.ejecta_mass
        f_red = float(np.clip(ejecta_state.lanthanide_rich_fraction, 0.0, 1.0))
        m_red = m_ej * f_red
        m_blue = m_ej * (1.0 - f_red)

        # Velocities
        v_mean = max(ejecta_state.mean_velocity, 1.0e4)
        v_blue = min(v_mean * self.v_blue_ratio, 0.5 * c)
        v_red = min(v_mean * self.v_red_ratio, 0.5 * c)

        # Diffusion timescales
        t_diff_blue = self._compute_diffusion_time(m_blue, v_blue, self.kappa_blue)
        t_diff_red = self._compute_diffusion_time(m_red, v_red, self.kappa_red)

        # Heating rate at time t
        eps_dot = self._heating_rate(t)

        # Diffusion efficiency trapping factor: eta(t) = 1 - exp(-(t / t_diff)^2)
        eta_blue = (1.0 - np.exp(-(t / t_diff_blue)**2)) if t_diff_blue > 0.0 else 1.0
        eta_red = (1.0 - np.exp(-(t / t_diff_red)**2)) if t_diff_red > 0.0 else 1.0

        # Luminosities [W]
        L_blue = float(m_blue * eps_dot * eta_blue)
        L_red = float(m_red * eps_dot * eta_red)
        L_total = L_blue + L_red

        # Photospheric radii R(t) = v * t
        r_blue = float(v_blue * t)
        r_red = float(v_red * t)

        # Effective temperatures from Stefan-Boltzmann law: L = 4 * pi * R^2 * sigma * T^4
        def calc_temp(L: float, R: float) -> float:
            if L <= 0.0 or R <= 0.0:
                return 0.0
            val = L / (4.0 * np.pi * (R**2) * sigma_SB)
            return float(np.power(val, 0.25))

        t_blue = calc_temp(L_blue, r_blue)
        t_red = calc_temp(L_red, r_red)

        # Safety checks
        if not np.isfinite(L_total) or not np.isfinite(t_blue) or not np.isfinite(t_red):
            raise ValueError("Calculated KilonovaState contains NaN or Inf")

        return KilonovaState(
            time=t,
            L_blue=L_blue,
            L_red=L_red,
            L_total=L_total,
            T_blue=t_blue,
            T_red=t_red,
            R_blue=r_blue,
            R_red=r_red,
            t_diff_blue=t_diff_blue,
            t_diff_red=t_diff_red,
            M_blue=m_blue,
            M_red=m_red,
            kappa_blue=self.kappa_blue,
            kappa_red=self.kappa_red
        )
