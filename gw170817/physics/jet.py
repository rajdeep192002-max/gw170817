"""
Reduced-order structured relativistic jet model for GW170817.

REDUCED-ORDER APPROXIMATION:
This is a reduced-order phenomenological structured relativistic jet model.
It does NOT perform:
- GRMHD
- relativistic radiation transport
- neutrino transport
- magnetic-field evolution
- detailed jet launching
- synchrotron radiative transfer
- detector response modeling

Its purpose is to provide a computationally inexpensive bridge between:
    merger -> relativistic jet -> off-axis prompt GRB
"""
from dataclasses import dataclass
from typing import Dict, Any, Union, Tuple
import numpy as np
from gw170817.constants import c, Mpc
from gw170817.config import SimConfig
from gw170817.physics.merger import MergerState


@dataclass
class JetState:
    """Instantaneous state of the structured relativistic jet and prompt GRB emission."""
    time: float                    # Physical time [s]
    launched: bool                 # True if jet has launched after merger
    grb_triggered: bool            # True if GRB delay (1.7s) has elapsed after merger
    theta: float                   # Off-axis angle evaluated [rad]
    viewing_angle: float           # Observer viewing angle [rad]
    E_core: float                  # Core isotropic-equivalent energy [J]
    E_theta: float                 # Isotropic-equivalent energy at angle theta [J]
    Gamma: float                   # Lorentz factor Gamma(theta)
    beta: float                    # Velocity ratio beta(theta) = v/c
    doppler_factor: float          # Relativistic Doppler factor delta
    intrinsic_luminosity: float    # Intrinsic prompt GRB luminosity L_iso(theta) [W]
    observed_flux: float           # Observed prompt GRB flux F_obs [W/m^2]
    distance: float                # Luminosity distance D [m]
    jet_delay: float               # GW-to-GRB prompt delay delta_t [s]


class StructuredJetModel:
    """
    Physics-informed reduced-order Gaussian structured relativistic jet model.
    """

    def __init__(
        self,
        config: SimConfig = None,
        E_core_erg: float = 1.0e50,      # Core isotropic energy [erg]
        Gamma_core: float = 100.0,       # Core Lorentz factor
        theta_core_rad: float = 0.08,    # Core half-opening angle [rad] (~4.58 deg)
        viewing_angle_deg: float = 22.0, # Observer viewing angle [deg]
        jet_delay_s: float = 1.7,        # Prompt GW-to-GRB delay [s]
        duration_s: float = 2.0          # Prompt GRB duration [s]
    ):
        if config is None:
            config = SimConfig()
        self.config = config

        # Convert energy: 1 erg = 1.0e-7 J
        self.E_core = float(E_core_erg * 1.0e-7)  # [J]
        self.Gamma_core = float(Gamma_core)
        self.theta_core = float(theta_core_rad)
        self.viewing_angle = float(np.deg2rad(viewing_angle_deg))
        self.jet_delay = float(jet_delay_s)
        self.duration = float(duration_s)
        self.distance = float(config.distance)

        # Safety checks
        if self.E_core <= 0.0:
            raise ValueError(f"Core energy must be positive, got {self.E_core}")
        if self.Gamma_core < 1.0:
            raise ValueError(f"Core Lorentz factor must be >= 1.0, got {self.Gamma_core}")
        if self.theta_core <= 0.0:
            raise ValueError(f"Core angle must be positive, got {self.theta_core}")
        if self.distance <= 0.0:
            raise ValueError(f"Distance must be positive, got {self.distance}")
        if self.duration <= 0.0:
            raise ValueError(f"Duration must be positive, got {self.duration}")

    def energy_profile(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Gaussian structured jet energy profile:
        E(theta) = E_core * exp(-theta^2 / (2 * theta_core^2))
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        E_val = self.E_core * np.exp(-(theta_arr**2) / (2.0 * self.theta_core**2))
        return float(E_val) if np.isscalar(theta) or theta_arr.ndim == 0 else E_val

    def lorentz_profile(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Gaussian structured Lorentz factor profile:
        Gamma(theta) = 1 + (Gamma_core - 1) * exp(-theta^2 / (2 * theta_core^2))
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        Gamma_val = 1.0 + (self.Gamma_core - 1.0) * np.exp(-(theta_arr**2) / (2.0 * self.theta_core**2))
        return float(Gamma_val) if np.isscalar(theta) or theta_arr.ndim == 0 else Gamma_val

    def beta(self, Gamma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Relativistic velocity ratio beta = sqrt(1 - 1 / Gamma^2).
        Numerically protected against precision underflow.
        """
        g_arr = np.asarray(Gamma, dtype=np.float64)
        g_clamped = np.maximum(g_arr, 1.0)
        inv_g2 = 1.0 / (g_clamped**2)
        beta_val = np.sqrt(np.maximum(0.0, 1.0 - inv_g2))
        return float(beta_val) if np.isscalar(Gamma) or g_arr.ndim == 0 else beta_val

    def doppler_factor(
        self,
        Gamma: Union[float, np.ndarray],
        alpha: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Relativistic Doppler factor:
        delta = 1 / (Gamma * (1 - beta * cos(alpha)))
        """
        g_arr = np.asarray(Gamma, dtype=np.float64)
        a_arr = np.asarray(alpha, dtype=np.float64)
        beta_arr = np.asarray(self.beta(g_arr), dtype=np.float64)

        denom = g_arr * (1.0 - beta_arr * np.cos(a_arr))
        denom_clamped = np.maximum(denom, 1.0e-12)
        delta_val = 1.0 / denom_clamped

        return float(delta_val) if (np.isscalar(Gamma) and np.isscalar(alpha)) else delta_val

    def launch_from_merger(self, merger_state: MergerState) -> bool:
        """
        Phenomenological jet launching condition:
        Returns True if merger is complete, in contact, or merger started.
        """
        return bool(merger_state.merger_complete or merger_state.in_contact or merger_state.merger_started)

    def evaluate(
        self,
        theta: float = None,
        alpha: float = None,
        time: float = 0.0,
        merger_state: MergerState = None
    ) -> JetState:
        """
        Evaluate full JetState at angle theta and observer angle alpha.
        """
        if theta is None:
            theta = self.viewing_angle
        if alpha is None:
            alpha = theta

        theta_val = float(theta)
        alpha_val = float(alpha)
        time_val = float(time)

        # Merger launch coupling
        launched = False
        grb_triggered = False

        if merger_state is not None:
            launched = self.launch_from_merger(merger_state)
            grb_triggered = bool(launched and (time_val >= self.jet_delay))

        # Structure profiles
        E_th = self.energy_profile(theta_val)
        Gamma_th = self.lorentz_profile(theta_val)
        beta_th = self.beta(Gamma_th)
        delta_val = self.doppler_factor(Gamma_th, alpha_val)

        # Intrinsic prompt luminosity: L_iso = E(theta) / duration
        L_iso = E_th / self.duration

        # Observed flux proxy: F_obs = L_iso * delta^3 / (4 * pi * D^2)
        flux_obs = (L_iso * (delta_val**3)) / (4.0 * np.pi * (self.distance**2))

        # Safety checks
        if not np.isfinite(flux_obs) or not np.isfinite(delta_val) or not np.isfinite(Gamma_th):
            raise ValueError("Calculated JetState contains NaN or Inf")

        return JetState(
            time=time_val,
            launched=launched,
            grb_triggered=grb_triggered,
            theta=theta_val,
            viewing_angle=self.viewing_angle,
            E_core=self.E_core,
            E_theta=E_th,
            Gamma=Gamma_th,
            beta=beta_th,
            doppler_factor=delta_val,
            intrinsic_luminosity=L_iso,
            observed_flux=flux_obs,
            distance=self.distance,
            jet_delay=self.jet_delay
        )
