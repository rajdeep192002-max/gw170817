"""
Reduced-order broadband synchrotron afterglow model for GW170817.

REDUCED-ORDER APPROXIMATION:
This is a reduced-order phenomenological broadband afterglow model.
It does NOT perform:
- relativistic hydrodynamics
- detailed shock evolution
- particle acceleration microphysics
- synchrotron radiative transfer
- inverse-Compton calculations
- detailed jet lateral dynamics
- detector response
- full radiative-transfer calculations

Its purpose is to reproduce the dominant observable temporal/spectral scaling
(slow rise -> peak around 150 days -> steep decline) at extremely low computational cost.
"""
from dataclasses import dataclass
from typing import Dict, Any, Union, Tuple
import numpy as np
from gw170817.constants import day, Mpc
from gw170817.config import SimConfig
from gw170817.physics.jet import JetState


@dataclass
class AfterglowState:
    """Instantaneous state of the phenomenological broadband afterglow."""
    time: float                 # Physical time since merger [s]
    time_days: float            # Physical time since merger [days]
    frequency: float            # Observing frequency [Hz]
    spectral_index: float       # Spectral index beta (F_nu ~ nu^-beta)
    flux_density: float         # Observed flux density F_nu [W m^-2 Hz^-1]
    luminosity_density: float   # Spectral luminosity density L_nu [W Hz^-1]
    t_peak: float               # Peak time [s]
    peak_flux: float            # Peak flux density at reference frequency [W m^-2 Hz^-1]
    phase: str                  # "pre-peak" or "post-peak"


class AfterglowModel:
    """
    Phenomenological broken-power-law broadband afterglow model.
    Reproduces slow rise (t^0.8), peak (~150 days), and steep decay (t^-2.2).
    """

    def __init__(
        self,
        config: SimConfig = None,
        t_peak_days: float = 150.0,
        rise_index: float = 0.8,
        decay_index: float = -2.2,
        beta: float = 0.585,
        nu_ref: float = 1.0e9,            # 1 GHz reference
        F_peak_ref: float = 1.0e-28,      # Ref peak flux [W m^-2 Hz^-1] at 40 Mpc
        delta_ref: float = 0.1372          # Ref Doppler factor for off-axis jet
    ):
        if config is None:
            config = SimConfig()
        self.config = config

        self.t_peak_days = float(t_peak_days)
        self.t_peak = float(t_peak_days * day)   # [s]
        self.rise_index = float(rise_index)
        self.decay_index = float(decay_index)
        self.beta = float(beta)
        self.nu_ref = float(nu_ref)
        self.F_peak_ref = float(F_peak_ref)
        self.delta_ref = float(delta_ref)
        self.distance = float(config.distance)

        # Validation
        if self.t_peak <= 0.0:
            raise ValueError(f"t_peak_days must be positive, got {t_peak_days}")
        if self.nu_ref <= 0.0:
            raise ValueError(f"Reference frequency must be positive, got {nu_ref}")
        if self.F_peak_ref < 0.0:
            raise ValueError(f"Reference peak flux must be non-negative, got {F_peak_ref}")
        if self.distance <= 0.0:
            raise ValueError(f"Distance must be positive, got {self.distance}")

    def temporal_flux(self, time_seconds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Continuous broken-power-law temporal scaling:
        F(t) / F_peak = (t / t_peak)^rise_index    if t <= t_peak
        F(t) / F_peak = (t / t_peak)^decay_index   if t > t_peak
        """
        t_arr = np.asarray(time_seconds, dtype=np.float64)
        t_clamped = np.maximum(t_arr, 1.0e-12)

        ratio = t_clamped / self.t_peak
        scaled = np.where(t_arr <= self.t_peak, ratio**self.rise_index, ratio**self.decay_index)
        scaled = np.where(t_arr <= 0.0, 0.0, scaled)

        return float(scaled) if np.isscalar(time_seconds) or t_arr.ndim == 0 else scaled

    def spectral_factor(self, frequency_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Synchrotron spectral scaling factor:
        S(nu) = (nu / nu_ref)^(-beta)
        """
        nu_arr = np.asarray(frequency_hz, dtype=np.float64)
        nu_clamped = np.maximum(nu_arr, 1.0e-12)

        ratio = nu_clamped / self.nu_ref
        spec = np.power(ratio, -self.beta)
        spec = np.where(nu_arr <= 0.0, 0.0, spec)

        return float(spec) if np.isscalar(frequency_hz) or nu_arr.ndim == 0 else spec

    def evaluate(
        self,
        time_seconds: float,
        frequency_hz: float = None,
        jet_state: JetState = None
    ) -> AfterglowState:
        """
        Evaluate single AfterglowState at given time [s] and frequency [Hz].
        """
        if frequency_hz is None:
            frequency_hz = self.nu_ref

        t_val = float(time_seconds)
        nu_val = float(frequency_hz)

        # Distance scaling: F ~ 1 / D^2 relative to 40 Mpc reference
        ref_distance = 40.0 * Mpc
        dist_factor = (ref_distance / self.distance)**2

        # Optional Jet Doppler beaming coupling
        beaming_factor = 1.0
        if jet_state is not None:
            if jet_state.doppler_factor > 0.0:
                raw_beaming = (jet_state.doppler_factor / self.delta_ref)**3
                beaming_factor = float(np.clip(raw_beaming, 1.0e-4, 1.0e4))

        # Peak flux density at reference frequency [W m^-2 Hz^-1]
        F_peak = self.F_peak_ref * dist_factor * beaming_factor

        # Temporal and spectral factors
        T_fac = self.temporal_flux(t_val)
        S_fac = self.spectral_factor(nu_val)

        flux_density = F_peak * T_fac * S_fac
        luminosity_density = flux_density * (4.0 * np.pi * (self.distance**2))

        phase = "pre-peak" if t_val <= self.t_peak else "post-peak"

        # Numerical safety checks
        if not np.isfinite(flux_density) or not np.isfinite(luminosity_density):
            raise ValueError("Calculated AfterglowState contains NaN or Inf")

        return AfterglowState(
            time=t_val,
            time_days=t_val / day,
            frequency=nu_val,
            spectral_index=self.beta,
            flux_density=flux_density,
            luminosity_density=luminosity_density,
            t_peak=self.t_peak,
            peak_flux=F_peak,
            phase=phase
        )
