"""
Gravitational-wave waveform model and rolling buffer for GW170817.

REDUCED-ORDER APPROXIMATION:
This is a leading-order quadrupole reduced-order waveform, not a
numerical-relativity waveform.
It is generated directly from the orbital state (separation, phase, frequency)
of the binary inspiral model.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from gw170817.constants import G, c
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralState


@dataclass
class WaveformSample:
    """Instantaneous gravitational-wave sample."""
    time: float       # Physical time [s]
    f_gw: float       # GW frequency [Hz]
    phase: float      # Orbital phase phi [rad]
    amplitude: float  # Characteristic strain amplitude A [dimensionless]
    h_plus: float     # Plus polarization strain h+ [dimensionless]
    h_cross: float    # Cross polarization strain hx [dimensionless]


class GravitationalWaveModel:
    """
    Leading-order quadrupole GW strain model for compact circular binary inspiral.
    Driven directly by an InspiralState object.
    """

    def __init__(self, config: SimConfig = None):
        if config is None:
            config = SimConfig()
        self.config = config

        if self.config.distance <= 0.0:
            raise ValueError(f"Luminosity distance must be strictly positive, got {self.config.distance}")

        self.mu = (self.config.m1 * self.config.m2) / self.config.M_total
        self.distance = self.config.distance
        self.iota_rad = np.deg2rad(self.config.inclination_deg)

        # Pre-compute inclination factors
        cos_i = np.cos(self.iota_rad)
        self._plus_factor = 0.5 * (1.0 + cos_i**2)
        self._cross_factor = cos_i

        # Pre-compute constant factor: 4 * G * mu / (c^4 * D)
        self._amp_coeff = (4.0 * G * self.mu) / (c**4 * self.distance)

    def characteristic_strain(self, state: InspiralState) -> float:
        """
        Calculate characteristic strain amplitude A:
        A = (4 * G * mu / (c^4 * D)) * (Omega * a)^2
        """
        if state.separation <= 0.0:
            raise ValueError(f"Separation must be positive, got {state.separation}")
        if state.f_gw < 0.0 or np.isnan(state.f_gw) or np.isinf(state.f_gw):
            raise ValueError(f"Invalid GW frequency: {state.f_gw}")

        omega_a = state.omega_orb * state.separation
        amp = self._amp_coeff * (omega_a**2)

        if np.isnan(amp) or np.isinf(amp):
            raise ValueError("Calculated characteristic strain is NaN or Inf")

        return amp

    def strain(self, state: InspiralState) -> Tuple[float, float]:
        """
        Calculate instantaneous h_plus and h_cross.
        h_+ = A * ((1 + cos(i)^2)/2) * cos(2 * phi)
        h_x = A * cos(i) * sin(2 * phi)
        Includes exponential post-merger ringdown decay to zero to prevent flat tails.
        """
        amp = self.characteristic_strain(state)
        phi = state.orbital_phase
        two_phi = 2.0 * phi

        # Post-merger ringdown suppression if f_gw >= f_max or time >= 0
        ringdown = 1.0
        f_max = getattr(self.config, "f_max", 1500.0)
        if state.f_gw >= f_max or state.time >= 0.0:
            t_post = max(0.0, state.time)
            ringdown = np.exp(-t_post / 0.001)

        h_plus = amp * self._plus_factor * np.cos(two_phi) * ringdown
        h_cross = amp * self._cross_factor * np.sin(two_phi) * ringdown

        return float(h_plus), float(h_cross)

    def sample(self, state: InspiralState) -> WaveformSample:
        """Construct a complete WaveformSample from InspiralState."""
        amp = self.characteristic_strain(state)
        h_plus, h_cross = self.strain(state)

        return WaveformSample(
            time=state.time,
            f_gw=state.f_gw,
            phase=state.orbital_phase,
            amplitude=amp,
            h_plus=h_plus,
            h_cross=h_cross
        )


class WaveformBuffer:
    """
    Fixed-capacity circular ring buffer for storing GW waveform samples.
    Preallocates NumPy arrays to avoid per-sample allocations and unbounded memory growth.
    """

    def __init__(self, capacity: int = 1024):
        if capacity <= 0:
            raise ValueError(f"Buffer capacity must be > 0, got {capacity}")

        self.capacity = capacity
        self._time = np.zeros(capacity, dtype=np.float64)
        self._h_plus = np.zeros(capacity, dtype=np.float64)
        self._h_cross = np.zeros(capacity, dtype=np.float64)
        self._f_gw = np.zeros(capacity, dtype=np.float64)

        self._head = 0       # Write position
        self._size = 0       # Current number of valid samples stored
        self._total_appended = 0

    def append(self, sample: WaveformSample):
        """Append a WaveformSample into the circular buffer."""
        idx = self._head
        self._time[idx] = sample.time
        self._h_plus[idx] = sample.h_plus
        self._h_cross[idx] = sample.h_cross
        self._f_gw[idx] = sample.f_gw

        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        self._total_appended += 1

    def get_chronological(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve chronological arrays (time, h_plus, h_cross, f_gw) for display.
        Returns slices/copies ordered from oldest to newest sample.
        """
        if self._size == 0:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty, empty

        if self._size < self.capacity:
            # Buffer not yet wrapped
            return (
                self._time[:self._size].copy(),
                self._h_plus[:self._size].copy(),
                self._h_cross[:self._size].copy(),
                self._f_gw[:self._size].copy(),
            )
        else:
            # Buffer wrapped: head points to oldest sample
            idx_start = self._head
            order = np.roll(np.arange(self.capacity), -idx_start)
            return (
                self._time[order].copy(),
                self._h_plus[order].copy(),
                self._h_cross[order].copy(),
                self._f_gw[order].copy(),
            )

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def count(self) -> int:
        return self._size
