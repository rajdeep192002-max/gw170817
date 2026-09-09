"""
Simulation configuration for GW170817 reduced-order model.
Contains system parameters, timesteps, particle counts, quality levels, and backend settings.
"""
from dataclasses import dataclass, field
import numpy as np
from gw170817.constants import G, c, M_sun, Mpc


@dataclass
class SimConfig:
    # --- Binary Parameters (GW170817) ---
    m1_solar: float = 1.36               # Primary NS mass [M_sun]
    m2_solar: float = 1.36               # Secondary NS mass [M_sun]
    distance_Mpc: float = 40.0           # Luminosity distance [Mpc]
    
    # Initial Frequencies
    f_gw_start: float = 40.0             # Initial GW frequency [Hz]
    
    # Observer & Inclination
    inclination_deg: float = 20.0        # Inclination angle [deg]
    
    # --- Timesteps [s] ---
    dt_physics: float = 1.0e-4           # Physics simulation timestep [s]
    dt_render: float = 1.0 / 60.0        # Render frame interval [s] (60 FPS)
    dt_diagnostics: float = 0.01         # Diagnostics update interval [s]
    
    # --- Particle Resolution Modes ---
    DEV_PARTICLES: int = 20_000
    NORMAL_PARTICLES: int = 75_000
    HIGH_PARTICLES: int = 150_000
    
    # Active mode & quality
    mode: str = "DEV"                    # Options: "DEV", "NORMAL", "HIGH"
    quality_level: str = "MEDIUM"        # Options: "LOW", "MEDIUM", "HIGH"
    
    # --- Taichi Backend ---
    backend_preference: str = "vulkan"   # Preferred backend ("vulkan" or "cpu")
    
    # --- Random Seed ---
    seed: int = 42                       # Deterministic random seed

    # --- Derived Properties ---
    @property
    def m1(self) -> float:
        """Primary NS mass in kg."""
        return self.m1_solar * M_sun

    @property
    def m2(self) -> float:
        """Secondary NS mass in kg."""
        return self.m2_solar * M_sun

    @property
    def M_total(self) -> float:
        """Total mass in kg."""
        return self.m1 + self.m2

    @property
    def distance(self) -> float:
        """Luminosity distance in metres."""
        return self.distance_Mpc * Mpc

    @property
    def chirp_mass(self) -> float:
        """Chirp mass M_c in kg."""
        return (self.m1 * self.m2)**0.6 / (self.M_total)**0.2

    @property
    def particle_count(self) -> int:
        """Get particle count corresponding to current mode."""
        mapping = {
            "DEV": self.DEV_PARTICLES,
            "NORMAL": self.NORMAL_PARTICLES,
            "HIGH": self.HIGH_PARTICLES,
        }
        return mapping.get(self.mode.upper(), self.DEV_PARTICLES)

    @property
    def n_particles(self) -> int:
        """Alias for particle_count."""
        return self.particle_count

    @property
    def initial_separation(self) -> float:
        """
        Initial orbital separation a0 [m] calculated from Keplerian orbit at f_gw_start.
        f_gw = 2 * f_orb = (1 / pi) * sqrt(G * M_total / a^3)
        => a0 = (G * M_total / (pi * f_gw_start)^2)^(1/3)
        """
        return (G * self.M_total / (np.pi * self.f_gw_start)**2)**(1.0 / 3.0)
