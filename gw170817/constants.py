"""
Physical constants in SI units for GW170817 reduced-order simulation.
Single source of truth — import from here only.
Values from NIST CODATA 2018 / IAU standard.
"""

# Fundamental physical constants
G: float = 6.67430e-11        # Gravitational constant [m^3 kg^-1 s^-2]
c: float = 2.99792458e8       # Speed of light [m/s]
hbar: float = 1.05457182e-34  # Reduced Planck constant [J s]
k_B: float = 1.38064852e-23   # Boltzmann constant [J/K]
sigma_SB: float = 5.67037442e-8 # Stefan-Boltzmann constant [W m^-2 K^-4]

# Astronomical & Mass constants
M_sun: float = 1.98892e30     # Solar mass [kg]
m_p: float = 1.67262192e-27   # Proton mass [kg]
m_e: float = 9.10938371e-31   # Electron mass [kg]

# Distances
pc: float = 3.08567758e16     # Parsec [m]
kpc: float = 3.08567758e19    # Kiloparsec [m]
Mpc: float = 3.08567758e22    # Megaparsec [m]
ly: float = 9.46073047e15     # Light-year [m]

# Time
yr: float = 3.15576e7         # Julian year [s]
day: float = 86400.0          # Day [s]

# Derived constants
G_over_c2: float = G / (c**2) # [m/kg]
G_over_c4: float = G / (c**4) # [s^2 m^-1 kg^-1]
