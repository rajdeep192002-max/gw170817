"""
Magnetic Field Winding, MRI & Dynamo Physics Model for BNS Remnants.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Models differential rotation magnetic field winding (dB_phi/dt ~ B_p * shear),
magnetic stress feedback, Magnetorotational Instability (MRI) surrogate diagnostic
(gamma_MRI ~ 3/4 Omega), alpha-Omega dynamo amplification, and polar magnetosphere Poynting flux output.

References:
  Hayashi et al., Phys. Rev. Lett. 134, 211407 (2025)
  Balbus & Hawley, Astrophys. J. 376, 214 (1991)
  Blandford & Znajek, Mon. Not. R. Astron. Soc. 179, 433 (1977)
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.physics.remnant import RemnantState
from gw170817.physics.disk import DiskState
from gw170817.physics.rotation import RotationalState


@dataclass
class MagneticFieldState:
    """Instantaneous state of the magnetic field, MRI surrogate, and magnetosphere."""
    time: float                  # Event time relative to merger [s]
    b_poloidal: float            # Poloidal magnetic field magnitude B_p [Gauss]
    b_toroidal: float            # Toroidal magnetic field magnitude B_phi [Gauss]
    b_total: float               # Total magnetic field magnitude B [Gauss]
    b_ratio: float               # Winding ratio B_phi / B_p
    magnetic_energy_proxy: float # Magnetic energy proxy E_B [J]
    magnetic_stress: float       # Maxwell magnetic stress T_rphi [Pa]
    mri_active: bool             # True if Magnetorotational Instability active
    mri_growth_rate: float       # MRI growth rate gamma_MRI ~ 3/4 Omega [s^-1]
    mri_amplification: float     # Amplification factor B / B0
    mri_saturation: float        # Saturation fraction B / B_max
    dynamo_active: bool          # True if alpha-Omega dynamo active
    poynting_luminosity: float   # Polar Poynting flux luminosity L_poynting [W]
    collimation_angle_deg: float # Magnetic jet collimation half-opening angle [deg]


class MagneticFieldModel:
    """
    Reduced-order magnetic field amplification, winding, and Poynting flux model.
    Coupled directly to rotational shear and disk rotation.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        if config is None:
            config = SimConfig()
        self.config = config

        self.B0_pol = getattr(config, "B_POLOIDAL_INIT", 1.0e12)       # Initial NS B-field [G]
        self.B_max = getattr(config, "B_SATURATION_MAX", 5.0e15)       # Saturation max B-field [G]
        self.t_mri_growth = getattr(config, "MRI_GROWTH_TIME", 0.005)  # MRI growth timescale [s]

    def evaluate(
        self,
        remnant_state: RemnantState,
        disk_state: DiskState,
        event_time: float,
        rotational_state: Optional[RotationalState] = None
    ) -> MagneticFieldState:
        """
        Evaluate magnetic field state at event time t relative to merger [s].
        """
        if event_time < 0.0 or not disk_state.is_active:
            return MagneticFieldState(
                time=event_time,
                b_poloidal=self.B0_pol,
                b_toroidal=0.0,
                b_total=self.B0_pol,
                b_ratio=0.0,
                magnetic_energy_proxy=0.0,
                magnetic_stress=0.0,
                mri_active=False,
                mri_growth_rate=0.0,
                mri_amplification=1.0,
                mri_saturation=self.B0_pol / self.B_max,
                dynamo_active=False,
                poynting_luminosity=0.0,
                collimation_angle_deg=25.0
            )

        # 1. Shear-driven Winding dB_phi/dt ~ B_p * shear
        shear = 0.0
        if rotational_state is not None:
            shear = rotational_state.shear
        else:
            shear = 500.0 * np.exp(-event_time / 0.02)

        # Poloidal field grows via alpha-Omega dynamo saturation:
        growth = 1.0 + 500.0 * (1.0 - np.exp(-event_time / self.t_mri_growth))
        b_pol = float(min(self.B_max * 0.2, self.B0_pol * growth))

        # Toroidal field wound up by differential rotation shear:
        # B_phi = B_p * integral(shear dt)
        winding_factor = min(20.0, shear * event_time * 0.1 + event_time / 0.002)
        b_tor = float(min(self.B_max, b_pol * winding_factor * (1.0 + 0.5 * remnant_state.spin_parameter)))

        b_tot = float(np.sqrt(b_pol**2 + b_tor**2))
        b_ratio = float(b_tor / max(1.0, b_pol))

        # 2. MRI-inspired surrogate diagnostic [MOD]
        # gamma_MRI ~ 3/4 Omega_disk
        omega_ref = disk_state.omega_disk if disk_state.omega_disk > 0 else remnant_state.omega_rot
        mri_gamma = 0.75 * omega_ref
        mri_active = bool(event_time > 0.001 and b_tot < self.B_max)
        amplification = float(b_tot / self.B0_pol)
        saturation = float(min(1.0, b_tot / self.B_max))

        # 3. Maxwell Magnetic Stress T_rphi ~ B_p B_phi / (4 pi) (in SI Pa: (B_G * 1e-4)^2 / mu_0)
        mu_0 = 4.0 * np.pi * 1.0e-7
        b_p_si = b_pol * 1.0e-4
        b_t_si = b_tor * 1.0e-4
        b_tot_si = b_tot * 1.0e-4
        stress = (b_p_si * b_t_si) / mu_0

        # Magnetic Energy Proxy E_B ~ (B^2 / (2 mu_0)) * Volume
        vol_rem = (4.0 / 3.0) * np.pi * (remnant_state.radius**3)
        e_mag = (b_tot_si**2 / (2.0 * mu_0)) * vol_rem

        # 4. Poynting-flux outflow calculation (Blandford-Znajek / HMNS force-free magnetosphere)
        r_rem = remnant_state.radius
        omega = remnant_state.omega_rot

        poynting_lum = (1.0 / (6.0 * c)) * (b_p_si**2) * (r_rem**4) * (omega**2) * (max(0.1, remnant_state.spin_parameter)**2)
        poynting_lum = float(min(1.0e46, max(0.0, poynting_lum)))

        # 5. Dynamic jet collimation angle (winds up and narrows over ~0.05-0.2 s)
        coll_angle = 25.0 - 21.5 * (1.0 - np.exp(-event_time / 0.08))
        coll_angle = float(max(3.5, min(25.0, coll_angle)))

        return MagneticFieldState(
            time=event_time,
            b_poloidal=b_pol,
            b_toroidal=b_tor,
            b_total=b_tot,
            b_ratio=b_ratio,
            magnetic_energy_proxy=float(e_mag),
            magnetic_stress=float(stress),
            mri_active=mri_active,
            mri_growth_rate=float(mri_gamma),
            mri_amplification=amplification,
            mri_saturation=saturation,
            dynamo_active=True,
            poynting_luminosity=poynting_lum,
            collimation_angle_deg=coll_angle
        )
