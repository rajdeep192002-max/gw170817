"""
Neutrino Emission & Pair Annihilation Physics Model for BNS Mergers.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Models post-merger neutrino cooling luminosity, mean neutrino energies,
neutrino-pair annihilation energy deposition (nu - nubar -> e- - e+),
and neutrino-driven disk wind mass loss.

References:
  Hayashi et al., Phys. Rev. Lett. 134, 211407 (2025)
  Sekiguchi et al., Phys. Rev. D 93, 044046 (2016)
  Perego et al., Mon. Not. R. Astron. Soc. 443, 3134 (2014)
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.physics.remnant import RemnantState
from gw170817.physics.disk import DiskState


@dataclass
class NeutrinoState:
    """Instantaneous state of the neutrino emission channel."""
    time: float                  # Event time relative to merger [s]
    luminosity_total: float      # Total neutrino luminosity L_nu [W]
    luminosity_nue: float        # Electron neutrino luminosity L_nue [W]
    luminosity_nue_bar: float    # Electron antineutrino luminosity L_nue_bar [W]
    luminosity_nux: float = 0.0  # Heavy-flavor neutrino luminosity L_nux (mu, tau) [W]
    mean_energy_mev: float = 0.0 # Mean neutrino energy <E_nu> [MeV]
    annihilation_heating: float = 0.0 # Pair annihilation energy deposition rate [W]
    wind_mass_loss_rate: float = 0.0  # Neutrino-driven wind mass-loss rate M_dot_nu [kg/s]
    is_active: bool = False      # True if neutrino emission active post-merger
    transport_active: bool = False # True if neutrino transport field is propagating


class NeutrinoModel:
    """
    Reduced-order neutrino emission and heating surrogate model.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        if config is None:
            config = SimConfig()
        self.config = config

        self.L0_nu = getattr(config, "INITIAL_NEUTRINO_LUMINOSITY", 5.0e45) # ~5e52 erg/s [W]
        self.tau_prompt = getattr(config, "NEUTRINO_PROMPT_TIMESCALE", 0.08) # Prompt cooling timescale [s]
        self.tau_disk = getattr(config, "NEUTRINO_DISK_TIMESCALE", 1.5)      # Disk accretion cooling timescale [s]
        self.tau_nu = self.tau_prompt  # Backwards-compatible alias [s]

    def evaluate(
        self,
        remnant_state: RemnantState,
        disk_state: DiskState,
        event_time: float
    ) -> NeutrinoState:
        """
        Evaluate neutrino emission state at event time t relative to merger [s].
        """
        if event_time < 0.0 or not disk_state.is_active:
            return NeutrinoState(
                time=event_time,
                luminosity_total=0.0,
                luminosity_nue=0.0,
                luminosity_nue_bar=0.0,
                luminosity_nux=0.0,
                mean_energy_mev=0.0,
                annihilation_heating=0.0,
                wind_mass_loss_rate=0.0,
                is_active=False,
                transport_active=False
            )

        # Dual-timescale neutrino emission post-merger:
        # 1. Prompt thermal peak from shocked remnant (tau_prompt ~ 0.08 s, ~85% peak)
        # 2. Viscous accretion disk neutrino cooling tail (tau_disk ~ 1.5 s, ~15% peak)
        rise = 1.0 - np.exp(-event_time / 0.002)
        prompt_comp = 0.85 * np.exp(-event_time / self.tau_prompt)
        disk_mass_factor = min(2.0, max(0.2, disk_state.disk_mass_msun / 0.06)) if disk_state.disk_mass_msun > 0.0 else 0.5
        disk_comp = 0.15 * disk_mass_factor * ((1.0 + event_time / self.tau_disk) ** (-1.333))
        l_tot = self.L0_nu * rise * (prompt_comp + disk_comp)

        # Species fractions based on modern BNS simulations (e.g. Perego+2014, Sekiguchi+2016):
        # nu_e: ~35%, anti-nu_e: ~40%, nu_x (sum of nu_mu, nu_tau, anti-nu_mu, anti-nu_tau): ~25%
        l_nue = l_tot * 0.35
        l_nue_bar = l_tot * 0.40
        l_nux = l_tot * 0.25

        # Mean neutrino energy <E_nu> ~ 11 - 15 MeV (hotter at prompt merger, cooling in disk wind phase)
        e_mean = float(11.0 + 4.0 * np.exp(-event_time / 0.25))

        # Neutrino-pair annihilation heating rate Q_ann ~ K_ann * L_nue * L_nue_bar ~ 10^42 W
        ann_heating = float(1.5e42 * (l_tot / self.L0_nu)**2)

        # Neutrino-driven wind mass-loss rate M_dot ~ 10^-3 - 10^-2 M_sun/s
        wind_mdot = float(0.01 * M_sun * (prompt_comp + disk_comp))

        # Strict activity flags synchronized with emission threshold
        is_act = bool(event_time >= 0.0 and l_tot > 1.0e38)
        transport_act = bool(event_time >= 0.0 and (l_tot > 1.0e38 or event_time < 5.0))

        return NeutrinoState(
            time=event_time,
            luminosity_total=float(l_tot),
            luminosity_nue=float(l_nue),
            luminosity_nue_bar=float(l_nue_bar),
            luminosity_nux=float(l_nux),
            mean_energy_mev=e_mean,
            annihilation_heating=ann_heating,
            wind_mass_loss_rate=wind_mdot,
            is_active=is_act,
            transport_active=transport_act
        )
