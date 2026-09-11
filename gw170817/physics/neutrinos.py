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
    mean_energy_mev: float       # Mean neutrino energy <E_nu> [MeV]
    annihilation_heating: float  # Pair annihilation energy deposition rate [W]
    wind_mass_loss_rate: float   # Neutrino-driven wind mass-loss rate M_dot_nu [kg/s]
    is_active: bool              # True if neutrino emission active post-merger


class NeutrinoModel:
    """
    Reduced-order neutrino emission and heating surrogate model.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        if config is None:
            config = SimConfig()
        self.config = config

        self.L0_nu = getattr(config, "INITIAL_NEUTRINO_LUMINOSITY", 5.0e45) # ~5e52 erg/s [W]
        self.tau_nu = getattr(config, "NEUTRINO_COOLING_TIMESCALE", 0.05)    # Cooling timescale [s]

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
                mean_energy_mev=0.0,
                annihilation_heating=0.0,
                wind_mass_loss_rate=0.0,
                is_active=False
            )

        # Neutrino cooling decay post-merger
        # Rapid rise at contact (~2 ms) -> exponential thermal cooling decay
        rise = 1.0 - np.exp(-event_time / 0.002)
        decay = np.exp(-event_time / self.tau_nu)
        l_tot = self.L0_nu * rise * decay

        l_nue = l_tot * 0.35
        l_nue_bar = l_tot * 0.40

        # Mean neutrino energy <E_nu> ~ 10 - 15 MeV
        e_mean = float(14.0 * (0.5 + 0.5 * decay))

        # Neutrino-pair annihilation heating rate Q_ann ~ K_ann * L_nue * L_nue_bar ~ 10^42 W
        ann_heating = float(1.5e42 * (l_tot / self.L0_nu)**2)

        # Neutrino-driven wind mass-loss rate M_dot ~ 10^-3 - 10^-2 M_sun/s
        wind_mdot = float(0.01 * M_sun * decay)

        return NeutrinoState(
            time=event_time,
            luminosity_total=float(l_tot),
            luminosity_nue=float(l_nue),
            luminosity_nue_bar=float(l_nue_bar),
            mean_energy_mev=e_mean,
            annihilation_heating=ann_heating,
            wind_mass_loss_rate=wind_mdot,
            is_active=True
        )
