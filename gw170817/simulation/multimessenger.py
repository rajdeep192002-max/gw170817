"""
GW170817 Multi-Messenger Event Coordinator & Integrated Event State.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
This module orchestrates the project's physics-informed reduced-order modules
(gravitational waves, merger dynamics, ejecta, two-component kilonova, structured relativistic jet,
and broadband afterglow) into a unified multi-messenger event timeline and observational validation layer.

It does NOT perform full numerical relativity, GRHD, MHD, or neutrino radiation transport.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from gw170817.constants import c, day, Mpc, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation, SimulationState
from gw170817.physics.ejecta import EjectaState
from gw170817.validation.report import ValidationReport


@dataclass
class MultiMessengerEventState:
    """Comprehensive instantaneous state of the GW170817 multi-messenger event."""
    event_time: float              # Time relative to merger [s] (0.0 = merger, <0 = inspiral)
    elapsed_time: float            # Engine cumulative simulation time [s]
    step_count: int                # Physics step count
    phase: str                     # Evolution phase ("INSPIRAL", "MERGER", "POST_MERGER", "GRB", "AFTERGLOW")

    # Gravitational-Wave Channel
    gw_frequency: float            # Instantaneous GW frequency [Hz]
    separation: float              # Orbital separation a [m]
    h_plus: float                  # Plus polarization strain h+
    h_cross: float                 # Cross polarization strain hx

    # Merger & Hydro State
    contact_fraction: float        # Contact fraction [0.0 to 1.0]
    merger_started: bool           # True if contact/merger regime entered
    merger_complete: bool          # True if compact remnant formed

    # Ejecta & Kilonova Channel
    ejecta_mass: float             # Dynamically unbound ejecta mass [kg]
    ejecta_fraction: float         # Ejecta mass fraction of total binary mass
    kilonova_luminosity: float     # Kilonova bolometric luminosity [W]
    kilonova_t_blue: float         # Blue ejecta effective temperature [K]
    kilonova_t_red: float          # Red ejecta effective temperature [K]

    # Relativistic Jet & Prompt GRB Channel
    grb_launched: bool             # True if jet launched post-merger
    grb_triggered: bool            # True if 1.7 s prompt delay elapsed
    grb_delay: float               # GW-GRB prompt delay [s] (1.74 s)
    jet_viewing_angle_deg: float   # Observer viewing angle [deg] (22.0 deg)

    # Broadband Afterglow Channel
    afterglow_flux: float          # Observed afterglow flux density F_nu [W m^-2 Hz^-1] (1 GHz)
    afterglow_peak_time_days: float# Reference peak time [days] (150.0 d)

    # Observational Validation Channel
    validation_passed: bool        # True if all validation checks pass
    validation_passed_count: int   # Number of passing checks
    validation_total_count: int    # Total validation checks count


class MultiMessengerCoordinator:
    """
    Central Integration Coordinator for GW170817 Multi-Messenger Event.
    Orchestrates engine stepping, messenger evaluation, and observational validation.
    """

    def __init__(
        self,
        engine: Optional[GW170817Simulation] = None,
        config: Optional[SimConfig] = None
    ):
        if config is None:
            if engine is not None:
                config = engine.config
            else:
                config = SimConfig()
        self.config = config

        if engine is None:
            engine = GW170817Simulation(config=self.config)
        self.engine = engine

        self._cached_validation_report: Optional[ValidationReport] = None
        self._update_event_state()

    def reset(self):
        """Reset engine and coordinator state."""
        self.engine.reset()
        self._cached_validation_report = None
        return self._update_event_state()

    def validation_report(self, force_reevaluate: bool = False) -> ValidationReport:
        """Request observational ValidationReport comparing current model against GW170817 data."""
        if self._cached_validation_report is None or force_reevaluate:
            self._cached_validation_report = ValidationReport(engine=self.engine, config=self.config)
        return self._cached_validation_report

    def _get_ejecta_state(self, t_val: float) -> EjectaState:
        """Retrieve particle ejecta state or fallback to phenomenological config ejecta state post-merger."""
        ej_state = self.engine.ejecta.compute_state(self.engine.psys, max(0.0, t_val))
        if ej_state.ejecta_mass <= 0.0 and t_val > 0.0:
            m_blue = getattr(self.config, "M_EJ_BLUE", 0.02 * M_sun)
            m_red = getattr(self.config, "M_EJ_RED", 0.04 * M_sun)
            m_tot = m_blue + m_red
            ej_state = EjectaState(
                time=t_val,
                ejecta_mass=m_tot,
                ejecta_fraction=m_tot / self.config.M_total,
                ejecta_particle_count=0,
                mean_velocity=0.2 * c,
                max_velocity=0.4 * c,
                kinetic_energy=0.5 * m_tot * (0.2 * c)**2,
                angular_momentum_proxy=0.0,
                mean_Ye=0.25,
                lanthanide_rich_fraction=m_red / m_tot if m_tot > 0 else 0.5
            )
        return ej_state

    def _update_event_state(self) -> MultiMessengerEventState:
        """Construct full MultiMessengerEventState from engine and physics modules."""
        st = self.engine.current_state
        val_rep = self.validation_report()

        # Strain waveform sample
        h_plus = 0.0
        h_cross = 0.0
        if self.engine.waveform_buffer.count > 0:
            h_plus = float(self.engine.waveform_buffer._h_plus[self.engine.waveform_buffer._head - 1])
            h_cross = float(self.engine.waveform_buffer._h_cross[self.engine.waveform_buffer._head - 1])

        # Kilonova thermal evaluation at current event time
        ej_state = self._get_ejecta_state(st.event_time)
        kn_state = self.engine.kilonova.evaluate(ej_state, max(0.0, st.event_time))

        viewing_angle_deg = float(np.rad2deg(self.engine.jet.viewing_angle))

        self._event_state = MultiMessengerEventState(
            event_time=st.event_time,
            elapsed_time=st.elapsed_time,
            step_count=st.step_count,
            phase=st.phase,
            gw_frequency=st.gw_frequency,
            separation=st.separation,
            h_plus=h_plus,
            h_cross=h_cross,
            contact_fraction=st.merger_contact_fraction,
            merger_started=st.merger_started,
            merger_complete=st.merger_complete,
            ejecta_mass=ej_state.ejecta_mass,
            ejecta_fraction=ej_state.ejecta_fraction,
            kilonova_luminosity=kn_state.L_total,
            kilonova_t_blue=kn_state.T_blue,
            kilonova_t_red=kn_state.T_red,
            grb_launched=st.grb_launched,
            grb_triggered=st.grb_triggered,
            grb_delay=self.engine.jet.jet_delay,
            jet_viewing_angle_deg=viewing_angle_deg,
            afterglow_flux=st.afterglow_flux,
            afterglow_peak_time_days=self.engine.afterglow.t_peak_days,
            validation_passed=val_rep.all_passed,
            validation_passed_count=sum(1 for r in val_rep.results if r.passed),
            validation_total_count=len(val_rep.results)
        )
        return self._event_state

    def step(self, dt: float = None) -> MultiMessengerEventState:
        """Advance simulation clock and update multi-messenger event state."""
        self.engine.step(dt)
        return self._update_event_state()

    def jump_to_demo_phase(self, f_gw: float = 1200.0, separation: float = 30.0e3) -> MultiMessengerEventState:
        """Perform accelerated demo jump to near-merger state."""
        self.engine.jump_to_demo_phase(f_gw=f_gw, separation=separation)
        return self._update_event_state()

    def evaluate_at_event_time(self, t_seconds: float) -> MultiMessengerEventState:
        """
        Evaluate messenger channels (Timeline phase, Jet, Afterglow, Kilonova)
        at a specific event time t_seconds relative to merger without particle stepping.
        """
        t_val = float(t_seconds)
        merg_state = self.engine.dynamics.merger_state
        phase = self.engine.timeline.current_phase(t_val)

        ej_state = self._get_ejecta_state(t_val)
        kn_state = self.engine.kilonova.evaluate(ej_state, max(0.0, t_val))
        j_state = self.engine.jet.evaluate(time=t_val, merger_state=merg_state)
        ag_state = self.engine.afterglow.evaluate(time_seconds=max(0.0, t_val), jet_state=j_state)
        val_rep = self.validation_report()

        return MultiMessengerEventState(
            event_time=t_val,
            elapsed_time=self.engine.elapsed_time,
            step_count=self.engine.step_count,
            phase=phase,
            gw_frequency=self.engine.dynamics.inspiral_state.f_gw,
            separation=self.engine.dynamics.inspiral_state.separation,
            h_plus=self.engine.current_state.h_plus if hasattr(self.engine.current_state, "h_plus") else 0.0,
            h_cross=0.0,
            contact_fraction=merg_state.contact_fraction,
            merger_started=merg_state.merger_started,
            merger_complete=merg_state.merger_complete,
            ejecta_mass=ej_state.ejecta_mass,
            ejecta_fraction=ej_state.ejecta_fraction,
            kilonova_luminosity=kn_state.L_total,
            kilonova_t_blue=kn_state.T_blue,
            kilonova_t_red=kn_state.T_red,
            grb_launched=j_state.launched,
            grb_triggered=j_state.grb_triggered,
            grb_delay=self.engine.jet.jet_delay,
            jet_viewing_angle_deg=float(np.rad2deg(self.engine.jet.viewing_angle)),
            afterglow_flux=ag_state.flux_density,
            afterglow_peak_time_days=self.engine.afterglow.t_peak_days,
            validation_passed=val_rep.all_passed,
            validation_passed_count=sum(1 for r in val_rep.results if r.passed),
            validation_total_count=len(val_rep.results)
        )

    @property
    def current_state(self) -> MultiMessengerEventState:
        """Return current cached MultiMessengerEventState."""
        return self._event_state
