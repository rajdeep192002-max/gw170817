"""
GW170817 Multi-Messenger Event Coordinator & Integrated Event State.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Orchestrates gravitational waves, rotational dynamics, magnetic winding, disk evolution,
neutrinos, ejecta, kilonova, relativistic jet, and afterglow.
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
    phase: str                     # Authoritative primary NSM phase ("INSPIRAL", "LATE_INSPIRAL", "MERGER", "RINGDOWN")

    # Gravitational-Wave Channel
    gw_frequency: float            # Instantaneous GW frequency [Hz]
    separation: float              # Orbital separation a [m]
    h_plus: float                  # Plus polarization strain h+
    h_cross: float                 # Cross polarization strain hx

    # Merger & Hydro State
    contact_fraction: float        # Contact fraction [0.0 to 1.0]
    merger_started: bool           # True if contact/merger regime entered
    merger_complete: bool          # True if compact remnant formed

    # Rotational Dynamics Channel [MOD]
    omega_core: float              # Central core angular velocity [rad/s]
    omega_outer: float             # Outer envelope angular velocity [rad/s]
    differential_rotation: float   # Delta Omega [rad/s]
    angular_momentum: float        # Total system angular momentum [kg m^2 / s]
    t_over_w: float                # T / |W| kinetic-to-potential ratio
    shear: float                   # Shearing rate [s^-1]
    gravitational_redshift: float  # Redshift z = 1 / sqrt(1 - u) - 1

    # Ejecta & Kilonova Channel
    ejecta_mass: float             # Dynamically unbound ejecta mass [kg]
    ejecta_fraction: float         # Ejecta mass fraction of total binary mass
    ejecta_mean_ye: float          # Mass-weighted electron fraction from EjectaState
    ejecta_radioactive_heating_rate: float  # Specific r-process heating [W kg^-1]
    ejecta_opacity_mean: float     # Composition-weighted opacity proxy [m^2 kg^-1]
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

    # Remnant, Disk, Magnetic & Neutrino Subsystems
    remnant_scenario: str          # Remnant scenario ("GW170817_LIKE", "PROMPT_BH_REFERENCE", "HMNS_REFERENCE")
    remnant_type: str              # Physical remnant state ("INSPIRAL", "HMNS", "BH")
    disk_mass_msun: float          # Accretion disk mass [M_sun]
    b_poloidal: float              # Poloidal magnetic field magnitude [Gauss]
    b_toroidal: float              # Toroidal magnetic field magnitude [Gauss]
    b_ratio: float                 # Magnetic winding ratio B_phi / B_p
    mri_active: bool               # True if Magnetorotational Instability active
    poynting_luminosity: float     # Poynting flux output [W]
    neutrino_luminosity: float     # Neutrino emission luminosity [W]

    # Observational Validation Channel
    validation_passed: bool        # True if all validation checks pass
    validation_passed_count: int   # Number of passing checks
    validation_total_count: int    # Total validation checks count

    @property
    def is_black_hole(self) -> bool:
        """Return True if remnant has collapsed to a black hole."""
        return self.remnant_type == "BH"

    @property
    def gw_active(self) -> bool:
        """True if gravitational wave emission channel is active."""
        return True

    @property
    def gw_propagation_active(self) -> bool:
        """True if outward 3D GW metric propagation wavefront is active (t >= 0)."""
        return bool(self.event_time >= 0.0 or self.merger_started)

    @property
    def ejecta_active(self) -> bool:
        """True if dynamically unbound ejecta is active (t >= 0 or contact > 0.05)."""
        return bool(self.event_time >= 0.0 or self.contact_fraction > 0.05 or self.ejecta_mass > 0.0)

    @property
    def bh_active(self) -> bool:
        """True if compact remnant has collapsed to a Black Hole."""
        return bool(self.remnant_type == "BH")

    @property
    def disk_active(self) -> bool:
        """True if post-merger accretion disk is active."""
        return bool(self.event_time >= 0.0 and (self.contact_fraction > 0.05 or self.disk_mass_msun > 0.0))

    @property
    def magnetic_active(self) -> bool:
        """True if magnetic winding / MRI instability is active."""
        return bool(self.event_time >= 0.0 and (self.contact_fraction > 0.05 or self.b_ratio > 1.0))

    @property
    def jet_active(self) -> bool:
        """True if prompt GRB relativistic jet is triggered (+1.74s delay)."""
        return bool(self.grb_triggered)

    @property
    def kilonova_active(self) -> bool:
        """True if r-process kilonova emission is actively evolving."""
        return bool(self.event_time > 0.0 and self.kilonova_luminosity > 0.0)

    @property
    def afterglow_active(self) -> bool:
        """True if broadband afterglow synchrotron flux is active."""
        return bool(self.event_time > 0.0 and self.afterglow_flux > 0.0)

    @property
    def lensing_active(self) -> bool:
        """True if gravitational lensing model is active (BNS dual-lens or BH Schwarzschild)."""
        return True

    @property
    def orchestration(self) -> "MultiMessengerOrchestrationState":
        """Return unified multi-messenger orchestration state snapshot."""
        return MultiMessengerOrchestrationState(
            gw_active=self.gw_active,
            gw_propagation_active=self.gw_propagation_active,
            ejecta_active=self.ejecta_active,
            bh_active=self.bh_active,
            disk_active=self.disk_active,
            magnetic_active=self.magnetic_active,
            jet_active=self.jet_active,
            kilonova_active=self.kilonova_active,
            afterglow_active=self.afterglow_active,
            lensing_active=self.lensing_active
        )


@dataclass
class MultiMessengerOrchestrationState:
    """Unified snapshot of active multi-messenger channels derived from physics state."""
    gw_active: bool
    gw_propagation_active: bool
    ejecta_active: bool
    bh_active: bool
    disk_active: bool
    magnetic_active: bool
    jet_active: bool
    kilonova_active: bool
    afterglow_active: bool
    lensing_active: bool




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
            m_rich_frac = m_red / m_tot if m_tot > 0 else 0.5
            m_poor_frac = m_blue / m_tot if m_tot > 0 else 0.5
            eps_dot = 2.0e6 * ((max(0.0, t_val) + 10.0) / 86400.0) ** (-1.3)
            kappa = 0.1 * m_poor_frac + 1.0 * m_rich_frac
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
                lanthanide_rich_fraction=m_rich_frac,
                lanthanide_poor_fraction=m_poor_frac,
                dynamical_ejecta_mass=m_tot,
                disk_wind_ejecta_mass=0.0,
                radioactive_heating_rate=eps_dot,
                opacity_mean=kappa,
                temperature_proxy=0.0,
                density_proxy=0.0,
                r_process_active=True
            )
        return ej_state

    def _update_event_state(self) -> MultiMessengerEventState:
        """Construct full MultiMessengerEventState from engine and physics modules."""
        st = self.engine.current_state
        val_rep = self.validation_report()

        h_plus = 0.0
        h_cross = 0.0
        if self.engine.waveform_buffer.count > 0:
            h_plus = float(self.engine.waveform_buffer._h_plus[self.engine.waveform_buffer._head - 1])
            h_cross = float(self.engine.waveform_buffer._h_cross[self.engine.waveform_buffer._head - 1])

        insp_st = self.engine.dynamics.inspiral_state
        rem_st = self.engine.remnant.evaluate(insp_st, st.event_time)
        rot_st = self.engine.rotation.evaluate(insp_st, rem_st, st.event_time)
        disk_st = self.engine.disk.evaluate(rem_st, st.event_time, rot_st)
        mag_st = self.engine.magnetic_field.evaluate(rem_st, disk_st, st.event_time, rot_st)
        nu_st = self.engine.neutrinos.evaluate(rem_st, disk_st, st.event_time)

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
            omega_core=rot_st.omega_core,
            omega_outer=rot_st.omega_outer,
            differential_rotation=rot_st.differential_rotation,
            angular_momentum=rot_st.angular_momentum,
            t_over_w=rot_st.T_over_W,
            shear=rot_st.shear,
            gravitational_redshift=rot_st.gravitational_redshift,
            ejecta_mass=ej_state.ejecta_mass,
            ejecta_fraction=ej_state.ejecta_fraction,
            ejecta_mean_ye=ej_state.mean_Ye,
            ejecta_radioactive_heating_rate=ej_state.radioactive_heating_rate,
            ejecta_opacity_mean=ej_state.opacity_mean,
            kilonova_luminosity=kn_state.L_total,
            kilonova_t_blue=kn_state.T_blue,
            kilonova_t_red=kn_state.T_red,
            grb_launched=st.grb_launched,
            grb_triggered=st.grb_triggered,
            grb_delay=self.engine.jet.jet_delay,
            jet_viewing_angle_deg=viewing_angle_deg,
            afterglow_flux=st.afterglow_flux,
            afterglow_peak_time_days=self.engine.afterglow.t_peak_days,
            remnant_scenario=rem_st.scenario,
            remnant_type=rem_st.remnant_type,
            disk_mass_msun=disk_st.disk_mass_msun,
            b_poloidal=mag_st.b_poloidal,
            b_toroidal=mag_st.b_toroidal,
            b_ratio=mag_st.b_ratio,
            mri_active=mag_st.mri_active,
            poynting_luminosity=mag_st.poynting_luminosity,
            neutrino_luminosity=nu_st.luminosity_total,
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
        Synchronize all channels to a specific event time without particle stepping.

        The engine cache is updated first so engine, coordinator, and renderer
        consumers observe the same authoritative presentation event time.
        """
        t_val = float(t_seconds)
        if t_val < 0.0:
            raise ValueError("Event-time evaluation is post-merger only")
        self.engine.set_demo_event_time(t_val)
        return self._update_event_state()

    @property
    def current_state(self) -> MultiMessengerEventState:
        """Return current cached MultiMessengerEventState."""
        return self._event_state

    @property
    def grb_state(self):
        """Current relativistic jet / prompt GRB state object derived from event_time."""
        t_val = self.current_state.event_time
        return self.engine.jet.evaluate(time=t_val, merger_state=self.engine.dynamics.merger_state)

    @property
    def kilonova_state(self):
        """Current two-component kilonova emission state object derived from event_time."""
        t_val = max(0.0, self.current_state.event_time)
        ej_state = self._get_ejecta_state(t_val)
        return self.engine.kilonova.evaluate(ej_state, t_val)

    @property
    def afterglow_state(self):
        """Current broadband afterglow emission state object derived from event_time."""
        t_val = max(0.0, self.current_state.event_time)
        j_state = self.grb_state
        return self.engine.afterglow.evaluate(time_seconds=t_val, jet_state=j_state)
