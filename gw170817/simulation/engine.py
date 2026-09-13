"""
Central Integrated Simulation Engine for GW170817.
Orchestrates all physics modules, particle dynamics, and multi-messenger event timeline.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem, initialize_taichi
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.tidal import TidalModel
from gw170817.physics.merger import MergerModel
from gw170817.simulation.merger_dynamics import MergerDynamics
from gw170817.physics.rotation import RotationalModel
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.disk import DiskModel
from gw170817.physics.magnetic_field import MagneticFieldModel
from gw170817.physics.neutrinos import NeutrinoModel
from gw170817.physics.ejecta import EjectaModel
from gw170817.physics.kilonova import KilonovaModel
from gw170817.physics.jet import StructuredJetModel
from gw170817.physics.afterglow import AfterglowModel
from gw170817.simulation.event_timeline import EventTimeline
from gw170817.physics.gravitational_waves import GravitationalWaveModel, WaveformBuffer


@dataclass
class SimulationState:
    """Comprehensive snapshot of simulation engine state."""
    time: float                    # Engine cumulative simulation time [s]
    elapsed_time: float            # Engine cumulative simulation time [s]
    event_time: float              # Event time relative to merger [s] (0.0 = merger)
    step_count: int                # Total physics steps executed
    phase: str                     # Timeline phase ("INSPIRAL", "MERGER", etc.)

    gw_frequency: float            # Instantaneous GW frequency [Hz]
    separation: float              # Binary separation a [m]
    orbital_phase: float           # Binary orbital phase [rad]

    merger_contact_fraction: float # Contact fraction [0.0 to 1.0]
    merger_started: bool           # True if contact/merger initiated
    merger_complete: bool          # True if compact remnant formed

    ejecta_mass: float             # Dynamically unbound ejecta mass [kg]
    ejecta_fraction: float         # Ejecta mass fraction of total binary mass
    ejecta_mean_velocity: float    # Ejecta mean expansion velocity [m/s]
    kilonova_luminosity: float     # Kilonova bolometric luminosity [W]

    grb_launched: bool             # True if relativistic jet launched
    grb_triggered: bool            # True if 1.7 s GW-GRB prompt delay elapsed

    afterglow_flux: float          # Observed broadband afterglow flux density [W m^-2 Hz^-1]


class GW170817Simulation:
    """
    Central Integrated Simulation Engine for GW170817.
    Orchestrates all physics modules, particle dynamics, and multi-messenger event timeline.
    """

    def __init__(
        self,
        config: SimConfig = None,
        particle_system: ParticleSystem = None,
        backend_preference: str = "vulkan"
    ):
        if config is None:
            config = SimConfig()
        self.config = config

        # 1. Initialize Taichi backend
        self.backend = initialize_taichi(backend_preference)

        # 2. Instantiate Particle System
        if particle_system is None:
            particle_system = ParticleSystem(config)
        self.psys = particle_system

        # 3. Instantiate Physics Modules
        self.inspiral = InspiralModel(config)
        self.tidal = TidalModel(config)
        self.merger = MergerModel(config, self.tidal)
        self.dynamics = MergerDynamics(self.psys, self.inspiral, self.tidal, self.merger)
        self.rotation = RotationalModel(config)
        self.remnant = RemnantModel(config)
        self.disk = DiskModel(config)
        self.magnetic_field = MagneticFieldModel(config)
        self.neutrinos = NeutrinoModel(config)
        self.ejecta = EjectaModel(config)
        self.kilonova = KilonovaModel(config)
        self.jet = StructuredJetModel(config)
        self.afterglow = AfterglowModel(config)

        # 4. Instantiate Multi-Messenger Event Timeline
        self.timeline = EventTimeline(config, self.jet, self.afterglow)

        # 5. Instantiate Gravitational Wave Model & Buffer
        self.gw_model = GravitationalWaveModel(config)
        self.buffer_capacity = getattr(config, "WAVEFORM_BUFFER_LEN", 1024)
        self.waveform_buffer = WaveformBuffer(capacity=self.buffer_capacity)

        # Simulation Clocks & Counters
        self.elapsed_time = 0.0
        self.time = 0.0
        self.step_count = 0
        self.ejecta_update_interval = 10

        self._post_merger_event_time = 0.0
        self.is_demo_phase = False

        # Initialize Simulation State
        self.reset()

    def reset(self):
        """Reset simulation time, step counter, particle fields, and model states."""
        self.elapsed_time = 0.0
        self.time = 0.0
        self.step_count = 0
        self._post_merger_event_time = 0.0
        self.is_demo_phase = False

        self.dynamics.initialize()
        self.ejecta.initialize()
        self.waveform_buffer = WaveformBuffer(capacity=self.buffer_capacity)

        # Cache initial state
        self._update_cached_state()

    @property
    def current_state(self) -> SimulationState:
        """Return currently cached SimulationState."""
        return self._state

    def _compute_time_to_merger(self, f_gw: float) -> float:
        """Calculate remaining inspiral time to merger in seconds using Peters quadrupole formula."""
        if f_gw >= self.inspiral.f_max:
            return 0.0
        f_clamped = max(f_gw, 1.0)
        c_coeff = getattr(self.inspiral, "_df_coeff", 0.0)
        if c_coeff <= 0.0:
            return 0.0
        f_max = self.inspiral.f_max
        tau = (3.0 / (8.0 * c_coeff)) * ((f_clamped ** (-8.0 / 3.0)) - (f_max ** (-8.0 / 3.0)))
        return max(0.0, float(tau))

    def _compute_event_time(self) -> float:
        """
        Compute simulation event time relative to merger [s].
        """
        if self.is_demo_phase:
            return self._post_merger_event_time

        merg_st = self.dynamics.merger_state
        if merg_st.merger_started or merg_st.merger_complete:
            return self._post_merger_event_time
        else:
            return -self._compute_time_to_merger(self.dynamics.inspiral_state.f_gw)

    def set_inspiral_time(self, time_to_merger_s: float):
        """
        Jump inspiral physics directly to a target time-to-merger in seconds.
        """
        tau_target = max(0.0, float(time_to_merger_s))
        c_coeff = getattr(self.inspiral, "_df_coeff", 0.0)
        f_max = self.inspiral.f_max
        if c_coeff > 0.0:
            term = (f_max ** (-8.0 / 3.0)) + (8.0 * c_coeff / 3.0) * tau_target
            f_target = float(term ** (-3.0 / 8.0))
        else:
            f_target = 40.0

        f_clamped = min(f_max - 1.0, max(self.config.f_gw_start, f_target))
        from gw170817.constants import G
        a_target = (G * self.config.M_total / (np.pi * f_clamped)**2)**(1.0 / 3.0)

        self.dynamics.inspiral_state.f_gw = f_clamped
        self.dynamics.inspiral_state.orbital_frequency = f_clamped / 2.0
        self.dynamics.inspiral_state.omega_orb = np.pi * f_clamped
        self.dynamics.inspiral_state.separation = a_target
        self.dynamics.inspiral_state.df_dt = self.inspiral._compute_df_dt(f_clamped)
        self.dynamics.inspiral_state.time = -tau_target

        self.dynamics.merger_state.merger_started = False
        self.dynamics.merger_state.merger_complete = False
        self.dynamics.merger_state.contact_fraction = 0.0

        self._post_merger_event_time = -tau_target
        self.is_demo_phase = False
        self.dynamics.update_particles()
        return self._update_cached_state()

    def jump_to_demo_phase(self, f_gw: float = 1200.0, separation: float = 30.0e3):
        """Perform accelerated demo jump to near-merger state."""
        self.dynamics.inspiral_state.f_gw = f_gw
        self.dynamics.inspiral_state.orbital_frequency = f_gw / 2.0
        self.dynamics.inspiral_state.omega_orb = np.pi * f_gw
        self.dynamics.inspiral_state.separation = separation

        self.dynamics.merger_state.merger_started = True
        self.dynamics.merger_state.merger_complete = False
        self.dynamics.merger_state.contact_fraction = 1.0
        self.dynamics.v_clamp = 0.05

        self.is_demo_phase = True
        self._post_merger_event_time = 0.0
        self.dynamics.update_particles()
        return self._update_cached_state()

    def set_post_merger_event_time(self, t_seconds: float):
        """Set post-merger event time in seconds."""
        t_val = float(t_seconds)
        self.is_demo_phase = True
        self._post_merger_event_time = t_val

        self.dynamics.merger_state.merger_started = True
        self.dynamics.merger_state.merger_complete = True
        self.dynamics.merger_state.contact_fraction = 1.0
        self.dynamics.v_clamp = min(0.40, 0.05 + 0.35 * (1.0 - np.exp(-max(0.0, t_val) / 0.1)))

        f_max = self.inspiral.f_max
        self.dynamics.inspiral_state.f_gw = f_max
        self.dynamics.inspiral_state.orbital_frequency = f_max / 2.0
        self.dynamics.inspiral_state.omega_orb = np.pi * f_max
        self.dynamics.inspiral_state.separation = 20.0e3

        self.dynamics.update_particles()
        return self._update_cached_state()

    def set_demo_event_time(self, t_seconds: float):
        """Synchronize the non-stepping presentation state to one event timestamp.

        The presentation director uses this path while replaying its compressed
        timeline.  It deliberately preserves the reduced-order physics models,
        but ramps contact over the existing 60 ms dynamical-merger interval
        instead of forcing the visual state to fully merged at event time zero.
        """
        t_val = float(t_seconds)
        if t_val < 0.0:
            raise ValueError("Demo event time must be non-negative")

        if not self.is_demo_phase:
            self.dynamics.inspiral_state.f_gw = 1200.0
            self.dynamics.inspiral_state.orbital_frequency = 600.0
            self.dynamics.inspiral_state.omega_orb = np.pi * 1200.0
            self.dynamics.inspiral_state.separation = 30.0e3

        self.is_demo_phase = True
        self._post_merger_event_time = t_val

        contact_fraction = float(np.clip(t_val / 0.060, 0.0, 1.0))
        merger_state = self.dynamics.merger_state
        merger_state.contact_fraction = contact_fraction
        merger_state.in_contact = bool(contact_fraction >= 0.5)
        merger_state.merger_started = True
        merger_state.merger_complete = bool(contact_fraction >= 1.0)
        self.dynamics.v_clamp = min(
            0.40, 0.05 + 0.35 * (1.0 - np.exp(-t_val / 0.1))
        )

        self.dynamics.update_particles()
        return self._update_cached_state()

    def set_synthetic_inspiral_state(
        self,
        r1: np.ndarray, r2: np.ndarray,
        v1: np.ndarray, v2: np.ndarray,
        synthetic_state
    ):
        """Inject explicit orbital state for visual timeline syncing."""
        self.dynamics.inspiral_state.f_gw = float(synthetic_state.f_gw)
        self.dynamics.inspiral_state.orbital_frequency = float(synthetic_state.orbital_frequency)
        self.dynamics.inspiral_state.omega_orb = float(synthetic_state.omega_orb)
        self.dynamics.inspiral_state.separation = float(synthetic_state.separation)
        self.dynamics.inspiral_state.orbital_phase = float(synthetic_state.orbital_phase)

        eps1 = getattr(synthetic_state, 'eps1', 0.0)
        eps2 = getattr(synthetic_state, 'eps2', 0.0)

        self.psys.set_particle_positions_rigid(
            float(r1[0]), float(r1[1]), float(r1[2]),
            float(r2[0]), float(r2[1]), float(r2[2]),
            float(v1[0]), float(v1[1]), float(v1[2]),
            float(v2[0]), float(v2[1]), float(v2[2]),
            float(synthetic_state.orbital_phase),
            float(synthetic_state.omega_orb),
            float(eps1), float(eps2),
            float(self.dynamics.merger_state.contact_fraction),
            float(self.dynamics.v_clamp)
        )

        self.is_demo_phase = False
        return self._update_cached_state()

    def _update_cached_state(self) -> SimulationState:
        """Internal helper to construct and cache current SimulationState."""
        insp_state = self.dynamics.inspiral_state
        merg_state = self.dynamics.merger_state

        event_time = self._compute_event_time()

        phase = self.timeline.current_phase(event_time)

        if self.step_count > 0 and (self.step_count % self.ejecta_update_interval == 0):
            diag_com = self.psys.compute_diagnostics_numpy()
            self.ejecta.classify(self.psys, diag_com['com_pos'], diag_com['com_vel'])

        rem_state = self.remnant.evaluate(insp_state, event_time)
        rot_state = self.rotation.evaluate(insp_state, rem_state, event_time)
        disk_state = self.disk.evaluate(rem_state, event_time, rot_state)
        mag_state = self.magnetic_field.evaluate(rem_state, disk_state, event_time, rot_state)

        ej_state = self.ejecta.compute_state(self.psys, max(0.0, event_time))
        kn_state = self.kilonova.evaluate(ej_state, max(0.0, event_time))

        j_state = self.jet.evaluate(time=event_time, merger_state=merg_state)
        ag_state = self.afterglow.evaluate(time_seconds=max(0.0, event_time), jet_state=j_state)

        gw_sample = self.gw_model.sample(insp_state)
        self.waveform_buffer.append(gw_sample)

        self._state = SimulationState(
            time=self.elapsed_time,
            elapsed_time=self.elapsed_time,
            event_time=event_time,
            step_count=self.step_count,
            phase=phase,
            gw_frequency=insp_state.f_gw,
            separation=insp_state.separation,
            orbital_phase=insp_state.orbital_phase,
            merger_contact_fraction=merg_state.contact_fraction,
            merger_started=merg_state.merger_started,
            merger_complete=merg_state.merger_complete,
            ejecta_mass=ej_state.ejecta_mass,
            ejecta_fraction=ej_state.ejecta_fraction,
            ejecta_mean_velocity=ej_state.mean_velocity,
            kilonova_luminosity=kn_state.L_total,
            grb_launched=j_state.launched,
            grb_triggered=j_state.grb_triggered,
            afterglow_flux=ag_state.flux_density
        )
        return self._state

    def step(self, dt: float = None) -> SimulationState:
        """
        Advance simulation clock and all coupled physics by dt seconds.
        """
        if dt is None:
            dt = self.config.dt_physics

        self.dynamics.step(dt)

        if self.dynamics.merger_state.merger_started or self.is_demo_phase:
            self._post_merger_event_time += dt

        self.elapsed_time += dt
        self.time += dt
        self.step_count += 1

        return self._update_cached_state()

    def run(self, duration: float, dt: float = None) -> SimulationState:
        """Run simulation for a total physical duration in seconds."""
        if duration < 0.0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
        if dt is not None and dt <= 0.0:
            raise ValueError(f"Timestep dt must be positive, got {dt}")
        if dt is None:
            dt = self.config.dt_physics
        start_time = self.elapsed_time
        target_time = start_time + duration
        while self.elapsed_time < target_time - 1.0e-12:
            step_dt = min(dt, target_time - self.elapsed_time)
            self.step(step_dt)
        return self.current_state
