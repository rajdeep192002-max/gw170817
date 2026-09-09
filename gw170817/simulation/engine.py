"""
Integrated GW170817 Reduced-Order Simulation Engine.

REDUCED-ORDER APPROXIMATION:
This engine integrates the project's reduced-order physics modules into a single deterministic simulation.
It is NOT a full numerical relativity simulation, GRHD, neutrino transport, or full GRMHD calculation.
The accelerated/demo transition to the merger regime is a demonstration mechanism, not a claim
that the laptop directly resolved the complete astrophysical inspiral.
"""
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np
from gw170817.constants import day, Mpc, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.particles import initialize_taichi, ParticleSystem
from gw170817.physics.inspiral import InspiralModel, InspiralState
from gw170817.physics.gravitational_waves import GravitationalWaveModel, WaveformBuffer
from gw170817.physics.tidal import TidalModel
from gw170817.physics.merger import MergerModel
from gw170817.simulation.merger_dynamics import MergerDynamics
from gw170817.physics.ejecta import EjectaModel, EjectaState
from gw170817.physics.kilonova import KilonovaModel, KilonovaState
from gw170817.physics.jet import StructuredJetModel, JetState
from gw170817.physics.afterglow import AfterglowModel, AfterglowState
from gw170817.simulation.event_timeline import EventTimeline, EventPhase


@dataclass
class SimulationState:
    """Integrated instantaneous state of the GW170817 multi-messenger simulation."""
    time: float                    # Engine cumulative elapsed simulation time [s]
    elapsed_time: float            # Engine cumulative elapsed simulation time [s]
    event_time: float              # Multi-messenger event time relative to merger [s] (0.0 = merger, <0 = inspiral)
    step_count: int                # Total physics steps executed
    phase: str                     # Current multi-messenger evolution phase (INSPIRAL, MERGER, etc.)

    gw_frequency: float            # Instantaneous GW frequency [Hz]
    separation: float              # Orbital separation a [m]
    orbital_phase: float           # Orbital phase phi [rad]

    merger_contact_fraction: float # Contact fraction [0.0 to 1.0]
    merger_started: bool           # True if merger regime entered
    merger_complete: bool          # True if compact remnant formed

    ejecta_mass: float             # Dynamically unbound ejecta mass [kg]
    ejecta_fraction: float         # Ejecta mass fraction of total binary mass [0.0 to 1.0]
    ejecta_mean_velocity: float    # Mass-weighted ejecta speed [m/s]

    kilonova_luminosity: float     # Total kilonova bolometric luminosity [W]

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
        self.ejecta_update_interval = 10  # Compute particle ejecta classification every 10 steps

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
        - During inspiral (before merger start): event_time < 0.0 (t = - remaining time to merger)
        - During / post merger: event_time >= 0.0 (t = time since merger)
        """
        insp_state = self.dynamics.inspiral_state
        merg_state = self.dynamics.merger_state

        if merg_state.merger_started or self.is_demo_phase:
            return self._post_merger_event_time
        else:
            t_rem = self._compute_time_to_merger(insp_state.f_gw)
            return -t_rem

    def jump_to_demo_phase(self, f_gw: float = 1200.0, separation: float = 30.0e3):
        """
        Accelerated demonstration state jump to near-merger regime.
        Allows immediate visual & multi-messenger evaluation without requiring tiny-step integration.
        
        This is an accelerated demonstration state transition and is not a claim
        that the full astrophysical inspiral was numerically resolved.
        """
        self.is_demo_phase = True
        self._post_merger_event_time = 0.0

        synthetic_state = InspiralState(
            time=0.0,
            f_gw=float(f_gw),
            orbital_frequency=float(f_gw / 2.0),
            omega_orb=float(np.pi * f_gw),
            separation=float(separation),
            orbital_phase=np.pi,
            df_dt=self.inspiral._compute_df_dt(float(f_gw)),
            chirp_mass=self.config.chirp_mass
        )

        self.dynamics.inspiral_state = synthetic_state
        self.dynamics.tidal_state = self.tidal.evaluate(synthetic_state)
        self.dynamics.merger_state = self.merger.evaluate(synthetic_state)

        # Synchronize particle state without performing an inspiral timestep that changes requested parameters
        r1, r2 = self.inspiral.orbital_positions(synthetic_state, self.config.m1, self.config.m2)
        v1, v2 = self.inspiral.orbital_velocities(synthetic_state, self.config.m1, self.config.m2)
        eps1 = min(0.5 * self.dynamics.tidal_state.tidal_distortion_1, 0.4)
        eps2 = min(0.5 * self.dynamics.tidal_state.tidal_distortion_2, 0.4)

        self.dynamics._update_particles_kernel(
            self.psys.n_particles_1,
            self.psys.max_particles,
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

        self._update_cached_state()

    def _update_cached_state(self) -> SimulationState:
        """Internal helper to construct and cache current SimulationState."""
        insp_state = self.dynamics.inspiral_state
        merg_state = self.dynamics.merger_state

        event_time = self._compute_event_time()

        # Phase from Timeline based on event time relative to merger
        phase = self.timeline.current_phase(event_time)

        # Ejecta & Kilonova
        if (self.step_count % self.ejecta_update_interval == 0) or (self.step_count == 0):
            diag_com = self.psys.compute_diagnostics_numpy()
            self.ejecta.classify(self.psys, diag_com['com_pos'], diag_com['com_vel'])

        ej_state = self.ejecta.compute_state(self.psys, max(0.0, event_time))
        kn_state = self.kilonova.evaluate(ej_state, max(0.0, event_time))

        # Jet & Afterglow
        j_state = self.jet.evaluate(time=event_time, merger_state=merg_state)
        ag_state = self.afterglow.evaluate(time_seconds=max(0.0, event_time), jet_state=j_state)

        # GW Waveform
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

        dt_val = float(dt)
        if dt_val <= 0.0 or not np.isfinite(dt_val):
            raise ValueError(f"Step timestep dt must be strictly positive and finite, got {dt}")

        self.elapsed_time += dt_val
        self.time = self.elapsed_time
        self.step_count += 1

        # Advance particle & orbital dynamics
        self.dynamics.step(dt_val)

        if self.dynamics.merger_state.merger_started or self.is_demo_phase:
            self._post_merger_event_time += dt_val

        return self._update_cached_state()

    def run(
        self,
        duration: float,
        dt: float = None,
        callback: Optional[Callable[[SimulationState], None]] = None
    ) -> SimulationState:
        """
        Run simulation for requested duration in seconds without duration overshoot.
        Deterministic, bounded memory.
        """
        dur_val = float(duration)
        if dur_val <= 0.0 or not np.isfinite(dur_val):
            raise ValueError(f"Run duration must be strictly positive and finite, got {duration}")

        if dt is None:
            dt = self.config.dt_physics
        else:
            dt = float(dt)
            if dt <= 0.0 or not np.isfinite(dt):
                raise ValueError(f"Run timestep dt must be strictly positive and finite, got {dt}")

        remaining = dur_val
        eps = 1e-15
        while remaining > eps:
            step_dt = min(dt, remaining)
            st = self.step(step_dt)
            remaining -= step_dt
            if callback is not None:
                callback(st)

        return self._state

    @property
    def current_state(self) -> SimulationState:
        """Get current cached SimulationState."""
        return self._state

    def compute_diagnostics(self) -> Dict[str, Any]:
        """Expose lightweight engine-level diagnostics."""
        dyn_diag = self.dynamics.compute_diagnostics()
        return {
            "backend": self.backend,
            "particle_count": self.psys.max_particles,
            "simulation_time": self.elapsed_time,
            "event_time": self._compute_event_time(),
            "step_count": self.step_count,
            "current_phase": self._state.phase,
            "dynamics_diagnostics": dyn_diag
        }
