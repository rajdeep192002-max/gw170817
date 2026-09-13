"""
GW170817 Multi-Messenger Demonstration Playback Director (Task 022 Visual Comprehensibility Pass).

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Provides deterministic presentation playback control across the continuous GW170817 multi-messenger sequence.
Maintains strict separation between Presentation Time (0.00 / 5.00 s visual timeline) and Physical Event Time (-5.0 s -> +200 d).
Presentation speed scaling (REAL TIME 1.0x vs SLOW MOTION 0.10x) scales animation progress directly without altering dt_physics (1e-4 s).
"""
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.multimessenger import MultiMessengerCoordinator, MultiMessengerEventState
from gw170817.simulation.demo_scenario import DemoScenario


class DemoStage(Enum):
    """Authoritative primary physical NSM demonstration playback stages."""
    IDLE = "IDLE"
    INSPIRAL = "INSPIRAL"
    LATE_INSPIRAL = "LATE_INSPIRAL"
    MERGER = "MERGER"
    RINGDOWN = "RINGDOWN"
    CONTINUOUS_POST_MERGER = "CONTINUOUS_POST_MERGER"
    COMPLETE = "COMPLETE"


class PlaybackMode(Enum):
    """Presentation playback speed modes."""
    REAL_TIME = "REAL TIME"
    SLOW_MOTION = "SLOW MOTION"


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smoothstep easing function mapping x to [0.0, 1.0]."""
    t = float(np.clip((x - edge0) / max(1.0e-6, edge1 - edge0), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class DemoDirector:
    """
    Deterministic Playback Controller for GW170817 Multi-Messenger Demonstration.
    Orchestrates continuous presentation progress t_pres in [0.0, 15.0] seconds across all phase channels.
    """

    MERGER_PRESENTATION_TIME: float = 6.0

    STAGE_SEQUENCE: List[DemoStage] = [
        DemoStage.INSPIRAL,
        DemoStage.LATE_INSPIRAL,
        DemoStage.MERGER,
        DemoStage.RINGDOWN
    ]

    CHECKPOINT_MAP: Dict[DemoStage, str] = {
        DemoStage.INSPIRAL:      "INSPIRAL_START",
        DemoStage.LATE_INSPIRAL: "INSPIRAL_LATE",
        DemoStage.MERGER:        "MERGER_DEMO",
        DemoStage.RINGDOWN:      "GRB_PROMPT"
    }

    # Presentation playback durations per stage [s] (Total = 15.0 s base duration)
    STAGE_DURATIONS: Dict[DemoStage, float] = {
        DemoStage.INSPIRAL:      4.0,
        DemoStage.LATE_INSPIRAL: 2.0,
        DemoStage.MERGER:        3.0,
        DemoStage.RINGDOWN:      6.0
    }

    def __init__(
        self,
        scenario: Optional[DemoScenario] = None,
        coordinator: Optional[MultiMessengerCoordinator] = None,
        config: Optional[SimConfig] = None
    ):
        if config is None:
            if coordinator is not None:
                config = coordinator.config
            elif scenario is not None:
                config = scenario.config
            else:
                config = SimConfig()
        self.config = config

        if coordinator is None:
            if scenario is not None:
                coordinator = scenario.coordinator
            else:
                coordinator = MultiMessengerCoordinator(config=self.config)
        self.coordinator = coordinator

        if scenario is None:
            scenario = DemoScenario(coordinator=self.coordinator, config=self.config)
        self.scenario = scenario

        self._stage: DemoStage = DemoStage.IDLE
        self._stage_index: int = -1
        self._stage_elapsed: float = 0.0
        self._presentation_time: float = 0.0
        self.total_presentation_duration: float = sum(self.STAGE_DURATIONS.values())  # 15.0 s

        self._is_paused: bool = False
        self._playback_mode: PlaybackMode = PlaybackMode.REAL_TIME
        self._speed_multiplier: float = 1.0

        # Accumulated orbital phase for continuous binary revolution during inspiral demo.
        # Updated every update() frame so NS positions orbit smoothly without resetting.
        self._accumulated_orbital_phase: float = 0.0

        self.reset()

    STAGE_START_TIMES: Dict[DemoStage, float] = {
        DemoStage.INSPIRAL:      0.0,
        DemoStage.LATE_INSPIRAL: 4.0,
        DemoStage.MERGER:        6.0,
        DemoStage.RINGDOWN:      9.0
    }

    def reset(self) -> MultiMessengerEventState:
        """Reset director to IDLE state at -5.0 s event_time."""
        self._stage = DemoStage.IDLE
        self._stage_index = -1
        self._stage_elapsed = 0.0
        self._presentation_time = 0.0
        self._is_paused = False
        self._accumulated_orbital_phase = 0.0

        self.coordinator.reset()
        self.coordinator.engine.set_inspiral_time(5.0)
        return self.coordinator._update_event_state()

    def _sync_physics_for_presentation_time(
        self, t_pres: float, dt_pres: float = 0.0
    ) -> MultiMessengerEventState:
        """
        Synchronize engine & coordinator physical event state to presentation time t_pres in [0.0, 15.0] s.

        Primary NSM physical stages continuously map to event_time:
          INSPIRAL       0–4 s pres  →  event_time −5 s → −1 s   (tau_rem 5 → 1 s)
          LATE_INSPIRAL  4–6 s pres  →  event_time −1 s →  0 s   (tau_rem 1 → 0)
          MERGER         6–9 s pres  →  event_time  0   → 1.74 s (merger, remnant, ejecta, disk, jet launch)
          RINGDOWN       9–15 s pres →  event_time  1.74 s → 200 days (continuous post-merger evolution)
        """
        self._presentation_time = max(0.0, float(t_pres))
        p_tot = float(np.clip(self._presentation_time / self.total_presentation_duration, 0.0, 1.0))

        # Pre-fetch current omega_orb for phase accumulation (used in inspiral bands)
        _omega_orb = float(
            self.coordinator.engine.dynamics.inspiral_state.omega_orb
        )

        p_inspiral = 4.0 / 15.0    # 0.2667
        p_late     = 6.0 / 15.0    # 0.4000
        p_merger   = 9.0 / 15.0    # 0.6000

        if p_tot < p_inspiral:
            # INSPIRAL (0-4s pres): tau_rem = remaining time to merger: 5 s → 1 s
            frac = p_tot / p_inspiral
            tau_rem = 5.0 - frac * 4.0          # 5 → 1
            self.coordinator.engine.set_inspiral_time(max(0.001, tau_rem))
            # Accumulate orbital phase using omega_orb *before* state was updated
            self._accumulated_orbital_phase += _omega_orb * dt_pres
            self.coordinator.engine.dynamics.inspiral_state.orbital_phase = (
                self._accumulated_orbital_phase
            )
            self.coordinator.engine.dynamics.update_particles()
            self.coordinator._update_event_state()

        elif p_tot < p_late:
            # LATE_INSPIRAL (4-6s pres): tau_rem 1 s → 0 s
            frac = (p_tot - p_inspiral) / (p_late - p_inspiral)
            tau_rem = 1.0 * (1.0 - frac)
            self.coordinator.engine.set_inspiral_time(max(0.001, tau_rem))
            self._accumulated_orbital_phase += _omega_orb * dt_pres
            self.coordinator.engine.dynamics.inspiral_state.orbital_phase = (
                self._accumulated_orbital_phase
            )
            self.coordinator.engine.dynamics.update_particles()
            self.coordinator._update_event_state()

        elif p_tot < p_merger:
            # MERGER (6-9s pres): Slow-motion physical merger evolution (0 ms -> 60 ms -> 1.74 s)
            frac_merger = (p_tot - p_late) / (p_merger - p_late)
            u = max(0.0, min(1.0, float(frac_merger)))
            if u <= 5.0 / 6.0:
                x = u / (5.0 / 6.0)
                t_event_merger = 0.010 * x + 0.050 * (x ** 2)
            else:
                y = min(1.0, (u - 5.0 / 6.0) / (1.0 / 6.0))
                t_event_merger = min(1.740, 0.060 + (1.740 - 0.060) * (y ** 2))

            self.coordinator.evaluate_at_event_time(t_event_merger)

        else:
            # RINGDOWN / CONTINUOUS_POST_MERGER (>=9s pres): event_time 1.74 s → 200+ days
            if self._presentation_time > self.total_presentation_duration:
                self._stage = DemoStage.CONTINUOUS_POST_MERGER
                dt_ext = self._presentation_time - self.total_presentation_duration
                t_ag = 200.0 * 86400.0 + dt_ext * 86400.0
            else:
                frac_ringdown = (p_tot - p_merger) / (1.0 - p_merger)
                t_ag = 1.74 + frac_ringdown * (200.0 * 86400.0 - 1.74)
            self.coordinator.evaluate_at_event_time(t_ag)

        # Sample GW model from the exact InspiralState driving the NS positions & event state
        insp_state = self.coordinator.engine.dynamics.inspiral_state
        insp_state.time = self.coordinator.current_state.event_time
        gw_sample = self.coordinator.engine.gw_model.sample(insp_state)

        # Append to WaveformBuffer (WaveformBuffer handles deduplication automatically)
        buf = self.coordinator.engine.waveform_buffer
        buf.append(gw_sample)

        return self.coordinator.current_state

    def _compute_integrated_phase(self, t_target: float) -> float:
        """Compute integrated orbital phase over presentation duration [0.0, t_target]."""
        if t_target <= 0.0:
            return 0.0
        n_steps = 100
        dt = t_target / n_steps
        phase = 0.0
        c_coeff = getattr(self.coordinator.engine.inspiral, "_df_coeff", 0.0)
        f_max = self.coordinator.engine.inspiral.f_max
        f_start = self.config.f_gw_start

        for i in range(n_steps):
            t_mid = (i + 0.5) * dt
            if t_mid < 4.0:
                tau_rem = 5.0 - t_mid
            elif t_mid < 6.0:
                tau_rem = 6.0 - t_mid
            else:
                tau_rem = 0.0

            if tau_rem > 0.0 and c_coeff > 0.0:
                term = (f_max ** (-8.0 / 3.0)) + (8.0 * c_coeff / 3.0) * tau_rem
                f_gw = min(f_max - 1.0, max(f_start, float(term ** (-3.0 / 8.0))))
            else:
                f_gw = f_max

            omega_orb = np.pi * f_gw
            phase += omega_orb * dt

        return phase

    def jump_to_stage_index(self, index: int) -> MultiMessengerEventState:
        """Jump deterministically to a specific stage index in STAGE_SEQUENCE."""
        if index < 0:
            return self.reset()
        elif index >= len(self.STAGE_SEQUENCE):
            self._stage = DemoStage.CONTINUOUS_POST_MERGER
            self._stage_index = len(self.STAGE_SEQUENCE)
            self._is_paused = False
            if self._presentation_time < self.total_presentation_duration:
                self._presentation_time = self.total_presentation_duration
            self._accumulated_orbital_phase = self._compute_integrated_phase(self._presentation_time)
            return self._sync_physics_for_presentation_time(self._presentation_time)

        self._stage_index = index
        self._stage = self.STAGE_SEQUENCE[index]
        self._stage_elapsed = 0.0
        self._presentation_time = self.STAGE_START_TIMES[self._stage]
        self._accumulated_orbital_phase = self._compute_integrated_phase(self._presentation_time)

        checkpoint = self.CHECKPOINT_MAP[self._stage]
        self.scenario.jump_to_checkpoint(checkpoint, preserve_waveform=True)
        return self._sync_physics_for_presentation_time(self._presentation_time)

    def start(self) -> MultiMessengerEventState:
        """Start continuous presentation playback from -5.0 s event_time."""
        self._is_paused = False
        return self.jump_to_stage_index(0)

    def pause(self):
        """Pause presentation playback."""
        if self.is_running:
            self._is_paused = True

    def resume(self):
        """Resume presentation playback."""
        if self.is_running:
            self._is_paused = False

    def toggle_pause(self):
        """Toggle pause/resume state."""
        if self._is_paused:
            self.resume()
        else:
            self.pause()

    def set_real_time(self):
        """Set playback mode to REAL TIME (1.00x)."""
        self._playback_mode = PlaybackMode.REAL_TIME
        self._speed_multiplier = 1.00

    def set_slow_motion(self):
        """Set playback mode to SLOW MOTION (0.10x)."""
        self._playback_mode = PlaybackMode.SLOW_MOTION
        self._speed_multiplier = 0.10

    def toggle_playback_mode(self):
        """Toggle playback mode between REAL TIME and SLOW MOTION."""
        if self._playback_mode == PlaybackMode.REAL_TIME:
            self.set_slow_motion()
        else:
            self.set_real_time()

    def increase_speed(self):
        """Increase presentation speed multiplier."""
        self._speed_multiplier = min(10.0, float(np.round(self._speed_multiplier + 0.25, 2)))

    def decrease_speed(self):
        """Decrease presentation speed multiplier."""
        self._speed_multiplier = max(0.05, float(np.round(self._speed_multiplier - 0.25, 2)))

    def next_stage(self) -> MultiMessengerEventState:
        """Advance deterministically to the next demonstration stage."""
        if not self.is_running and self._stage != DemoStage.COMPLETE:
            return self.start()
        return self.jump_to_stage_index(self._stage_index + 1)

    def previous_stage(self) -> MultiMessengerEventState:
        """Step backward to the previous demonstration stage."""
        if self._stage == DemoStage.COMPLETE:
            return self.jump_to_stage_index(len(self.STAGE_SEQUENCE) - 1)
        return self.jump_to_stage_index(self._stage_index - 1)

    def update(self, dt: float) -> MultiMessengerEventState:
        """
        Advance presentation clock by dt * speed_multiplier seconds.
        Continuously maps presentation_time in [0.0, 15.0] to physical event timestamps
        and phase progress variables.
        """
        if not self.is_running or self._is_paused:
            return self.coordinator.current_state

        dt_val = max(0.0, float(dt))
        dt_pres = dt_val * self._speed_multiplier

        self._stage_elapsed += dt_pres
        self._presentation_time += dt_pres

        if self._stage in self.STAGE_SEQUENCE:
            stage_dur = self.STAGE_DURATIONS.get(self._stage, 1.0)
            if self._stage_elapsed >= stage_dur:
                rem = self._stage_elapsed - stage_dur
                self.next_stage()
                self._stage_elapsed = rem
                self._presentation_time = self.STAGE_START_TIMES.get(self._stage, self.total_presentation_duration) + rem

        self._sync_physics_for_presentation_time(self._presentation_time, dt_pres=dt_pres)

        return self.coordinator.current_state

    def step(self, dt: float) -> MultiMessengerEventState:
        """Step presentation playback by dt seconds. Alias for update()."""
        return self.update(dt)

    # --- Continuous Visual Progress Variables ---
    @property
    def presentation_time(self) -> float:
        """Current presentation playback timestamp in [0.0, 15.0] seconds."""
        return float(self._presentation_time)

    @property
    def inspiral_progress(self) -> float:
        """Smooth inspiral phase progress [0.0, 1.0]."""
        p = float(self._presentation_time / 6.0)  # 0 to 6s
        return smoothstep(0.0, 1.0, p)

    @property
    def merger_progress(self) -> float:
        """Smooth contact & merger collision progress [0.0, 1.0]."""
        p = float((self._presentation_time - 6.0) / 3.0)  # 6 to 9s
        return smoothstep(0.0, 1.0, p)

    @property
    def ejecta_progress(self) -> float:
        """Smooth dynamic ejecta expansion progress [0.0, 1.0]."""
        p = float((self._presentation_time - 6.0) / 3.0)  # 6 to 9s
        return smoothstep(0.0, 1.0, p)

    @property
    def disk_progress(self) -> float:
        """Smooth accretion disk torus formation progress [0.0, 1.0]."""
        p = float((self._presentation_time - 6.0) / 3.0)  # 6 to 9s
        return smoothstep(0.0, 1.0, p)

    @property
    def b_winding_progress(self) -> float:
        """Smooth magnetic field winding progress [0.0, 1.0]."""
        p = float((self._presentation_time - 6.0) / 3.0)  # 6 to 9s
        return smoothstep(0.0, 1.0, p)

    @property
    def jet_progress(self) -> float:
        """Smooth structured GRB jet launch & expansion progress [0.0, 1.0]."""
        p = float((self._presentation_time - 7.0) / 6.0)
        return smoothstep(0.0, 1.0, p)

    @property
    def kilonova_progress(self) -> float:
        """Smooth thermal kilonova envelope expansion progress [0.0, 1.0]."""
        p = float((self._presentation_time - 9.0) / 6.0)
        return smoothstep(0.0, 1.0, p)

    @property
    def afterglow_progress(self) -> float:
        """Smooth broadband afterglow rise & peak progress [0.0, 1.0]."""
        p = float((self._presentation_time - 11.0) / 4.0)
        return smoothstep(0.0, 1.0, p)

    @property
    def current_stage(self) -> str:
        """Return name of current stage."""
        return self._stage.value

    @property
    def current_stage_name(self) -> str:
        """Return name of current stage for HUD overlays."""
        return self._stage.value

    @property
    def stage_enum(self) -> DemoStage:
        """Return current DemoStage enum."""
        return self._stage

    @property
    def playback_mode_str(self) -> str:
        """Return playback mode name."""
        return self._playback_mode.value

    @property
    def speed_multiplier(self) -> float:
        """Return presentation speed multiplier."""
        return self._speed_multiplier

    @property
    def is_slow_motion(self) -> bool:
        """True if playback mode is SLOW MOTION (speed multiplier < 1.0)."""
        return self._playback_mode == PlaybackMode.SLOW_MOTION or self._speed_multiplier < 0.99

    @property
    def is_running(self) -> bool:
        """True if presentation playback is active in a valid stage."""
        return self._stage in self.STAGE_SEQUENCE or self._stage == DemoStage.CONTINUOUS_POST_MERGER

    @property
    def is_paused(self) -> bool:
        """True if playback is currently paused."""
        return self._is_paused

    @property
    def is_complete(self) -> bool:
        """True if demo presentation sequence has completed (transitioned into continuous post-merger)."""
        return self._stage in (DemoStage.COMPLETE, DemoStage.CONTINUOUS_POST_MERGER)

    @property
    def stage_progress(self) -> float:
        """Progress fraction within current stage [0.0 to 1.0]."""
        if not self.is_running:
            return 1.0 if self.is_complete else 0.0
        dur = self.STAGE_DURATIONS.get(self._stage, 0.5)
        if dur <= 0.0:
            return 1.0
        return float(np.clip(self._stage_elapsed / dur, 0.0, 1.0))

    @property
    def progress(self) -> float:
        """Overall demonstration playback progress fraction [0.0 to 1.0]."""
        if self.is_complete:
            return 1.0
        if not self.is_running or self.total_presentation_duration <= 0.0:
            return 0.0
        return float(np.clip(self._presentation_time / self.total_presentation_duration, 0.0, 1.0))

    def summary_dict(self) -> Dict[str, Any]:
        """Return current director status summary."""
        return {
            "current_stage": self.current_stage,
            "stage_index": self._stage_index,
            "total_stages": len(self.STAGE_SEQUENCE),
            "playback_mode": self.playback_mode_str,
            "speed_multiplier": self._speed_multiplier,
            "is_slow_motion": self.is_slow_motion,
            "presentation_time": self._presentation_time,
            "total_presentation_duration": self.total_presentation_duration,
            "stage_progress": self.stage_progress,
            "overall_progress": self.progress,
            "disk_progress": self.disk_progress,
            "b_winding_progress": self.b_winding_progress,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "is_complete": self.is_complete
        }
