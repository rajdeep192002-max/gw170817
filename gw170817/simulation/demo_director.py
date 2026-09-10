"""
GW170817 Multi-Messenger Demonstration Playback Director.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Provides deterministic presentation playback control across the GW170817 multi-messenger sequence
(Inspiral -> Late Inspiral -> Merger -> GRB Prompt -> Kilonova -> Afterglow).

Presentation timing (wall-clock playback seconds per stage) and presentation speed scaling (REAL TIME / SLOW MOTION)
are kept strictly separate from physical event timestamps and physical integration timestep dt_physics (1e-4 s).
"""
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.multimessenger import MultiMessengerCoordinator, MultiMessengerEventState
from gw170817.simulation.demo_scenario import DemoScenario


class DemoStage(Enum):
    """Deterministic demonstration playback stages."""
    IDLE = "IDLE"
    INSPIRAL = "INSPIRAL"
    LATE_INSPIRAL = "LATE_INSPIRAL"
    MERGER = "MERGER"
    GRB = "GRB"
    KILONOVA = "KILONOVA"
    AFTERGLOW = "AFTERGLOW"
    COMPLETE = "COMPLETE"


class PlaybackMode(Enum):
    """Presentation playback speed modes."""
    REAL_TIME = "REAL TIME"
    SLOW_MOTION = "SLOW MOTION"


class DemoDirector:
    """
    Deterministic Playback Controller for GW170817 Multi-Messenger Demonstration.
    Orchestrates DemoScenario checkpoints into a continuous presentation playback sequence.
    Supports REAL TIME (1.0x) and SLOW MOTION (0.1x) playback modes.
    """

    STAGE_SEQUENCE: List[DemoStage] = [
        DemoStage.INSPIRAL,
        DemoStage.LATE_INSPIRAL,
        DemoStage.MERGER,
        DemoStage.GRB,
        DemoStage.KILONOVA,
        DemoStage.AFTERGLOW
    ]

    CHECKPOINT_MAP: Dict[DemoStage, str] = {
        DemoStage.INSPIRAL:      "INSPIRAL_START",
        DemoStage.LATE_INSPIRAL: "INSPIRAL_LATE",
        DemoStage.MERGER:        "MERGER_DEMO",
        DemoStage.GRB:           "GRB_PROMPT",
        DemoStage.KILONOVA:      "KILONOVA_PEAK",
        DemoStage.AFTERGLOW:     "AFTERGLOW_PEAK"
    }

    STAGE_DURATIONS: Dict[DemoStage, float] = {
        DemoStage.INSPIRAL:      8.0,
        DemoStage.LATE_INSPIRAL: 5.0,
        DemoStage.MERGER:        5.0,
        DemoStage.GRB:           4.0,
        DemoStage.KILONOVA:      5.0,
        DemoStage.AFTERGLOW:     5.0
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
        self._is_paused: bool = False

        # Playback speed modes
        self._playback_mode: PlaybackMode = PlaybackMode.REAL_TIME
        self._speed_multiplier: float = 1.0

        self._total_duration: float = sum(self.STAGE_DURATIONS.values())
        self.reset()

    def reset(self) -> MultiMessengerEventState:
        """Reset director to IDLE state and reset underlying coordinator."""
        self._stage = DemoStage.IDLE
        self._stage_index = -1
        self._stage_elapsed = 0.0
        self._is_paused = False
        self.coordinator.reset()
        return self.coordinator.current_state

    def start(self) -> MultiMessengerEventState:
        """Start presentation playback from the first stage (INSPIRAL)."""
        self._stage_index = 0
        self._stage = self.STAGE_SEQUENCE[0]
        self._stage_elapsed = 0.0
        self._is_paused = False
        checkpoint = self.CHECKPOINT_MAP[self._stage]
        return self.scenario.jump_to_checkpoint(checkpoint, preserve_waveform=True)

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
        """Set playback mode to REAL TIME (1.0x)."""
        self._playback_mode = PlaybackMode.REAL_TIME
        self._speed_multiplier = 1.0

    def set_slow_motion(self):
        """Set playback mode to SLOW MOTION (0.1x)."""
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
        if not self.is_running:
            return self.start()

        if self._stage_index < len(self.STAGE_SEQUENCE) - 1:
            self._stage_index += 1
            self._stage = self.STAGE_SEQUENCE[self._stage_index]
            self._stage_elapsed = 0.0
            checkpoint = self.CHECKPOINT_MAP[self._stage]
            return self.scenario.jump_to_checkpoint(checkpoint, preserve_waveform=True)
        else:
            self._stage = DemoStage.COMPLETE
            self._stage_index = len(self.STAGE_SEQUENCE)
            self._stage_elapsed = self.STAGE_DURATIONS.get(self.STAGE_SEQUENCE[-1], 0.0)
            self._is_paused = False
            return self.coordinator.current_state

    def previous_stage(self) -> MultiMessengerEventState:
        """Step backward to the previous demonstration stage."""
        if self._stage == DemoStage.COMPLETE:
            self._stage_index = len(self.STAGE_SEQUENCE) - 1
            self._stage = self.STAGE_SEQUENCE[self._stage_index]
            self._stage_elapsed = 0.0
            checkpoint = self.CHECKPOINT_MAP[self._stage]
            return self.scenario.jump_to_checkpoint(checkpoint, preserve_waveform=True)

        if self._stage_index > 0:
            self._stage_index -= 1
            self._stage = self.STAGE_SEQUENCE[self._stage_index]
            self._stage_elapsed = 0.0
            checkpoint = self.CHECKPOINT_MAP[self._stage]
            return self.scenario.jump_to_checkpoint(checkpoint, preserve_waveform=True)
        else:
            return self.reset()

    def update(self, dt: float) -> MultiMessengerEventState:
        """
        Advance presentation timer by dt * speed_multiplier seconds.
        Automatically transitions to the next stage when presentation stage duration elapses.
        Does NOT modify physical event timestamps or dt_physics.
        """
        if not self.is_running or self._is_paused:
            return self.coordinator.current_state

        dt_val = max(0.0, float(dt))
        dt_pres = dt_val * self._speed_multiplier
        self._stage_elapsed += dt_pres

        # Smooth presentation-time mapping for inspiral stages without continuous 0.1 ms integration loops
        if self._stage == DemoStage.INSPIRAL:
            p = min(1.0, max(0.0, self._stage_elapsed / 8.0))
            tau_rem = 12.0 - 10.0 * p
            self.coordinator.engine.set_inspiral_time(tau_rem, dt_phase=dt_pres)
        elif self._stage == DemoStage.LATE_INSPIRAL:
            p = min(1.0, max(0.0, self._stage_elapsed / 5.0))
            tau_rem = 2.0 - 1.9 * p
            self.coordinator.engine.set_inspiral_time(tau_rem, dt_phase=dt_pres)

        stage_dur = self.STAGE_DURATIONS.get(self._stage, 5.0)
        if self._stage_elapsed >= stage_dur:
            self.next_stage()

        return self.coordinator.current_state

    @property
    def current_stage(self) -> str:
        """Return name of current stage."""
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
    def is_running(self) -> bool:
        """True if presentation playback is active in a valid stage."""
        return self._stage in self.STAGE_SEQUENCE

    @property
    def is_paused(self) -> bool:
        """True if playback is currently paused."""
        return self._is_paused

    @property
    def is_complete(self) -> bool:
        """True if demo playback sequence has finished."""
        return self._stage == DemoStage.COMPLETE

    @property
    def stage_progress(self) -> float:
        """Progress fraction within current stage [0.0 to 1.0]."""
        if not self.is_running:
            return 1.0 if self.is_complete else 0.0
        dur = self.STAGE_DURATIONS.get(self._stage, 5.0)
        if dur <= 0.0:
            return 1.0
        return float(np.clip(self._stage_elapsed / dur, 0.0, 1.0))

    @property
    def progress(self) -> float:
        """Overall demonstration playback progress fraction [0.0 to 1.0]."""
        if self.is_complete:
            return 1.0
        if not self.is_running or self._total_duration <= 0.0:
            return 0.0

        elapsed_prev = sum(self.STAGE_DURATIONS[st] for st in self.STAGE_SEQUENCE[:self._stage_index])
        total_elapsed = elapsed_prev + min(self._stage_elapsed, self.STAGE_DURATIONS.get(self._stage, 0.0))
        return float(np.clip(total_elapsed / self._total_duration, 0.0, 1.0))

    def summary_dict(self) -> Dict[str, Any]:
        """Return current director status summary."""
        return {
            "current_stage": self.current_stage,
            "stage_index": self._stage_index,
            "total_stages": len(self.STAGE_SEQUENCE),
            "playback_mode": self.playback_mode_str,
            "speed_multiplier": self._speed_multiplier,
            "stage_elapsed": self._stage_elapsed,
            "stage_duration": self.STAGE_DURATIONS.get(self._stage, 0.0),
            "stage_progress": self.stage_progress,
            "overall_progress": self.progress,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "is_complete": self.is_complete
        }
