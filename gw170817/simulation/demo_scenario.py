"""
GW170817 Multi-Messenger Demonstration Scenario Manager.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Provides deterministic checkpoint jumps across the single continuous multi-messenger timeline
(-12.0 s Inspiral -> Merger -> +1.7 s GRB prompt delay -> Kilonova -> +150 d Afterglow peak)
to showcase the reduced-order simulation capabilities.
Preserves waveform buffer history and event clock continuity across stage jumps.
"""
from typing import Dict, Any, Optional
from gw170817.constants import day
from gw170817.config import SimConfig
from gw170817.simulation.multimessenger import MultiMessengerCoordinator, MultiMessengerEventState


class DemoScenario:
    """
    Deterministic Multi-Messenger Checkpoint Scenario Manager for GW170817.
    """

    CHECKPOINTS = {
        "INSPIRAL_START":   "Initial inspiral phase (t = -12.0 s, ~233.5 km separation)",
        "INSPIRAL_MID":     "Mid inspiral phase (t = -6.0 s, ~185.0 km separation)",
        "INSPIRAL_LATE":    "Late inspiral regime (t = -2.0 s, ~132.0 km separation)",
        "FINAL_APPROACH":   "Final approach (t = -0.5 s, ~78.0 km separation)",
        "MERGER_DEMO":      "BNS Merger contact regime (t = 0.0 s)",
        "GRB_PROMPT":       "Prompt GRB 170817A trigger (+1.74 s delay after merger)",
        "EARLY_KILONOVA":   "Early kilonova thermal diffusion (+1.0 day post-merger)",
        "KILONOVA_PEAK":    "Kilonova thermal emission peak (+10.0 days post-merger)",
        "AFTERGLOW_RISE":   "Broadband off-axis afterglow rise (+30.0 days post-merger)",
        "AFTERGLOW_PEAK":   "Broadband off-axis afterglow maximum (+150.0 days post-merger)",
        "AFTERGLOW_DECLINE": "Broadband off-axis afterglow decline (+200.0 days post-merger)"
    }

    def __init__(self, coordinator: Optional[MultiMessengerCoordinator] = None, config: Optional[SimConfig] = None):
        if config is None:
            if coordinator is not None:
                config = coordinator.config
            else:
                config = SimConfig()
        self.config = config

        if coordinator is None:
            coordinator = MultiMessengerCoordinator(config=self.config)
        self.coordinator = coordinator

    def jump_to_checkpoint(self, checkpoint_name: str, preserve_waveform: bool = True) -> MultiMessengerEventState:
        """
        Jump to a named timeline checkpoint deterministically.
        Preserves rolling waveform buffer history unless preserve_waveform=False.
        """
        name = checkpoint_name.upper()
        if name not in self.CHECKPOINTS:
            raise KeyError(f"Unknown checkpoint '{checkpoint_name}'. Available: {list(self.CHECKPOINTS.keys())}")

        # Backup waveform buffer if preserve_waveform is True
        buf_backup = None
        if preserve_waveform and hasattr(self.coordinator.engine, "waveform_buffer"):
            buf_backup = self.coordinator.engine.waveform_buffer

        if name == "INSPIRAL_START":
            self.coordinator.engine.set_inspiral_time(12.0)
            st = self.coordinator._update_event_state()
        elif name == "INSPIRAL_MID":
            self.coordinator.engine.set_inspiral_time(6.0)
            st = self.coordinator._update_event_state()
        elif name == "INSPIRAL_LATE":
            self.coordinator.engine.set_inspiral_time(2.0)
            st = self.coordinator._update_event_state()
        elif name == "FINAL_APPROACH":
            self.coordinator.engine.set_inspiral_time(0.5)
            st = self.coordinator._update_event_state()
        elif name == "MERGER_DEMO":
            self.coordinator.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
            st = self.coordinator.current_state
        elif name == "GRB_PROMPT":
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            st = self.coordinator.evaluate_at_event_time(1.74)
        elif name == "EARLY_KILONOVA":
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            st = self.coordinator.evaluate_at_event_time(1.0 * day)
        elif name == "KILONOVA_PEAK":
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            st = self.coordinator.evaluate_at_event_time(10.0 * day)
        elif name == "AFTERGLOW_RISE":
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            st = self.coordinator.evaluate_at_event_time(30.0 * day)
        elif name == "AFTERGLOW_PEAK":
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            st = self.coordinator.evaluate_at_event_time(150.0 * day)
        elif name == "AFTERGLOW_DECLINE":
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            st = self.coordinator.evaluate_at_event_time(200.0 * day)
        else:
            st = self.coordinator.current_state

        if buf_backup is not None:
            self.coordinator.engine.waveform_buffer = buf_backup

        return st
