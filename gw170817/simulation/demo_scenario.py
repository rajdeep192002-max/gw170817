"""
GW170817 Multi-Messenger Demonstration Scenario Manager.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Provides deterministic checkpoint jumps across the multi-messenger timeline
(Inspiral -> Merger -> GRB prompt delay -> Kilonova -> Afterglow peak)
to showcase the reduced-order simulation capabilities.
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
        "INSPIRAL_START": "Initial inspiral phase at 40 Hz (283.8 km separation)",
        "INSPIRAL_LATE":  "Late inspiral regime at 400 Hz (~61 km separation)",
        "MERGER_DEMO":    "Accelerated near-merger demonstration state (1200 Hz, 30 km)",
        "GRB_PROMPT":     "Prompt GRB 170817A trigger (+1.74 s delay after merger)",
        "KILONOVA_PEAK":  "Kilonova thermal emission evolution (+1.0 day post-merger)",
        "AFTERGLOW_PEAK": "Broadband off-axis afterglow maximum (+150.0 days post-merger)"
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

    def jump_to_checkpoint(self, checkpoint_name: str) -> MultiMessengerEventState:
        """
        Jump to a named timeline checkpoint deterministically.
        """
        name = checkpoint_name.upper()
        if name not in self.CHECKPOINTS:
            raise KeyError(f"Unknown checkpoint '{checkpoint_name}'. Available: {list(self.CHECKPOINTS.keys())}")

        if name == "INSPIRAL_START":
            self.coordinator.reset()
            return self.coordinator.current_state

        elif name == "INSPIRAL_LATE":
            self.coordinator.reset()
            # Jump synthetic inspiral state to 400 Hz (~61 km) without starting merger
            self.coordinator.engine.jump_to_demo_phase(f_gw=400.0, separation=61.1e3)
            # Re-evaluate as inspiral since 61 km is well outside contact threshold (46 km)
            self.coordinator.engine.is_demo_phase = False
            return self.coordinator._update_event_state()

        elif name == "MERGER_DEMO":
            self.coordinator.reset()
            self.coordinator.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
            return self.coordinator.current_state

        elif name == "GRB_PROMPT":
            self.coordinator.reset()
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            return self.coordinator.evaluate_at_event_time(1.74)

        elif name == "KILONOVA_PEAK":
            self.coordinator.reset()
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            return self.coordinator.evaluate_at_event_time(1.0 * day)

        elif name == "AFTERGLOW_PEAK":
            self.coordinator.reset()
            self.coordinator.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
            return self.coordinator.evaluate_at_event_time(150.0 * day)

        return self.coordinator.current_state
