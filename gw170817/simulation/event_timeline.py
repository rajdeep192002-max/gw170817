"""
Multi-messenger GW170817 event timeline and coordination layer.

REDUCED-ORDER APPROXIMATION:
This timeline is an event-coordination layer for a reduced-order multi-messenger simulation.
It does NOT:
- solve general relativity
- derive the observed 1.7 s GW-GRB delay
- derive the 150-day afterglow peak
- model radiative transfer
- model jet propagation in detail
- determine the physical kilonova duration
The timestamps are observationally motivated/reference values used to synchronize the reduced-order modules.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any, Optional
import numpy as np
from gw170817.constants import day, Mpc
from gw170817.config import SimConfig


class EventPhase(Enum):
    """Simulation-relative multi-messenger evolution phases."""
    INSPIRAL = "INSPIRAL"
    MERGER = "MERGER"
    POST_MERGER = "POST_MERGER"
    GRB = "GRB"
    AFTERGLOW = "AFTERGLOW"


@dataclass(frozen=True)
class MessengerEvent:
    """Dataclass representing a discrete multi-messenger event marker."""
    name: str
    time: float          # Simulation-relative time [s] (0.0 = merger)
    messenger: str     # "GW", "GRB", "KILONOVA", "AFTERGLOW"
    description: str


class EventTimeline:
    """
    Central Event Coordinator for GW170817 Multi-Messenger Simulation.
    Synchronizes all physics modules around a unified simulation clock (t = 0 at merger).
    """

    def __init__(
        self,
        config: SimConfig = None,
        jet_model=None,
        afterglow_model=None
    ):
        if config is None:
            config = SimConfig()
        self.config = config
        self.jet_model = jet_model
        self.afterglow_model = afterglow_model

        # Event Metadata
        self.event_name = "GW170817"
        self.event_type = "Binary neutron star merger"
        self.reference_utc = "2017-08-17 12:41:04 UTC"
        self.distance = float(config.distance)

        # Reference Timings [s]
        self.merger_time = 0.0
        self.grb_delay = 1.7
        self.grb_time = self.merger_time + self.grb_delay  # 1.7 s
        self.grb_duration = float(jet_model.duration) if jet_model is not None else 2.0
        
        self.afterglow_peak_days = float(afterglow_model.t_peak_days) if afterglow_model is not None else 150.0
        self.afterglow_peak_time = self.afterglow_peak_days * day  # 12,960,000.0 s

        # Preallocate immutable chronological event sequence
        self._events: List[MessengerEvent] = [
            MessengerEvent(
                name="GW170817 merger",
                time=self.merger_time,
                messenger="GW",
                description="Binary neutron star merger / gravitational-wave reference event"
            ),
            MessengerEvent(
                name="Kilonova",
                time=self.merger_time,
                messenger="KILONOVA",
                description="Electromagnetic ejecta-powered kilonova begins after merger"
            ),
            MessengerEvent(
                name="GRB 170817A",
                time=self.grb_time,
                messenger="GRB",
                description="Short gamma-ray burst observed approximately 1.7 s after GW170817"
            ),
            MessengerEvent(
                name="Afterglow peak",
                time=self.afterglow_peak_time,
                messenger="AFTERGLOW",
                description="Reduced-order broadband afterglow reference peak"
            )
        ]

        # Fast lookup map
        self._event_map: Dict[str, MessengerEvent] = {ev.name: ev for ev in self._events}

    def current_phase(self, time_seconds: float) -> str:
        """
        Classify deterministic multi-messenger phase based on simulation time t [s]:
        - t < 0.0           -> INSPIRAL
        - t == 0.0          -> MERGER
        - 0.0 < t < 1.7     -> POST_MERGER
        - 1.7 <= t < 3.7    -> GRB
        - 3.7 <= t < 150d   -> POST_MERGER
        - t >= 150.0 days   -> AFTERGLOW
        """
        t = float(time_seconds)
        if t < 0.0:
            return EventPhase.INSPIRAL.value
        elif t == 0.0:
            return EventPhase.MERGER.value
        elif 0.0 < t < self.grb_time:
            return EventPhase.POST_MERGER.value
        elif self.grb_time <= t < (self.grb_time + self.grb_duration):
            return EventPhase.GRB.value
        elif (self.grb_time + self.grb_duration) <= t < self.afterglow_peak_time:
            return EventPhase.POST_MERGER.value
        else:
            return EventPhase.AFTERGLOW.value

    def get_events(self) -> List[MessengerEvent]:
        """Return chronological list of pre-constructed timeline events."""
        return list(self._events)

    def get_event(self, name: str) -> MessengerEvent:
        """Get event by name. Raises KeyError if unknown."""
        if name not in self._event_map:
            raise KeyError(f"Unknown timeline event: '{name}'. Available events: {list(self._event_map.keys())}")
        return self._event_map[name]

    def events_up_to(self, time_seconds: float) -> List[MessengerEvent]:
        """Return all events with event.time <= time_seconds in chronological order."""
        t = float(time_seconds)
        return [ev for ev in self._events if ev.time <= t]

    def time_since_merger(self, time_seconds: float) -> float:
        """Return time elapsed since merger reference event [s]."""
        return float(time_seconds) - self.merger_time

    def time_since_grb(self, time_seconds: float) -> float:
        """Return time elapsed since GRB 170817A trigger event [s]."""
        return float(time_seconds) - self.grb_time

    def afterglow_time_days(self, time_seconds: float) -> float:
        """Convert simulation time in seconds to days post-merger."""
        return float(time_seconds) / day

    def is_grb_active(self, time_seconds: float) -> bool:
        """Check if prompt GRB emission is active at time_seconds."""
        t = float(time_seconds)
        return self.grb_time <= t < (self.grb_time + self.grb_duration)

    def is_afterglow_active(self, time_seconds: float) -> bool:
        """Check if broadband afterglow emission is active (t >= 1.0 day)."""
        t = float(time_seconds)
        return t >= day
