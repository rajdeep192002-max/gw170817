"""
Simulation sub-package for GW170817 particle system, backend management, merger dynamics, event timeline, and integrated simulation engine.
"""
from gw170817.simulation.particles import initialize_taichi, ParticleSystem
from gw170817.simulation.merger_dynamics import MergerDynamics
from gw170817.simulation.event_timeline import EventTimeline, MessengerEvent, EventPhase
from gw170817.simulation.engine import GW170817Simulation, SimulationState

__all__ = [
    "initialize_taichi",
    "ParticleSystem",
    "MergerDynamics",
    "EventTimeline",
    "MessengerEvent",
    "EventPhase",
    "GW170817Simulation",
    "SimulationState"
]
