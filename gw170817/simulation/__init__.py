"""
Simulation sub-package for GW170817 particle system, backend management, merger dynamics, event timeline, integrated engine, and multi-messenger coordinator.
"""
from gw170817.simulation.particles import initialize_taichi, ParticleSystem
from gw170817.simulation.merger_dynamics import MergerDynamics
from gw170817.simulation.event_timeline import EventTimeline, MessengerEvent, EventPhase
from gw170817.simulation.engine import GW170817Simulation, SimulationState
from gw170817.simulation.multimessenger import MultiMessengerCoordinator, MultiMessengerEventState
from gw170817.simulation.demo_scenario import DemoScenario
from gw170817.simulation.demo_director import DemoDirector, DemoStage

__all__ = [
    "initialize_taichi",
    "ParticleSystem",
    "MergerDynamics",
    "EventTimeline",
    "MessengerEvent",
    "EventPhase",
    "GW170817Simulation",
    "SimulationState",
    "MultiMessengerCoordinator",
    "MultiMessengerEventState",
    "DemoScenario",
    "DemoDirector",
    "DemoStage"
]
