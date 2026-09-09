"""
Simulation sub-package for GW170817 particle system, backend management, and merger dynamics.
"""
from gw170817.simulation.particles import initialize_taichi, ParticleSystem
from gw170817.simulation.merger_dynamics import MergerDynamics

__all__ = ["initialize_taichi", "ParticleSystem", "MergerDynamics"]
