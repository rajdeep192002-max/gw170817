"""
Visualization sub-package for GW170817 real-time scientific dashboard and GGUI renderer.
"""
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.dashboard import ScientificDashboard

__all__ = [
    "ParticleRenderer",
    "ScientificDashboard"
]
