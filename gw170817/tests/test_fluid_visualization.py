"""
Unit tests for Fluid Ejecta Visualization, Starfield Determinism, and Jet Propagation.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.background import BackgroundStarfield


def test_fluid_ejecta_components():
    cfg = SimConfig()
    psys = ParticleSystem(cfg)
    renderer = ParticleRenderer(psys)

    # Verify 2500 fluid particles initialized
    assert renderer.n_ejecta_fluid_particles == 2500

    # Test update of fluid ejecta
    renderer.update_ejecta_fluid(event_time=0.5, ejecta_progress=0.5, is_active=True)

    # Check component assignment: 0=Blue, 1=Purple, 2=Red
    comp_arr = renderer.ejecta_fluid_comp.to_numpy()
    assert 0 in comp_arr
    assert 1 in comp_arr
    assert 2 in comp_arr


def test_starfield_determinism():
    bg1 = BackgroundStarfield(width=256, height=144)
    bg2 = BackgroundStarfield(width=256, height=144)

    img1 = bg1.sky_texture.to_numpy()
    img2 = bg2.sky_texture.to_numpy()

    # Starfield texture generation must be strictly deterministic between calls
    assert np.allclose(img1, img2, atol=1.0e-5)


def test_jet_line_expansion():
    cfg = SimConfig()
    psys = ParticleSystem(cfg)
    renderer = ParticleRenderer(psys)

    # Use event_times after jet_delay (default 1.7 s) so jet propagation is non-zero
    # early: 2.0 s post-merger (0.3 s propagation)
    # late:  10.0 s post-merger (8.3 s propagation)
    renderer.update_jet_lines(event_time=2.0, jet_progress=0.2, is_active=True, jet_delay=1.7)
    verts_early = renderer.jet_vertices.to_numpy()

    renderer.update_jet_lines(event_time=10.0, jet_progress=0.8, is_active=True, jet_delay=1.7)
    verts_late = renderer.jet_vertices.to_numpy()

    # Jet shock front propagates outward (z-coordinates expand)
    max_z_early = np.max(np.abs(verts_early[:, 2]))
    max_z_late = np.max(np.abs(verts_late[:, 2]))

    assert max_z_late > max_z_early, (
        f"Jet did not propagate: max|z| early={max_z_early:.1f}m, late={max_z_late:.1f}m"
    )
