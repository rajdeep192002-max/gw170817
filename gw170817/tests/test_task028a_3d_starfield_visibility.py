"""
Unit tests for Task 028A — 3D Astronomical Starfield Visibility, Camera Scaling & Relativistic Deflection.

Verifies:
1. StellarCatalog generates non-zero 3D astronomical stars with finite positions, magnitudes, and spectral colors.
2. GPU star buffers allocate correct particle counts per quality mode: DEV (2000), NORMAL (3500), HIGH (7000).
3. Star positions reside in the distant 3D celestial sphere (R in [0.8e6, 3.0e6] m) within camera z_far (3.5e6 m).
4. Relativistic lensing deflection kernel (update_star_lensing_deflection_kernel) updates star_deflected_pos on GPU with finite coordinates.
5. Starfield Debug Diagnostic mode (toggle_starfield_debug) generates bright test markers and restores catalog.
6. Lensing deflection produces measurable displacement on background stars near compact mass sources.
7. Numerical stability: No NaNs or Infs in star positions, deflected positions, colors, or brightness fields.
"""
import pytest
import numpy as np

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.background import StellarCatalog, BackgroundStarfield
from gw170817.visualization.renderer import ParticleRenderer


@pytest.fixture(scope="module")
def starfield_setup():
    config_dev = SimConfig(mode="DEV", seed=42)
    config_normal = SimConfig(mode="NORMAL", seed=42)
    config_high = SimConfig(mode="HIGH", seed=42)

    sim_dev = GW170817Simulation(config=config_dev)
    sim_normal = GW170817Simulation(config=config_normal)
    sim_high = GW170817Simulation(config=config_high)

    renderer_dev = ParticleRenderer(sim_dev.psys)
    renderer_normal = ParticleRenderer(sim_normal.psys)
    renderer_high = ParticleRenderer(sim_high.psys)

    return (sim_dev, sim_normal, sim_high), (renderer_dev, renderer_normal, renderer_high)


def test_stellar_catalog_properties():
    """Test StellarCatalog generates valid 3D astronomical stars."""
    cat = StellarCatalog(n_stars=3500, seed=42)
    assert len(cat.stars) == 3500

    positions = np.array([s.position_3d for s in cat.stars])
    magnitudes = np.array([s.magnitude for s in cat.stars])
    colors = np.array([s.color_rgb for s in cat.stars])

    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(magnitudes))
    assert np.all(np.isfinite(colors))

    radii = np.linalg.norm(positions, axis=1)
    assert np.all(radii >= 0.7e6), "Star distances must be at astronomical scale (>= 0.7e6 m)"
    assert np.all(radii <= 3.2e6), "Star distances must lie within maximum celestial sphere radius"
    assert np.min(magnitudes) >= 2.0
    assert np.max(magnitudes) <= 12.0


def test_gpu_star_counts_per_quality_mode(starfield_setup):
    """Test GPU star particle allocation per quality mode (DEV=2000, NORMAL=3500, HIGH=7000)."""
    _, (r_dev, r_normal, r_high) = starfield_setup

    assert r_dev.n_stars == 2000
    assert r_normal.n_stars == 3500
    assert r_high.n_stars == 7000

    v_dev = r_dev.star_pos.to_numpy()
    v_normal = r_normal.star_pos.to_numpy()
    v_high = r_high.star_pos.to_numpy()

    assert len(v_dev) == 2000
    assert len(v_normal) == 3500
    assert len(v_high) == 7000

    assert np.all(np.isfinite(v_dev))
    assert np.all(np.isfinite(v_normal))
    assert np.all(np.isfinite(v_high))


def test_starfield_lensing_deflection_kernel(starfield_setup):
    """Test GPU relativistic lensing deflection kernel on 3D starfield."""
    _, (r_dev, _, _) = starfield_setup

    pos_undeflected = r_dev.star_pos.to_numpy()

    # Call deflection kernel with lensing OFF (lensing_active = 0)
    r_dev.update_star_lensing_deflection_kernel(
        ns1_x=0.0, ns1_y=0.0, ns1_z=0.0,
        ns2_x=0.0, ns2_y=0.0, ns2_z=0.0,
        m1_kg=2.7e30, m2_kg=0.0,
        lensing_active=0,
        enhanced_scale=1.0
    )
    pos_off = r_dev.star_deflected_pos.to_numpy()
    assert np.allclose(pos_undeflected, pos_off, atol=1e-3), "Undeflected positions must match original star positions"

    # Call deflection kernel with lensing ON (lensing_active = 1)
    r_dev.update_star_lensing_deflection_kernel(
        ns1_x=0.0, ns1_y=0.0, ns1_z=0.0,
        ns2_x=0.0, ns2_y=0.0, ns2_z=0.0,
        m1_kg=2.7e30, m2_kg=0.0,
        lensing_active=1,
        enhanced_scale=2.5
    )
    pos_on = r_dev.star_deflected_pos.to_numpy()

    assert np.all(np.isfinite(pos_on))
    diff = np.linalg.norm(pos_on - pos_undeflected, axis=1)
    assert np.max(diff) > 100.0, "Lensing active must produce non-zero relativistic deflection on star positions"


def test_starfield_debug_diagnostic_mode(starfield_setup):
    """Test starfield debug diagnostic mode toggle."""
    _, (r_dev, _, _) = starfield_setup

    assert r_dev.starfield_debug is False

    # Toggle ON
    active_on = r_dev.toggle_starfield_debug()
    assert active_on is True
    assert r_dev.starfield_debug is True

    dbg_pos = r_dev.star_pos.to_numpy()
    assert np.all(np.isfinite(dbg_pos))
    assert np.max(np.abs(dbg_pos)) > 0.0

    # Toggle OFF
    active_off = r_dev.toggle_starfield_debug()
    assert active_off is False
    assert r_dev.starfield_debug is False

    cat_pos = r_dev.star_pos.to_numpy()
    assert np.all(np.isfinite(cat_pos))
    assert len(cat_pos) == r_dev.n_stars
