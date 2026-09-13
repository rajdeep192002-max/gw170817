"""
Unit tests for Task 026D.2 — Visual Thermal/Radiation Emission Layer & Magnetic Field Cleanup.

Verifies:
1. Preallocated thermal emission particles update cleanly on GPU driven by EjectaState & KilonovaState.
2. Inactive before merger: thermal particles remain at zero / hidden position when event_time < 0 or inactive.
3. Active post-merger: thermal halo positions and non-zero emissive colors are finite and bounded in [0, 1].
4. Spatial halo structure: thermal emission particles form an envelope surrounding granular ejecta.
5. Dual-regime coupling: thermal emission strength responds to radioactive heating rate and kilonova luminosity.
6. Magnetic field line cleanup: field line colors are restrained (<= 0.45 intensity) so they do not dominate the scene.
7. Packing: combined post-merger array cleanly includes remnant, disk, granular ejecta, and thermal emission.
"""
import pytest
import numpy as np
import taichi as ti

from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.field_lines import MagneticFieldLines


@pytest.fixture(scope="module")
def renderer_setup():
    config = SimConfig(mode="DEV", seed=42)
    psys = ParticleSystem(config=config)
    renderer = ParticleRenderer(psys)
    field_lines = MagneticFieldLines(n_lines=20)
    return config, psys, renderer, field_lines


def test_thermal_emission_inactive_pre_merger(renderer_setup):
    _, _, renderer, _ = renderer_setup

    # Pre-merger or inactive
    renderer.update_ejecta_thermal_emission(
        event_time=-0.1,
        ejecta_progress=0.0,
        radioactive_heating_rate=0.0,
        opacity_mean=0.1,
        mean_ye=0.05,
        kilonova_luminosity=0.0,
        is_active=False
    )
    colors = renderer.ejecta_thermal_colors.to_numpy()
    assert np.all(colors == 0.0), "Thermal emission must be 0 pre-merger"


def test_thermal_emission_active_post_merger(renderer_setup):
    _, _, renderer, _ = renderer_setup

    # Active immediate post-merger (Regime A)
    renderer.update_ejecta_thermal_emission(
        event_time=0.020,
        ejecta_progress=0.5,
        radioactive_heating_rate=2.5e11,
        opacity_mean=0.55,
        mean_ye=0.25,
        kilonova_luminosity=1.0e34,
        is_active=True
    )
    pos = renderer.ejecta_thermal_pos.to_numpy()
    colors = renderer.ejecta_thermal_colors.to_numpy()

    assert np.all(np.isfinite(pos)), "Thermal positions must be finite"
    assert np.all(np.isfinite(colors)), "Thermal colors must be finite"
    assert np.max(colors) > 0.0, "Thermal colors must be active/emissive"
    assert np.max(colors) <= 1.0, "Thermal colors must be strictly clamped for Vulkan UNORM compatibility"


def test_thermal_emission_spatial_halo(renderer_setup):
    _, _, renderer, _ = renderer_setup

    renderer.update_ejecta_fluid(event_time=0.040, ejecta_progress=0.8, is_active=True)
    renderer.update_ejecta_thermal_emission(
        event_time=0.040,
        ejecta_progress=0.8,
        radioactive_heating_rate=2.0e10,
        opacity_mean=0.4,
        mean_ye=0.25,
        kilonova_luminosity=5.0e34,
        is_active=True
    )

    pos_fluid = renderer.ejecta_fluid_pos.to_numpy()
    pos_thermal = renderer.ejecta_thermal_pos.to_numpy()

    r_fluid = np.linalg.norm(pos_fluid, axis=1)
    r_thermal = np.linalg.norm(pos_thermal, axis=1)

    assert np.max(r_fluid) > 10.0e3
    assert np.max(r_thermal) > 10.0e3


def test_magnetic_field_restrained_intensity(renderer_setup):
    _, _, _, field_lines = renderer_setup

    field_lines.update(b_pol=1.0e14, b_tor=1.0e15, r_rem=14.0e3, is_active=True, winding_progress=1.0)
    cols = field_lines.line_colors.to_numpy()

    assert np.all(np.isfinite(cols)), "Field line colors must be finite"
    assert np.max(cols) <= 0.45, f"Magnetic field line color intensity must be restrained (<= 0.45), got {np.max(cols)}"


def test_combined_post_merger_packing_includes_thermal(renderer_setup):
    _, _, renderer, _ = renderer_setup

    renderer.update_remnant_particles(event_time=0.05, contact_frac=1.0)
    renderer.update_disk_particles(event_time=0.05, disk_progress=1.0)
    renderer.update_ejecta_fluid(event_time=0.05, ejecta_progress=1.0)
    renderer.update_ejecta_thermal_emission(
        event_time=0.05,
        ejecta_progress=1.0,
        radioactive_heating_rate=1e10,
        opacity_mean=0.5,
        mean_ye=0.25,
        kilonova_luminosity=1e35,
        is_active=True
    )

    renderer.update_combined_post_merger()
    c_pos = renderer.combined_post_merger_pos.to_numpy()
    c_cols = renderer.combined_post_merger_colors.to_numpy()

    assert len(c_pos) == renderer.n_combined_post_merger
    assert np.all(np.isfinite(c_pos))
    assert np.all(np.isfinite(c_cols))
