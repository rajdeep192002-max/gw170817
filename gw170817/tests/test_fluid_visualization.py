"""
Unit tests for Fluid Ejecta Visualization, Starfield Determinism, and Jet Propagation.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem, initialize_taichi
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.background import BackgroundStarfield
from gw170817.simulation.demo_director import DemoDirector


initialize_taichi()


@pytest.mark.parametrize(
    ("mode", "expected_samples", "expected_physics_particles"),
    [("DEV", 6_000, 20_000), ("NORMAL", 16_000, 75_000), ("HIGH", 32_000, 100_000)],
)
def test_granular_ejecta_sample_counts_preserve_physics_resolution(
    mode, expected_samples, expected_physics_particles
):
    cfg = SimConfig(mode=mode, seed=42)
    psys = ParticleSystem(cfg)
    renderer = ParticleRenderer(psys)

    assert renderer.n_ejecta_fluid_particles == expected_samples
    assert psys.max_particles == expected_physics_particles

    renderer.update_ejecta_fluid(
        event_time=0.5, ejecta_progress=0.5, is_active=True, ejecta_mass_fraction=0.02
    )

    # Check component assignment: 0=Blue, 1=Purple, 2=Red
    comp_arr = renderer.ejecta_fluid_comp.to_numpy()
    assert 0 in comp_arr
    assert 1 in comp_arr
    assert 2 in comp_arr
    assert np.all(np.isfinite(renderer.ejecta_fluid_pos.to_numpy()))
    radii = renderer.ejecta_fluid_radius.to_numpy()
    assert np.all(np.isfinite(radii))
    assert np.all((radii >= 0.45e3) & (radii <= 0.80e3))


def test_granular_ejecta_activation_and_deterministic_seeding():
    cfg = SimConfig(mode="DEV", seed=2468)
    renderer_a = ParticleRenderer(ParticleSystem(cfg))
    renderer_b = ParticleRenderer(ParticleSystem(SimConfig(mode="DEV", seed=2468)))

    np.testing.assert_allclose(
        renderer_a.ejecta_fluid_dir.to_numpy(), renderer_b.ejecta_fluid_dir.to_numpy()
    )
    np.testing.assert_allclose(
        renderer_a.ejecta_fluid_radial_fill.to_numpy(), renderer_b.ejecta_fluid_radial_fill.to_numpy()
    )

    renderer_a.update_ejecta_fluid(event_time=0.0, ejecta_progress=0.0, is_active=True)
    assert np.all(renderer_a.ejecta_fluid_pos.to_numpy()[:, 2] < -1.0e8)

    renderer_a.update_ejecta_fluid(event_time=0.02, ejecta_progress=0.10, is_active=True)
    assert np.any(renderer_a.ejecta_fluid_pos.to_numpy()[:, 2] > -1.0e8)


@pytest.mark.parametrize(
    ("mode", "expected_thermal_samples"),
    [("DEV", 1_500), ("NORMAL", 4_000), ("HIGH", 8_000)],
)
def test_thermal_emission_samples_reuse_granular_ejecta_without_physics_growth(
    mode, expected_thermal_samples
):
    cfg = SimConfig(mode=mode, seed=42)
    psys = ParticleSystem(cfg)
    renderer = ParticleRenderer(psys)

    assert renderer.n_ejecta_thermal_particles == expected_thermal_samples
    assert psys.max_particles == cfg.n_particles

    renderer.update_ejecta_fluid(0.10, 0.50, is_active=True, ejecta_mass_fraction=0.02)
    renderer.update_ejecta_thermal_emission(
        event_time=0.10,
        ejecta_progress=0.50,
        radioactive_heating_rate=2.0e10,
        opacity_mean=0.7,
        mean_ye=0.25,
        kilonova_luminosity=1.0e34,
        is_active=True,
    )
    assert np.all(np.isfinite(renderer.ejecta_thermal_pos.to_numpy()))
    assert np.all(np.isfinite(renderer.ejecta_thermal_colors.to_numpy()))
    assert np.any(renderer.ejecta_thermal_pos.to_numpy()[:, 2] > -1.0e8)


def test_thermal_emission_uses_existing_state_and_respects_merger_start():
    renderer = ParticleRenderer(ParticleSystem(SimConfig(mode="DEV", seed=2468)))

    renderer.update_ejecta_fluid(0.0, 0.0, is_active=True)
    renderer.update_ejecta_thermal_emission(
        event_time=0.0, ejecta_progress=0.0, radioactive_heating_rate=2.0e10,
        opacity_mean=0.7, mean_ye=0.25, kilonova_luminosity=1.0e34, is_active=True,
    )
    assert np.all(renderer.ejecta_thermal_pos.to_numpy()[:, 2] < -1.0e8)

    renderer.update_ejecta_fluid(0.10, 0.50, is_active=True)
    renderer.update_ejecta_thermal_emission(
        event_time=0.10, ejecta_progress=0.50, radioactive_heating_rate=1.0e6,
        opacity_mean=3.0, mean_ye=0.10, kilonova_luminosity=0.0, is_active=True,
    )
    dim_colors = renderer.ejecta_thermal_colors.to_numpy()
    renderer.update_ejecta_thermal_emission(
        event_time=0.10, ejecta_progress=0.50, radioactive_heating_rate=2.0e10,
        opacity_mean=0.2, mean_ye=0.40, kilonova_luminosity=1.0e35, is_active=True,
    )
    bright_colors = renderer.ejecta_thermal_colors.to_numpy()
    assert np.any(renderer.ejecta_thermal_pos.to_numpy()[:, 2] > -1.0e8)
    assert np.sum(bright_colors) > np.sum(dim_colors)


def test_granular_ejecta_respects_progressive_merger_timing():
    cfg = SimConfig(mode="DEV", seed=42)
    director = DemoDirector(config=cfg)
    renderer = ParticleRenderer(director.coordinator.engine.psys)

    # At exactly 6.0 s, 026C.2 maps to event time zero and ejecta
    # progress is zero, so no granular samples may be visible.
    state_at_start = director.jump_to_stage_index(2)
    renderer.update_ejecta_fluid(
        event_time=state_at_start.event_time,
        ejecta_progress=director.ejecta_progress,
        is_active=True,
        ejecta_mass_fraction=state_at_start.ejecta_fraction,
    )
    assert np.all(renderer.ejecta_fluid_pos.to_numpy()[:, 2] < -1.0e8)
    renderer.update_ejecta_thermal_emission(
        event_time=state_at_start.event_time,
        ejecta_progress=director.ejecta_progress,
        radioactive_heating_rate=state_at_start.ejecta_radioactive_heating_rate,
        opacity_mean=state_at_start.ejecta_opacity_mean,
        mean_ye=state_at_start.ejecta_mean_ye,
        kilonova_luminosity=state_at_start.kilonova_luminosity,
        is_active=True,
    )
    assert np.all(renderer.ejecta_thermal_pos.to_numpy()[:, 2] < -1.0e8)

    # The same authoritative director mapping activates the buffer during
    # the progressive 6–9 s merger interval.
    state_later = director._sync_physics_for_presentation_time(6.5)
    renderer.update_ejecta_fluid(
        event_time=state_later.event_time,
        ejecta_progress=director.ejecta_progress,
        is_active=True,
        ejecta_mass_fraction=state_later.ejecta_fraction,
    )
    renderer.update_ejecta_thermal_emission(
        event_time=state_later.event_time,
        ejecta_progress=director.ejecta_progress,
        radioactive_heating_rate=state_later.ejecta_radioactive_heating_rate,
        opacity_mean=state_later.ejecta_opacity_mean,
        mean_ye=state_later.ejecta_mean_ye,
        kilonova_luminosity=state_later.kilonova_luminosity,
        is_active=True,
    )
    assert director.ejecta_progress > 0.0
    assert np.any(renderer.ejecta_fluid_pos.to_numpy()[:, 2] > -1.0e8)
    assert np.any(renderer.ejecta_thermal_pos.to_numpy()[:, 2] > -1.0e8)


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
