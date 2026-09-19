"""
Comprehensive test suite for Task 046: Scientific Neutrino Transport Redesign.

Verifies:
1. NeutrinoModel data flow, dual-timescale cooling, species breakdown, and synchronized flags.
2. MultiMessengerCoordinator propagation of neutrino_luminosity_nux and neutrino_transport_active.
3. GPU neutrino radiation transport particle fields: 600 tracers, species counts (35%, 40%, 25%).
4. Outward radial transport streaming: dr/dt > 0 for all non-wrapped tracers (no inward motion).
5. Monotonic spatial volume density falloff from inner source to outer boundary (> 50x falloff).
6. Central engine clearance: no neutrino particles inside r < 16 km (BH shadow remains unobstructed).
7. Species color palettes: distinct Cyan, Teal-White, and Violet.
8. Dashboard visual hierarchy: restrained ejecta (0.08 km), small neutrino tracers (0.35 km),
   and canvas waveform line suppression in NEUTRINO mode.
"""
from unittest.mock import MagicMock, patch
import pytest
import math
import numpy as np

from gw170817.config import SimConfig
from gw170817.physics.neutrinos import NeutrinoModel, NeutrinoState
from gw170817.physics.remnant import RemnantModel, RemnantState
from gw170817.physics.disk import DiskModel, DiskState
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.multimessenger import MultiMessengerCoordinator, MultiMessengerEventState
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.dashboard import ScientificDashboard


def create_mock_dashboard():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)

    mock_window = MagicMock()
    mock_window.running = True
    mock_window.is_pressed.return_value = False
    mock_window.get_event.return_value = False
    mock_window.get_cursor_pos.return_value = (0.5, 0.5)

    mock_scene = MagicMock()
    mock_canvas = MagicMock()
    mock_gui = MagicMock()
    mock_window.get_gui.return_value = mock_gui

    with patch("taichi.ui.Window", return_value=mock_window), \
         patch("taichi.ui.Scene", return_value=mock_scene), \
         patch("taichi.ui.Canvas", return_value=mock_canvas):
        dashboard = ScientificDashboard(engine=engine, config=config)
        dashboard.window = mock_window
        dashboard.scene = mock_scene
        dashboard.canvas = mock_canvas
        return dashboard


# -----------------------------------------------------------------------------
# 1. Physics Model Tests
# -----------------------------------------------------------------------------

def test_neutrino_physics_dual_timescale():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    nu_model = NeutrinoModel(config)

    # Pre-merger: emission is inactive
    engine.set_inspiral_time(2.0)
    t_pre = engine.current_state.event_time
    rem_pre = engine.remnant.evaluate(engine.dynamics.inspiral_state, t_pre)
    disk_pre = engine.disk.evaluate(rem_pre, t_pre)
    st_pre = nu_model.evaluate(rem_pre, disk_pre, t_pre)

    assert not st_pre.is_active
    assert not st_pre.transport_active
    assert st_pre.luminosity_total == 0.0
    assert st_pre.luminosity_nue == 0.0
    assert st_pre.luminosity_nue_bar == 0.0
    assert st_pre.luminosity_nux == 0.0

    # Post-merger at t = 0.05 s (prompt peak)
    engine.set_post_merger_event_time(0.05)
    t_prompt = engine.current_state.event_time
    rem_prompt = engine.remnant.evaluate(engine.dynamics.inspiral_state, t_prompt)
    disk_prompt = engine.disk.evaluate(rem_prompt, t_prompt)
    st_prompt = nu_model.evaluate(rem_prompt, disk_prompt, t_prompt)

    assert st_prompt.is_active
    assert st_prompt.transport_active
    assert st_prompt.luminosity_total > 1.0e44
    assert math.isclose(
        st_prompt.luminosity_total,
        st_prompt.luminosity_nue + st_prompt.luminosity_nue_bar + st_prompt.luminosity_nux,
        rel_tol=1e-5
    )
    assert abs(st_prompt.luminosity_nue / st_prompt.luminosity_total - 0.35) < 1e-4
    assert abs(st_prompt.luminosity_nue_bar / st_prompt.luminosity_total - 0.40) < 1e-4
    assert abs(st_prompt.luminosity_nux / st_prompt.luminosity_total - 0.25) < 1e-4

    # Post-merger at t = 1.0 s (disk accretion cooling tail)
    engine.set_post_merger_event_time(1.0)
    t_disk = engine.current_state.event_time
    rem_disk = engine.remnant.evaluate(engine.dynamics.inspiral_state, t_disk)
    disk_disk = engine.disk.evaluate(rem_disk, t_disk)
    st_disk = nu_model.evaluate(rem_disk, disk_disk, t_disk)

    assert st_disk.is_active
    assert st_disk.transport_active
    assert st_disk.luminosity_total > 1.0e43, "Disk wind accretion tail should maintain physical luminosity"
    assert st_disk.luminosity_total < st_prompt.luminosity_total, "Luminosity must decay over time"
    assert 10.0 <= st_disk.mean_energy_mev <= 15.0


def test_multimessenger_state_population():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    coord = MultiMessengerCoordinator(engine=engine, config=config)

    # Pre-merger state
    st0 = coord.current_state
    assert hasattr(st0, "neutrino_luminosity_nux")
    assert hasattr(st0, "neutrino_transport_active")
    assert not st0.neutrino_is_active
    assert not st0.neutrino_transport_active

    # Post-merger state via evaluate_at_event_time
    st_pm = coord.evaluate_at_event_time(0.2)
    assert st_pm.neutrino_is_active
    assert st_pm.neutrino_transport_active
    assert st_pm.neutrino_luminosity_nux > 0.0
    assert math.isclose(
        st_pm.neutrino_luminosity,
        st_pm.neutrino_luminosity_nue + st_pm.neutrino_luminosity_nue_bar + st_pm.neutrino_luminosity_nux,
        rel_tol=1e-4
    )


# -----------------------------------------------------------------------------
# 2. Particle Renderer & GPU Transport Kernel Tests
# -----------------------------------------------------------------------------

def test_particle_renderer_neutrino_fields():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    renderer = ParticleRenderer(engine.psys)

    assert renderer.n_nu_particles == 600
    assert renderer.nu_pos.shape[0] == 600
    assert renderer.nu_colors.shape[0] == 600
    assert renderer.nu_dir.shape[0] == 600
    assert renderer.nu_phase.shape[0] == 600
    assert renderer.nu_species.shape[0] == 600
    assert renderer.nu_speed.shape[0] == 600

    # Verify species breakdown
    species_arr = renderer.nu_species.to_numpy()
    n_nue = np.sum(species_arr == 0)
    n_nuebar = np.sum(species_arr == 1)
    n_nux = np.sum(species_arr == 2)

    assert n_nue == 210  # 35% of 600
    assert n_nuebar == 240  # 40% of 600
    assert n_nux == 150  # 25% of 600
    assert n_nue + n_nuebar + n_nux == 600


def test_neutrino_radial_outward_transport():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    renderer = ParticleRenderer(engine.psys)

    # Step at t1 = 0.10 s
    renderer.update_nu_particles(event_time=0.10, nu_luminosity_w=2.0e45, is_active=True)
    pos1 = renderer.nu_pos.to_numpy()
    r1 = np.linalg.norm(pos1, axis=1)

    # Step at t2 = 0.15 s
    renderer.update_nu_particles(event_time=0.15, nu_luminosity_w=2.0e45, is_active=True)
    pos2 = renderer.nu_pos.to_numpy()
    r2 = np.linalg.norm(pos2, axis=1)

    # All tracers outside merger/BH horizon: r >= 16 km
    assert np.all(r1 >= 15.9e3), f"Minimum r1 = {np.min(r1):.1f} m, should be >= 16.0 km"
    assert np.all(r2 >= 15.9e3), f"Minimum r2 = {np.min(r2):.1f} m, should be >= 16.0 km"
    assert np.all(r1 <= 260.1e3), f"Maximum r1 = {np.max(r1):.1f} m, should be <= 260.0 km"
    assert np.all(r2 <= 260.1e3), f"Maximum r2 = {np.max(r2):.1f} m, should be <= 260.0 km"

    # Non-wrapped tracers must have strictly positive radial expansion dr/dt > 0
    non_wrapped = r2 > r1
    fraction_advancing = np.mean(non_wrapped)
    # With 0.05s delta, ~95%+ of tracers should advance radially outward without wrapping
    assert fraction_advancing > 0.90, f"Only {fraction_advancing*100:.1f}% advancing outward"

    # For advancing tracers, verify outward direction matches unit direction vector
    dirs = renderer.nu_dir.to_numpy()
    for i in np.where(non_wrapped)[0][:20]:
        unit_p = pos2[i] / np.linalg.norm(pos2[i])
        expected_dir = dirs[i]
        cos_sim = np.dot(unit_p, expected_dir)
        assert cos_sim > 0.999, f"Tracer {i} direction does not match radial unit vector"


def test_neutrino_density_falloff():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    renderer = ParticleRenderer(engine.psys)

    renderer.update_nu_particles(event_time=0.30, nu_luminosity_w=2.0e45, is_active=True)
    pos = renderer.nu_pos.to_numpy()
    r_km = np.linalg.norm(pos, axis=1) / 1.0e3

    # Shell 1: near engine (16 to 50 km)
    vol1 = (4.0 / 3.0) * np.pi * (50.0**3 - 16.0**3)
    count1 = np.sum((r_km >= 16.0) & (r_km < 50.0))
    dens1 = count1 / vol1

    # Shell 2: mid-field (50 to 150 km)
    vol2 = (4.0 / 3.0) * np.pi * (150.0**3 - 50.0**3)
    count2 = np.sum((r_km >= 50.0) & (r_km < 150.0))
    dens2 = count2 / vol2

    # Shell 3: outer boundary (150 to 260 km)
    vol3 = (4.0 / 3.0) * np.pi * (260.0**3 - 150.0**3)
    count3 = np.sum((r_km >= 150.0) & (r_km <= 260.0))
    dens3 = count3 / vol3

    assert dens1 > dens2 > dens3, f"Density not monotonically falling: {dens1:.2e} > {dens2:.2e} > {dens3:.2e}"
    ratio_inner_to_outer = dens1 / dens3
    assert ratio_inner_to_outer > 50.0, f"Density falloff ratio {ratio_inner_to_outer:.1f}x should exceed 50x"


def test_neutrino_species_colors():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    renderer = ParticleRenderer(engine.psys)

    renderer.update_nu_particles(event_time=0.20, nu_luminosity_w=2.0e45, is_active=True)
    colors = renderer.nu_colors.to_numpy()
    species = renderer.nu_species.to_numpy()

    # Species 0: nu_e (Electric Cyan: base [0.25, 0.75, 1.00] -> blue/green dominant)
    nue_idx = np.where(species == 0)[0][0]
    c_nue = colors[nue_idx]
    assert c_nue[2] > c_nue[0], "nu_e blue channel should exceed red"
    assert c_nue[1] > c_nue[0], "nu_e green channel should exceed red"

    # Species 1: anti-nu_e (Teal-White: base [0.45, 0.95, 0.85] -> strong green/blue, subtle red)
    nuebar_idx = np.where(species == 1)[0][0]
    c_nuebar = colors[nuebar_idx]
    assert c_nuebar[1] > c_nuebar[0], "anti-nu_e green channel should exceed red"

    # Species 2: nu_x (Violet-Indigo: base [0.70, 0.40, 0.95] -> red and blue dominant, low green)
    nux_idx = np.where(species == 2)[0][0]
    c_nux = colors[nux_idx]
    assert c_nux[0] > c_nux[1], "nu_x red channel should exceed green"
    assert c_nux[2] > c_nux[1], "nu_x blue channel should exceed green"


# -----------------------------------------------------------------------------
# 3. Dashboard Integration & Hierarchy Tests
# -----------------------------------------------------------------------------

def test_dashboard_neutrino_hierarchy_and_clean_viewport():
    dashboard = create_mock_dashboard()

    # Enter post-merger state at t = 0.5 s with presentation time advancing ejecta
    dashboard.engine.set_post_merger_event_time(0.5)
    dashboard.director._presentation_time = 8.0
    dashboard.coordinator._update_event_state()
    dashboard.set_view_mode("NEUTRINO")

    dashboard.scene.reset_mock()
    dashboard.canvas.reset_mock()
    dashboard.render_frame()

    # Check particles calls
    particle_calls = dashboard.scene.particles.call_args_list
    assert len(particle_calls) > 0

    # 1. Neutrino tracers rendered with small discrete radius 0.35e3
    nu_calls = [c for c in particle_calls if c[0][0] is dashboard.renderer.nu_pos]
    assert len(nu_calls) == 1, "Neutrino tracers should be rendered once in NEUTRINO mode"
    nu_radius = nu_calls[0][1].get("radius", 0.0)
    assert abs(nu_radius - 0.35e3) < 1.0, f"Neutrino radius should be 0.35 km, got {nu_radius/1e3:.2f} km"

    # 2. Ejecta fluid rendered with restrained micro-dots 0.08e3
    ej_calls = [c for c in particle_calls if c[0][0] is dashboard.renderer.ejecta_fluid_pos]
    assert len(ej_calls) == 1, "Ejecta fluid should be rendered in NEUTRINO mode as background context"
    ej_radius = ej_calls[0][1].get("radius", 0.0)
    assert abs(ej_radius - 0.08e3) < 1.0, f"Ejecta fluid radius should be 0.08 km, got {ej_radius/1e3:.2f} km"

    # 3. Remnant core rendered prominently (2.0 km)
    rem_calls = [c for c in particle_calls if c[0][0] is dashboard.renderer.remnant_pos]
    assert len(rem_calls) == 1
    rem_radius = rem_calls[0][1].get("radius", 0.0)
    assert abs(rem_radius - 2.0e3) < 1.0, f"Remnant radius should be 2.0 km, got {rem_radius/1e3:.2f} km"

    # 4. 2D Waveform lines on canvas should be suppressed in NEUTRINO mode
    canvas_lines_calls = dashboard.canvas.lines.call_args_list
    wf_rendered_on_canvas = any(
        c[0][0] is dashboard.renderer.waveform_vertices
        for c in canvas_lines_calls
    )
    assert not wf_rendered_on_canvas, "2D waveform lines on canvas should be suppressed in NEUTRINO mode"
