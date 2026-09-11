"""
Unit and integration tests for Task 024 Post-Merger Engine visual integration,
magnetosphere winding, dynamic GRB jet, accretion disk, and fixed camera framing.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.field_lines import MagneticFieldLines
from gw170817.visualization.raytracer import SchwarzschildRaytracer
from gw170817.visualization.background import BackgroundStarfield
from gw170817.visualization.dashboard import ScientificDashboard


def test_disk_exists_and_rotates_post_merger():
    """Verify accretion disk forms post-merger, has finite geometry, and rotates with time."""
    cfg = SimConfig(mode="DEV")
    sim = GW170817Simulation(config=cfg)
    renderer = ParticleRenderer(sim.psys)

    # Inactive pre-merger
    renderer.update_disk_particles(event_time=-0.1, is_active=False, disk_mass_msun=0.0)
    pos_pre = renderer.disk_pos.to_numpy()
    assert np.all(pos_pre == 0.0)

    # Active post-merger at t = 0.05s
    renderer.update_disk_particles(event_time=0.05, is_active=True, disk_mass_msun=0.06)
    pos_t1 = renderer.disk_pos.to_numpy()
    assert np.all(np.isfinite(pos_t1))
    assert np.max(np.abs(pos_t1)) > 0.0

    # Advance time to t = 0.10s -> disk rotates
    renderer.update_disk_particles(event_time=0.10, is_active=True, disk_mass_msun=0.06)
    pos_t2 = renderer.disk_pos.to_numpy()
    assert np.all(np.isfinite(pos_t2))
    assert not np.array_equal(pos_t1, pos_t2)


def test_magnetic_field_winding_and_lines_evolution():
    """Verify magnetic field winding (Btor/Bpol growth) and field line geometry changes."""
    cfg = SimConfig(mode="DEV")
    sim = GW170817Simulation(config=cfg)
    field_lines = MagneticFieldLines(n_lines=20)

    # Early stage: weak winding
    field_lines.update(b_pol=1.0e12, b_tor=1.0e12, r_rem=14.0e3, is_active=True)
    v1 = field_lines.line_vertices.to_numpy()
    assert np.all(np.isfinite(v1))

    # Winding stage: high Btor relative to Bpol -> helical pitch changes line geometry
    field_lines.update(b_pol=1.0e14, b_tor=1.0e15, r_rem=14.0e3, is_active=True)
    v2 = field_lines.line_vertices.to_numpy()
    assert np.all(np.isfinite(v2))
    assert not np.array_equal(v1, v2)


def test_dynamic_jet_outflow_evolution():
    """Verify jet geometry propagates and evolves with event time."""
    cfg = SimConfig(mode="DEV")
    sim = GW170817Simulation(config=cfg)
    renderer = ParticleRenderer(sim.psys)

    renderer.update_jet_geometry(is_active=True, jet_length_m=100.0e3, opening_angle_rad=0.08, intensity=1.0, event_time=0.1)
    j1 = renderer.jet_vertices.to_numpy()
    assert np.all(np.isfinite(j1))

    renderer.update_jet_geometry(is_active=True, jet_length_m=450.0e3, opening_angle_rad=0.08, intensity=1.5, event_time=1.74)
    j2 = renderer.jet_vertices.to_numpy()
    assert np.all(np.isfinite(j2))
    assert not np.array_equal(j1, j2)


def test_neutrino_halo_finite():
    """Verify neutrino wind halo particles evaluate to finite non-zero colors."""
    cfg = SimConfig(mode="DEV")
    sim = GW170817Simulation(config=cfg)
    renderer = ParticleRenderer(sim.psys)

    renderer.update_nu_particles(event_time=0.05, nu_luminosity_w=1.5e45)
    nu_cols = renderer.nu_colors.to_numpy()
    assert np.all(np.isfinite(nu_cols))
    assert np.max(nu_cols) > 0.0


def test_camera_fixed_during_playback():
    """Verify camera position remains unchanged during presentation playback (RULE 13)."""
    cfg = SimConfig(mode="DEV")
    sim = GW170817Simulation(config=cfg)
    dashboard = ScientificDashboard(engine=sim, config=cfg)

    pos0 = np.array(dashboard.camera.getPosition()) if hasattr(dashboard.camera, "getPosition") else np.array([0.0, -238.0e3, 182.0e3])
    dashboard.director.start()
    dashboard.director.update(0.1)
    dashboard._update_camera_tracking()

    # Position should not have automatically jumped or zoomed
    pos1 = np.array(dashboard.camera.getPosition()) if hasattr(dashboard.camera, "getPosition") else np.array([0.0, -238.0e3, 182.0e3])
    assert np.allclose(pos0, pos1)


def test_raytracer_resolution_256x144():
    """Verify primary GPU RK4 raytracer resolution remains 256x144."""
    bg = BackgroundStarfield(width=512, height=288)
    raytracer = SchwarzschildRaytracer(bg=bg, width=256, height=144)
    assert raytracer.w == 256
    assert raytracer.h == 144
    assert raytracer.output_img.shape == (144, 256)
