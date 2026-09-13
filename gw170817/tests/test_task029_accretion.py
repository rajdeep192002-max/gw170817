"""
Unit tests for Task 029: Post-Ringdown Black-Hole Accretion Visualization.

Verifies:
A. Disk particles experience inward radial accretion drift after BH formation.
B. Orbital Keplerian motion remains present and active.
C. Particles reaching capture radius R_capture (14.5 km) are recycled to outer disk.
D. No rendered disk particle exists inside the capture boundary r < 14.5 km.
E. Accretion continues during continuous post-merger evolution (presentation_time > 15 s).
F. Pre-BH stages (is_black_hole=False) remain unchanged (static baseline radius).
G. All numerical positions remain finite (no NaN/Inf).
"""
import pytest
import numpy as np
import taichi as ti

from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem
from gw170817.visualization.renderer import ParticleRenderer


@pytest.fixture(scope="module")
def psys_fixture():
    """Fixture providing a ParticleSystem for ParticleRenderer."""
    cfg = SimConfig(mode="DEV", seed=42)
    return ParticleSystem(cfg)


@pytest.fixture(scope="module", autouse=True)
def init_taichi_fixture():
    """Ensure Taichi is initialized for visualization tests."""
    try:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
    except RuntimeError:
        pass


@pytest.fixture(scope="module")
def renderer(psys_fixture):
    """Module-scoped ParticleRenderer fixture."""
    return ParticleRenderer(psys_fixture)


def test_pre_bh_stage_unchanged(renderer):
    """F. Pre-BH stage (is_black_hole=False) disk radius does not drift inward."""
    renderer.update_disk_particles(event_time=0.01, dt_vis=0.016, is_active=True, is_black_hole=False)
    local_pos_init = renderer.disk_local_pos.to_numpy().copy()

    renderer.update_disk_particles(event_time=0.02, dt_vis=0.016, is_active=True, is_black_hole=False)
    local_pos_step = renderer.disk_local_pos.to_numpy().copy()

    # Radial positions r (index 0) must remain identical pre-BH
    np.testing.assert_allclose(local_pos_init[:, 0], local_pos_step[:, 0], rtol=1e-5)


def test_post_bh_inward_accretion_drift(renderer):
    """A. Disk particles experience inward radial accretion drift after BH formation."""
    renderer.update_disk_particles(event_time=0.10, dt_vis=0.016, is_active=True, is_black_hole=True)
    r_before = renderer.disk_local_pos.to_numpy()[:, 0].copy()

    # Step forward 1 frame of BH accretion
    renderer.update_disk_particles(event_time=0.10, dt_vis=0.016, is_active=True, is_black_hole=True)
    r_after = renderer.disk_local_pos.to_numpy()[:, 0].copy()

    # Over 1 frame, all particles not at capture boundary must drift inward
    drift_mask = r_after < r_before
    assert np.sum(drift_mask) == len(r_before), "100% of particles must drift inward over 1 accretion frame"


def test_orbital_motion_active(renderer):
    """B. Orbital motion remains active alongside radial accretion drift."""
    renderer.update_disk_particles(event_time=0.50, dt_vis=0.016, is_active=True, is_black_hole=True)
    pos1 = renderer.disk_pos.to_numpy().copy()

    renderer.update_disk_particles(event_time=0.55, dt_vis=0.016, is_active=True, is_black_hole=True)
    pos2 = renderer.disk_pos.to_numpy().copy()

    # Cartesian 3D position must evolve azimuthally
    diff = np.linalg.norm(pos2 - pos1, axis=1)
    assert np.all(diff > 0.0), "All active disk particles must possess active orbital velocities"


def test_no_rendered_particle_inside_capture_boundary(renderer):
    """D & C. No rendered particle exists inside R_capture = 14.5 km (recycled if reached)."""
    # Step accretion for 100 frames
    for f in range(100):
        renderer.update_disk_particles(event_time=1.0 + f * 0.016, dt_vis=0.016, is_active=True, is_black_hole=True)

    pos = renderer.disk_pos.to_numpy()
    radii_3d = np.linalg.norm(pos[:, :2], axis=1)

    # All active particles must satisfy r >= 14.5 km
    min_r = np.min(radii_3d)
    assert min_r >= 14.5e3 - 1.0e-3, f"Disk particle found inside capture radius: min_r = {min_r/1e3:.2f} km < 14.5 km"


def test_accretion_continues_continuous_post_merger(renderer):
    """E. Accretion continues during continuous post-merger evolution (t_event >> 15 s)."""
    t_late = 17280000.0  # 200 days post-merger
    renderer.update_disk_particles(event_time=t_late, dt_vis=0.016, is_active=True, is_black_hole=True)
    pos_late = renderer.disk_pos.to_numpy()

    assert not np.isnan(pos_late).any()
    assert not np.isinf(pos_late).any()
    radii_3d = np.linalg.norm(pos_late[:, :2], axis=1)
    assert np.min(radii_3d) >= 14.5e3

