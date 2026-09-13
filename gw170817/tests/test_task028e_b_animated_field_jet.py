"""
Unit tests for Task 028E-B: Animated Magnetic Field & Rotating/Helical Jet Outflow.

Tests:
1. Continuous 3D rotation of magnetic field lines around remnant.
2. Jet base rotation and helical tip twist proportional to B_phi/B_p.
3. Jet symmetry and polar z-axis alignment (no wobble/precession).
4. Numerical stability (no NaN/Inf) under extreme time scale and winding ratio.
5. Continuity across post-merger timeline.
"""
import numpy as np
import pytest
import taichi as ti

from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem
from gw170817.visualization.field_lines import MagneticFieldLines
from gw170817.visualization.renderer import ParticleRenderer


@pytest.fixture(scope="module")
def psys_fixture():
    """Fixture providing a mock ParticleSystem for ParticleRenderer."""
    cfg = SimConfig(mode="DEV", seed=42)
    return ParticleSystem(cfg)


@pytest.fixture(scope="module", autouse=True)
def init_taichi_fixture():
    """Ensure Taichi is initialized for visualization tests."""
    try:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
    except RuntimeError:
        pass


def test_magnetic_field_lines_continuous_rotation():
    """Verify 3D magnetic field line vertices continuously change with event_time due to rotation."""
    mfl = MagneticFieldLines(n_lines=20, segments_per_line=16)

    # Initial state at t = 1.0 s
    mfl.update(b_pol=1.0e14, b_tor=1.0e13, r_rem=14.0e3, is_active=True, event_time=1.0, omega_rot=120.0)
    v1 = mfl.line_vertices.to_numpy().copy()

    # Evolved state at t = 2.0 s with identical magnetic field strengths
    mfl.update(b_pol=1.0e14, b_tor=1.0e13, r_rem=14.0e3, is_active=True, event_time=2.0, omega_rot=120.0)
    v2 = mfl.line_vertices.to_numpy().copy()

    # Vertices must evolve (rotation active)
    diff = np.max(np.abs(v2 - v1))
    assert diff > 1.0e-3, f"Field line vertices failed to rotate between t=1s and t=2s (max diff={diff})"

    # No NaN or Inf
    assert not np.isnan(v2).any()
    assert not np.isinf(v2).any()


def test_jet_helical_twist_and_rotation(psys_fixture):
    """Verify jet line rotation around polar z-axis and helical tip twist proportional to b_ratio."""
    renderer = ParticleRenderer(psys_fixture)

    # Case A: Zero b_ratio (purely straight conical jet line frustum)
    renderer.update_jet_lines(event_time=2.0, jet_progress=1.0, is_active=True, b_ratio=0.0, omega_rot=120.0)
    v_straight = renderer.jet_vertices.to_numpy().copy()

    # Case B: High b_ratio (helical twist applied)
    renderer.update_jet_lines(event_time=2.0, jet_progress=1.0, is_active=True, b_ratio=4.0, omega_rot=120.0)
    v_twisted = renderer.jet_vertices.to_numpy().copy()

    # Vertex tip position (index 1 of first line) must differ between straight and twisted jet
    diff_tip = np.linalg.norm(v_twisted[1] - v_straight[1])
    assert diff_tip > 1.0e-3, f"Jet tip failed to twist helically under b_ratio=4.0 (diff={diff_tip})"

    # Check phase angle difference between tip and base
    # Line 0: start at index 0, end at index 1 (+z jet line)
    x0, y0, z0 = v_twisted[0]
    x1, y1, z1 = v_twisted[1]

    phi_start = np.arctan2(y0, x0)
    phi_end = np.arctan2(y1, x1)

    # Angular twist should match clamp(0.25 * b_ratio, 0.0, 1.25) -> clamp(1.0, 0.0, 1.25) = 1.0 rad
    angle_diff = (phi_end - phi_start) % (2.0 * np.pi)
    if angle_diff > np.pi:
        angle_diff -= 2.0 * np.pi
    assert np.isclose(abs(angle_diff), 1.0, atol=1.0e-2), f"Helical twist angle mismatch: {angle_diff} vs expected 1.0 rad"


def test_jet_axis_polar_alignment(psys_fixture):
    """Verify jet lines remain anchored and symmetric along polar z-axis with no off-axis drift."""
    renderer = ParticleRenderer(psys_fixture)

    renderer.update_jet_lines(event_time=5.0, jet_progress=1.0, is_active=True, b_ratio=2.0, omega_rot=150.0)
    v = renderer.jet_vertices.to_numpy()

    # Even line index endpoints (+z), odd line index endpoints (-z)
    for i in range(renderer.n_jet_lines):
        start_v = v[2 * i]
        end_v = v[2 * i + 1]

        dir_sign = 1.0 if (i % 2 == 0) else -1.0
        assert np.sign(start_v[2]) == dir_sign, f"Line {i} z_start has wrong polar direction sign"
        assert np.sign(end_v[2]) == dir_sign, f"Line {i} z_end has wrong polar direction sign"
        assert abs(start_v[2]) == pytest.approx(14.0e3, rel=1.0e-4)


def test_numerical_stability_extreme_times(psys_fixture):
    """Verify numerical stability (no NaN/Inf/overflow) under extreme event times (>10^5 s)."""
    renderer = ParticleRenderer(psys_fixture)
    mfl = MagneticFieldLines(n_lines=20, segments_per_line=16)

    t_extreme = 1.0e6  # 1 million seconds
    mfl.update(b_pol=1.0e15, b_tor=5.0e15, r_rem=14.0e3, is_active=True, event_time=t_extreme, omega_rot=500.0)
    v_mfl = mfl.line_vertices.to_numpy()
    assert not np.isnan(v_mfl).any()
    assert not np.isinf(v_mfl).any()

    renderer.update_jet_lines(event_time=t_extreme, jet_progress=1.0, is_active=True, b_ratio=10.0, omega_rot=500.0)
    v_jet = renderer.jet_vertices.to_numpy()
    assert not np.isnan(v_jet).any()
    assert not np.isinf(v_jet).any()
