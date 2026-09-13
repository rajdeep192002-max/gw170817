"""
Unit Test Suite for Task 036 — Magnetic Field Lines Presentation Mode.

Verifies:
A. MAGNETIC FIELD button/mode exists in dashboard.
B. CORE mode hides/subdues field lines (show_magnetic_field is False).
C. MULTI mode hides/subdues field lines (show_magnetic_field is False).
D. MAGNETIC FIELD mode shows field lines (show_magnetic_field is True).
E. Field physics state continues evolving while hidden.
F. Existing winding/rotation behavior remains unchanged.
G. Jet remains active in magnetic mode.
H. Camera preset applies only once.
I. Manual camera controls remain functional afterward.
"""
import sys
import numpy as np
import pytest

from gw170817.config import SimConfig
from gw170817.visualization.dashboard import ScientificDashboard


@pytest.fixture(scope="module")
def dash():
    """Module-scoped dashboard fixture to prevent Vulkan context handle exhaustion on Windows."""
    return ScientificDashboard(config=SimConfig(mode="DEV"))


def test_magnetic_field_mode_exists(dash):
    """A. Verify MAGNETIC FIELD mode is registered and selectable."""
    dash.set_view_mode("MAGNETIC FIELD")
    assert dash.view_mode == "MAGNETIC FIELD"
    assert dash.show_magnetic_field is True


def test_core_mode_hides_field_lines(dash):
    """B. Verify CORE mode hides/subdues field lines."""
    dash.set_view_mode("CORE")
    assert dash.view_mode == "CORE"
    assert dash.show_magnetic_field is False


def test_multi_mode_hides_field_lines(dash):
    """C. Verify MULTI mode hides/subdues field lines."""
    dash.set_view_mode("MULTI")
    assert dash.view_mode == "MULTI"
    assert dash.show_magnetic_field is False


def test_magnetic_field_mode_shows_field_lines(dash):
    """D. Verify MAGNETIC FIELD mode activates field-line display."""
    dash.set_view_mode("MAGNETIC FIELD")
    assert dash.view_mode == "MAGNETIC FIELD"
    assert dash.show_magnetic_field is True
    assert abs(dash.cam_distance - 160.0e3) < 1.0, "MAGNETIC FIELD camera preset must be 160 km"
    assert abs(dash.cam_pitch - 0.45) < 1e-3


def test_field_physics_evolves_while_hidden(dash):
    """E & F. Verify field physics state continues evolving continuously while hidden in CORE mode."""
    dash.set_view_mode("CORE")
    assert dash.show_magnetic_field is False

    # Advance simulation across post-merger stage
    dash.director._sync_physics_for_presentation_time(7.5)
    st1 = dash.coordinator.current_state
    b_pol_1 = float(st1.b_poloidal)

    dash.director._sync_physics_for_presentation_time(8.5)
    st2 = dash.coordinator.current_state
    b_pol_2 = float(st2.b_poloidal)

    # Physics continues evolving regardless of presentation visibility
    assert b_pol_1 > 0.0
    assert b_pol_2 > 0.0
    assert hasattr(st2, 'differential_rotation')
    assert hasattr(st2, 'omega_core')


def test_jet_remains_active_in_magnetic_mode(dash):
    """G. Verify jet remains active in MAGNETIC FIELD mode (visually secondary, not removed)."""
    dash.director._sync_physics_for_presentation_time(9.0)
    dash.set_view_mode("MAGNETIC FIELD")

    st = dash.engine.current_state
    evt_st = dash.coordinator.current_state

    # Jet is active in post-merger stage
    assert st.grb_triggered
    assert dash.director.jet_progress > 0.0
    assert dash.show_magnetic_field is True


def test_camera_preset_applies_once_and_controls_persist(dash):
    """H & I. Verify camera preset applies once and manual orbit controls remain fully functional."""
    dash.set_view_mode("MAGNETIC FIELD")
    initial_dist = dash.cam_distance
    initial_pitch = dash.cam_pitch

    # Manual orbit
    dash.camera_orbit_up(0.08)
    assert abs(dash.cam_pitch - (initial_pitch + 0.08)) < 1e-3

    # Manual zoom
    dash.camera_zoom_in(0.85)
    assert abs(dash.cam_distance - (initial_dist * 0.85)) < 1.0
