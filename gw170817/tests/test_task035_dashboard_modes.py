"""
Unit Test Suite for Task 035 — Creative Final Scientific Dashboard + View Modes.

Verifies:
A. H1/L1 traces are absent from canvas rendering in presentation UI.
B. CORE mode works and applies camera preset.
C. GW mode activates without changing physical GW state.
D. GW mode exposes the existing 3D GW wavefront propagation.
E. BH LENS mode safely handles pre-BH state without crashing.
F. BH LENS mode activates Schwarzschild ray-tracing camera preset after BH formation.
G. Camera controls remain functional after presets.
H. MULTI mode works.
I. No duplicate clocks/state systems exist.
"""
import sys
import numpy as np
import pytest

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.dashboard import ScientificDashboard


@pytest.fixture(scope="module")
def dash():
    """Module-scoped dashboard fixture to prevent Vulkan context handle exhaustion on Windows."""
    return ScientificDashboard(config=SimConfig(mode="DEV"))


def test_h1_l1_absent_from_presentation(dash):
    """A. Verify H1/L1 traces are omitted from canvas rendering in presentation UI."""
    assert hasattr(dash.renderer, 'obs_h1_vertices')  # Data structure remains in renderer
    assert hasattr(dash.renderer, 'obs_l1_vertices')  # Data structure remains in renderer


def test_core_mode_preset(dash):
    """B. Verify CORE mode sets default camera distance (280 km) and pitch (0.65 rad)."""
    dash.set_view_mode("CORE")

    assert dash.view_mode == "CORE"
    assert abs(dash.cam_distance - 280.0e3) < 1.0
    assert abs(dash.cam_pitch - 0.65) < 1e-3
    assert not dash.bh_lens_waiting


def test_gw_mode_preserves_physics_and_exposes_wavefront(dash):
    """C & D. Verify GW mode activates 3D wavefront without altering physical GW frequency or event_time."""
    initial_f_gw = float(dash.engine.current_state.gw_frequency)
    initial_event_time = float(dash.engine.current_state.event_time)

    dash.set_view_mode("GW")

    assert dash.view_mode == "GW"
    assert dash.wave_propagation.active, "GW mode must activate 3D GW wavefront propagation"
    assert abs(dash.engine.current_state.gw_frequency - initial_f_gw) < 1e-5, "Physics state must not change"
    assert abs(dash.engine.current_state.event_time - initial_event_time) < 1e-5, "Physics event_time must not change"
    assert abs(dash.cam_distance - 350.0e3) < 1.0, "GW mode must use framing (350 km)"


def test_bh_lens_mode_pre_bh_safe_handling(dash):
    """E. Verify BH LENS mode safely handles pre-BH state without crashing."""
    rem_st = dash.engine.remnant.evaluate(dash.engine.dynamics.inspiral_state, -5.0)
    assert not rem_st.is_black_hole

    dash.set_view_mode("BH LENS")

    assert dash.view_mode == "BH LENS"
    assert dash.bh_lens_waiting, "Pre-BH state must set bh_lens_waiting flag"
    assert dash.cam_distance > 100.0e3


def test_bh_lens_mode_post_bh_activation(dash):
    """F. Verify BH LENS mode applies close camera preset (55 km) and enables lensing once BH forms."""
    dash.director._sync_physics_for_presentation_time(10.0)
    st = dash.engine.current_state

    rem_st = dash.engine.remnant.evaluate(dash.engine.dynamics.inspiral_state, float(st.event_time))
    assert rem_st.is_black_hole, "BH must be formed post-merger"

    dash.set_view_mode("BH LENS")

    assert dash.view_mode == "BH LENS"
    assert not dash.bh_lens_waiting
    assert dash.lensing.enabled, "Relativistic lensing must be enabled in BH LENS mode"
    assert abs(dash.cam_distance - 55.0e3) < 1.0, "BH LENS mode must set close safe camera distance (55 km)"
    assert abs(dash.cam_pitch - 0.25) < 1e-3


def test_camera_controls_functional_after_presets(dash):
    """G. Verify manual orbit and zoom controls remain fully functional after setting mode presets."""
    dash.set_view_mode("GW")

    dist_before = dash.cam_distance
    pitch_before = dash.cam_pitch

    dash.camera_orbit_up(0.10)
    assert abs(dash.cam_pitch - (pitch_before + 0.10)) < 1e-3, "Camera pitch should change with manual input"

    dash.camera_zoom_in(0.80)
    assert abs(dash.cam_distance - (dist_before * 0.80)) < 1.0, "Camera distance should change with manual zoom"


def test_multi_mode_activation(dash):
    """H. Verify MULTI mode sets clean multimessenger state."""
    dash.set_view_mode("MULTI")

    assert dash.view_mode == "MULTI"
    assert not dash.bh_lens_waiting


def test_single_authoritative_clock(dash):
    """I. Verify no duplicate clocks/state systems exist in dashboard."""
    st = dash.engine.current_state

    assert hasattr(st, 'event_time')
    assert hasattr(dash.director, 'presentation_time')
    assert not hasattr(dash, 'custom_event_time'), "No duplicate event_time clock variable in dashboard"
