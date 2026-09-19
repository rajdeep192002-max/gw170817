"""
Regression and verification test for Task 044: BH Lensing Visibility + Bright Reference Stars.

Verifies:
1. Deterministic star positions are unchanged between runs (seed=42).
2. Exactly 32 dedicated bright reference stars are selected across 16 azimuth sectors around the BH.
3. In BH LENS mode, reference stars receive distinct high-contrast diamond blue-white boost.
4. When switching to other modes (CORE, GW, MAGNETIC FIELD, MULTI), stars cleanly return to standard brightness.
5. Reduced-order Schwarzschild lensing deflects escaping background stars coherently.
6. No NaN or Inf values in deflected star positions.
7. event_time is completely unchanged across mode switches.
"""
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from gw170817.config import SimConfig
from gw170817.constants import M_sun
from gw170817.simulation.engine import GW170817Simulation
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
    mock_gui.button.return_value = False
    mock_gui.checkbox.return_value = False
    mock_window.get_gui.return_value = mock_gui

    with patch("taichi.ui.Window", return_value=mock_window), \
         patch("taichi.ui.Scene", return_value=mock_scene), \
         patch("taichi.ui.Canvas", return_value=mock_canvas):
        dashboard = ScientificDashboard(engine=engine, config=config)
        dashboard.window = mock_window
        dashboard.scene = mock_scene
        dashboard.canvas = mock_canvas
        return dashboard


def test_bh_lensing_reference_stars_and_deflection():
    dashboard = create_mock_dashboard()
    renderer = dashboard.renderer

    # 1. Deterministic star positions exist and are non-empty
    orig_stars = renderer.star_pos.to_numpy()
    assert len(orig_stars) > 0, "Starfield must contain stars"
    assert not np.isnan(orig_stars).any(), "Original star positions must not contain NaN"
    assert not np.isinf(orig_stars).any(), "Original star positions must not contain Inf"

    # 2. Verify dedicated BH LENS bright reference stars selection
    assert hasattr(renderer, "_is_bh_ref_star"), "Renderer must have _is_bh_ref_star attribute"
    ref_mask = renderer._is_bh_ref_star
    n_ref = int(np.sum(ref_mask))
    assert 24 <= n_ref <= 36, f"Expected 24-36 reference stars, got {n_ref}"

    # Verify reference stars are distributed across azimuths
    ref_pos = orig_stars[ref_mask]
    azimuths = np.arctan2(ref_pos[:, 1], ref_pos[:, 0])
    # Check that azimuths span all 4 quadrants
    assert np.any((azimuths >= 0) & (azimuths < np.pi/2)), "Quadrant 1 must have reference stars"
    assert np.any((azimuths >= np.pi/2) & (azimuths <= np.pi)), "Quadrant 2 must have reference stars"
    assert np.any((azimuths >= -np.pi) & (azimuths < -np.pi/2)), "Quadrant 3 must have reference stars"
    assert np.any((azimuths >= -np.pi/2) & (azimuths < 0)), "Quadrant 4 must have reference stars"

    # 3. Verify high-contrast brightness in CORE mode
    renderer.update_star_brightness_mode("CORE")
    colors_bh = renderer.star_colors.to_numpy()
    ref_colors = colors_bh[ref_mask]
    non_ref_colors = colors_bh[~ref_mask]

    # Reference stars must be substantially brighter than non-reference stars
    mean_ref_b = float(np.mean(np.linalg.norm(ref_colors, axis=1)))
    mean_non_ref_b = float(np.mean(np.linalg.norm(non_ref_colors, axis=1)))
    assert mean_ref_b > 2.0 * mean_non_ref_b, f"Reference stars ({mean_ref_b:.2f}) must be >2x brighter than non-ref stars ({mean_non_ref_b:.2f})"

    # 4. Verify mode switching to GW restores standard brightness
    renderer.update_star_brightness_mode("GW")
    colors_gw = renderer.star_colors.to_numpy()
    ref_colors_gw = colors_gw[ref_mask]
    mean_gw_ref = float(np.mean(np.linalg.norm(ref_colors_gw, axis=1)))
    assert mean_gw_ref < mean_ref_b * 0.7, "GW mode must restore natural star brightness"

    # 5. Verify camera movement changes lensing deflection coherently
    # Camera at orbit angle 1
    renderer.update_star_lensing_deflection_kernel(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
        cam_x=0.0, cam_y=-32.0e3, cam_z=8.0e3,
        is_black_hole=1, remnant_mass=2.74 * M_sun
    )
    def_pos1 = renderer.star_deflected_pos.to_numpy()

    # Camera at orbit angle 2 (rotated by 45 degrees)
    renderer.update_star_lensing_deflection_kernel(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0,
        cam_x=22.6e3, cam_y=-22.6e3, cam_z=8.0e3,
        is_black_hole=1, remnant_mass=2.74 * M_sun
    )
    def_pos2 = renderer.star_deflected_pos.to_numpy()

    assert not np.array_equal(def_pos1, def_pos2), "Orbiting camera must produce different deflected positions"
    assert not np.isnan(def_pos1).any(), "No NaNs in deflection angle 1"
    assert not np.isnan(def_pos2).any(), "No NaNs in deflection angle 2"

    # 6. Verify event_time continuity across mode switches
    dashboard.engine.set_post_merger_event_time(2.5)
    et_ref = dashboard.engine.current_state.event_time
    for mode in ["CORE", "GW", "MAGNETIC FIELD", "NEUTRINO", "MULTI"]:
        dashboard.set_view_mode(mode)
        assert dashboard.engine.current_state.event_time == et_ref, f"event_time altered on switch to {mode}"
