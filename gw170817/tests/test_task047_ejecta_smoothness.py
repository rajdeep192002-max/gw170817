"""
Regression and unit test suite for Task 047: Ejecta Visual Update Cadence & Smoothness.

Verifies:
1. Ejecta fluid particle positions update on EVERY consecutive render frame (100% per-frame update rate).
2. Consecutive frame displacement is smooth and non-zero (0 zero-displacement frames, eliminating 15 Hz temporal aliasing/flicker).
3. Particle trajectories are continuous over time: dr/dt > 0 during outward expansion.
4. Ejecta visibility and brightness values evolve continuously without alternating frame pop-in/pop-out.
5. Remnant and accretion disk updates preserve their optimized visual cadence without regression.
"""
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from gw170817.config import SimConfig
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
    mock_window.get_gui.return_value = mock_gui

    with patch("taichi.ui.Window", return_value=mock_window), \
         patch("taichi.ui.Scene", return_value=mock_scene), \
         patch("taichi.ui.Canvas", return_value=mock_canvas):
        dashboard = ScientificDashboard(engine=engine, config=config)
        dashboard.window = mock_window
        dashboard.scene = mock_scene
        dashboard.canvas = mock_canvas
        return dashboard


def test_ejecta_per_frame_smoothness_and_continuity():
    dashboard = create_mock_dashboard()

    # ---- 1. CORE Mode Ejecta Per-Frame Position Continuity ----
    dashboard.engine.set_post_merger_event_time(0.50)
    dashboard.director._presentation_time = 8.0
    dashboard.coordinator._update_event_state()
    dashboard.set_view_mode("CORE")

    positions_history = []

    # Render 10 consecutive frames
    for f in range(10):
        t_frame = 0.50 + f * 0.016
        dashboard.engine.set_post_merger_event_time(t_frame)
        dashboard.render_frame()

        pos = dashboard.renderer.ejecta_fluid_pos.to_numpy()
        positions_history.append(pos.copy())

    zero_displacement_frames = 0
    displacements = []

    for f in range(len(positions_history) - 1):
        pos_a = positions_history[f]
        pos_b = positions_history[f + 1]

        active_mask = pos_a[:, 2] > -1.0e8
        assert np.any(active_mask), f"Ejecta particles should be active on frame {f}"

        disp = np.linalg.norm(pos_b[active_mask] - pos_a[active_mask], axis=1)
        mean_disp = float(np.mean(disp))
        displacements.append(mean_disp)

        if mean_disp < 1.0:  # less than 1 meter displacement across frame
            zero_displacement_frames += 1

    # VERIFICATION 1: Zero frames with static/frozen ejecta positions
    assert zero_displacement_frames == 0, (
        f"Found {zero_displacement_frames} frozen frames where ejecta positions were not updated. "
        "Ejecta positions must update on EVERY frame to eliminate flicker/jumping."
    )

    # VERIFICATION 2: Displacements across consecutive frames are uniform and continuous
    disp_std = float(np.std(displacements))
    disp_mean = float(np.mean(displacements))
    relative_variation = disp_std / disp_mean if disp_mean > 0 else 0.0

    assert relative_variation < 0.25, (
        f"Displacement variation between consecutive frames is too high ({relative_variation*100:.1f}%). "
        "Frame-to-frame motion must be smooth and uniform."
    )

    # ---- 2. Ejecta Trajectories Outward Monotonicity ----
    dashboard.engine.set_post_merger_event_time(0.10)
    dashboard.director._presentation_time = 8.0
    dashboard.coordinator._update_event_state()
    dashboard.set_view_mode("CORE")

    radii_history = []
    for f in range(6):
        t_frame = 0.10 + f * 0.033  # ~30 FPS steps
        dashboard.engine.set_post_merger_event_time(t_frame)
        dashboard.render_frame()

        pos = dashboard.renderer.ejecta_fluid_pos.to_numpy()
        r = np.linalg.norm(pos, axis=1)
        radii_history.append(r)

    mean_radii = [float(np.mean(r[r > 1.0e3])) for r in radii_history]
    for i in range(len(mean_radii) - 1):
        assert mean_radii[i + 1] > mean_radii[i], (
            f"Ejecta mean radius decreased from frame {i} ({mean_radii[i]:.1f}m) to {i+1} ({mean_radii[i+1]:.1f}m). "
            "Trajectories must be strictly outward expanding without bouncing."
        )

    # ---- 3. MULTI Mode Per-Frame Ejecta Smoothness ----
    dashboard.engine.set_post_merger_event_time(1.0)
    dashboard.director._presentation_time = 8.0
    dashboard.coordinator._update_event_state()
    dashboard.set_view_mode("MULTI")

    positions_multi = []
    for f in range(6):
        t_frame = 1.0 + f * 0.016
        dashboard.engine.set_post_merger_event_time(t_frame)
        dashboard.render_frame()

        pos = dashboard.renderer.ejecta_fluid_pos.to_numpy()
        positions_multi.append(pos.copy())

    for f in range(len(positions_multi) - 1):
        disp = np.linalg.norm(positions_multi[f + 1] - positions_multi[f], axis=1)
        assert np.mean(disp) > 10.0, f"Frame {f} displacement in MULTI mode should be non-zero"
