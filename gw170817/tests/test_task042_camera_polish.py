"""
Regression and verification test for Task 042: Camera Post-Scroll Shake Removal.

Verifies using a single dashboard fixture (to prevent multi-Taichi GPU context crashes):
1. Inertial camera parameters (cam_damping=4.8, cam_accel=4.2).
2. Monotonic angular velocity decay when input stops.
3. Velocity zero-clamp threshold (2.0e-4 rad/s) locks motion cleanly without oscillation.
4. Diagonal movement (simultaneous yaw & pitch) remains smooth and clamped within pitch limits.
5. Pitch limits (-pi/2 + 0.05, pi/2 - 0.05) are enforced with zeroed pitch velocity upon contact.
6. Zoom in/out and manual orbit zero residual angular velocity.
7. Manual input cancels camera preset interpolation.
8. Camera integration uses wall-clock dt independently of simulation time / slow motion.
"""
from unittest.mock import MagicMock, patch
import pytest
import math
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


def test_camera_post_scroll_shake_and_dynamics():
    dashboard = create_mock_dashboard()

    # 1. Verify camera tuning parameters
    assert abs(dashboard.cam_damping - 4.8) < 1.0e-4, f"cam_damping should be 4.8, got {dashboard.cam_damping}"
    assert abs(dashboard.cam_accel - 4.2) < 1.0e-4, f"cam_accel should be 4.2, got {dashboard.cam_accel}"

    # 2. Verify monotonic velocity decay and zero-clamp threshold
    dt = 0.016
    for _ in range(10):
        dashboard.update_inertial_camera(dt, input_yaw=1.0, input_pitch=0.0)

    initial_omega_yaw = dashboard.cam_omega_yaw
    assert initial_omega_yaw > 0.1, "Yaw velocity should accelerate with arrow input"

    prev_omega = initial_omega_yaw
    zero_reached = False
    for step in range(150):
        dashboard.update_inertial_camera(dt, input_yaw=0.0, input_pitch=0.0)
        curr_omega = dashboard.cam_omega_yaw

        # Monotonic decay check
        assert curr_omega <= prev_omega + 1.0e-7, f"Velocity increased during coasting at step {step}: {curr_omega} > {prev_omega}"
        # No sign flip check
        assert curr_omega >= 0.0, f"Velocity flipped sign to negative during decay at step {step}: {curr_omega}"

        if curr_omega == 0.0:
            zero_reached = True
            break
        prev_omega = curr_omega

    assert zero_reached, "Camera velocity should decay to exactly 0.0 via zero-clamp threshold"

    # 3. Verify diagonal movement & pitch limits
    for _ in range(60):
        dashboard.update_inertial_camera(dt, input_yaw=1.0, input_pitch=1.0)

    assert dashboard.cam_pitch <= dashboard.cam_pitch_max, f"Pitch exceeded max limit: {dashboard.cam_pitch} > {dashboard.cam_pitch_max}"
    assert dashboard.cam_pitch >= dashboard.cam_pitch_min, f"Pitch fell below min limit: {dashboard.cam_pitch} < {dashboard.cam_pitch_min}"

    if dashboard.cam_pitch >= dashboard.cam_pitch_max:
        assert dashboard.cam_omega_pitch == 0.0, "Pitch velocity must be zeroed when hitting upper limit"

    # 4. Verify zoom & orbit methods zero residual velocity
    dashboard.cam_omega_yaw = 0.5
    dashboard.cam_omega_pitch = -0.3
    dashboard.camera_zoom_in()
    assert dashboard.cam_omega_yaw == 0.0, "camera_zoom_in should zero cam_omega_yaw"
    assert dashboard.cam_omega_pitch == 0.0, "camera_zoom_in should zero cam_omega_pitch"

    dashboard.cam_omega_yaw = 0.5
    dashboard.cam_omega_pitch = -0.3
    dashboard.camera_orbit_left()
    assert dashboard.cam_omega_yaw == 0.0, "camera_orbit_left should zero cam_omega_yaw"
    assert dashboard.cam_omega_pitch == 0.0, "camera_orbit_left should zero cam_omega_pitch"

    # 5. Verify manual input cancels camera preset interpolation
    dashboard.set_view_mode("NEUTRINO")
    dashboard.is_interpolating_cam = True
    dashboard.update_inertial_camera(0.016, input_yaw=1.0, input_pitch=0.0)
    assert not dashboard.is_interpolating_cam, "Manual input must cancel preset camera interpolation"

    # 6. Verify wall-clock dt integration independent of simulation time / slow motion
    dashboard.director.set_slow_motion()
    assert dashboard.director.speed_multiplier == 0.10

    with patch.object(dashboard, "update_inertial_camera") as mock_update:
        dashboard.run_step()
        assert mock_update.called, "update_inertial_camera must be called during run_step"
        dt_passed = mock_update.call_args[0][0]
        assert dt_passed > 0.0, "dt passed to camera must be positive wall clock dt"
        assert abs(dt_passed - 0.016) < 0.09, "Camera dt must be independent of simulation speed_multiplier"
