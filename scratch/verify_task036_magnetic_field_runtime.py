"""
Runtime Visual Verification Script for Task 036 — Magnetic Field Lines Presentation Mode.
Tests MAGNETIC FIELD mode, camera preset, manual navigation, mode transitions, and frame rendering.
"""
import sys
import time
sys.path.insert(0, '.')
import numpy as np

from gw170817.config import SimConfig
from gw170817.visualization.dashboard import ScientificDashboard

def verify_magnetic_field_runtime():
    print("=== TASK 036 MAGNETIC FIELD RUNTIME VERIFICATION ===")
    config = SimConfig(mode="DEV")
    dash = ScientificDashboard(config=config)

    print("1. Advance simulation to post-merger stage (t = 8.5s presentation time)...")
    dash.director._sync_physics_for_presentation_time(8.5)
    st = dash.engine.current_state
    evt_st = dash.coordinator.current_state

    print("2. Verifying default CORE mode field-line status...")
    dash.set_view_mode("CORE")
    assert dash.show_magnetic_field is False, "Field lines must be hidden in CORE mode"

    print("3. Entering MAGNETIC FIELD mode...")
    dash.set_view_mode("MAGNETIC FIELD")
    assert dash.view_mode == "MAGNETIC FIELD"
    assert dash.show_magnetic_field is True, "Field lines must be visible in MAGNETIC FIELD mode"
    assert abs(dash.cam_distance - 160.0e3) < 1.0, "MAGNETIC FIELD camera preset must be 160 km"
    assert abs(dash.cam_pitch - 0.45) < 1e-3

    print("4. Testing manual camera movement in MAGNETIC FIELD mode...")
    dash.camera_orbit_up(0.06)
    dash.camera_orbit_left(0.08)
    dash.camera_zoom_in(0.90)
    assert abs(dash.cam_distance - 160.0e3 * 0.90) < 1.0
    assert abs(dash.cam_pitch - (0.45 + 0.06)) < 1e-3

    print("5. Verifying mode transitions: MAGNETIC FIELD -> CORE -> MULTI...")
    dash.set_view_mode("CORE")
    assert dash.show_magnetic_field is False, "Field lines must be hidden in CORE mode"
    dash.set_view_mode("MULTI")
    assert dash.show_magnetic_field is False, "Field lines must be hidden in MULTI mode"
    dash.set_view_mode("MAGNETIC FIELD")
    assert dash.show_magnetic_field is True, "Field lines must be visible in MAGNETIC FIELD mode"

    print("6. Simulating Frame Execution in MAGNETIC FIELD mode & Measuring FPS...")
    n_frames = 60
    t_start = time.time()
    for f in range(n_frames):
        dash.render_frame()
    t_elapsed = time.time() - t_start

    fps = n_frames / max(t_elapsed, 1e-4)
    print(f"Executed {n_frames} frames in {t_elapsed:.2f} s -> Framerate: {fps:.1f} FPS")

    print("\n[SUCCESS] TASK 036 MAGNETIC FIELD RUNTIME VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    verify_magnetic_field_runtime()
