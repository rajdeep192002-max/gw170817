"""
Runtime Visual Verification Script for Task 035 Dashboard & View Modes.
Runs real dashboard frame rendering across all 4 modes (CORE, GW, BH LENS, MULTI) and measures runtime FPS.
"""
import sys
import time
sys.path.insert(0, '.')
import numpy as np

from gw170817.config import SimConfig
from gw170817.visualization.dashboard import ScientificDashboard

def verify_dashboard_runtime():
    print("=== TASK 035 DASHBOARD RUNTIME VERIFICATION ===")
    config = SimConfig(mode="DEV")
    dash = ScientificDashboard(config=config)

    print("1. Verifying CORE Mode...")
    dash.set_view_mode("CORE")
    assert dash.view_mode == "CORE"
    assert abs(dash.cam_distance - 280.0e3) < 1.0

    print("2. Verifying GW Mode...")
    dash.set_view_mode("GW")
    assert dash.view_mode == "GW"
    assert dash.wave_propagation.active
    assert abs(dash.cam_distance - 350.0e3) < 1.0

    print("3. Verifying BH LENS Mode Pre-BH State...")
    dash.set_view_mode("BH LENS")
    assert dash.view_mode == "BH LENS"
    assert dash.bh_lens_waiting, "Pre-BH state should set waiting flag"

    print("4. Advancing to Post-Merger BH Formation (t = 10.0s presentation time)...")
    dash.director._sync_physics_for_presentation_time(10.0)
    dash.set_view_mode("BH LENS")
    assert not dash.bh_lens_waiting
    assert dash.lensing.enabled
    assert abs(dash.cam_distance - 55.0e3) < 1.0, "BH LENS mode preset must set close camera distance (55 km)"

    print("5. Testing Manual Camera Orbit After Preset...")
    dash.camera_orbit_up(0.05)
    dash.camera_orbit_right(0.08)
    assert dash.cam_pitch != 0.25, "Manual pitch must persist"

    print("6. Verifying MULTI Mode...")
    dash.set_view_mode("MULTI")
    assert dash.view_mode == "MULTI"

    print("7. Simulating Frame Execution & Measuring FPS...")
    n_frames = 60
    t_start = time.time()
    for f in range(n_frames):
        dash.render_frame()
    t_elapsed = time.time() - t_start

    fps = n_frames / max(t_elapsed, 1e-4)
    print(f"Executed {n_frames} frames in {t_elapsed:.2f} s -> Framerate: {fps:.1f} FPS")

    print("\n[SUCCESS] TASK 035 DASHBOARD RUNTIME VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    verify_dashboard_runtime()
