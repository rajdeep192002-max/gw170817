"""
Test script to verify camera z_near / z_far clipping fix.
"""
import taichi as ti
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.dashboard import ScientificDashboard

def test_camera_zfar():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    dashboard = ScientificDashboard(engine=engine, config=config)

    # Check camera position
    cam = dashboard.camera
    cam_dist = 280.0e3
    cam.position(0.0, -cam_dist * 0.85, cam_dist * 0.65)
    cam.lookat(0.0, 0.0, 0.0)
    cam.up(0.0, 0.0, 1.0)
    cam.z_near(1.0e3)
    cam.z_far(1.0e6)

    print("[Test] Camera z_near set to 1.0e3 (1 km), z_far set to 1.0e6 (1000 km)")
    print("[Test] Camera position distance from origin:", cam_dist / 1.0e3, "km")

    # Step 1 frame
    dashboard.run_step()
    print("[Test] Frame 1 executed without error!")

if __name__ == "__main__":
    test_camera_zfar()
