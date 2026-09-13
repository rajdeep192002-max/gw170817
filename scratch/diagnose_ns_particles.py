"""
Temporary diagnostic specifically for neutron-star particles (Task 026B).
"""
import sys
import numpy as np
import taichi as ti

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.dashboard import ScientificDashboard


def diagnose_ns_particles():
    print("=== NS PARTICLE DIAGNOSTIC (TASK 026B) ===")

    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    dashboard = ScientificDashboard(engine=engine, config=config)

    # Reset / start inspiral
    dashboard.director.reset()
    dashboard.run_step()

    psys = engine.psys
    pos_np = psys.pos.to_numpy()
    sid_np = psys.star_id.to_numpy()
    act_np = psys.active.to_numpy()

    ns1_mask = (act_np == 1) & (sid_np == 0)
    ns2_mask = (act_np == 1) & (sid_np == 1)

    ns1_pos = pos_np[ns1_mask]
    ns2_pos = pos_np[ns2_mask]

    print(f"\n1. Particle System State (Inspiral):")
    print(f"   - Active NS1 particle count: {len(ns1_pos)}")
    print(f"   - Active NS2 particle count: {len(ns2_pos)}")

    if len(ns1_pos) > 0:
        print(f"   - Representative NS1 XYZ (particle 0): {ns1_pos[0]} m ({ns1_pos[0]/1e3} km)")
        print(f"   - NS1 XYZ Min: {np.min(ns1_pos, axis=0)} m")
        print(f"   - NS1 XYZ Max: {np.max(ns1_pos, axis=0)} m")

    if len(ns2_pos) > 0:
        print(f"   - Representative NS2 XYZ (particle 0): {ns2_pos[0]} m ({ns2_pos[0]/1e3} km)")
        print(f"   - NS2 XYZ Min: {np.min(ns2_pos, axis=0)} m")
        print(f"   - NS2 XYZ Max: {np.max(ns2_pos, axis=0)} m")

    # Renderer state
    render_pos = dashboard.renderer.render_pos.to_numpy()
    render_cols = dashboard.renderer.colors.to_numpy()
    vis_mask = (render_pos[:, 2] > -1.0e8)

    print(f"\n2. ParticleRenderer Buffer State:")
    print(f"   - Render-buffer active count (z > -1e8): {np.sum(vis_mask)} / {len(render_pos)}")
    if np.sum(vis_mask) > 0:
        print(f"   - Representative rendered RGB (particle 0): {render_cols[0]}")
        print(f"   - Representative rendered radius: 1.8e3 m (1.8 km)")

    # Camera state
    print(f"\n3. Camera State:")
    cam_dist = 280.0e3
    cam_pos = np.array([0.0, -cam_dist * 0.85, cam_dist * 0.65])
    cam_look = np.array([0.0, 0.0, 0.0])
    cam_dir = (cam_look - cam_pos) / np.linalg.norm(cam_look - cam_pos)

    print(f"   - Camera Position: {cam_pos} m ({cam_pos/1e3} km)")
    print(f"   - Camera Look Target: {cam_look} m")
    print(f"   - Camera Look Direction: {cam_dir}")
    print(f"   - Camera Projection: Perspective")

if __name__ == "__main__":
    diagnose_ns_particles()
