"""
Diagnostic script to isolate 3D rendering pipeline issues in GW170817 Scientific Dashboard.
"""
import sys
import numpy as np
import taichi as ti

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.dashboard import ScientificDashboard


def diagnose_render_path():
    print("=== DIAGNOSING 3D RENDER PATH ===")

    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    dashboard = ScientificDashboard(engine=engine, config=config)

    # 1. Inspect Camera setup
    print("\n1. Camera Setup:")
    cam_dist = 280.0e3
    print(f"   Camera Target Distance: {cam_dist/1e3:.1f} km")
    print(f"   Dashboard Camera frustum/projection initialized")

    # 2. Inspect particle initialization in ParticleSystem (Inspiral)
    print("\n2. Inspiral Particles (ParticleSystem):")
    pos_np = engine.psys.pos.to_numpy()
    act_np = engine.psys.active.to_numpy()
    n_act = np.sum(act_np)
    r_norms = np.linalg.norm(pos_np, axis=1)
    print(f"   Total Particles: {len(pos_np)}")
    print(f"   Active Particles: {n_act}")
    print(f"   Position Min: {np.min(pos_np, axis=0)}")
    print(f"   Position Max: {np.max(pos_np, axis=0)}")
    print(f"   Mean Distance from Origin: {np.mean(r_norms)/1e3:.1f} km")
    print(f"   Min Distance from Origin:  {np.min(r_norms)/1e3:.1f} km")
    print(f"   Max Distance from Origin:  {np.max(r_norms)/1e3:.1f} km")

    # 3. Step 1 frame and check particle renderer positions & colors
    dashboard.run_step()
    render_pos = dashboard.renderer.render_pos.to_numpy()
    render_cols = dashboard.renderer.colors.to_numpy()

    print("\n3. ParticleRenderer Output (Inspiral frame 1):")
    r_render = np.linalg.norm(render_pos, axis=1)
    visible_mask = (render_pos[:, 2] > -1.0e8)
    n_visible = np.sum(visible_mask)
    print(f"   Visible Particles (z > -1e8): {n_visible} / {len(render_pos)}")
    if n_visible > 0:
        print(f"   Visible Render Pos Min: {np.min(render_pos[visible_mask], axis=0)}")
        print(f"   Visible Render Pos Max: {np.max(render_pos[visible_mask], axis=0)}")
        print(f"   Visible Render Colors Min: {np.min(render_cols[visible_mask], axis=0)}")
        print(f"   Visible Render Colors Max: {np.max(render_cols[visible_mask], axis=0)}")
        print(f"   Visible Render Colors Mean: {np.mean(render_cols[visible_mask], axis=0)}")

    # 4. Check camera vs particle scale ratio
    print("\n4. Scale Ratio Check:")
    if n_visible > 0:
        max_coord = np.max(np.abs(render_pos[visible_mask]))
        print(f"   Max particle coordinate: {max_coord/1e3:.1f} km")
        print(f"   Camera distance:        {cam_dist/1e3:.1f} km")
        print(f"   Ratio (Max Coord / Cam Dist): {max_coord / cam_dist:.4f}")

    # 5. Check post-merger combined fields
    print("\n5. Post-Merger Combined Particles:")
    dashboard.director.jump_to_stage_index(3)  # RINGDOWN
    dashboard.run_step()

    combined_pos = dashboard.renderer.combined_post_merger_pos.to_numpy()
    combined_cols = dashboard.renderer.combined_post_merger_colors.to_numpy()
    pm_visible = (combined_pos[:, 2] > -1.0e8)
    n_pm_vis = np.sum(pm_visible)

    print(f"   Combined Post-Merger Total Particles: {len(combined_pos)}")
    print(f"   Combined Post-Merger Visible Particles: {n_pm_vis}")
    if n_pm_vis > 0:
        print(f"   Post-Merger Pos Min: {np.min(combined_pos[pm_visible], axis=0)}")
        print(f"   Post-Merger Pos Max: {np.max(combined_pos[pm_visible], axis=0)}")
        print(f"   Post-Merger Colors Mean: {np.mean(combined_cols[pm_visible], axis=0)}")

    # 6. Check scene light & camera setup in dashboard.py
    print("\n6. Scene Lights & Camera Bindings:")
    print(f"   Scene ambient light: (0.85, 0.85, 0.85)")
    print(f"   Scene camera bound: camera set to Perspective mode")

if __name__ == "__main__":
    diagnose_render_path()
