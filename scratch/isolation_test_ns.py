"""
Isolation test for Neutron Star rendering path (Task 026B Follow-up).
Renders ONLY the neutron star particle buffers in GGUI scene with:
- Star background DISABLED
- Ejecta DISABLED
- Disk DISABLED
- Jet DISABLED
- HUD DISABLED

Keeps normal camera.
"""
import time
import numpy as np
import taichi as ti

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.renderer import ParticleRenderer

def test_ns_isolation():
    print("=== TASK 026B ISOLATION TEST: NEUTRON STARS ONLY ===")

    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    renderer = ParticleRenderer(engine.psys)

    # Initialize GGUI Window & Scene
    window = ti.ui.Window("GW170817 NS Isolation Test", (1600, 900), vsync=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Normal camera setup
    cam_dist = 280.0e3
    camera.position(0.0, -cam_dist * 0.85, cam_dist * 0.65)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 0.0, 1.0)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    # Reset inspiral
    engine.reset()
    engine.set_inspiral_time(5.0)

    print("\nRunning 60 frames of NS Isolation Rendering...")

    for frame in range(60):
        if not window.running:
            break

        # Advance inspiral physics
        engine.step(1.0e-3)
        st = engine.current_state

        # Update NS particle render colors & positions
        renderer.update_particle_colors(
            engine.psys.pos,
            engine.psys.star_id,
            engine.psys.active,
            engine.psys.ye,
            engine.psys.max_particles,
            float(st.merger_contact_fraction),
            0.0  # ejecta_progress = 0
        )

        # Submit ONLY NS particles to scene (no background, no ejecta, no disk, no jet, no HUD)
        scene.set_camera(camera)
        scene.ambient_light((0.85, 0.85, 0.85))

        scene.particles(
            renderer.render_pos,
            radius=1.8e3,
            per_vertex_color=renderer.colors
        )

        canvas.scene(scene)
        window.show()

    pos_np = engine.psys.pos.to_numpy()
    render_pos = renderer.render_pos.to_numpy()
    render_cols = renderer.colors.to_numpy()
    vis_mask = (render_pos[:, 2] > -1.0e8)

    print(f"\nISOLATION TEST COMPLETED:")
    print(f"  - Active Inspirating NS Particles in Buffer: {np.sum(vis_mask)} / {len(render_pos)}")
    print(f"  - Particle Position Min: {np.min(render_pos[vis_mask], axis=0)}")
    print(f"  - Particle Position Max: {np.max(render_pos[vis_mask], axis=0)}")
    print(f"  - Particle Color Mean:   {np.mean(render_cols[vis_mask], axis=0)}")
    print(f"  - Camera Distance:       {cam_dist/1e3} km")

if __name__ == "__main__":
    test_ns_isolation()
