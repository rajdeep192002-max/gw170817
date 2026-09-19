"""
Unit tests for Lightweight Reduced-Order Volumetric Accretion-Disk Renderer (Task 049).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import taichi as ti
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem
from gw170817.visualization.renderer import ParticleRenderer


def test_volumetric_accretion_disk():
    print("=== TEST TASK 049 VOLUMETRIC ACCRETION DISK VALIDATION ===")

    # Initialize Taichi GPU backend safely
    try:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
    except Exception:
        pass

    config = SimConfig(mode="DEV")
    psys = ParticleSystem(config)
    renderer = ParticleRenderer(psys)

    # 1. Structure Audit
    assert renderer.n_disk_mesh_layers == 3, f"Expected 3 mesh layers, got {renderer.n_disk_mesh_layers}"
    assert renderer.n_disk_mesh_verts == 9216, f"Expected 9216 vertices, got {renderer.n_disk_mesh_verts}"
    assert renderer.n_disk_mesh_indices == 53568, f"Expected 53568 indices, got {renderer.n_disk_mesh_indices}"

    # 2. Update Volumetric Mesh & Particles
    renderer.update_disk_particles(
        event_time=1.25,
        dt_vis=0.016,
        is_active=True,
        disk_progress=1.0,
        disk_mass_msun=0.06,
        is_black_hole=True,
        cam_x=0.0,
        cam_y=-250.0e3,
        cam_z=80.0e3,
        lensing_enabled=True,
        intensity_scale=1.0,
        remnant_mass_kg=2.7 * 1.989e30
    )

    mesh_pos = renderer.disk_mesh_pos.to_numpy()
    mesh_col = renderer.disk_mesh_colors.to_numpy()

    assert mesh_pos.shape == (9216, 3)
    assert mesh_col.shape == (9216, 3)

    # 3. Layer 0 (Direct Volumetric Surface) Validation
    l0_pos = mesh_pos[:3072]
    l0_col = mesh_col[:3072]

    # Non-zero volumetric emission
    l0_norms = np.linalg.norm(l0_col, axis=1)
    assert np.all(np.isfinite(l0_pos)), "Mesh positions contain non-finite numbers"
    assert np.all(np.isfinite(l0_col)), "Mesh colors contain non-finite numbers"
    assert np.mean(l0_norms) > 0.05, f"Volumetric disk emission too faint: mean norm={np.mean(l0_norms):.4f}"

    # 4. Layer 1 (Upper Lensed Arch) & Layer 2 (Lower Lensed Arc) Lensing Validation
    l1_pos = mesh_pos[3072:6144]
    l2_pos = mesh_pos[6144:9216]

    # Lensed vertices should wrap visually above (z > 0) and below (z < 0) BH shadow
    valid_l1 = l1_pos[l1_pos[:, 2] > -1.0e8]
    valid_l2 = l2_pos[l2_pos[:, 2] > -1.0e8]

    assert len(valid_l1) > 0, "Layer 1 primary upper lensed arch has no active vertices"
    assert len(valid_l2) > 0, "Layer 2 secondary lower lensed arc has no active vertices"

    mean_z_upper = np.mean(valid_l1[:, 2])
    mean_z_lower = np.mean(valid_l2[:, 2])

    assert mean_z_upper > 0.0, f"Primary upper lensed arch should wrap above BH shadow (mean z={mean_z_upper:.1f})"
    assert mean_z_lower < 0.0, f"Secondary lower lensed arc should wrap below BH shadow (mean z={mean_z_lower:.1f})"

    # 5. Tracer Particles Subdued Intensity
    tracer_col = renderer.disk_colors.to_numpy()
    mean_tracer_brightness = np.mean(np.linalg.norm(tracer_col, axis=1))

    assert mean_tracer_brightness < 0.35, f"Tracer particles should be subdued, got mean brightness {mean_tracer_brightness:.4f}"

    print("ALL VOLUMETRIC ACCRETION DISK TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_volumetric_accretion_disk()
