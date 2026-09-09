"""
Test suite for GW170817 Scientific Dashboard v0.9 non-GUI logic (Task 014).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import M_sun
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.dashboard import ScientificDashboard, launch_dashboard


def test_dashboard_logic():
    print("=== TASK 014 DASHBOARD LOGIC VALIDATION ===")

    # 1. Component Imports Check
    from gw170817.visualization import ParticleRenderer, ScientificDashboard
    assert ParticleRenderer is not None
    assert ScientificDashboard is not None

    # 2. Engine Supply & Dashboard Construction
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config)

    assert engine is not None
    assert engine.current_state.phase == "INSPIRAL"

    renderer = ParticleRenderer(engine.psys)
    assert renderer is not None
    assert renderer.colors.shape[0] == engine.psys.max_particles

    # 3. Particle Renderer GPU Update Test
    st_init = engine.current_state
    renderer.update_particle_colors(
        engine.psys.star_id,
        engine.psys.active,
        engine.psys.max_particles,
        float(st_init.merger_contact_fraction)
    )

    colors_np = renderer.colors.to_numpy()
    assert colors_np.shape == (engine.psys.max_particles, 3)
    assert np.all(np.isfinite(colors_np))

    # 4. Waveform Renderer Buffer Update Test
    engine.step(0.001)
    _, hp_arr, _, _ = engine.waveform_buffer.get_chronological()
    assert len(hp_arr) > 0
    renderer.update_waveform_buffer(hp_arr)
    wf_verts = renderer.waveform_vertices.to_numpy()
    assert wf_verts.shape[0] == renderer.n_wave_vertices
    assert np.all(np.isfinite(wf_verts))

    # 5. Demo Jump Preservation Test
    engine.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
    st_demo = engine.current_state

    assert abs(st_demo.gw_frequency - 1200.0) < 1e-5
    assert abs(st_demo.separation - 30.0e3) < 1.0
    assert st_demo.merger_started
    assert st_demo.phase in ["MERGER", "POST_MERGER"]

    renderer.update_particle_colors(
        engine.psys.star_id,
        engine.psys.active,
        engine.psys.max_particles,
        float(st_demo.merger_contact_fraction)
    )
    colors_demo = renderer.colors.to_numpy()
    assert np.all(np.isfinite(colors_demo))

    # 6. Reset State Preservation Test
    engine.reset()
    st_reset = engine.current_state

    assert st_reset.elapsed_time == 0.0
    assert st_reset.phase == "INSPIRAL"
    assert abs(st_reset.gw_frequency - 40.0) < 1e-5

    # 7. Numerical Safety Check across all state fields
    for field, val in st_reset.__dict__.items():
        if isinstance(val, (float, int)):
            assert np.isfinite(val), f"SimulationState field {field} non-finite: {val}"

    print("Dashboard Imports: PASS")
    print("Engine Integration: PASS")
    print("Particle Color GPU Kernel: PASS")
    print("Waveform Trace GPU Kernel: PASS")
    print("Demo Jump Preservation: PASS")
    print("Reset State Preservation: PASS")
    print("Numerical Safety: PASS")
    print("\nALL TASK 014 DASHBOARD LOGIC CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_dashboard_logic()
