"""
Test suite for GW170817 Scientific Dashboard logic and rendering kernels (Task 014, Task 021, Task 022).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import M_sun
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.lensing import RelativisticLensingModel
from gw170817.visualization.dashboard import ScientificDashboard, launch_dashboard


def test_dashboard_logic():
    print("=== TASK 014, 021 & 022 DASHBOARD LOGIC VALIDATION ===")

    # 1. Component Imports Check
    from gw170817.visualization import ParticleRenderer, ScientificDashboard, RelativisticLensingModel
    assert ParticleRenderer is not None
    assert ScientificDashboard is not None
    assert RelativisticLensingModel is not None

    # 2. Engine Supply & Dashboard Construction
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config)

    assert engine is not None
    assert engine.current_state.phase == "INSPIRAL"

    renderer = ParticleRenderer(engine.psys)
    assert renderer is not None
    assert renderer.colors.shape[0] == engine.psys.max_particles

    # 3. Particle Renderer Luminous GPU Update Test
    st_init = engine.current_state
    renderer.update_particle_colors(
        engine.psys.pos,
        engine.psys.star_id,
        engine.psys.active,
        engine.psys.ye,
        engine.psys.max_particles,
        float(st_init.merger_contact_fraction),
        0.0
    )

    colors_np = renderer.colors.to_numpy()
    assert colors_np.shape == (engine.psys.max_particles, 3)
    assert np.all(np.isfinite(colors_np))
    assert np.mean(colors_np) > 0.4

    # 4. Waveform Renderer Buffer Update Test
    engine.step(0.001)
    _, hp_arr, _, _ = engine.waveform_buffer.get_chronological()
    assert len(hp_arr) > 0
    renderer.update_waveform_buffer(hp_arr)
    wf_verts = renderer.waveform_vertices.to_numpy()
    assert wf_verts.shape[0] == renderer.n_wave_vertices
    assert np.all(np.isfinite(wf_verts))

    # 5. Relativistic Jet Outflow Geometry Test
    renderer.update_jet_geometry(is_active=True, jet_length_m=400.0e3, opening_angle_rad=0.08, intensity=1.0)
    jet_verts = renderer.jet_vertices.to_numpy()
    assert jet_verts.shape[0] == renderer.n_jet_vertices
    assert np.all(np.isfinite(jet_verts))

    # 6. Relativistic Lensing & Starfield Integration Test
    lensing = RelativisticLensingModel(config=config)
    lens_st = lensing.evaluate()
    assert lens_st.enabled
    assert np.isfinite(lens_st.compactness1)

    # 7. Demo Jump Preservation Test
    engine.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
    st_demo = engine.current_state

    assert abs(st_demo.gw_frequency - 1200.0) < 1e-5
    assert abs(st_demo.separation - 30.0e3) < 1.0
    assert st_demo.merger_started
    assert st_demo.phase in ["MERGER", "RINGDOWN"]

    renderer.update_particle_colors(
        engine.psys.pos,
        engine.psys.star_id,
        engine.psys.active,
        engine.psys.ye,
        engine.psys.max_particles,
        float(st_demo.merger_contact_fraction),
        0.5
    )
    colors_demo = renderer.colors.to_numpy()
    assert np.all(np.isfinite(colors_demo))

    # 8. Reset State Preservation Test
    engine.reset()
    engine.set_inspiral_time(5.0)
    st_reset = engine.current_state

    assert st_reset.phase == "INSPIRAL"
    assert abs(st_reset.event_time - (-5.0)) < 0.5

    # 9. Numerical Safety Check across all state fields
    for field, val in st_reset.__dict__.items():
        if isinstance(val, (float, int)):
            assert np.isfinite(val), f"SimulationState field {field} non-finite: {val}"

    print("Dashboard Imports: PASS")
    print("Engine Integration: PASS")
    print("Particle Color GPU Kernel (Luminous Density/Temp): PASS")
    print("Waveform Trace GPU Kernel: PASS")
    print("Relativistic Jet Outflow Geometry: PASS")
    print("Relativistic Lensing Integration: PASS")
    print("Demo Jump Preservation: PASS")
    print("Reset State Preservation: PASS")
    print("Numerical Safety: PASS")
    print("\nALL TASK 014, 021 & 022 DASHBOARD LOGIC CHECKS PASSED SUCCESSFULLY!")


def test_control_mapping_no_d_trigger():
    """Verify Task 024.5 requirement: Camera key 'D' must NOT trigger DemoDirector start."""
    from gw170817.visualization.dashboard import ScientificDashboard

    class DummyEvent:
        def __init__(self, key):
            self.key = key

    class DummyWindow:
        def __init__(self):
            self.current_event = None
            self.running = True
        def get_event(self, event_type):
            if self.current_event is not None:
                return True
            return False
        def is_pressed(self, key):
            return False

    config = SimConfig(mode="DEV")
    engine = GW170817Simulation(config)
    dashboard = ScientificDashboard(engine=engine, config=config)
    dummy_win = DummyWindow()
    dashboard.window = dummy_win

    dashboard.director.reset()
    assert not dashboard.director.is_running

    dummy_win.current_event = DummyEvent('d')
    dashboard.process_input()
    dummy_win.current_event = None
    assert not dashboard.director.is_running, "'d' key must NOT start DemoDirector"

    dummy_win.current_event = DummyEvent('D')
    dashboard.process_input()
    dummy_win.current_event = None
    assert not dashboard.director.is_running, "'D' key must NOT start DemoDirector"

    dummy_win.current_event = DummyEvent('a')
    dashboard.process_input()
    dummy_win.current_event = None
    assert not dashboard.director.is_running, "'a' key must NOT start DemoDirector"

    dummy_win.current_event = DummyEvent('A')
    dashboard.process_input()
    dummy_win.current_event = None
    assert not dashboard.director.is_running, "'A' key must NOT start DemoDirector"

    dummy_win.current_event = DummyEvent(' ')
    dashboard.process_input()
    dummy_win.current_event = None
    assert dashboard.director.is_running, "SPACE key MUST start DemoDirector"
    assert dashboard.director.current_stage == "INSPIRAL"


def test_checkpoint_navigation_n_b():
    """Verify Task 024.4 requirement: N and B keys physically jump presentation time, stage, and event time."""
    from gw170817.visualization.dashboard import ScientificDashboard

    config = SimConfig(mode="DEV")
    engine = GW170817Simulation(config)
    dashboard = ScientificDashboard(engine=engine, config=config)

    # 1. Reset director
    dashboard.director.reset()
    assert dashboard.director.presentation_time == 0.0
    assert dashboard.director.current_stage == "IDLE"
    assert abs(dashboard.coordinator.current_state.event_time - (-5.0)) < 0.1

    # 2. Press N: Start / Jump to Stage 0 (INSPIRAL, 0.0s)
    st1 = dashboard.director.next_stage()
    t1 = dashboard.director.presentation_time
    stage1 = dashboard.director.current_stage
    evt1 = st1.event_time

    assert t1 == 0.0
    assert stage1 == "INSPIRAL"
    assert abs(evt1 - (-5.0)) < 0.1

    # 3. Press N: Jump to Stage 1 (LATE_INSPIRAL, 4.0s)
    st2 = dashboard.director.next_stage()
    t2 = dashboard.director.presentation_time
    stage2 = dashboard.director.current_stage
    evt2 = st2.event_time

    assert t2 > t1
    assert t2 == 4.0
    assert stage2 == "LATE_INSPIRAL"
    assert abs(evt2 - (-1.0)) < 0.5

    # 4. Press N: Jump to Stage 2 (MERGER, 6.0s)
    st3 = dashboard.director.next_stage()
    t3 = dashboard.director.presentation_time
    stage3 = dashboard.director.current_stage
    evt3 = st3.event_time

    assert t3 > t2
    assert t3 == 6.0
    assert stage3 == "MERGER"

    # 5. Press N: Jump to Stage 3 (RINGDOWN, 9.0s)
    st4 = dashboard.director.next_stage()
    t4 = dashboard.director.presentation_time
    stage4 = dashboard.director.current_stage
    evt4 = st4.event_time

    assert t4 > t3
    assert t4 == 9.0
    assert stage4 == "RINGDOWN"
    assert abs(evt4 - 1.74) < 0.1

    # 6. Press B: Step back to Stage 2 (MERGER, 6.0s)
    st5 = dashboard.director.previous_stage()
    t5 = dashboard.director.presentation_time
    stage5 = dashboard.director.current_stage

    assert t5 < t4
    assert t5 == 6.0
    assert stage5 == "MERGER"

    print("Checkpoint Navigation (N/B physical state & time jump): PASS")


if __name__ == "__main__":
    test_dashboard_logic()
    test_control_mapping_no_d_trigger()
    test_checkpoint_navigation_n_b()
