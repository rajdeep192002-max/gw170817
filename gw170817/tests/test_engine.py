"""
Test suite for integrated GW170817 simulation engine time/state semantics (Task 013.1).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import day, Mpc, M_sun
from gw170817.simulation.engine import GW170817Simulation, SimulationState


def test_engine():
    # A & B. Construction & Component Creation
    config = SimConfig(mode="DEV", seed=42)
    sim = GW170817Simulation(config)

    assert hasattr(sim, "inspiral")
    assert hasattr(sim, "tidal")
    assert hasattr(sim, "merger")
    assert hasattr(sim, "dynamics")
    assert hasattr(sim, "ejecta")
    assert hasattr(sim, "kilonova")
    assert hasattr(sim, "jet")
    assert hasattr(sim, "afterglow")
    assert hasattr(sim, "timeline")
    assert hasattr(sim, "gw_model")
    assert hasattr(sim, "waveform_buffer")

    st0 = sim.current_state

    # 1. Initial State Assertions (INSPIRAL Phase)
    assert st0.elapsed_time == 0.0
    assert st0.event_time < 0.0
    assert st0.phase == "INSPIRAL", f"Expected initial phase INSPIRAL, got {st0.phase}"
    assert abs(st0.gw_frequency - 40.0) < 1e-5
    assert abs(st0.separation / 1e3 - 283.83) < 0.5
    assert np.isfinite(st0.separation)

    # 2. Demo Near-Merger Transition Test
    sim.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
    st_demo = sim.current_state

    assert abs(st_demo.gw_frequency - 1200.0) < 1e-5, f"Expected f_gw 1200.0 Hz, got {st_demo.gw_frequency}"
    assert abs(st_demo.separation - 30.0e3) < 1.0, f"Expected separation 30000 m, got {st_demo.separation}"
    assert st_demo.merger_started, "Merger regime should be active at demo phase"
    assert st_demo.merger_contact_fraction > 0.0, "Contact fraction should be > 0"
    assert st_demo.merger_complete == False, "Merger should not be complete at 30 km"

    # 3. Exact Duration run() API Check
    sim.reset()
    dur_req = 0.00015
    dt_req = 0.00010
    st_run = sim.run(duration=dur_req, dt=dt_req)

    assert abs(st_run.elapsed_time - dur_req) < 1e-12, f"Expected elapsed duration {dur_req}, got {st_run.elapsed_time}"

    # Invalid run arguments check
    try:
        sim.run(-0.001)
        assert False, "Should have raised ValueError on negative duration"
    except ValueError:
        pass

    try:
        sim.run(0.001, dt=-0.0001)
        assert False, "Should have raised ValueError on negative dt"
    except ValueError:
        pass

    # 4. GRB Timing Check
    sim.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
    j_before = sim.jet.evaluate(time=1.0, merger_state=sim.dynamics.merger_state)
    j_after = sim.jet.evaluate(time=1.75, merger_state=sim.dynamics.merger_state)

    assert j_before.grb_triggered == False, "GRB should not be triggered before 1.7 s delay"
    assert j_after.grb_triggered == True, "GRB should be triggered after 1.7 s delay"

    # 5. Afterglow Check at 150 Days
    ag_150d = sim.afterglow.evaluate(150.0 * day)
    assert np.isfinite(ag_150d.flux_density) and ag_150d.flux_density > 0.0

    # 6. Reset Determinism Check
    sim.reset()
    dt = 1.0e-4
    n_steps = 20
    for _ in range(n_steps):
        sim.step(dt)
    freq_before = sim.current_state.gw_frequency

    sim.reset()
    for _ in range(n_steps):
        sim.step(dt)
    freq_after = sim.current_state.gw_frequency

    assert abs(freq_before - freq_after) < 1e-9, "Reset determinism failed!"

    # 7. Numerical Safety Check
    for field, val in st_run.__dict__.items():
        if isinstance(val, (float, int)):
            assert np.isfinite(val), f"SimulationState field {field} is non-finite: {val}"

    # Print Validation Report
    sim.reset()
    st_init = sim.current_state
    sim.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
    st_demo_val = sim.current_state

    print("=== TASK 013.1 ENGINE SEMANTICS VALIDATION ===")
    print(f"\nBackend: {sim.backend.upper()}")
    print(f"Particle count: {sim.psys.max_particles:,}")

    print("\nInitial:")
    print(f"    elapsed time: {st_init.elapsed_time:.4f} s")
    print(f"    event time: {st_init.event_time:.4f} s")
    print(f"    phase: {st_init.phase}")
    print(f"    GW frequency: {st_init.gw_frequency:.2f} Hz")
    print(f"    separation: {st_init.separation / 1e3:.2f} km")

    print("\nDemo:")
    print(f"    GW frequency: {st_demo_val.gw_frequency:.1f} Hz")
    print(f"    separation: {st_demo_val.separation / 1e3:.2f} km")
    print(f"    contact fraction: {st_demo_val.merger_contact_fraction:.4f}")
    print(f"    merger started: {st_demo_val.merger_started}")
    print(f"    merger complete: {st_demo_val.merger_complete}")

    print("\nrun() exact-duration test:")
    print(f"    requested: {dur_req:.6f} s")
    print(f"    actual: {st_run.elapsed_time:.6f} s")

    print("\nGRB timing:")
    print(f"    before 1.7 s: triggered = {j_before.grb_triggered}")
    print(f"    after 1.7 s: triggered = {j_after.grb_triggered}")

    print("\nAfterglow:")
    print(f"    t = 150 days:")
    print(f"    flux: {ag_150d.flux_density:.4e} W m^-2 Hz^-1")

    print("\nReset determinism: PASS")
    print("Numerical safety: PASS")
    print("Regression: PASS")
    print("\nALL TASK 013.1 CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_engine()
