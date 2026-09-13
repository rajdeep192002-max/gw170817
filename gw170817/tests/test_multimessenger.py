"""
Test suite for GW170817 Multi-Messenger Event Coordinator & Scenario Integration (Task 016).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.constants import day, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.multimessenger import MultiMessengerCoordinator, MultiMessengerEventState
from gw170817.simulation.demo_scenario import DemoScenario
from gw170817.validation.report import ValidationReport


def test_multimessenger():
    print("=== TASK 016 MULTI-MESSENGER EVENT INTEGRATION TEST ===")

    # 1. Integration Initializes Successfully
    config = SimConfig(mode="DEV", seed=42)
    coord = MultiMessengerCoordinator(config=config)
    scenario = DemoScenario(coordinator=coord, config=config)

    assert coord is not None
    assert scenario is not None

    st0 = coord.current_state
    assert isinstance(st0, MultiMessengerEventState)
    print("1. Initialization: PASS")

    # 2. Initial State is INSPIRAL
    assert st0.phase == "INSPIRAL", f"Expected initial phase INSPIRAL, got {st0.phase}"
    assert st0.event_time < 0.0
    print("2. Initial INSPIRAL phase: PASS")

    # 3. GW Frequency is Finite and Positive
    assert np.isfinite(st0.gw_frequency) and st0.gw_frequency > 0.0
    assert abs(st0.gw_frequency - 40.0) < 1e-5
    print("3. Finite positive GW frequency: PASS")

    # 4. Inspiral Frequency Increases with Time
    st_prev = st0
    for _ in range(50):
        st_next = coord.step(1.0e-4)
    assert st_next.gw_frequency > st_prev.gw_frequency, "GW frequency must increase during inspiral"
    print("4. Inspiral frequency increase: PASS")

    # 5. Merger Transition Works
    coord.reset()
    st_merger = coord.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
    assert st_merger.merger_started, "Merger regime should be started"
    assert st_merger.contact_fraction > 0.0
    assert st_merger.phase in ["MERGER", "RINGDOWN"]
    print("5. Merger transition: PASS")

    # 6. Event Time Consistency with EventTimeline
    assert abs(st_merger.event_time - 0.0) < 1e-9
    assert coord.engine.timeline.current_phase(0.0) == "MERGER"
    print("6. Event time consistency: PASS")

    # 7. GRB Activates at ~ +1.7 s
    coord.jump_to_demo_phase(f_gw=1500.0, separation=15.0e3)
    st_grb_before = coord.evaluate_at_event_time(1.0)
    st_grb_after = coord.evaluate_at_event_time(1.74)

    assert not st_grb_before.grb_triggered, "GRB should not trigger before delay"
    assert st_grb_after.grb_triggered, "GRB should trigger at +1.74 s"
    assert st_grb_after.phase == "RINGDOWN"
    print("7. GRB prompt activation at +1.7 s: PASS")

    # 8. Kilonova State Available After Merger
    st_kn = coord.evaluate_at_event_time(1.0 * day)
    assert st_kn.kilonova_luminosity > 0.0
    assert st_kn.kilonova_t_blue > 0.0
    assert st_kn.kilonova_t_red > 0.0
    assert np.isfinite(st_kn.ejecta_mean_ye)
    assert st_kn.ejecta_radioactive_heating_rate > 0.0
    assert st_kn.ejecta_opacity_mean >= 0.0
    print("8. Post-merger kilonova evaluation: PASS")

    # 9. Afterglow State Available at Late Time
    st_ag = coord.evaluate_at_event_time(150.0 * day)
    assert st_ag.afterglow_flux > 0.0
    assert st_ag.phase == "RINGDOWN"
    print("9. Late-time afterglow evaluation: PASS")

    # 10. Afterglow Peak Occurs at ~150 Days
    assert abs(st_ag.afterglow_peak_time_days - 150.0) < 1e-5
    print("10. Afterglow peak time 150 days: PASS")

    # 11. Validation Report Can Be Requested
    val_rep = coord.validation_report()
    assert isinstance(val_rep, ValidationReport)
    assert val_rep.all_passed
    assert st0.validation_passed
    assert st0.validation_passed_count == st0.validation_total_count
    print("11. Validation report request: PASS")

    # 12. All Numerical Outputs are Finite
    for field, val in st_ag.__dict__.items():
        if isinstance(val, (float, int)):
            assert np.isfinite(val), f"Field {field} is non-finite: {val}"
    print("12. Numerical safety across fields: PASS")

    # 13. Reset is Deterministic
    coord.reset()
    st_res1 = coord.current_state
    coord.step(0.001)
    st_step1 = coord.current_state

    coord.reset()
    st_res2 = coord.current_state
    coord.step(0.001)
    st_step2 = coord.current_state

    assert abs(st_res1.gw_frequency - st_res2.gw_frequency) < 1e-9
    assert abs(st_step1.gw_frequency - st_step2.gw_frequency) < 1e-9
    print("13. Reset determinism: PASS")

    # Demo Scenario Checkpoints Test
    st_cp1 = scenario.jump_to_checkpoint("INSPIRAL_START")
    assert st_cp1.phase == "INSPIRAL"

    st_cp3 = scenario.jump_to_checkpoint("MERGER_DEMO")
    assert abs(st_cp3.gw_frequency - 1200.0) < 1e-5

    st_cp4 = scenario.jump_to_checkpoint("GRB_PROMPT")
    assert st_cp4.grb_triggered

    st_cp6 = scenario.jump_to_checkpoint("AFTERGLOW_PEAK")
    assert st_cp6.phase == "RINGDOWN"
    print("Demo Scenario Checkpoints: PASS")

    print("\nALL TASK 016 MULTI-MESSENGER INTEGRATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_multimessenger()
