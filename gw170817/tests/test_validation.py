"""
Test suite for GW170817 Observational Validation Layer (Task 015).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.constants import M_sun, Mpc, day
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.validation.observational_data import GW170817ReferenceData, ValueReference
from gw170817.validation.comparison import compare_value, ComparisonResult
from gw170817.validation.report import ValidationReport


def test_validation():
    print("=== TASK 015 OBSERVATIONAL VALIDATION LAYER TEST ===")

    # 1. Reference Data Internal Validity Check
    ref = GW170817ReferenceData
    ref_list = [
        ref.TOTAL_MASS, ref.CHIRP_MASS, ref.DISTANCE, ref.GRB_DELAY,
        ref.VIEWING_ANGLE, ref.AFTERGLOW_PEAK_TIME, ref.AFTERGLOW_RISE_SLOPE,
        ref.AFTERGLOW_DECAY_SLOPE, ref.EJECTA_MASS_TOTAL
    ]
    for item in ref_list:
        assert isinstance(item, ValueReference)
        min_v = min(item.min_val, item.max_val)
        max_v = max(item.min_val, item.max_val)
        assert min_v <= item.nominal <= max_v, f"Reference {item.name} nominal {item.nominal} outside range [{min_v}, {max_v}]"
    print("1. Reference data internal validity: PASS")

    # 2. Total Mass 2.72 M_sun vs 2.74 M_sun Check
    res_mass = compare_value(ref.TOTAL_MASS, 2.72 * M_sun, tolerance_rel=0.05, unit_scale=M_sun, unit_label="M_sun")
    assert res_mass.passed, f"Total mass 2.72 M_sun should pass against 2.74 M_sun target, got {res_mass}"
    assert res_mass.status == "PASS"
    print("2. Total mass tolerance check (2.72 vs 2.74 M_sun): PASS")

    # 3. Distance 40 Mpc Check
    res_dist = compare_value(ref.DISTANCE, 40.0 * Mpc, tolerance_rel=0.15, unit_scale=Mpc, unit_label="Mpc")
    assert res_dist.passed, "Distance 40 Mpc should pass"
    print("3. Distance 40 Mpc check: PASS")

    # 4. GRB Delay 1.7 s Check
    res_delay = compare_value(ref.GRB_DELAY, 1.70, tolerance_rel=0.15, unit_scale=1.0, unit_label="s")
    assert res_delay.passed, "GRB delay 1.7 s should pass"
    print("4. GRB delay 1.7 s check: PASS")

    # 5. Afterglow Peak 150 d Range Check
    res_peak = compare_value(ref.AFTERGLOW_PEAK_TIME, 150.0, tolerance_rel=0.10, unit_scale=1.0, unit_label="days")
    assert res_peak.passed, "Afterglow peak 150 d should fall within reference range [150, 160]"
    print("5. Afterglow peak 150 d range check: PASS")

    # 6. Afterglow Rise Exponent 0.8 Check
    res_rise = compare_value(ref.AFTERGLOW_RISE_SLOPE, 0.80, tolerance_rel=0.20, unit_scale=1.0, unit_label="dimensionless")
    assert res_rise.passed, "Rise exponent 0.8 should fall within reference range [0.7, 1.0]"
    print("6. Afterglow rise exponent 0.8 check: PASS")

    # 7. Afterglow Decline Exponent -2.2 Check
    res_decay = compare_value(ref.AFTERGLOW_DECAY_SLOPE, -2.20, tolerance_rel=0.20, unit_scale=1.0, unit_label="dimensionless")
    assert res_decay.passed, "Decline exponent -2.2 should fall within reference range [-2.4, -1.9]"
    print("7. Afterglow decline exponent -2.2 check: PASS")

    # 8. Viewing Angle 22 deg Check
    res_angle = compare_value(ref.VIEWING_ANGLE, 22.0, tolerance_rel=0.25, unit_scale=1.0, unit_label="deg")
    assert res_angle.passed, "Viewing angle 22 deg should fall within reference range [15, 28]"
    print("8. Viewing angle 22 deg check: PASS")

    # 9. Comparison Function Rejection of Deliberately Bad Values
    bad_mass = compare_value(ref.TOTAL_MASS, 10.0 * M_sun, tolerance_rel=0.05, unit_scale=M_sun, unit_label="M_sun")
    assert not bad_mass.passed, "10 M_sun total mass should fail"
    assert bad_mass.status == "FAIL"

    bad_delay = compare_value(ref.GRB_DELAY, 100.0, tolerance_rel=0.15, unit_scale=1.0, unit_label="s")
    assert not bad_delay.passed, "100 s GRB delay should fail"
    assert bad_delay.status == "FAIL"
    print("9. Rejection of bad values (10 M_sun, 100 s delay): PASS")

    # 10. Full Report Generation & Numerical Safety Check
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config)
    report = ValidationReport(engine=engine, config=config)

    assert report.all_passed, "Validation report should pass for default simulation engine"
    for r in report.results:
        assert np.isfinite(r.model_value), f"Non-finite model_value in {r.name}"
        assert np.isfinite(r.difference), f"Non-finite difference in {r.name}"
        assert np.isfinite(r.relative_error), f"Non-finite relative_error in {r.name}"

    summary = report.summary_dict()
    assert summary["all_passed"]
    assert summary["passed_count"] == summary["total_count"]
    print("10. Full ValidationReport generation & numerical safety check: PASS")

    print("\n--- FORMATTED VALIDATION REPORT PRINTOUT ---")
    report.print_report()

    print("\nALL TASK 015 OBSERVATIONAL VALIDATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_validation()
