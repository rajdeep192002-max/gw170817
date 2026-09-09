"""
Test suite for reduced-order kilonova light curve model (Task 009).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import c, day, M_sun
from gw170817.physics.ejecta import EjectaState
from gw170817.physics.kilonova import KilonovaModel, KilonovaState

L_sun = 3.828e26  # Solar luminosity [W]


def test_kilonova_model():
    print("=== TEST 009 REDUCED-ORDER KILONOVA MODEL VALIDATION ===")

    config = SimConfig()
    kn_model = KilonovaModel(config)

    # 1. Opacity unit conversion check
    print("1. Opacity Unit Conversion Check:")
    print(f"   - kappa_blue: {kn_model.kappa_blue:.4f} m^2/kg (Expected: 0.1 m^2/kg)")
    print(f"   - kappa_red:  {kn_model.kappa_red:.4f} m^2/kg (Expected: 1.0 m^2/kg)")
    assert abs(kn_model.kappa_blue - 0.1) < 1e-6
    assert abs(kn_model.kappa_red - 1.0) < 1e-6

    # 2. Zero Ejecta Test
    state_zero_ej = EjectaState(
        time=0.0, ejecta_mass=0.0, ejecta_fraction=0.0, ejecta_particle_count=0,
        mean_velocity=0.0, max_velocity=0.0, kinetic_energy=0.0, angular_momentum_proxy=0.0,
        mean_Ye=0.05, lanthanide_rich_fraction=0.5
    )
    kn_zero = kn_model.evaluate(state_zero_ej, t_seconds=day)
    print("\n2. Zero Ejecta Test:")
    print(f"   - L_total = {kn_zero.L_total:.2f} W, T_blue = {kn_zero.T_blue:.2f} K, R_blue = {kn_zero.R_blue:.2f} m")
    assert kn_zero.L_blue == 0.0 and kn_zero.L_red == 0.0 and kn_zero.L_total == 0.0
    assert kn_zero.T_blue == 0.0 and kn_zero.T_red == 0.0
    assert kn_zero.R_blue == 0.0 and kn_zero.R_red == 0.0
    assert not np.isnan(kn_zero.L_total) and not np.isinf(kn_zero.L_total)

    # 3. Positive Ejecta & Diffusion Timescale Test
    m_ej_test = 0.03 * M_sun  # 0.03 M_sun
    v_ej_test = 0.15 * c      # 0.15 c
    state_pos_ej = EjectaState(
        time=0.0, ejecta_mass=m_ej_test, ejecta_fraction=0.01, ejecta_particle_count=1000,
        mean_velocity=v_ej_test, max_velocity=0.25 * c, kinetic_energy=1e43, angular_momentum_proxy=1e40,
        mean_Ye=0.20, lanthanide_rich_fraction=0.5
    )

    kn_pos = kn_model.evaluate(state_pos_ej, t_seconds=1.0 * day)

    print("\n3. Positive Ejecta Test (t = 1.0 day):")
    print(f"   - M_blue = {kn_pos.M_blue / M_sun:.4f} M_sun, M_red = {kn_pos.M_red / M_sun:.4f} M_sun")
    print(f"   - t_diff_blue = {kn_pos.t_diff_blue / day:.2f} days")
    print(f"   - t_diff_red  = {kn_pos.t_diff_red / day:.2f} days")
    print(f"   - L_blue = {kn_pos.L_blue:.4e} W ({kn_pos.L_blue / L_sun:.4e} L_sun)")
    print(f"   - L_red  = {kn_pos.L_red:.4e} W ({kn_pos.L_red / L_sun:.4e} L_sun)")
    print(f"   - L_total = {kn_pos.L_total:.4e} W ({kn_pos.L_total / L_sun:.4e} L_sun)")
    print(f"   - T_blue = {kn_pos.T_blue:.1f} K, T_red = {kn_pos.T_red:.1f} K")

    assert kn_pos.L_total > 0.0
    assert kn_pos.t_diff_red > kn_pos.t_diff_blue, "t_diff_red must be > t_diff_blue due to higher kappa_red"
    assert kn_pos.T_blue > kn_pos.T_red, "Blue component should generally be hotter at early times"

    # 4. Component Split Test (lanthanide_rich_fraction = 0 vs 1)
    state_pure_blue = EjectaState(0.0, m_ej_test, 0.01, 1000, v_ej_test, 0.25*c, 1e43, 1e40, 0.35, 0.0)
    state_pure_red = EjectaState(0.0, m_ej_test, 0.01, 1000, v_ej_test, 0.25*c, 1e43, 1e40, 0.10, 1.0)

    kn_blue_only = kn_model.evaluate(state_pure_blue, 1.0 * day)
    kn_red_only = kn_model.evaluate(state_pure_red, 1.0 * day)

    print("\n4. Component Mass Split Test:")
    print(f"   - Pure Blue (frac=0.0): M_blue = {kn_blue_only.M_blue / M_sun:.4f} M_sun, M_red = {kn_blue_only.M_red:.4f}")
    print(f"   - Pure Red  (frac=1.0): M_blue = {kn_red_only.M_blue:.4f}, M_red = {kn_red_only.M_red / M_sun:.4f} M_sun")

    assert kn_blue_only.M_red == 0.0 and abs(kn_blue_only.M_blue - m_ej_test) < 1e-6
    assert kn_red_only.M_blue == 0.0 and abs(kn_red_only.M_red - m_ej_test) < 1e-6

    # 5. Time Evolution Test (0.1 day, 1 day, 10 days)
    print("\n5. Time Evolution Test:")
    times = [0.1 * day, 1.0 * day, 10.0 * day]
    for t_val in times:
        st = kn_model.evaluate(state_pos_ej, t_val)
        print(f"   - t = {t_val / day:4.1f} days -> L_tot = {st.L_total:.4e} W, T_blue = {st.T_blue:6.1f} K, T_red = {st.T_red:6.1f} K")
        assert np.isfinite(st.L_total) and st.L_total >= 0.0
        assert np.isfinite(st.T_blue) and st.T_blue >= 0.0
        assert np.isfinite(st.T_red) and st.T_red >= 0.0

    # 6. Determinism Check
    kn_det1 = kn_model.evaluate(state_pos_ej, 2.5 * day)
    kn_det2 = kn_model.evaluate(state_pos_ej, 2.5 * day)
    assert kn_det1.L_total == kn_det2.L_total, "Deterministic luminosity mismatch!"

    print("\nALL TEST_KILONOVA CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_kilonova_model()
