"""
Test suite for reduced-order broadband synchrotron afterglow model (Task 011).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import day, Mpc
from gw170817.physics.jet import StructuredJetModel
from gw170817.physics.afterglow import AfterglowModel, AfterglowState


def test_afterglow_model():
    config = SimConfig()
    model = AfterglowModel(config)

    # 15. Validation Report Header
    print("=== TASK 011 BROADBAND AFTERGLOW VALIDATION ===")
    print(f"Peak time:           {model.t_peak_days:.1f} days ({model.t_peak:.4e} s)")
    print(f"Rise index:          {model.rise_index}")
    print(f"Decay index:         {model.decay_index}")
    print(f"Spectral index:      {model.beta}")
    print(f"Reference frequency: {model.nu_ref / 1e9:.1f} GHz")
    print(f"Reference peak flux: {model.F_peak_ref:.4e} W m^-2 Hz^-1")

    # A. Peak Definition
    st_peak = model.evaluate(model.t_peak)
    print(f"\nA. Peak Definition Check (t = 150 days):")
    print(f"   - Peak Flux F(150d) = {st_peak.flux_density:.4e} W m^-2 Hz^-1 (Expected: {st_peak.peak_flux:.4e})")
    assert abs(st_peak.flux_density - st_peak.peak_flux) < 1e-35

    # B & C. Temporal Evolution (Rising & Falling Phases)
    test_days = [10.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
    fluxes = {}
    print("\nB & C. Temporal Evolution:")
    for d in test_days:
        st = model.evaluate(d * day)
        fluxes[d] = st.flux_density
        print(f"   - {d:5.1f} day flux: {st.flux_density:.4e} W m^-2 Hz^-1 ({st.phase})")

    # Rising phase assertions
    assert fluxes[10.0] < fluxes[50.0] < fluxes[100.0] < fluxes[150.0]
    # Falling phase assertions
    assert fluxes[150.0] > fluxes[200.0] > fluxes[300.0] > fluxes[500.0]

    # D. Peak Location Grid Search
    grid_days = np.logspace(0, 3, 500)
    grid_fluxes = np.array([model.evaluate(d * day).flux_density for d in grid_days])
    peak_day_found = grid_days[np.argmax(grid_fluxes)]
    print(f"\nD. Peak Grid Location Check: Found peak at {peak_day_found:.2f} days (Expected: 150.0 days)")
    assert abs(peak_day_found - 150.0) < 5.0

    # E. Power-Law Slope Check
    ratio_rise_actual = fluxes[100.0] / fluxes[50.0]
    ratio_rise_theory = (100.0 / 50.0)**model.rise_index

    ratio_decay_actual = fluxes[300.0] / fluxes[150.0]
    ratio_decay_theory = (300.0 / 150.0)**model.decay_index

    print(f"\nE. Power-Law Slope Verification:")
    print(f"   - Pre-peak ratio  F(100d)/F(50d)  = {ratio_rise_actual:.4f} (Theory: {ratio_rise_theory:.4f})")
    print(f"   - Post-peak ratio F(300d)/F(150d) = {ratio_decay_actual:.4f} (Theory: {ratio_decay_theory:.4f})")

    assert abs(ratio_rise_actual - ratio_rise_theory) < 1e-4
    assert abs(ratio_decay_actual - ratio_decay_theory) < 1e-4

    # F. Spectral Scaling Check
    freqs = {
        "1 GHz":   1.0e9,
        "3 GHz":   3.0e9,
        "optical": 5.0e14,
        "X-ray":   1.0e18
    }
    print(f"\nF. Spectral Scaling (F_nu ~ nu^-{model.beta}):")
    st_1g = model.evaluate(150.0 * day, freqs["1 GHz"])
    for name, f_val in freqs.items():
        st_freq = model.evaluate(150.0 * day, f_val)
        ratio_freq_actual = st_freq.flux_density / st_1g.flux_density
        ratio_freq_theory = (f_val / freqs["1 GHz"])**(-model.beta)
        print(f"   - {name:7s} ({f_val:.1e} Hz): F_nu = {st_freq.flux_density:.4e} W m^-2 Hz^-1 (Ratio vs 1GHz: {ratio_freq_actual:.4e}, Theory: {ratio_freq_theory:.4e})")
        assert abs(ratio_freq_actual - ratio_freq_theory) < 1e-4

    # G. Jet Coupling Check
    jet_model = StructuredJetModel(config)
    jet_on_axis = jet_model.evaluate(theta=0.0, alpha=0.0)
    jet_off_axis = jet_model.evaluate(theta=np.deg2rad(22.0), alpha=np.deg2rad(22.0))

    st_jet_on = model.evaluate(150.0 * day, jet_state=jet_on_axis)
    st_jet_off = model.evaluate(150.0 * day, jet_state=jet_off_axis)

    print(f"\nG. Jet Coupling Check:")
    print(f"   - On-axis jet boosted peak flux  = {st_jet_on.peak_flux:.4e} W m^-2 Hz^-1")
    print(f"   - Off-axis jet boosted peak flux = {st_jet_off.peak_flux:.4e} W m^-2 Hz^-1")
    assert st_jet_on.peak_flux > st_jet_off.peak_flux

    # H. Distance Scaling Check F(2D) == F(D)/4
    config_double_d = SimConfig(distance_Mpc=80.0)
    model_double_d = AfterglowModel(config_double_d)
    st_d1 = model.evaluate(150.0 * day)
    st_d2 = model_double_d.evaluate(150.0 * day)

    ratio_dist = st_d1.flux_density / st_d2.flux_density
    print(f"\nH. Distance Scaling Check (D -> 2D): Ratio F(D)/F(2D) = {ratio_dist:.4f} (Expected: 4.0000)")
    assert abs(ratio_dist - 4.0) < 1e-5

    # I & J. Numerical Safety & Vectorization
    print(f"\nI & J. Numerical Safety & Vectorization Check:")
    early_times = [1.0, 100.0, 1.0 * day, 150.0 * day, 1000.0 * day]
    for t_s in early_times:
        st_t = model.evaluate(t_s)
        assert np.isfinite(st_t.flux_density) and st_t.flux_density >= 0.0

    # Test invalid time & frequency handling
    st_invalid_t = model.evaluate(-10.0)
    assert st_invalid_t.flux_density == 0.0

    st_invalid_nu = model.evaluate(150.0 * day, frequency_hz=-1.0)
    assert st_invalid_nu.flux_density == 0.0

    # Array vectorization test
    t_arr = np.array([10.0, 150.0, 300.0]) * day
    vec_fluxes = model.temporal_flux(t_arr)
    assert len(vec_fluxes) == 3 and np.all(np.isfinite(vec_fluxes))

    print("\nALL TEST_AFTERGLOW CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_afterglow_model()
