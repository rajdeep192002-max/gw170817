"""
Test suite for reduced-order structured relativistic jet model (Task 010).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import Mpc
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.tidal import TidalModel
from gw170817.physics.merger import MergerModel
from gw170817.physics.jet import StructuredJetModel, JetState


def test_structured_jet():
    print("=== TEST 010 STRUCTURED RELATIVISTIC JET VALIDATION ===")
    print("Phenomenological structured-jet validation report:")

    config = SimConfig()
    jet_model = StructuredJetModel(config)

    # A. Core Profile Test
    E_0 = jet_model.energy_profile(0.0)
    Gamma_0 = jet_model.lorentz_profile(0.0)
    print(f"\nA. Core Profile Check (theta = 0):")
    print(f"   - Core Energy E(0) = {E_0:.4e} J (Expected: {jet_model.E_core:.4e} J)")
    print(f"   - Core Gamma(0)   = {Gamma_0:.1f} (Expected: {jet_model.Gamma_core:.1f})")

    assert abs(E_0 - jet_model.E_core) < 1e-6
    assert abs(Gamma_0 - jet_model.Gamma_core) < 1e-6

    # B. Large-Angle Behavior Test
    theta_large = np.deg2rad(45.0)
    E_large = jet_model.energy_profile(theta_large)
    Gamma_large = jet_model.lorentz_profile(theta_large)

    print(f"\nB. Large-Angle Check (theta = 45 deg):")
    print(f"   - Energy E(45 deg) = {E_large:.4e} J (Decreased from E_core)")
    print(f"   - Gamma(45 deg)   = {Gamma_large:.4f} (Approaching 1.0)")

    assert E_large < E_0
    assert 1.0 <= Gamma_large < Gamma_0

    # C. Lorentz Factor & Beta Test
    beta_0 = jet_model.beta(Gamma_0)
    beta_large = jet_model.beta(Gamma_large)
    print(f"\nC. Relativistic Kinematics Check:")
    print(f"   - beta(Gamma=100) = {beta_0:.8f}")
    print(f"   - beta(Gamma={Gamma_large:.4f}) = {beta_large:.8f}")

    assert 0.0 <= beta_0 < 1.0
    assert 0.0 <= beta_large < 1.0

    # D & E. Doppler Factor & Viewing Angle Dependence
    delta_on_axis = jet_model.doppler_factor(Gamma_0, alpha=0.0)
    delta_off_axis = jet_model.doppler_factor(Gamma_0, alpha=np.deg2rad(22.0))

    state_on_axis = jet_model.evaluate(theta=0.0, alpha=0.0)
    state_off_axis = jet_model.evaluate(theta=np.deg2rad(22.0), alpha=np.deg2rad(22.0))

    print(f"\nD & E. Doppler Boosting & Viewing Angle Check:")
    print(f"   - On-axis Doppler factor delta(0 deg)   = {delta_on_axis:.2f}")
    print(f"   - Off-axis Doppler factor delta(22 deg) = {delta_off_axis:.4f}")
    print(f"   - On-axis Observed Flux  = {state_on_axis.observed_flux:.4e} W/m^2")
    print(f"   - Off-axis Observed Flux = {state_off_axis.observed_flux:.4e} W/m^2")

    assert delta_on_axis > delta_off_axis, "On-axis Doppler factor must exceed off-axis"
    assert state_on_axis.observed_flux > state_off_axis.observed_flux, "On-axis flux must exceed off-axis"

    # F. Distance Scaling Check F(2D) == F(D)/4
    state_dist1 = jet_model.evaluate(theta=0.0)
    config_double_d = SimConfig(distance_Mpc=80.0)
    jet_model_double_d = StructuredJetModel(config_double_d)
    state_dist2 = jet_model_double_d.evaluate(theta=0.0)

    ratio_flux = state_dist1.observed_flux / state_dist2.observed_flux
    print(f"\nF. Inverse-Square Distance Scaling Check (D -> 2D):")
    print(f"   - Flux Ratio F(D) / F(2D) = {ratio_flux:.4f} (Expected: 4.0000)")

    assert abs(ratio_flux - 4.0) < 1e-5

    # G & H. GW-GRB Delay & Merger Coupling Test
    inspiral_model = InspiralModel(config)
    tidal_model = TidalModel(config)
    merger_model = MergerModel(config, tidal_model)

    state_before_merger = inspiral_model.initial_state()
    merger_before = merger_model.evaluate(state_before_merger)

    # Synthetic completed merger state
    state_after_merger = inspiral_model.initial_state()
    state_after_merger.separation = 15.0e3  # 15 km < a_contact (24 km)
    merger_after = merger_model.evaluate(state_after_merger)

    jet_before = jet_model.evaluate(time=0.0, merger_state=merger_before)
    jet_after_prompt = jet_model.evaluate(time=merger_after.time + 1.0, merger_state=merger_after)
    jet_after_delay = jet_model.evaluate(time=merger_after.time + 1.74, merger_state=merger_after)

    print(f"\nG & H. Merger Coupling & 1.7s Delay Check:")
    print(f"   - GW-GRB Delay Configured = {jet_model.jet_delay:.2f} s")
    print(f"   - Before Merger (a = {merger_before.separation/1e3:.1f} km): launched = {jet_before.launched}, grb_triggered = {jet_before.grb_triggered}")
    print(f"   - After Merger (t = +1.0 s): launched = {jet_after_prompt.launched}, grb_triggered = {jet_after_prompt.grb_triggered}")
    print(f"   - After Delay  (t = +1.74 s): launched = {jet_after_delay.launched}, grb_triggered = {jet_after_delay.grb_triggered}")

    assert not jet_before.launched and not jet_before.grb_triggered
    assert jet_after_prompt.launched and not jet_after_prompt.grb_triggered
    assert jet_after_delay.launched and jet_after_delay.grb_triggered

    # I, J & K. Numerical Safety, Determinism & Vectorization
    print(f"\nI, J & K. Vectorization, Determinism & Numerical Safety Check:")
    theta_grid = np.linspace(0, np.pi/4, 100)
    E_grid = jet_model.energy_profile(theta_grid)
    Gamma_grid = jet_model.lorentz_profile(theta_grid)

    assert len(E_grid) == 100 and len(Gamma_grid) == 100
    assert np.all(np.isfinite(E_grid)) and np.all(np.isfinite(Gamma_grid))

    for field, val in state_off_axis.__dict__.items():
        if isinstance(val, (float, int)):
            assert np.isfinite(val), f"Field {field} is non-finite: {val}"

    state_det1 = jet_model.evaluate(theta=np.deg2rad(22.0))
    state_det2 = jet_model.evaluate(theta=np.deg2rad(22.0))
    assert state_det1.observed_flux == state_det2.observed_flux

    print("\nALL TEST_JET CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_structured_jet()
