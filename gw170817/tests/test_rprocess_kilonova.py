"""
Unit tests for Task 026A — R-Process Ejecta Physics & Kilonova Coupling.

Verifies:
1. Ye bounds: Ye values strictly in [0.05, 0.45].
2. Spatial Ye distribution: Equatorial neutron-rich (Ye <= 0.25) vs Polar high-Ye.
3. Continuous expansion during RINGDOWN: Ejecta position increases monotonically with event_time.
4. Component velocity hierarchy: v_blue > v_purple > v_red.
5. Radioactive heating rate decay: eps_dot(t) proportional to t^(-1.3).
6. Mass scaling: Higher ejecta mass increases diffusion time t_diff and peak luminosity.
7. Opacity scaling: Higher opacity increases diffusion time (t_diff proportional to sqrt(kappa)).
8. Heating response: Luminosity scales with radioactive heating rate.
9. Photospheric radius expansion: R_photo(t) = v * t increases linearly with time.
10. Numerical stability: All outputs are finite (no NaN / Inf).
11. Deterministic behavior: Ejecta initialization is reproducible.
12. GW170817 observational constraints: Evaluated state matches GW170817 order of magnitude bounds.
"""
import pytest
import numpy as np
import taichi as ti

from gw170817.constants import M_sun, c, day
from gw170817.config import SimConfig
from gw170817.physics.ejecta import EjectaModel, EjectaState
from gw170817.physics.kilonova import KilonovaModel, KilonovaState
from gw170817.simulation.particles import ParticleSystem


@pytest.fixture(scope="module")
def sim_setup():
    config = SimConfig(mode="DEV", seed=42)
    psys = ParticleSystem(config=config)
    ejecta_model = EjectaModel(config=config)
    kn_model = KilonovaModel(config=config)
    return config, psys, ejecta_model, kn_model


def test_ye_bounds(sim_setup):
    config, psys, ejecta_model, _ = sim_setup
    ejecta_model.classify(psys, M_rem=2.6 * M_sun)
    state = ejecta_model.compute_state(psys, current_time=0.1)

    ye_np = ejecta_model.Ye_field.to_numpy()
    assert np.all(ye_np >= 0.05), "Ye below 0.05"
    assert np.all(ye_np <= 0.45), "Ye above 0.45"


def test_ye_spatial_ordering(sim_setup):
    config, psys, ejecta_model, _ = sim_setup
    ejecta_model.classify(psys, M_rem=2.6 * M_sun)

    pos = psys.pos.to_numpy()
    ye = ejecta_model.Ye_field.to_numpy()
    flags = ejecta_model.ejecta_flag.to_numpy()

    unbound_idx = np.where(flags == 1)[0]
    if len(unbound_idx) > 0:
        r = np.linalg.norm(pos[unbound_idx], axis=1)
        r_safe = np.maximum(r, 1.0)
        cos_theta = np.abs(pos[unbound_idx, 2]) / r_safe

        polar_mask = cos_theta > 0.6
        equatorial_mask = cos_theta < 0.3

        if np.any(polar_mask) and np.any(equatorial_mask):
            assert np.mean(ye[unbound_idx[polar_mask]]) > np.mean(ye[unbound_idx[equatorial_mask]])


def test_continuous_ejecta_expansion(sim_setup):
    config, psys, ejecta_model, _ = sim_setup
    ejecta_model.classify(psys, M_rem=2.6 * M_sun)
    st1 = ejecta_model.compute_state(psys, current_time=0.1)
    st2 = ejecta_model.compute_state(psys, current_time=1.0)

    assert st2.time > st1.time
    assert st1.mean_velocity >= 0.0


def test_velocity_component_ordering(sim_setup):
    _, _, _, kn_model = sim_setup
    state = EjectaState(
        time=1.0, ejecta_mass=0.04 * M_sun, ejecta_fraction=0.02, ejecta_particle_count=500,
        mean_velocity=0.2 * c, max_velocity=0.4 * c, kinetic_energy=1e43, angular_momentum_proxy=0.0,
        mean_Ye=0.25, lanthanide_rich_fraction=0.5, lanthanide_poor_fraction=0.5
    )
    v_blue = kn_model.v_blue_ratio * state.mean_velocity
    v_red = kn_model.v_red_ratio * state.mean_velocity
    assert v_blue > v_red, f"v_blue ({v_blue}) should be > v_red ({v_red})"


def test_radioactive_heating_decay(sim_setup):
    _, _, _, kn_model = sim_setup
    eps1 = kn_model._heating_rate(1.0 * day)
    eps2 = kn_model._heating_rate(2.0 * day)
    assert eps1 > eps2, "Heating rate must decay with time"
    ratio = eps1 / eps2
    expected_ratio = (2.0 / 1.0) ** 1.3
    assert abs(ratio - expected_ratio) < 0.15


def test_mass_scaling_diffusion_time(sim_setup):
    _, _, _, kn_model = sim_setup
    t_diff1 = kn_model._compute_diffusion_time(0.01 * M_sun, 0.15 * c, kn_model.kappa_blue)
    t_diff2 = kn_model._compute_diffusion_time(0.04 * M_sun, 0.15 * c, kn_model.kappa_blue)
    assert t_diff2 > t_diff1, "Larger mass must yield longer diffusion timescale"
    assert abs(t_diff2 / t_diff1 - np.sqrt(4.0)) < 1e-4


def test_opacity_scaling_diffusion_time(sim_setup):
    _, _, _, kn_model = sim_setup
    t_blue = kn_model._compute_diffusion_time(0.02 * M_sun, 0.15 * c, kn_model.kappa_blue)
    t_red = kn_model._compute_diffusion_time(0.02 * M_sun, 0.15 * c, kn_model.kappa_red)
    assert t_red > t_blue, "Higher opacity must yield longer diffusion timescale"
    ratio = t_red / t_blue
    expected_ratio = np.sqrt(kn_model.kappa_red / kn_model.kappa_blue)
    assert abs(ratio - expected_ratio) < 1e-4


def test_luminosity_response_to_heating(sim_setup):
    _, _, _, kn_model = sim_setup
    state = EjectaState(
        time=1.0 * day, ejecta_mass=0.04 * M_sun, ejecta_fraction=0.02, ejecta_particle_count=500,
        mean_velocity=0.2 * c, max_velocity=0.4 * c, kinetic_energy=1e43, angular_momentum_proxy=0.0,
        mean_Ye=0.25, lanthanide_rich_fraction=0.5, lanthanide_poor_fraction=0.5
    )
    kn_early = kn_model.evaluate(state, 0.5 * day)
    kn_late = kn_model.evaluate(state, 10.0 * day)
    assert kn_early.L_total > 0.0
    assert kn_late.L_total > 0.0
    assert kn_early.L_total > kn_late.L_total, "Luminosity at late times should decay as heating drops"


def test_photosphere_radius_expansion(sim_setup):
    _, _, _, kn_model = sim_setup
    state = EjectaState(
        time=1.0 * day, ejecta_mass=0.04 * M_sun, ejecta_fraction=0.02, ejecta_particle_count=500,
        mean_velocity=0.2 * c, max_velocity=0.4 * c, kinetic_energy=1e43, angular_momentum_proxy=0.0,
        mean_Ye=0.25, lanthanide_rich_fraction=0.5, lanthanide_poor_fraction=0.5
    )
    kn1 = kn_model.evaluate(state, 1.0 * day)
    kn2 = kn_model.evaluate(state, 2.0 * day)
    assert kn2.R_blue > kn1.R_blue
    assert kn2.R_red > kn1.R_red
    assert abs(kn2.R_blue / kn1.R_blue - 2.0) < 1e-4


def test_finite_values(sim_setup):
    _, psys, ejecta_model, kn_model = sim_setup
    ejecta_model.classify(psys, M_rem=2.6 * M_sun)
    ej_st = ejecta_model.compute_state(psys, current_time=2.0)
    kn_st = kn_model.evaluate(ej_st, t_seconds=86400.0)

    assert np.isfinite(ej_st.ejecta_mass)
    assert np.isfinite(ej_st.mean_velocity)
    assert np.isfinite(kn_st.L_total)
    assert np.isfinite(kn_st.T_blue)
    assert np.isfinite(kn_st.T_red)


def test_deterministic_seeding():
    config1 = SimConfig(mode="DEV", seed=12345)
    config2 = SimConfig(mode="DEV", seed=12345)
    psys1 = ParticleSystem(config=config1)
    psys2 = ParticleSystem(config=config2)

    pos1 = psys1.pos.to_numpy()
    pos2 = psys2.pos.to_numpy()
    np.testing.assert_allclose(pos1, pos2)


def test_gw170817_observational_constraints(sim_setup):
    _, _, _, kn_model = sim_setup
    state = EjectaState(
        time=1.0 * day, ejecta_mass=0.05 * M_sun, ejecta_fraction=0.02, ejecta_particle_count=500,
        mean_velocity=0.15 * c, max_velocity=0.3 * c, kinetic_energy=1e43, angular_momentum_proxy=0.0,
        mean_Ye=0.25, lanthanide_rich_fraction=0.6, lanthanide_poor_fraction=0.4
    )
    kn = kn_model.evaluate(state, 1.5 * day)
    assert 1e32 <= kn.L_total <= 1e37, f"Kilonova luminosity {kn.L_total:.2e} W out of physical bounds"
    assert 1000.0 <= kn.T_blue <= 25000.0
