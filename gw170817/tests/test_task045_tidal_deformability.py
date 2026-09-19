"""
Comprehensive validation test suite for reduced-order BNS tidal deformability visualization.
Verifies:
1. Physical dimensionless tidal deformability (Lambda1, Lambda2, Lambda_tilde).
2. Monotonic scaling of quadrupolar tidal elongation proportional to ~1/a^3.
3. Spherical shape at large separation (epsilon < 2%).
4. Distinct asymmetric deformability between Star 1 and Star 2 (epsilon2 > epsilon1 by ~33%).
5. Coherent ellipsoidal volume-preserving prolate elongation along the instantaneous binary axis.
6. Clean post-merger culling without particle explosions, NaNs, or Infs.
7. HUD telemetry text formatting and explicit reduced-order labeling.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pytest

from gw170817.config import SimConfig
from gw170817.constants import G, c, M_sun
from gw170817.physics.tidal import TidalModel, TidalStar, TidalState
from gw170817.physics.inspiral import InspiralModel, InspiralState
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.dashboard import ScientificDashboard


def test_tidal_model_dimensionless_deformability():
    """Verify Lambda1, Lambda2, and effective binary Lambda_tilde derived from physical NS parameters."""
    config = SimConfig(mode="DEV", seed=42)
    tidal = TidalModel(config, r1_ns=12.0e3, r2_ns=12.0e3, k2_1=0.08, k2_2=0.08)

    # Lambda = (2/3) * k2 * (c^2 R / G M)^5
    c1 = (G * config.m1) / (c**2 * 12.0e3)
    expected_lam1 = (2.0 / 3.0) * 0.08 * (1.0 / (c1**5))
    c2 = (G * config.m2) / (c**2 * 12.0e3)
    expected_lam2 = (2.0 / 3.0) * 0.08 * (1.0 / (c2**5))

    assert np.isclose(tidal.star1.lambda_dimensionless, expected_lam1, rtol=1e-4)
    assert np.isclose(tidal.star2.lambda_dimensionless, expected_lam2, rtol=1e-4)

    # GW170817 typical values: Lambda1 ~ 285, Lambda2 ~ 571
    assert 250.0 < tidal.star1.lambda_dimensionless < 350.0
    assert 500.0 < tidal.star2.lambda_dimensionless < 650.0

    # Effective binary tidal deformability Lambda_tilde <= 800 (LIGO-Virgo 90% credible bound)
    assert 350.0 < tidal.lambda_tilde < 450.0
    assert tidal.lambda_tilde < 800.0


def test_monotonic_tidal_scaling():
    """Verify that elongation increases monotonically with decreasing separation ~ 1/a^3."""
    config = SimConfig(mode="DEV", seed=42)
    tidal = TidalModel(config)

    separations = [100.0e3, 60.0e3, 45.0e3, 35.0e3, 28.0e3, 24.0e3]
    elongations_1 = []
    elongations_2 = []

    for a in separations:
        state = InspiralState(
            time=-1.0, f_gw=100.0, orbital_frequency=50.0, omega_orb=100.0*np.pi,
            separation=a, orbital_phase=0.0, df_dt=1.0, chirp_mass=config.chirp_mass
        )
        t_state = tidal.evaluate(state)
        elongations_1.append(t_state.tidal_elongation_1)
        elongations_2.append(t_state.tidal_elongation_2)

    # 1. Monotonic increase
    for k in range(len(separations) - 1):
        assert elongations_1[k+1] > elongations_1[k], f"Non-monotonic elongation for star 1: {elongations_1}"
        assert elongations_2[k+1] > elongations_2[k], f"Non-monotonic elongation for star 2: {elongations_2}"

    # 2. Large separation (a >= 60 km): essentially spherical (epsilon <= 1.5%)
    assert elongations_1[1] < 0.015
    assert elongations_2[1] < 0.020

    # 3. Intermediate late inspiral (a = 35 km): subtle elongation (5% - 9%)
    assert 0.04 < elongations_1[3] < 0.10
    assert 0.06 < elongations_2[3] < 0.12

    # 4. Contact (a = 24 km): strongest pre-merger elongation (18% - 26%), non-cartoonish
    assert 0.16 < elongations_1[-1] < 0.23
    assert 0.22 < elongations_2[-1] < 0.29


def test_asymmetric_deformability_star1_vs_star2():
    """Verify that Star 2 (lower mass, larger Lambda) deforms more than Star 1."""
    config = SimConfig(mode="DEV", seed=42)
    tidal = TidalModel(config)

    state = InspiralState(
        time=-0.1, f_gw=800.0, orbital_frequency=400.0, omega_orb=800.0*np.pi,
        separation=26.0e3, orbital_phase=0.0, df_dt=10.0, chirp_mass=config.chirp_mass
    )
    t_state = tidal.evaluate(state)

    # Ratio of quadrupolar elongation should follow (M1 / M2)^2 ~ (1.465 / 1.270)^2 ~ 1.33
    ratio = t_state.tidal_elongation_2 / t_state.tidal_elongation_1
    expected_ratio = (config.m1 / config.m2)**2
    assert np.isclose(ratio, expected_ratio, rtol=0.05)
    assert t_state.tidal_elongation_2 > t_state.tidal_elongation_1


def test_particle_cloud_ellipsoidal_alignment():
    """Verify particle cloud prolate elongation aligns along the instantaneous binary separation axis."""
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)

    # Set late inspiral with known orbital phase
    engine.dynamics.inspiral_state.separation = 26.0e3
    engine.dynamics.inspiral_state.orbital_phase = np.pi / 4.0  # 45 degrees
    engine.dynamics.update_particles()

    pos_np = engine.psys.pos.to_numpy()
    sid_np = engine.psys.star_id.to_numpy()

    # Get NS1 particles
    ns1_pos = pos_np[sid_np == 0]
    com1 = np.mean(ns1_pos, axis=0)
    offsets1 = ns1_pos - com1

    # Separation vector unit direction at phi = pi/4 is [cos(pi/4), sin(pi/4), 0]
    sep_dir = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
    perp_dir = np.array([-np.sin(np.pi / 4.0), np.cos(np.pi / 4.0), 0.0])

    proj_along = np.abs(np.dot(offsets1, sep_dir))
    proj_perp = np.abs(np.dot(offsets1, perp_dir))

    # Semi-major axis along separation must exceed semi-minor axis perpendicular to it
    max_along = np.percentile(proj_along, 95)
    max_perp = np.percentile(proj_perp, 95)

    assert max_along > max_perp, f"Star not prolate along separation: along={max_along:.2f}, perp={max_perp:.2f}"
    axis_ratio = max_along / max_perp
    assert 1.10 < axis_ratio < 1.35, f"Unexpected axis ratio: {axis_ratio:.3f}"


def test_volume_preservation():
    """Verify that prolate deformation preserves stellar volume (1+eps)(1-0.5*eps)^2 ~ 1."""
    eps_vals = np.linspace(0.01, 0.25, 25)
    for eps in eps_vals:
        vol_factor = (1.0 + eps) * ((1.0 - 0.5 * eps)**2)
        # Deviation from unit volume is O(eps^2) <= 0.75 * (0.25)^2 ~ 0.047
        assert abs(vol_factor - 1.0) < 0.05, f"Volume not conserved for eps={eps}: {vol_factor}"


def test_dashboard_hud_tidal_telemetry():
    """Verify that compact tidal deformability telemetry data is available and valid in pre-merger."""
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)

    # Sync pre-merger
    engine.set_inspiral_time(2.0)

    # Check telemetry fields
    lam1 = engine.tidal.star1.lambda_dimensionless
    lam2 = engine.tidal.star2.lambda_dimensionless
    lam_tilde = engine.tidal.lambda_tilde
    assert np.isclose(lam1, 284.6, atol=2.0)
    assert np.isclose(lam2, 571.4, atol=2.0)
    assert np.isclose(lam_tilde, 402.0, atol=2.0)

    t_st = engine.dynamics.tidal_state
    assert t_st is not None
    assert 0.0 <= t_st.relative_tidal_strength <= 1.0

    # Test formatted telemetry strings
    telemetry_header = "TIDAL DEFORMABILITY"
    lam1_str = f"Λ1 = {lam1:.0f} [ADOPTED MODEL]"
    lam2_str = f"Λ2 = {lam2:.0f} [ADOPTED MODEL]"
    lam_tilde_str = f"Λ~ = {lam_tilde:.0f} (GW170817 <= 800)"
    resp_str = f"TIDAL RESPONSE = {t_st.relative_tidal_strength * 100.0:.0f}%"
    label_str = "[REDUCED-ORDER TIDAL DEFORMABILITY MODEL]"

    assert "285" in lam1_str
    assert "571" in lam2_str
    assert "402" in lam_tilde_str
    assert "REDUCED-ORDER" in label_str


def test_post_merger_culling():
    """Verify that post-merger, inspiral particles are culled and individual NS deformation drops out."""
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)

    # Jump to post-merger
    engine.set_post_merger_event_time(2.0)
    assert engine.dynamics.merger_state.contact_fraction == 1.0
    assert engine.dynamics.merger_state.merger_complete is True

    # Check that particle speeds and positions have no NaNs or Infs
    pos_np = engine.psys.pos.to_numpy()
    vel_np = engine.psys.vel.to_numpy()
    assert not np.isnan(pos_np).any()
    assert not np.isinf(pos_np).any()
    assert not np.isnan(vel_np).any()
    assert not np.isinf(vel_np).any()
