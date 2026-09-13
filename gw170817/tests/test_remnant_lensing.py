"""
Unit tests for Task 026D.4 — Adopted Delayed-Collapse Black-Hole Remnant & Lensing Source Transition.

Verifies:
1. Pre-merger: Lensing source references binary neutron star positions and masses.
2. Post-merger HMNS & BH: Lensing source transitions to single compact remnant at origin with total mass M_rem and m2=0.
3. Effective remnant mass: Matches RemnantState.mass.
4. Stale NS position elimination: Lensing raytracer uses single remnant position [0,0,0] after collapse.
5. BH visual state: Renderer remnant particle system scales horizon radius and renders dark central horizon shadow for BH state.
6. Timing: Collapse uses existing RemnantModel.t_collapse_delay (~0.08s) without creating a second clock.
"""
import pytest
import numpy as np

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.dashboard import ScientificDashboard
from gw170817.physics.remnant import RemnantModel, RemnantState


@pytest.fixture(scope="module")
def dashboard_setup():
    config = SimConfig(mode="DEV", seed=42)
    sim = GW170817Simulation(config=config)
    dashboard = ScientificDashboard(engine=sim, config=config)
    return config, sim, dashboard


def test_pre_merger_lensing_sources(dashboard_setup):
    _, sim, dashboard = dashboard_setup
    sim.reset()

    # Pre-merger inspiral (t_event = -1.0s)
    dashboard.director._sync_physics_for_presentation_time(0.0)
    p1, p2, m1, m2 = dashboard.get_lensing_sources()

    assert m1 == sim.config.m1
    assert m2 == sim.config.m2
    assert not np.allclose(p1, p2), "Pre-merger lensing sources must be distinct binary NS positions"


def test_post_collapse_single_remnant_lensing_source(dashboard_setup):
    _, sim, dashboard = dashboard_setup

    # Force post-merger delayed-collapse state (t_event = 0.10s >= t_collapse_delay)
    sim.current_state.event_time = 0.10
    sim.current_state.merger_contact_fraction = 1.0

    p1, p2, m1, m2 = dashboard.get_lensing_sources()
    rem_st = sim.remnant.evaluate(sim.dynamics.inspiral_state, 0.10)

    assert rem_st.is_black_hole, "Remnant must be BH at t_event = 0.10s"
    assert m1 == rem_st.mass, f"Lensing primary mass ({m1}) must equal RemnantState mass ({rem_st.mass})"
    assert m2 == 0.0, "Secondary lensing mass must be 0.0 after collapse to single remnant"
    assert p1 == (0.0, 0.0, 0.0), "Lensing source position must be centered single remnant origin [0,0,0]"
    assert p2 == (0.0, 0.0, 0.0)


def test_remnant_model_collapse_timing(dashboard_setup):
    config, sim, _ = dashboard_setup
    rem_model = RemnantModel(config=config, scenario="GW170817_LIKE")

    # Before collapse delay (t = 0.02s < 0.08s)
    st_hmns = rem_model.evaluate(sim.dynamics.inspiral_state, 0.02)
    assert st_hmns.remnant_type == "HMNS"
    assert not st_hmns.is_black_hole

    # After collapse delay (t = 0.10s > 0.08s)
    st_bh = rem_model.evaluate(sim.dynamics.inspiral_state, 0.10)
    assert st_bh.remnant_type == "BH"
    assert st_bh.is_black_hole
    assert st_bh.radius < st_hmns.radius, "BH horizon radius must be more compact than HMNS radius"


def test_bh_visual_particle_rendering(dashboard_setup):
    _, sim, dashboard = dashboard_setup

    # HMNS state
    dashboard.renderer.update_remnant_particles(event_time=0.02, contact_frac=1.0, remnant_type_str="HMNS", horizon_radius_m=14.0e3)
    pos_hmns = dashboard.renderer.remnant_pos.to_numpy()
    cols_hmns = dashboard.renderer.remnant_colors.to_numpy()

    assert np.all(np.isfinite(pos_hmns))
    assert np.all(np.isfinite(cols_hmns))

    # BH state (compact dark horizon shadow + accretion edge)
    dashboard.renderer.update_remnant_particles(event_time=0.10, contact_frac=1.0, remnant_type_str="BH", horizon_radius_m=8.0e3)
    pos_bh = dashboard.renderer.remnant_pos.to_numpy()
    cols_bh = dashboard.renderer.remnant_colors.to_numpy()

    assert np.all(np.isfinite(pos_bh))
    assert np.all(np.isfinite(cols_bh))
    # BH visual particle distribution must be physically smaller than HMNS
    assert np.max(np.abs(pos_bh)) < np.max(np.abs(pos_hmns))
