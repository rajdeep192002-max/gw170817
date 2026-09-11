"""
Unit tests for RemnantModel and configurable scenarios.
"""
import pytest
from gw170817.config import SimConfig
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.inspiral import InspiralState


def test_gw170817_like_scenario_default():
    """Verify default GW170817_LIKE scenario forms HMNS with delayed collapse (not prompt BH)."""
    cfg = SimConfig()
    remnant = RemnantModel(config=cfg, scenario="GW170817_LIKE")
    
    dummy_insp = InspiralState(
        time=0.0, f_gw=1500.0, orbital_frequency=750.0, omega_orb=4712.0,
        separation=15.0e3, orbital_phase=0.0, df_dt=1e4, chirp_mass=1.188 * 1.989e30
    )
    
    # Pre-merger
    st_pre = remnant.evaluate(dummy_insp, event_time=-0.1)
    assert st_pre.remnant_type == "INSPIRAL"
    assert not st_pre.is_black_hole

    # Post-merger HMNS stage (t = +0.02 s < t_collapse_delay ~ 0.08 s)
    st_hmns = remnant.evaluate(dummy_insp, event_time=0.02)
    assert st_hmns.remnant_type == "HMNS"
    assert not st_hmns.is_black_hole
    assert st_hmns.scenario == "GW170817_LIKE"

    # Delayed BH collapse (t = +0.10 s > t_collapse_delay)
    st_bh = remnant.evaluate(dummy_insp, event_time=0.10)
    assert st_bh.remnant_type == "BH"
    assert st_bh.is_black_hole


def test_prompt_bh_reference_scenario():
    """Verify PROMPT_BH_REFERENCE scenario collapses promptly (~1 ms)."""
    cfg = SimConfig()
    remnant = RemnantModel(config=cfg, scenario="PROMPT_BH_REFERENCE")

    dummy_insp = InspiralState(
        time=0.0, f_gw=1500.0, orbital_frequency=750.0, omega_orb=4712.0,
        separation=15.0e3, orbital_phase=0.0, df_dt=1e4, chirp_mass=1.188 * 1.989e30
    )

    st_prompt = remnant.evaluate(dummy_insp, event_time=0.005)
    assert st_prompt.remnant_type == "BH"
    assert st_prompt.is_black_hole
    assert st_prompt.scenario == "PROMPT_BH_REFERENCE"


def test_hmns_reference_scenario():
    """Verify HMNS_REFERENCE scenario remains an HMNS without prompt/early BH collapse."""
    cfg = SimConfig()
    remnant = RemnantModel(config=cfg, scenario="HMNS_REFERENCE")

    dummy_insp = InspiralState(
        time=0.0, f_gw=1500.0, orbital_frequency=750.0, omega_orb=4712.0,
        separation=15.0e3, orbital_phase=0.0, df_dt=1e4, chirp_mass=1.188 * 1.989e30
    )

    st_long = remnant.evaluate(dummy_insp, event_time=1.0)
    assert st_long.remnant_type == "HMNS"
    assert not st_long.is_black_hole
    assert st_long.scenario == "HMNS_REFERENCE"
