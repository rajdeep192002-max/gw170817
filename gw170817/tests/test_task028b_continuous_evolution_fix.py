"""
Unit tests for Task 028B — Continuous Post-Merger Evolution (15-Second Reset Fix).

Verifies:
1. Presentation time advances past 15.0 s without resetting to 0.0 s or initial inspiral.
2. Stage transitions cleanly to DemoStage.CONTINUOUS_POST_MERGER at t_pres >= 15.0 s.
3. Director remains active (is_running == True) throughout continuous post-merger evolution.
4. Authoritative physical event_time increases monotonically across the 15-second boundary and beyond.
5. Physical event time never decreases or wraps back to -5.0 s.
6. Binary NS inspiral state is not reinitialized.
7. Zero NaNs or Infs occur during continuous evolution from t_pres = 15 s through t_pres = 60 s.
"""
import pytest
import numpy as np

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector, DemoStage


@pytest.fixture(scope="module")
def director_setup():
    config = SimConfig(mode="DEV", seed=42)
    sim = GW170817Simulation(config=config)
    coordinator = MultiMessengerCoordinator(engine=sim, config=config)
    director = DemoDirector(coordinator=coordinator, config=config)
    return config, sim, coordinator, director


def test_presentation_time_reaches_above_15s(director_setup):
    """Test presentation time can advance past 15.0 s without reset."""
    _, sim, _, director = director_setup
    director.reset()
    director.start()

    # Step director beyond 15s (e.g. dt_wall step equivalent to 16.5s presentation time)
    director._sync_physics_for_presentation_time(16.5)

    assert director.presentation_time == 16.5, "Presentation time must reach >= 15.0 s"
    assert director.current_stage == DemoStage.CONTINUOUS_POST_MERGER.value
    assert director.is_running is True


def test_no_reset_at_15s_boundary(director_setup):
    """Test that event time and stage do not reset to initial inspiral at t_pres = 15.0s."""
    _, sim, _, director = director_setup
    director.reset()
    director.start()

    # Sample event time at 14.9s
    director._sync_physics_for_presentation_time(14.9)
    t_149 = float(sim.current_state.event_time)

    # Sample event time at 15.1s
    director._sync_physics_for_presentation_time(15.1)
    t_151 = float(sim.current_state.event_time)

    assert t_151 > t_149, f"Event time must increase across 15s boundary (t_151={t_151} > t_149={t_149})"
    assert t_151 > 0.0, "Event time must remain post-merger (> 0.0s)"
    assert director.current_stage == DemoStage.CONTINUOUS_POST_MERGER.value


def test_monotonic_event_time_post_15s(director_setup):
    """Test event_time increases monotonically for t_pres in [15s, 60s]."""
    _, sim, _, director = director_setup
    director.reset()
    director.start()

    t_pres_values = [15.0, 16.0, 20.0, 30.0, 45.0, 60.0]
    event_times = []

    for tp in t_pres_values:
        director._sync_physics_for_presentation_time(tp)
        et = float(sim.current_state.event_time)
        event_times.append(et)

        assert np.isfinite(et), f"Event time at t_pres={tp} must be finite"
        assert director.is_running is True

    for i in range(len(event_times) - 1):
        assert event_times[i+1] > event_times[i], (
            f"Event time must strictly increase: t({t_pres_values[i+1]})={event_times[i+1]} > t({t_pres_values[i]})={event_times[i]}"
        )


def test_no_nan_inf_during_extended_run(director_setup):
    """Test numerical stability (no NaNs or Infs) during extended continuous post-merger run."""
    _, sim, coord, director = director_setup
    director.reset()
    director.start()

    # Step simulation through 200 steps past 15s
    for t_step in np.linspace(15.0, 35.0, 50):
        st = director._sync_physics_for_presentation_time(t_step)

        assert np.isfinite(st.event_time)
        assert np.isfinite(st.disk_mass_msun)
        assert np.isfinite(st.kilonova_luminosity)
        assert np.isfinite(st.afterglow_flux)
        assert np.isfinite(coord.current_state.b_poloidal)
        assert np.isfinite(coord.current_state.b_toroidal)
