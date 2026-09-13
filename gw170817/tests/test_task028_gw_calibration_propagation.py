"""
Unit tests for Task 028 — GW Waveform Calibration, Chirp Audio Sonification, Interactive GW Propagation & Continuous Evolution.

Verifies:
1. One authoritative merger clock drives orbital dynamics, f_gw = 2 * f_orb, and strain h_plus/h_cross without redundant clocks.
2. Presentation timeline calibration: t_event = 0.0 s corresponds to MERGER_PRESENTATION_TIME = 6.0 s.
3. Dual strain traces: Observed GW170817 (H1/L1) vs Model BNS Quadrupole are accessible without NameError.
4. GW Chirp audio sonification (GWChirpSonification) yields physical time-frequency track, peak at merger (1500 Hz), and post-merger decay.
5. Interactive 3D GW Wavefront Propagation (GWWavefrontPropagation):
   - Launches 3D quadrupolar expanding wavefront lines centered at world origin (0, 0, 0).
   - Monotonic radius growth with r = r_0 + c_vis * dt.
   - Quadrupolar 1/r amplitude attenuation.
   - Vertices and colors are valid, non-empty, and finite 3D coordinates.
6. Continuous evolution (CONTINUOUS_POST_MERGER): DemoDirector updates continuously beyond 15.0 s without resetting or freezing.
7. Numerical stability: No NaNs or Infs in state variables or vertices.
"""
import pytest
import numpy as np

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector, DemoStage
from gw170817.visualization.wave_propagation import GWWavefrontPropagation, WavefrontState
from gw170817.audio.chirp_sound import GWChirpSonification, AudioState
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.simulation.observational_data import ObservationalGWData


@pytest.fixture(scope="module")
def task028_setup():
    config = SimConfig(mode="DEV", seed=42)
    sim = GW170817Simulation(config=config)
    coordinator = MultiMessengerCoordinator(engine=sim, config=config)
    director = DemoDirector(coordinator=coordinator, config=config)
    renderer = ParticleRenderer(sim.psys)
    wave_prop = GWWavefrontPropagation()
    chirp_audio = GWChirpSonification()
    obs_data = ObservationalGWData()
    return config, sim, director, renderer, wave_prop, chirp_audio, obs_data


def test_authoritative_clock_and_gw_relation(task028_setup):
    """Test orbital frequency -> GW frequency relation f_gw = 2 * f_orb."""
    config, sim, _, _, _, _, _ = task028_setup
    sim.reset()

    st_inspiral = sim.inspiral.initial_state(config)
    f_orb = st_inspiral.orbital_frequency
    f_gw = st_inspiral.f_gw

    assert np.isclose(f_gw, 2.0 * f_orb, rtol=1e-3), f"f_gw ({f_gw}) must be 2 * f_orb ({f_orb})"
    assert sim.current_state.event_time < 0.0, "Initial inspiral event time must be negative relative to merger"


def test_merger_presentation_time_calibration(task028_setup):
    """Test presentation timeline calibration at t_presentation = 6.0s -> t_event = 0.0s."""
    config, sim, director, _, _, _, _ = task028_setup
    director.reset()
    director.start()

    # Jump to MERGER stage index (index 2: MERGER at presentation_time = 6.0s)
    director.jump_to_stage_index(2)
    st = sim.current_state

    assert np.isclose(st.event_time, 0.0, atol=1e-2), f"t_event ({st.event_time}) must be ~0.0s at presentation time 6.0s"
    assert director.current_stage == DemoStage.MERGER.value, "Stage must be MERGER at 6.0s presentation time"


def test_observed_vs_model_strain_separation(task028_setup):
    """Test that observed H1/L1 data and model strain are distinct and valid."""
    config, sim, _, _, _, _, obs_data = task028_setup
    sim.reset()

    t_event = -1.0  # 1s before merger
    h1_val, l1_val = obs_data.get_sample_at_time(t_event)

    st = sim.inspiral.initial_state(config)
    st.time = t_event
    h_p, h_c = sim.gw_model.strain(st)

    assert np.isfinite(h1_val), "Observed H1 strain must be finite"
    assert np.isfinite(l1_val), "Observed L1 strain must be finite"
    assert np.isfinite(h_p), "Model strain h_plus must be finite"
    assert np.isfinite(h_c), "Model strain h_cross must be finite"
    assert abs(h1_val - h_p) > 1.0e-25, "Observed and model strain values must be distinct data sources"


def test_chirp_audio_sonification(task028_setup):
    """Test GW chirp audio sonification provider frequency and amplitude response."""
    _, _, _, _, _, chirp_audio, _ = task028_setup

    # 1. Inspiriting at t = -5.0s
    state_in = chirp_audio.evaluate(-5.0)
    assert state_in.active is True
    assert 20.0 <= state_in.frequency <= 1500.0
    assert 0.01 <= state_in.amplitude <= 1.0

    # 2. Merger peak at t = 0.0s
    state_peak = chirp_audio.evaluate(0.0)
    assert state_peak.active is True
    assert np.isclose(state_peak.frequency, 1500.0)
    assert np.isclose(state_peak.amplitude, 1.0)

    # 3. Post-merger at t = +0.5s (decayed)
    state_post = chirp_audio.evaluate(0.5)
    assert state_post.active is False
    assert state_post.amplitude == 0.0


def test_3d_gw_wavefront_propagation(task028_setup):
    """Test 3D quadrupolar wavefront generation, expansion, and geometry."""
    _, _, _, renderer, wave_prop, _, _ = task028_setup

    # Manual trigger at t_event = 0.0s
    wave_prop.trigger(event_time=0.0)
    wf_st1 = wave_prop.update(event_time=0.0, f_gw=100.0)

    assert wf_st1.active is True
    assert wf_st1.radius >= 15.0e3, "Initial radius must be >= 15 km"
    assert wf_st1.n_vertices > 0, "Wavefront must generate line vertices"

    # Evolve time forward dt = 0.1s
    wf_st2 = wave_prop.update(event_time=0.1, f_gw=200.0)
    assert wf_st2.radius > wf_st1.radius, "Wavefront radius must grow monotonically with time"
    assert wf_st2.amplitude < wf_st1.amplitude, "Wavefront amplitude must attenuate with 1/r distance"

    # Test GPU line updating in ParticleRenderer
    renderer.update_gw_wavefront_lines(wave_prop.line_vertices, wave_prop.line_colors, wf_st2.n_vertices)
    v_arr = renderer.gw_wave_vertices.to_numpy()
    c_arr = renderer.gw_wave_colors.to_numpy()

    assert np.all(np.isfinite(v_arr))
    assert np.all(np.isfinite(c_arr))
    assert np.max(np.abs(v_arr)) > 0.0, "3D Wavefront line vertices must be non-zero"


def test_continuous_post_merger_evolution(task028_setup):
    """Test continuous evolution stage beyond 15s without freezing or resetting."""
    config, sim, director, _, _, _, _ = task028_setup
    director.reset()
    director.start()

    # Step presentation time beyond 15.0s (e.g. 18.0s)
    director._sync_physics_for_presentation_time(18.0)

    st = sim.current_state
    assert director.current_stage == DemoStage.CONTINUOUS_POST_MERGER.value
    assert director.is_running is True, "Director must remain running in continuous post-merger mode"
    assert st.event_time > 10.0, "Event time must continuously advance post-merger"
    assert np.isfinite(st.event_time), "Event time must be finite"


def test_wavefront_toggle(task028_setup):
    """Test toggle functionality of 3D GW wavefront propagation."""
    _, _, _, _, wave_prop, _, _ = task028_setup

    wave_prop.active = False
    new_st = wave_prop.toggle(event_time=0.0)
    assert new_st is True
    assert wave_prop.active is True

    new_st_off = wave_prop.toggle(event_time=0.0)
    assert new_st_off is False
    assert wave_prop.active is False
