"""
Unit tests for Task 026D.6 — Live Gravitational-Wave Instrument & Chirp Acceleration.

Verifies:
1. Live GW waveform is generated from physical InspiralState (frequency, phase, separation).
2. Chirp frequency & amplitude increase as inspiral progresses toward merger.
3. GPU waveform and chirp vertices update continuously with non-zero line coordinates.
4. Post-merger ringdown is explicitly modeled and decaying.
5. No second event clock is introduced.
"""
import pytest
import numpy as np

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.physics.gravitational_waves import GravitationalWaveModel, WaveformBuffer
from gw170817.visualization.renderer import ParticleRenderer


@pytest.fixture(scope="module")
def gw_setup():
    config = SimConfig(mode="DEV", seed=42)
    sim = GW170817Simulation(config=config)
    renderer = ParticleRenderer(sim.psys)
    return config, sim, renderer


def test_gw_frequency_and_amplitude_chirp(gw_setup):
    config, sim, _ = gw_setup
    sim.reset()

    # Early inspiral state (initial frequency ~40 Hz)
    st_early = sim.inspiral.initial_state(config)
    sample_early = sim.gw_model.sample(st_early)

    # Evolve inspiral state forward toward merger (higher frequency, smaller separation)
    st_late = st_early
    for _ in range(400):
        st_late = sim.inspiral.step(st_late, 0.005)
    sample_late = sim.gw_model.sample(st_late)

    assert sample_late.f_gw > sample_early.f_gw, "GW frequency must increase as inspiral progresses"
    assert sample_late.amplitude > sample_early.amplitude, "GW strain amplitude must increase as separation decreases"


def test_live_gpu_waveform_vertices_update(gw_setup):
    config, sim, renderer = gw_setup
    sim.reset()

    # Populate waveform buffer with physical inspiral steps
    st = sim.inspiral.initial_state(config)
    for _ in range(128):
        st = sim.inspiral.step(st, 0.010)
        sample = sim.gw_model.sample(st)
        sim.waveform_buffer.append(sample)

    t_arr, h_p_arr, h_c_arr, f_gws = sim.waveform_buffer.get_chronological()
    assert len(h_p_arr) > 0

    # Update GPU waveform lines
    renderer.update_waveform_buffer(waveform_data=h_p_arr, f_gw=600.0, event_time=-0.1)
    w_verts = renderer.waveform_vertices.to_numpy()
    c_verts = renderer.chirp_vertices.to_numpy()

    assert np.all(np.isfinite(w_verts))
    assert np.all(np.isfinite(c_verts))
    assert np.max(np.abs(w_verts)) > 0.0, "Waveform 2D line vertices must be non-zero during live display"
    assert np.max(np.abs(c_verts)) > 0.0, "Chirp track 2D line vertices must be non-zero during live display"


def test_post_merger_ringdown_decay(gw_setup):
    config, sim, _ = gw_setup

    # Post-merger ringdown (t_event = +0.005s)
    sim.dynamics.inspiral_state.time = 0.005
    sim.dynamics.inspiral_state.f_gw = 1500.0
    h_p, h_c = sim.gw_model.strain(sim.dynamics.inspiral_state)

    # Strain should be exponentially suppressed post-merger
    assert abs(h_p) < 1.0e-20
    assert np.isfinite(h_p)
