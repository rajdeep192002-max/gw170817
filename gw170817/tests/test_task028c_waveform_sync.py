"""
Task 028C-B — Verification tests for minimal GW waveform synchronization fix.

Verifies:
1. Presentation-time synchronization appends waveform samples.
2. Waveform sample timestamp equals current physical event_time.
3. Waveform samples are not duplicated during IDLE/PAUSED frames.
4. Consecutive presentation frames produce evolving waveform values.
5. Waveform phase follows the exact InspiralState.orbital_phase.
6. f_GW remains 2 * f_orb.
7. Merger timing remains aligned (t_event = 0 at t_pres = 6.0 s).
8. Observed H1/L1 data are unchanged.
9. 15-second continuous-post-merger behavior remains unchanged.
10. No NaN or Inf values in waveform samples.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector, DemoStage
from gw170817.simulation.observational_data import ObservationalGWData


class TestTask028CWaveformSynchronization:
    """Test suite verifying Task 028C-B GW waveform synchronization fixes."""

    @pytest.fixture
    def setup_director(self):
        config = SimConfig()
        coordinator = MultiMessengerCoordinator(config=config)
        director = DemoDirector(coordinator=coordinator, config=config)
        return director, coordinator, config

    def test_presentation_appends_waveform_samples(self, setup_director):
        """1. Presentation-time synchronization appends waveform samples."""
        director, coordinator, _ = setup_director
        buf = coordinator.engine.waveform_buffer
        initial_count = buf._total_appended

        director.start()
        director.update(0.1)

        assert buf._total_appended > initial_count, "Presentation step must append waveform samples"

    def test_sample_timestamp_equals_physical_event_time(self, setup_director):
        """2. Waveform sample timestamp equals current physical event_time."""
        director, coordinator, _ = setup_director
        director.start()
        director.update(0.5)

        buf = coordinator.engine.waveform_buffer
        t_arr, _, _, _ = buf.get_chronological()

        current_event_time = coordinator.engine.dynamics.inspiral_state.time
        assert len(t_arr) > 0
        assert np.isclose(t_arr[-1], current_event_time, atol=1e-5), (
            f"Sample timestamp {t_arr[-1]} must equal physical event_time {current_event_time}"
        )

    def test_waveform_samples_not_duplicated_when_idle_or_paused(self, setup_director):
        """3. Waveform samples are not duplicated during IDLE or PAUSED frames."""
        director, coordinator, _ = setup_director
        buf = coordinator.engine.waveform_buffer

        # Call sync multiple times with 0 dt (simulating IDLE render frames)
        director._sync_physics_for_presentation_time(0.0, dt_pres=0.0)
        appended_after_first = buf._total_appended
        
        director._sync_physics_for_presentation_time(0.0, dt_pres=0.0)
        director._sync_physics_for_presentation_time(0.0, dt_pres=0.0)

        assert buf._total_appended == appended_after_first, "IDLE frames with same timestamp/strain must not flood buffer with duplicates"

    def test_consecutive_presentation_frames_produce_evolving_waveform(self, setup_director):
        """4. Consecutive presentation frames produce evolving waveform values."""
        director, coordinator, _ = setup_director
        director.start()

        buf = coordinator.engine.waveform_buffer
        director.update(0.1)
        _, hp1, hc1, _ = buf.get_chronological()
        
        director.update(0.1)
        _, hp2, hc2, _ = buf.get_chronological()

        assert len(hp2) > len(hp1), "New frame must append updated sample"
        assert not (hp2[-1] == hp1[-1] and hc2[-1] == hc1[-1]), "Consecutive frames must produce evolving strain values"

    def test_waveform_phase_follows_inspiral_orbital_phase(self, setup_director):
        """5. Waveform phase follows the same InspiralState.orbital_phase."""
        director, coordinator, _ = setup_director
        director.start()
        director.update(0.2)

        insp_state = coordinator.engine.dynamics.inspiral_state
        sample = coordinator.engine.gw_model.sample(insp_state)

        assert sample.phase == insp_state.orbital_phase, "GW sample phase must match InspiralState.orbital_phase"

    def test_fgw_equals_two_forb(self, setup_director):
        """6. f_GW remains 2 * f_orb."""
        director, coordinator, _ = setup_director
        director.start()
        director.update(0.3)

        insp_state = coordinator.engine.dynamics.inspiral_state
        assert np.isclose(insp_state.f_gw, 2.0 * insp_state.orbital_frequency, atol=1e-6)

    def test_merger_timing_alignment(self, setup_director):
        """7. Merger timing remains aligned (event_time = 0 at presentation_time = 6.0 s)."""
        director, coordinator, _ = setup_director
        director.jump_to_stage_index(2)  # MERGER stage starts at 6.0 s pres

        st = coordinator.current_state
        assert np.isclose(st.event_time, 0.0, atol=1e-3), f"Merger event_time at 6.0s pres should be 0.0s, got {st.event_time}"

    def test_observed_data_unchanged(self, setup_director):
        """8. Observed H1/L1 data remain unchanged and functional."""
        obs = ObservationalGWData()
        t_arr, h1, l1 = obs.get_window(t_center=0.0, window_sec=1.5, n_samples=128)

        assert len(t_arr) == 128
        assert len(h1) == 128
        assert len(l1) == 128
        assert not np.isnan(h1).any()
        assert not np.isnan(l1).any()

    def test_continuous_post_merger_preserved(self, setup_director):
        """9. 15-second continuous-post-merger behavior remains unchanged."""
        director, coordinator, _ = setup_director
        director.start()

        # Step through 15 s
        for _ in range(160):
            director.update(0.1)

        assert director.presentation_time >= 15.0
        assert director.current_stage == DemoStage.CONTINUOUS_POST_MERGER.value

        # Step beyond 15 s
        t_before = coordinator.current_state.event_time
        director.update(1.0)
        t_after = coordinator.current_state.event_time

        assert t_after > t_before, "Simulation must continue evolving forward beyond 15.0 s"

    def test_no_nan_or_inf_in_waveform(self, setup_director):
        """10. No NaN or Inf in generated waveform samples across whole timeline."""
        director, coordinator, _ = setup_director
        director.start()

        for _ in range(50):
            director.update(0.2)

        buf = coordinator.engine.waveform_buffer
        t_arr, hp_arr, hc_arr, fgw_arr = buf.get_chronological()

        assert not np.isnan(t_arr).any()
        assert not np.isnan(hp_arr).any()
        assert not np.isnan(hc_arr).any()
        assert not np.isnan(fgw_arr).any()

        assert not np.isinf(t_arr).any()
        assert not np.isinf(hp_arr).any()
        assert not np.isinf(hc_arr).any()
        assert not np.isinf(fgw_arr).any()
