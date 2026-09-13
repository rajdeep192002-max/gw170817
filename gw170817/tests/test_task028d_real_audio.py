"""
Task 028C-D2 — Unit tests for real GW chirp audio output using authoritative MP3 recording.

Verifies:
A. Supplied audio asset loads successfully.
B. Decoded PCM has expected duration (~38 seconds).
C. Merger sample/offset is detected (sample ~1,346,770 at t ≈ 30.54 s).
D. event_time = -5, 0, +1 s map to sensible sample positions in the recording.
E. Repeated frame updates do not reopen/restart the stream or stall the thread.
F. Pause produces silence in output callback.
G. Mute produces silence in output callback.
H. Close is idempotent.
I. Hardware-unavailable mode does not crash.
J. No sd.play() / sd.stop() calls remain in active playback path.
"""
import pytest
import numpy as np
from gw170817.audio.chirp_sound import GWChirpSonification, AudioState


class TestTask028DRealAudioOutput:
    """Test suite for real GW chirp audio output integration."""

    @pytest.fixture
    def sonification(self):
        son = GWChirpSonification()
        yield son
        son.close()

    def test_supplied_audio_asset_loads_successfully(self, sonification):
        """A. Supplied audio asset loads successfully."""
        assert hasattr(sonification, "audio_buffer")
        assert isinstance(sonification.audio_buffer, np.ndarray)
        assert sonification.audio_buffer.dtype == np.float32

    def test_decoded_pcm_duration(self, sonification):
        """B. Decoded PCM has expected duration (~38 seconds)."""
        assert len(sonification.audio_buffer) > 0
        assert not np.isnan(sonification.audio_buffer).any()
        assert not np.isinf(sonification.audio_buffer).any()
        assert np.isclose(sonification.duration_sec, 38.127, atol=1.0)

    def test_merger_peak_sample_detection(self, sonification):
        """C. Merger sample/offset is detected (peak sample ~1,346,770)."""
        assert sonification.merger_peak_sample > 0
        assert np.isclose(sonification.merger_peak_time, 30.54, atol=0.5)

    def test_event_time_mapping(self, sonification):
        """D. event_time = -5, 0, +1 map to sensible positions in recording."""
        idx_neg5 = sonification.event_time_to_sample(-5.0)
        idx_zero = sonification.event_time_to_sample(0.0)
        idx_pos1 = sonification.event_time_to_sample(1.0)

        assert idx_zero == sonification.merger_peak_sample
        assert idx_neg5 < idx_zero
        assert idx_pos1 > idx_zero
        assert idx_zero - idx_neg5 == int(5.0 * sonification.sample_rate)
        assert idx_pos1 - idx_zero == int(1.0 * sonification.sample_rate)

        assert sonification.sample_to_event_time(idx_zero) == pytest.approx(0.0, abs=1e-4)
        assert sonification.sample_to_event_time(idx_neg5) == pytest.approx(-5.0, abs=1e-4)

    def test_repeated_frame_updates_no_reopen(self, sonification):
        """E. Repeated frame updates do not reopen/restart stream."""
        sonification.start_playback(-5.0, 1.0)
        st1 = sonification._stream

        # Evaluate repeatedly across 100 frames
        for f in range(100):
            sonification.evaluate(-5.0 + f * 0.016)

        st2 = sonification._stream
        assert st1 is st2, "Stream instance must remain persistent without reopening"

    def test_pause_produces_silence(self, sonification):
        """F. Pause produces silence in callback output."""
        sonification.start_playback(-5.0, 1.0)
        sonification.pause_playback()

        outdata = np.ones((1024, 1), dtype=np.float32)
        sonification._audio_callback(outdata, 1024, None, None)
        assert np.all(outdata == 0.0), "Pause must produce complete silence (zeros)"

    def test_mute_produces_silence(self, sonification):
        """G. Mute produces silence in callback output."""
        sonification.start_playback(-5.0, 1.0)
        sonification.toggle_mute()
        assert sonification.is_muted

        outdata = np.ones((1024, 1), dtype=np.float32)
        sonification._audio_callback(outdata, 1024, None, None)
        assert np.all(outdata == 0.0), "Mute must produce complete silence (zeros)"
        sonification.toggle_mute()

    def test_close_is_idempotent(self, sonification):
        """H. Close is idempotent and handles multiple calls safely."""
        sonification.close()
        assert sonification._stream is None
        # Second call must not raise exception
        sonification.close()
        assert sonification._stream is None

    def test_hardware_unavailable_mode(self, sonification):
        """I. Hardware-unavailable mode does not crash."""
        sonification.close()
        sonification.hardware_available = False

        # Calls must execute safely without raising
        sonification.start_playback(-5.0, 1.0)
        st = sonification.evaluate(-5.0)
        assert st.status_label == "UNAVAILABLE"
        sonification.pause_playback()
        sonification.stop_playback()

    def test_no_sd_play_in_active_path(self, sonification):
        """J. No sd.play() / sd.stop() calls remain in active playback path."""
        # Verify start_playback and stop_playback execute in < 1 ms on main thread
        import time
        t0 = time.time()
        for _ in range(50):
            sonification.start_playback(-5.0, 1.0)
            sonification.stop_playback()
        dt = (time.time() - t0) * 1000.0

        assert dt < 50.0, f"Playback state changes took {dt:.2f} ms; active path must not invoke blocking sd.stop/play"
