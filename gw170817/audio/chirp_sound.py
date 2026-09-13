"""
GW170817 Gravitational-Wave Chirp Audio Provider using Real Recording.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Provides calibrated strain audio playback for GW170817 using the authoritative project recording:
'GW170817 Spectrogram  Template Audio (1).mp3'

Loads the MP3 asset once during initialization, decodes it into a float32 mono PCM buffer,
and streams audio via a single persistent sounddevice.OutputStream callback.
Synchronization is anchored strictly to the simulation event clock:
event_time = 0.0 s maps directly to the merger peak sample (sample ~1,346,770 at t ≈ 30.54 s).
"""
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AudioState:
    """State of the GW chirp audio sonification."""
    active: bool          # True if audio sonification active
    sample_rate: int      # Audio sample rate [Hz]
    current_time: float   # Current physical event time [s]
    frequency: float      # Instantaneous audio chirp frequency [Hz] (for HUD telemetry)
    amplitude: float      # Instantaneous audio amplitude [0.0 to 1.0] (for HUD telemetry)
    status_label: str = "READY"            # Status label string for HUD telemetry
    hardware_available: bool = False       # True if audio hardware output device available
    is_playing: bool = False               # True if currently streaming PCM audio
    is_muted: bool = False                 # True if user muted audio output


_GLOBAL_AUDIO_CACHE = {}


class GWChirpSonification:
    """
    GW170817 Real Chirp Audio Provider.
    Synchronizes authoritative recording playback with simulation event_time.
    Uses a single persistent sounddevice.OutputStream callback (blocksize=1024) to eliminate latency and stalls.
    """

    DEFAULT_ASSET_NAME = "GW170817 Spectrogram  Template Audio (1).mp3"

    def __init__(self, sample_rate: int = 44100, asset_path: Optional[str] = None, master_volume: float = 0.20):
        self.sample_rate = sample_rate
        self.master_volume = master_volume
        self.enabled = True
        self.is_muted = False
        self.is_playing = False
        self.hardware_available = False
        self.status_label = "READY"

        self._stream = None
        self._sd = None
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.merger_peak_sample = 0
        self.merger_peak_time = 0.0
        self.duration_sec = 0.0

        # Thread-safe simulation time synchronization tracking
        self._current_sim_time = -5.0
        self._last_event_time_update = time.time()

        self._load_audio_asset(asset_path)
        self._init_hardware_audio()

    def _find_asset_file(self, asset_path: Optional[str]) -> Optional[str]:
        """Locate provided GW170817 MP3 audio recording file."""
        candidates = []
        if asset_path:
            candidates.append(asset_path)

        cwd = os.getcwd()
        candidates.append(os.path.join(cwd, self.DEFAULT_ASSET_NAME))
        candidates.append(r"C:\Users\rajde\Documents\Gravitational_wave_events\blackhole\GW170817 Spectrogram  Template Audio (1).mp3")

        mod_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(mod_dir, "..", "..", self.DEFAULT_ASSET_NAME)))

        for p in candidates:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        return None

    def _load_audio_asset(self, asset_path: Optional[str]):
        """Decode provided MP3 file ONCE into float32 mono PCM buffer and measure merger peak."""
        full_path = self._find_asset_file(asset_path)
        if not full_path:
            print(f"[GW Chirp Audio] Warning: Audio asset '{self.DEFAULT_ASSET_NAME}' not found.")
            return

        if full_path in _GLOBAL_AUDIO_CACHE:
            buf, sr, p_idx, p_t, dur = _GLOBAL_AUDIO_CACHE[full_path]
            self.audio_buffer = buf
            self.sample_rate = sr
            self.merger_peak_sample = p_idx
            self.merger_peak_time = p_t
            self.duration_sec = dur
            return

        try:
            import soundfile as sf
            data, sr = sf.read(full_path, dtype='float32')
            if data.ndim > 1:
                # Convert stereo to mono
                data = np.mean(data, axis=1).astype(np.float32)

            self.audio_buffer = data
            self.sample_rate = int(sr)
            self.duration_sec = float(len(data) / sr)

            # Locate authoritative merger peak in recording
            self.merger_peak_sample = int(np.argmax(np.abs(data)))
            self.merger_peak_time = float(self.merger_peak_sample / sr)

            _GLOBAL_AUDIO_CACHE[full_path] = (
                self.audio_buffer,
                self.sample_rate,
                self.merger_peak_sample,
                self.merger_peak_time,
                self.duration_sec
            )

            print(f"[GW Chirp Audio] Decoded recording '{os.path.basename(full_path)}': "
                  f"{self.duration_sec:.2f} s ({len(data)} samples) at {sr} Hz. "
                  f"Merger peak sample: {self.merger_peak_sample} (t = {self.merger_peak_time:.3f} s).")
        except Exception as e:
            print(f"[GW Chirp Audio] Failed to decode audio asset: {e}")
            self.audio_buffer = np.zeros(0, dtype=np.float32)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """
        Lightweight audio callback executing on PortAudio thread every ~23 ms (blocksize=1024).
        Determines current simulation event_time sample position and writes PCM data directly to outdata.
        Never calls sd.play(), sd.stop(), or device open/reopen operations.
        """
        if not self.is_playing or self.is_muted or not self.hardware_available or len(self.audio_buffer) == 0:
            outdata.fill(0)
            return

        # Compute current simulation event time dynamically
        now = time.time()
        dt = now - self._last_event_time_update
        curr_sim_time = self._current_sim_time + dt

        sample_idx = int(self.merger_peak_sample + round(curr_sim_time * self.sample_rate))

        if 0 <= sample_idx < len(self.audio_buffer):
            end_idx = min(sample_idx + frames, len(self.audio_buffer))
            n = end_idx - sample_idx
            raw_chunk = self.audio_buffer[sample_idx:end_idx]
            clamped_chunk = np.clip(raw_chunk * self.master_volume, -0.20, 0.20)
            outdata[:n, 0] = clamped_chunk
            if n < frames:
                outdata[n:, 0] = 0
        else:
            outdata.fill(0)

    def _init_hardware_audio(self):
        """Initialize single persistent sounddevice.OutputStream backend safely."""
        try:
            import sounddevice as sd
            self._sd = sd
            dev_info = sd.query_devices(kind='output')
            if dev_info is not None and len(self.audio_buffer) > 0:
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=1024,
                    callback=self._audio_callback
                )
                self._stream.start()
                self.hardware_available = True
                self.status_label = "READY"
            else:
                self.hardware_available = False
                self.status_label = "UNAVAILABLE"
        except Exception as e:
            print(f"[GW Chirp Audio] Output device unavailable: {e}")
            self._stream = None
            self.hardware_available = False
            self.status_label = "UNAVAILABLE"

    def event_time_to_sample(self, event_time: float) -> int:
        """Map simulation event_time to recording sample index (event_time = 0.0 s -> merger peak)."""
        return int(self.merger_peak_sample + round(event_time * self.sample_rate))

    def sample_to_event_time(self, sample_idx: int) -> float:
        """Map recording sample index to simulation event_time."""
        return float(sample_idx - self.merger_peak_sample) / self.sample_rate

    def start_playback(self, event_time: float = -5.0, speed_multiplier: float = 1.0):
        """Start hardware audio playback anchored to event_time without reopening device."""
        self._current_sim_time = float(event_time)
        self._last_event_time_update = time.time()

        if not self.enabled or not self.hardware_available:
            if not self.hardware_available:
                self.status_label = "UNAVAILABLE"
            return

        if self.is_muted:
            self.status_label = "MUTED"
            return

        if abs(speed_multiplier - 1.00) > 0.01:
            self.is_playing = False
            self.status_label = "NORMAL SPEED ONLY"
            return

        self.is_playing = True
        self.status_label = "ACTIVE — REAL CHIRP RECORDING"

    def pause_playback(self):
        """Pause hardware audio output by outputting silence in callback."""
        self.is_playing = False
        if self.hardware_available and not self.is_muted:
            self.status_label = "READY"

    def resume_playback(self, event_time: float, speed_multiplier: float = 1.0):
        """Resume hardware audio output from event_time."""
        self.start_playback(event_time=event_time, speed_multiplier=speed_multiplier)

    def stop_playback(self):
        """Stop hardware audio output by outputting silence in callback."""
        self.is_playing = False
        if self.hardware_available and not self.is_muted:
            self.status_label = "READY"

    def toggle_mute(self) -> bool:
        """Toggle audio mute state."""
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.status_label = "MUTED"
        else:
            if self.hardware_available:
                self.status_label = "READY"
        return self.is_muted

    def close(self):
        """Idempotently close persistent sounddevice OutputStream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.is_playing = False
        self.hardware_available = False

    def evaluate(self, event_time: float) -> AudioState:
        """
        Evaluate instantaneous audio telemetry state at event_time [s].
        Updates thread-safe event_time for audio callback and provides analytical telemetry for HUD.
        """
        self._current_sim_time = float(event_time)
        self._last_event_time_update = time.time()

        if not self.enabled:
            return AudioState(
                active=False, sample_rate=self.sample_rate, current_time=event_time,
                frequency=0.0, amplitude=0.0, status_label="DISABLED",
                hardware_available=self.hardware_available, is_playing=False, is_muted=self.is_muted
            )

        tau = max(0.0005, -event_time)
        if event_time < 0.0:
            f_inst = float(np.clip(24.0 * (10.0 / tau)**(3.0 / 8.0), 20.0, 1500.0))
            amp_inst = float(np.clip((10.0 / tau)**(1.0 / 4.0) * 0.05, 0.01, 1.0))
            active_flag = True
        elif event_time <= 0.1:
            f_inst = 1500.0
            amp_inst = float(np.exp(-event_time / 0.02))
            active_flag = True
        else:
            f_inst = 0.0
            amp_inst = 0.0
            active_flag = False

        if not self.hardware_available:
            lbl = "UNAVAILABLE"
        elif self.is_muted:
            lbl = "MUTED"
        elif self.is_playing:
            lbl = "ACTIVE — REAL CHIRP RECORDING"
        else:
            lbl = self.status_label

        return AudioState(
            active=active_flag,
            sample_rate=self.sample_rate,
            current_time=event_time,
            frequency=f_inst,
            amplitude=amp_inst,
            status_label=lbl,
            hardware_available=self.hardware_available,
            is_playing=self.is_playing,
            is_muted=self.is_muted
        )
