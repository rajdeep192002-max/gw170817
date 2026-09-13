import sys
sys.path.insert(0, '.')
import time
import threading
import sounddevice as sd
import numpy as np
from gw170817.audio.chirp_sound import GWChirpSonification


def run_standalone_diagnostic(device_index=None, device_name="Default"):
    print(f"\n========================================================")
    print(f"RUNNING STANDALONE DIAGNOSTIC FOR DEVICE: {device_name} (index: {device_index})")
    print(f"========================================================")

    # Instantiate GWChirpSonification
    audio = GWChirpSonification(asset_path=r"C:\Users\rajde\Documents\Gravitational_wave_events\blackhole\GW170817 Spectrogram  Template Audio (1).mp3")
    print(f"Audio asset buffer length: {len(audio.audio_buffer)} samples ({audio.duration_sec:.2f} s)")
    print(f"Merger peak sample: {audio.merger_peak_sample} (t = {audio.merger_peak_time:.3f} s)")

    # Diagnostics log container
    logs = []
    lock = threading.Lock()
    callback_count = 0
    t0_wall = time.time()

    # Safety volume & clamp variables
    master_vol = 0.15

    # Override internal callback to log detailed callback telemetry
    def test_callback(outdata, frames, time_info, status):
        nonlocal callback_count
        callback_count += 1
        now = time.time()
        wall_dt = now - t0_wall

        dt = now - audio._last_event_time_update
        curr_sim_time = audio._current_sim_time + dt
        sample_idx = int(audio.merger_peak_sample + round(curr_sim_time * audio.sample_rate))

        if not audio.is_playing or audio.is_muted or not audio.hardware_available or len(audio.audio_buffer) == 0:
            outdata.fill(0)
            rms = 0.0
            max_amp = 0.0
            first_nonzero = -1
        else:
            if 0 <= sample_idx < len(audio.audio_buffer):
                end_idx = min(sample_idx + frames, len(audio.audio_buffer))
                n = end_idx - sample_idx
                raw_chunk = audio.audio_buffer[sample_idx:end_idx]
                
                # Apply DEBUG AUDIO SAFETY MODE: 15% volume scale + hard amplitude clamp [-0.15, +0.15]
                clamped_chunk = np.clip(raw_chunk * master_vol, -0.15, 0.15)
                
                outdata[:n, 0] = clamped_chunk
                if n < frames:
                    outdata[n:, 0] = 0
                
                rms = float(np.sqrt(np.mean(clamped_chunk**2)))
                max_amp = float(np.max(np.abs(clamped_chunk)))
                nonzero_indices = np.where(np.abs(clamped_chunk) > 1e-5)[0]
                first_nonzero = int(nonzero_indices[0]) if len(nonzero_indices) > 0 else -1
            else:
                outdata.fill(0)
                rms = 0.0
                max_amp = 0.0
                first_nonzero = -1

        with lock:
            if len(logs) < 1000:
                logs.append({
                    'cb_count': callback_count,
                    'wall_t': wall_dt,
                    'dac_t': float(time_info.outputBufferDacTime) if time_info else 0.0,
                    'curr_sim_time': curr_sim_time,
                    'sample_idx': sample_idx,
                    'rms': rms,
                    'max_amp': max_amp,
                    'first_nonzero': first_nonzero,
                    'status': str(status) if status else 'OK'
                })

    # Close existing default stream if any
    audio.close()

    # Open specific test OutputStream
    try:
        kwargs = dict(samplerate=44100, channels=1, blocksize=1024, callback=test_callback)
        if device_index is not None:
            kwargs['device'] = device_index
        stream = sd.OutputStream(**kwargs)
        stream.start()
        audio._stream = stream
        audio.hardware_available = True
    except Exception as e:
        print(f"FAILED TO OPEN AUDIO DEVICE {device_name}: {e}")
        return

    print(f"Stream opened successfully on {device_name}. Latency = {stream.latency:.4f} s")

    # Step 1: Start playback at event_time = -5.0 s
    print("\n--- PHASE 1: Setting event_time = -5.0 s ---")
    audio.start_playback(-5.0, 1.0)
    time.sleep(2.0)

    # Step 2: Jump event_time to 0.0 s (MERGER CHIRP PEAK)
    print("\n--- PHASE 2: Forcing event_time = 0.0 s (MERGER CHIRP PEAK) ---")
    t_jump_merger = time.time() - t0_wall
    audio.start_playback(0.0, 1.0)
    time.sleep(1.0)

    # Step 3: Jump event_time to +5.0 s (POST-MERGER SILENCE)
    print("\n--- PHASE 3: Forcing event_time = +5.0 s (SILENCE SEEK TEST) ---")
    t_jump_silence = time.time() - t0_wall
    audio.start_playback(+5.0, 1.0)
    time.sleep(1.0)

    # Stop and close stream
    audio.close()

    # Analyze telemetry logs
    print("\n========================================================")
    print(f"TELEMETRY LOG ANALYSIS FOR {device_name}:")
    print(f"Total callbacks invoked: {callback_count}")
    
    # Filter logs around Phase 2 (merger peak) and Phase 3 (silence seek)
    merger_logs = [l for l in logs if l['wall_t'] >= t_jump_merger and l['wall_t'] < t_jump_merger + 0.5]
    silence_logs = [l for l in logs if l['wall_t'] >= t_jump_silence and l['wall_t'] < t_jump_silence + 0.5]

    print(f"\n--- Phase 2 (Merger Seek at wall={t_jump_merger:.2f}s) Callbacks ---")
    for l in merger_logs[:6]:
        print(f"  [CB #{l['cb_count']:4d} | Wall: {l['wall_t']:5.2f}s] EventTime: {l['curr_sim_time']:6.2f}s | Sample: {l['sample_idx']:7d} | RMS: {l['rms']:.4f} | Max: {l['max_amp']:.4f}")

    print(f"\n--- Phase 3 (Silence Seek at wall={t_jump_silence:.2f}s) Callbacks ---")
    for l in silence_logs[:6]:
        print(f"  [CB #{l['cb_count']:4d} | Wall: {l['wall_t']:5.2f}s] EventTime: {l['curr_sim_time']:6.2f}s | Sample: {l['sample_idx']:7d} | RMS: {l['rms']:.4f} | Max: {l['max_amp']:.4f}")

    # Measure latency until silence is output in callback
    silence_cbs = [l for l in silence_logs if l['rms'] == 0.0]
    if len(silence_cbs) > 0:
        first_silence_dt = (silence_cbs[0]['wall_t'] - t_jump_silence) * 1000.0
        print(f"\n>> CALLBACK LATENCY TO SILENCE: {first_silence_dt:.2f} ms after event_time change <<")
    else:
        print(f"\n>> WARNING: Callback failed to produce silence within 500 ms <<")


if __name__ == "__main__":
    # Test 1: Default device (Bluetooth Headphones)
    run_standalone_diagnostic(device_index=4, device_name="Bluetooth Headphones (realme Buds Wireless 5, MME)")
    
    # Test 2: Built-in Speakers (Realtek Audio)
    run_standalone_diagnostic(device_index=5, device_name="Built-in Speakers (Realtek Audio, MME)")
