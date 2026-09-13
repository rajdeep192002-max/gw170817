"""
Test suite for GW170817 Demo Director & Presentation Playback (Task 017, Task 021, Task 022).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.constants import day
from gw170817.config import SimConfig
from gw170817.simulation.demo_director import DemoDirector, DemoStage, PlaybackMode


def test_demo_director():
    print("=== TASK 017, 021 & 022 DEMO DIRECTOR TEST ===")

    config = SimConfig(mode="DEV", seed=42)
    director = DemoDirector(config=config)

    # 1. Reset returns to IDLE
    st_reset = director.reset()
    assert director.current_stage == "IDLE"
    assert director.stage_enum == DemoStage.IDLE
    assert not director.is_running
    assert not director.is_complete
    assert director.progress == 0.0
    print("1. reset returns to IDLE: PASS")

    # 2. Start enters INSPIRAL (Demo starts near -5.0 s event_time)
    st_start = director.start()
    assert director.current_stage == "INSPIRAL"
    assert director.stage_enum == DemoStage.INSPIRAL
    assert director.is_running
    assert not director.is_paused
    assert abs(st_start.event_time - (-5.0)) < 0.5, f"Expected event_time ~ -5 s, got {st_start.event_time}"
    assert 70.0 <= st_start.gw_frequency <= 76.0, f"Expected f_gw ~ 72.4 Hz, got {st_start.gw_frequency}"
    print("2. start enters INSPIRAL near -5.0 s: PASS")

    # 3. Pause/Resume works
    director.pause()
    assert director.is_paused
    director.resume()
    assert not director.is_paused
    director.toggle_pause()
    assert director.is_paused
    director.toggle_pause()
    assert not director.is_paused
    print("3. pause/resume works: PASS")

    # 4. Next stage advances deterministically
    st_late = director.next_stage()
    assert director.current_stage == "LATE_INSPIRAL"
    print("4. next_stage advances deterministically: PASS")

    # 5. Previous stage moves backward
    st_prev = director.previous_stage()
    assert director.current_stage == "INSPIRAL"
    print("5. previous_stage moves backward: PASS")

    # 6. All four presentation stages are reachable
    director.reset()
    visited_stages = []
    director.start()
    visited_stages.append(director.current_stage)
    for _ in range(3):
        st = director.next_stage()
        visited_stages.append(director.current_stage)

    expected_stages = ["INSPIRAL", "LATE_INSPIRAL", "MERGER", "RINGDOWN"]
    assert visited_stages == expected_stages, f"Expected {expected_stages}, got {visited_stages}"
    print("6. all four stages reachable: PASS")

    # 7. Progress remains in [0.0, 1.0]
    director.reset()
    director.start()
    for dt in [0.1, 0.5, 1.0, 2.0]:
        director.update(dt)
        p_overall = director.progress
        p_stage = director.stage_progress
        assert 0.0 <= p_overall <= 1.0, f"Overall progress out of bounds: {p_overall}"
        assert 0.0 <= p_stage <= 1.0, f"Stage progress out of bounds: {p_stage}"
    print("7. progress remains in [0,1]: PASS")

    # 8. Playback Modes (REAL TIME / SLOW MOTION)
    director.reset()
    assert director.playback_mode_str == "REAL TIME"
    assert director.speed_multiplier == 1.0

    director.set_slow_motion()
    assert director.playback_mode_str == "SLOW MOTION"
    assert director.speed_multiplier == 0.10

    director.toggle_playback_mode()
    assert director.playback_mode_str == "REAL TIME"
    assert director.speed_multiplier == 1.0

    director.increase_speed()
    assert director.speed_multiplier == 1.25
    director.decrease_speed()
    assert director.speed_multiplier == 1.00
    print("8. playback modes & speed multipliers: PASS")

    # 9. Waveform buffer preserved across stage jumps
    director.reset()
    director.start()
    buf_initial_count = director.coordinator.engine.waveform_buffer.count
    director.next_stage()
    buf_next_count = director.coordinator.engine.waveform_buffer.count
    assert buf_next_count >= buf_initial_count, "Waveform buffer must not be cleared on stage transition"
    print("9. waveform buffer preserved across stage jumps: PASS")

    # 10. Complete state is deterministic
    director.reset()
    director.start()
    for _ in range(len(expected_stages)):
        director.next_stage()
    assert director.is_complete
    assert director.current_stage in ("COMPLETE", "CONTINUOUS_POST_MERGER")
    assert director.progress == 1.0
    print("10. complete state is deterministic: PASS")

    # 11. Presentation timing does not modify physical event timestamps
    director.reset()
    st_init = director.start()
    t_phys_grb_ref = director.coordinator.engine.jet.jet_delay
    t_phys_ag_ref = director.coordinator.engine.afterglow.t_peak_days

    director.update(100.0)  # Advance presentation clock heavily

    assert director.coordinator.engine.jet.jet_delay == t_phys_grb_ref == 1.7
    assert director.coordinator.engine.afterglow.t_peak_days == t_phys_ag_ref == 150.0
    print("11. presentation timing does not modify physical event timestamps: PASS")

    # 12. No NaN/Inf in director state
    director.reset()
    director.start()
    for _ in range(4):
        summary = director.summary_dict()
        for k, v in summary.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"Non-finite summary value for {k}: {v}"
        st_ev = director.update(0.5)
        for k, v in st_ev.__dict__.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"Non-finite event state value for {k}: {v}"

    print("12. no NaN/Inf in director state: PASS")

    # 13. disk_progress property evolution across presentation checkpoints
    director.reset()
    assert hasattr(director, "disk_progress"), "director must have disk_progress property"
    assert director.disk_progress == 0.0, f"Expected 0.0 before merger, got {director.disk_progress}"

    director.jump_to_stage_index(0)  # INSPIRAL (t_pres = 0.0)
    assert director.disk_progress == 0.0

    director.jump_to_stage_index(2)  # MERGER (t_pres = 6.0)
    assert director.disk_progress == 0.0
    assert director.coordinator.current_state.event_time == 0.0
    assert director.coordinator.engine.current_state.event_time == 0.0
    assert director.coordinator.engine.current_state.merger_contact_fraction == 0.0

    director.jump_to_stage_index(3)  # RINGDOWN (t_pres = 9.0)
    assert director.disk_progress == 1.0
    print("13. disk_progress property evolution across checkpoints: PASS")

    print("\nALL TASK 017, 021 & 022 DEMO DIRECTOR CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_demo_director()
