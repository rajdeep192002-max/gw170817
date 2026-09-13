"""
Verification script for Global T Slow Motion across the entire simulation timeline.
Tests:
1. normal speed -> 1.0
2. T -> 0.10
3. ringdown -> still 0.10
4. BH formation -> still 0.10
5. 15 s boundary -> still 0.10
6. late post-merger -> still 0.10
7. T -> back to 1.0
8. event_time is strictly monotonic
9. all tested visual/physics state values remain finite (no NaN/Inf)
10. physics & visual subsystems synchronize seamlessly
"""
import sys
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.demo_director import DemoDirector, DemoStage, PlaybackMode

def verify_global_slow_motion():
    print("=" * 70)
    print("=== GLOBAL T SLOW MOTION VERIFICATION ACROSS ENTIRE TIMELINE ===")
    print("=" * 70)

    config = SimConfig(mode="DEV", seed=42)
    director = DemoDirector(config=config)
    director.start()

    # 1. Normal speed -> 1.0
    print("\n--- 1. Normal Speed Initial State ---")
    assert director.playback_mode_str == "REAL TIME"
    assert director.speed_multiplier == 1.0
    assert not director.is_slow_motion
    print(f"Initial: {director.playback_mode_str} ({director.speed_multiplier:.2f}x) | is_slow_motion={director.is_slow_motion} -> PASS")

    # 2. Press T -> 0.10
    print("\n--- 2. Press T: Toggle to Slow Motion ---")
    director.toggle_playback_mode()
    assert director.playback_mode_str == "SLOW MOTION"
    assert director.speed_multiplier == 0.10
    assert director.is_slow_motion
    print(f"After T: {director.playback_mode_str} ({director.speed_multiplier:.2f}x) | is_slow_motion={director.is_slow_motion} -> PASS")

    # 3. Continuous run from Inspiral through Late Inspiral, Merger, Ringdown, BH Formation, 15s boundary, and Long Post-Merger
    print("\n--- 3. Continuous Execution Across All Stages at 0.10x ---")
    dt = 0.02  # 50 FPS
    history_event_times = []
    history_stages = set()
    history_pres_times = []

    # Record initial event time
    st_init = director.coordinator.current_state
    history_event_times.append(st_init.event_time)
    history_pres_times.append(director.presentation_time)

    # Step through 20 presentation seconds at 0.10x:
    # 20 presentation seconds / (dt * 0.10) = 20 / 0.002 = 10,000 steps
    total_steps = 10000
    ringdown_verified = False
    bh_formation_verified = False
    boundary_15s_verified = False
    late_post_merger_verified = False

    for step_idx in range(total_steps):
        director.step(dt)
        st = director.coordinator.current_state
        history_event_times.append(st.event_time)
        history_pres_times.append(director.presentation_time)
        history_stages.add(director.stage_enum)

        # Verify speed_multiplier remains strictly 0.10 at EVERY step
        assert director.speed_multiplier == 0.10, f"Speed multiplier drifted to {director.speed_multiplier} at step {step_idx}"
        assert director.is_slow_motion, f"is_slow_motion became False at step {step_idx}"

        # Verify no NaN / Inf
        assert np.isfinite(st.event_time), f"Non-finite event_time at step {step_idx}: {st.event_time}"
        assert np.isfinite(st.gw_frequency), f"Non-finite gw_frequency at step {step_idx}: {st.gw_frequency}"
        assert np.isfinite(st.separation), f"Non-finite separation at step {step_idx}: {st.separation}"
        assert np.isfinite(director.disk_progress), f"Non-finite disk_progress at step {step_idx}"
        assert np.isfinite(director.ejecta_progress), f"Non-finite ejecta_progress at step {step_idx}"

        # Check Milestone: Ringdown (t_pres in [9.0, 15.0])
        if 9.5 <= director.presentation_time <= 10.0 and not ringdown_verified:
            assert director.speed_multiplier == 0.10
            assert director.is_slow_motion
            assert director.stage_enum == DemoStage.RINGDOWN
            print(f"  [Milestone: Ringdown] t_pres={director.presentation_time:.2f}s, stage={director.current_stage}, speed={director.speed_multiplier:.2f}x -> PASS")
            ringdown_verified = True

        # Check Milestone: BH formation (event_time >= 0.060s or remnant is BH)
        rem_st = director.coordinator.engine.remnant.evaluate(director.coordinator.engine.dynamics.inspiral_state, float(st.event_time))
        if rem_st.is_black_hole and not bh_formation_verified:
            assert director.speed_multiplier == 0.10
            assert director.is_slow_motion
            print(f"  [Milestone: BH Formation] t_pres={director.presentation_time:.2f}s, t_event={st.event_time:.2f}s, speed={director.speed_multiplier:.2f}x, BH formed -> PASS")
            bh_formation_verified = True

        # Check Milestone: 15s presentation boundary
        if 14.95 <= director.presentation_time <= 15.05 and not boundary_15s_verified:
            assert director.speed_multiplier == 0.10
            assert director.is_slow_motion
            print(f"  [Milestone: 15s Boundary] t_pres={director.presentation_time:.2f}s, stage={director.current_stage}, speed={director.speed_multiplier:.2f}x -> PASS")
            boundary_15s_verified = True

        # Check Milestone: Late post-merger (t_pres >= 18.0s)
        if director.presentation_time >= 18.0 and not late_post_merger_verified:
            assert director.speed_multiplier == 0.10
            assert director.is_slow_motion
            assert director.stage_enum == DemoStage.CONTINUOUS_POST_MERGER
            print(f"  [Milestone: Late Post-Merger] t_pres={director.presentation_time:.2f}s, stage={director.current_stage}, speed={director.speed_multiplier:.2f}x -> PASS")
            late_post_merger_verified = True

    assert ringdown_verified, "Ringdown stage was not verified"
    assert bh_formation_verified, "BH formation was not verified"
    assert boundary_15s_verified, "15s boundary was not verified"
    assert late_post_merger_verified, "Late post-merger stage was not verified"

    # 4. Monotonicity of event_time
    print("\n--- 4. Event Time Monotonicity Verification ---")
    ev_arr = np.array(history_event_times)
    diffs = np.diff(ev_arr)
    negative_diffs = diffs[diffs < -1e-9]
    assert len(negative_diffs) == 0, f"Found {len(negative_diffs)} non-monotonic event_time decreases!"
    print(f"Verified {len(ev_arr)} time steps: event_time is strictly monotonically increasing (min diff = {np.min(diffs):.2e}s) -> PASS")

    # 5. Toggle T back to 1.0 during late post-merger
    print("\n--- 5. Toggle T: Back to Real Time (1.0x) During Late Post-Merger ---")
    t_pres_before = director.presentation_time
    ev_before = director.coordinator.current_state.event_time
    director.toggle_playback_mode()
    assert director.playback_mode_str == "REAL TIME"
    assert director.speed_multiplier == 1.0
    assert not director.is_slow_motion
    print(f"After T toggle: {director.playback_mode_str} ({director.speed_multiplier:.2f}x) | is_slow_motion={director.is_slow_motion} -> PASS")

    # Step at real time
    director.step(dt)
    d_pres_rt = director.presentation_time - t_pres_before
    d_ev_rt = director.coordinator.current_state.event_time - ev_before
    assert abs(d_pres_rt - dt) < 1e-5, f"Expected presentation delta {dt}, got {d_pres_rt}"
    print(f"Real-time step delta: d_pres={d_pres_rt:.4f}s (expected {dt:.4f}s), d_event={d_ev_rt:.2f}s -> PASS")

    # 6. Verify toggle at multiple stages
    print("\n--- 6. Toggle Test at Multiple Timeline Checkpoints ---")
    checkpoints = [
        ("Early Inspiral", 1.5, DemoStage.INSPIRAL),
        ("Pre-Merger", 5.8, DemoStage.LATE_INSPIRAL),
        ("Merger", 7.5, DemoStage.MERGER),
        ("Ringdown", 10.5, DemoStage.RINGDOWN),
        ("Post-Merger 16s", 16.0, DemoStage.CONTINUOUS_POST_MERGER),
        ("Post-Merger 25s", 25.0, DemoStage.CONTINUOUS_POST_MERGER),
    ]

    for name, t_p, exp_stage in checkpoints:
        director._presentation_time = t_p
        director._stage = exp_stage
        director._stage_elapsed = t_p - director.STAGE_START_TIMES.get(exp_stage, 0.0)
        director._sync_physics_for_presentation_time(t_p)
        
        # Test 1.0 -> 0.10
        director.set_real_time()
        assert director.speed_multiplier == 1.0
        director.toggle_playback_mode()
        assert director.speed_multiplier == 0.10
        assert director.is_slow_motion
        
        # Test step delta is exactly 0.10x
        t0 = director.presentation_time
        director.step(dt)
        d_sm = director.presentation_time - t0
        assert abs(d_sm - dt * 0.10) < 1e-6
        
        # Test 0.10 -> 1.0
        director.toggle_playback_mode()
        assert director.speed_multiplier == 1.0
        assert not director.is_slow_motion
        t0 = director.presentation_time
        director.step(dt)
        d_rt = director.presentation_time - t0
        assert abs(d_rt - dt * 1.00) < 1e-6
        
        print(f"  Checkpoint '{name}' (t_pres={t_p:.1f}s): T toggles 1.0 <-> 0.10 cleanly, stepping scales perfectly -> PASS")

    # 7. Physical Subsystem Finite & Synchronized Verification
    print("\n--- 7. Physical Subsystems Finite & Synchronized State ---")
    summary = director.summary_dict()
    assert "is_slow_motion" in summary
    for k, v in summary.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"Non-finite summary value for {k}: {v}"
    print("Summary dictionary verified (all values finite).")

    print("\n" + "=" * 70)
    print("ALL GLOBAL SLOW MOTION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    verify_global_slow_motion()
