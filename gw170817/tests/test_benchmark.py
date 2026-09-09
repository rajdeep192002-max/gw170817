"""
Test suite for GW170817 Benchmark Machinery & Result Validation (Task 018).
"""
import sys
import os
import json
import numpy as np
from gw170817.config import SimConfig
from gw170817.benchmarks.benchmark_performance import run_benchmark_for_config


def test_benchmark_machinery():
    print("=== TASK 018 BENCHMARK MACHINERY TEST ===")

    # 1. Configuration Entries Exist & Particle Counts Correct
    cfg_dev = SimConfig(mode="DEV")
    cfg_norm = SimConfig(mode="NORMAL")
    cfg_high = SimConfig(mode="HIGH")

    assert cfg_dev.n_particles == 20_000, f"Expected 20,000 particles, got {cfg_dev.n_particles}"
    assert cfg_norm.n_particles == 75_000, f"Expected 75,000 particles, got {cfg_norm.n_particles}"
    assert cfg_high.n_particles == 150_000, f"Expected 150,000 particles, got {cfg_high.n_particles}"
    print("1. Configuration entries & particle counts: PASS")

    # 2. Lightweight Benchmark Run (5 warmup, 10 measurement frames)
    res = run_benchmark_for_config(mode="DEV", warmup_frames=5, measurement_frames=10, stability_steps=10)
    assert isinstance(res, dict)
    print("2. Lightweight benchmark execution: PASS")

    # 3. Schema & Keys Validation
    required_keys = [
        "mode", "n_particles", "backend", "init_time_sec", "warmup_frames", "measurement_frames",
        "avg_frame_ms", "min_frame_ms", "max_frame_ms", "p50_frame_ms", "p95_frame_ms", "p99_frame_ms",
        "avg_physics_ms", "avg_fps", "min_fps", "max_fps", "numerical_stability_passed", "longer_stability_passed"
    ]
    for key in required_keys:
        assert key in res, f"Missing key in benchmark result schema: {key}"
    print("3. Schema & keys validation: PASS")

    # 4. Non-Negative Timings & Finite Metrics
    for key, val in res.items():
        if isinstance(val, (float, int)) and key != "performance_degradation_pct":
            assert val >= 0, f"Negative benchmark timing/count for {key}: {val}"
            assert np.isfinite(val), f"Non-finite benchmark value for {key}: {val}"
    print("4. Non-negative timings & finite metrics: PASS")

    # 5. FPS / Frame-Time Mathematical Consistency
    # avg_fps ~ 1000 / avg_frame_ms
    expected_fps = 1000.0 / res["avg_frame_ms"]
    assert abs(res["avg_fps"] - expected_fps) < 0.1, f"FPS mismatch: got {res['avg_fps']}, expected {expected_fps}"
    print("5. FPS / frame-time consistency: PASS")

    # 6. Stability Results Structure Valid
    assert isinstance(res["numerical_stability_passed"], bool)
    assert isinstance(res["longer_stability_passed"], bool)
    assert res["numerical_stability_passed"] is True
    print("6. Stability results structure valid: PASS")

    print("\nALL TASK 018 BENCHMARK MACHINERY CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_benchmark_machinery()
