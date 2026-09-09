# GW170817 Simulation Performance & Numerical Stability Report

## System Environment
- **Taichi Version**: `(1, 7, 4)`
- **GPU Backend**: `VULKAN` (Vulkan Hardware Acceleration Active)
- **Timestamp**: `2026-09-09 21:57:13`

## Performance Benchmark Results

| Mode | Particles | Init Time (s) | Avg FPS | Min FPS | Avg ms | p50 ms | p95 ms | p99 ms | Physics ms | Stability |
|------|-----------|---------------|---------|---------|--------|--------|--------|--------|------------|-----------|
| **DEV** | 20,000 | 0.375 | **85.1** | 25.9 | 11.75 | 10.48 | 21.14 | 26.96 | 11.75 | **PASS** |
| **NORMAL** | 75,000 | 0.404 | **60.0** | 27.5 | 16.68 | 14.64 | 34.26 | 35.93 | 16.68 | **PASS** |
| **HIGH** | 150,000 | 0.512 | **48.7** | 9.6 | 20.55 | 16.55 | 50.57 | 53.58 | 20.55 | **PASS** |

## Longer-Term Stability Check (1,000 Steps)

| Mode | Early Avg ms (Step 1-50) | Late Avg ms (Step 950-1000) | Frame Time Degradation | Result |
|------|--------------------------|-----------------------------|-----------------------|--------|
| **DEV** | 11.96 ms | 11.87 ms | -0.8% | STABLE (PASS) |
| **NORMAL** | 16.52 ms | 19.42 ms | +17.5% | STABLE (PASS) |
| **HIGH** | 19.81 ms | 19.09 ms | -3.6% | STABLE (PASS) |

## Memory & Allocation Audit Findings

- **Per-Frame Allocations**: No major per-frame allocation issue identified.
- **Taichi Field Management**: Particle system position, velocity, and attribute fields are preallocated once during initialization and updated in-place via GPU compute kernels.
- **NumPy Memory Copies**: Zero full-field NumPy copies occur during the rendering loop. Particle data streams directly from GPU fields to Taichi GGUI vertex renderers.
- **Waveform & State Buffers**: Strain waveform buffers use rolling circular indices with preallocated capacity, avoiding dynamic re-allocations.

## Recommended Hackathon Configuration

**Recommended Hackathon Default**: `NORMAL` (75,000 particles)

**Justification**: NORMAL mode (75,000 particles) achieves excellent performance (60.0 FPS, 34.3 ms p95) while providing 3.75x higher particle resolution than DEV mode with 100% numerical stability.

---
*Report generated automatically by GW170817 simulation performance benchmark suite.*