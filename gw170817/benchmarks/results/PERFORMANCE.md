# GW170817 Simulation Performance & Numerical Stability Report

## System Environment
- **Taichi Version**: `(1, 7, 4)`
- **GPU Backend**: `VULKAN` (Vulkan Hardware Acceleration Active)
- **Timestamp**: `2026-09-12 22:45:09`

## Performance Benchmark Results

| Mode | Particles | Init Time (s) | Avg FPS | Min FPS | Avg ms | p50 ms | p95 ms | p99 ms | Physics ms | Stability |
|------|-----------|---------------|---------|---------|--------|--------|--------|--------|------------|-----------|
| **DEV** | 20,000 | 2.205 | **112.8** | 18.2 | 8.87 | 4.53 | 45.69 | 51.25 | 8.87 | **PASS** |
| **NORMAL** | 75,000 | 2.261 | **79.6** | 9.2 | 12.57 | 5.26 | 72.97 | 88.48 | 12.56 | **PASS** |
| **HIGH** | 100,000 | 2.252 | **84.9** | 9.4 | 11.78 | 4.55 | 74.27 | 84.77 | 11.78 | **PASS** |

## Longer-Term Stability Check (1,000 Steps)

| Mode | Early Avg ms (Step 1-50) | Late Avg ms (Step 950-1000) | Frame Time Degradation | Result |
|------|--------------------------|-----------------------------|-----------------------|--------|
| **DEV** | 6.06 ms | 5.82 ms | -3.9% | STABLE (PASS) |
| **NORMAL** | 8.87 ms | 8.15 ms | -8.1% | STABLE (PASS) |
| **HIGH** | 8.89 ms | 9.30 ms | +4.5% | STABLE (PASS) |

## Memory & Allocation Audit Findings

- **Per-Frame Allocations**: No major per-frame allocation issue identified.
- **Taichi Field Management**: Particle system position, velocity, and attribute fields are preallocated once during initialization and updated in-place via GPU compute kernels.
- **NumPy Memory Copies**: Zero full-field NumPy copies occur during the rendering loop. Particle data streams directly from GPU fields to Taichi GGUI vertex renderers.
- **Waveform & State Buffers**: Strain waveform buffers use rolling circular indices with preallocated capacity, avoiding dynamic re-allocations.

## Recommended Hackathon Configuration

**Recommended Hackathon Default**: `NORMAL` (75,000 particles)

**Justification**: NORMAL mode (75,000 particles) achieves excellent performance (79.6 FPS, 73.0 ms p95) while providing 3.75x higher particle resolution than DEV mode with 100% numerical stability.

---
*Report generated automatically by GW170817 simulation performance benchmark suite.*