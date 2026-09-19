"""
Regression test for Task E3: Caching Waveform Chronological Buffer.
Verifies that:
1. The first call to get_chronological() computes the chronological tuple.
2. Repeated calls to get_chronological() without new appends reuse the cached tuple (_cached_appended_count unchanged).
3. Appending a new sample invalidates/recomputes the cache.
4. Chronological ordering and returned values match uncached values exactly (pre-wrap and post-wrap).
5. Returned arrays are safe defensive copies.
"""
from unittest.mock import patch
import pytest
import numpy as np

from gw170817.physics.gravitational_waves import WaveformBuffer, WaveformSample


def test_waveform_buffer_caching_behavior():
    buf = WaveformBuffer(capacity=5)

    # Empty buffer check
    t, hp, hc, fgw = buf.get_chronological()
    assert len(t) == 0

    # Add 3 samples (buffer not wrapped)
    for i in range(3):
        sample = WaveformSample(
            time=float(i * 0.1),
            f_gw=100.0 + i * 10,
            phase=float(i * 0.5),
            amplitude=1e-21 * (i + 1),
            h_plus=1e-21 * (i + 1),
            h_cross=0.5e-21 * (i + 1)
        )
        buf.append(sample)

    assert buf._total_appended == 3

    # Spy on np.roll to verify it is NOT called when buffer is not wrapped or when cached
    with patch("numpy.roll", wraps=np.roll) as roll_spy:
        # First request computes cache
        t1, hp1, hc1, fgw1 = buf.get_chronological()
        assert len(t1) == 3
        assert np.allclose(t1, [0.0, 0.1, 0.2])
        assert buf._cached_appended_count == 3
        roll_spy.assert_not_called()

        # Second request without appends reuses cache
        t2, hp2, hc2, fgw2 = buf.get_chronological()
        assert np.array_equal(t1, t2)
        assert np.array_equal(hp1, hp2)
        roll_spy.assert_not_called()

    # Append 3 more samples to force buffer wrapping (capacity=5, total=6)
    for i in range(3, 6):
        sample = WaveformSample(
            time=float(i * 0.1),
            f_gw=100.0 + i * 10,
            phase=float(i * 0.5),
            amplitude=1e-21 * (i + 1),
            h_plus=1e-21 * (i + 1),
            h_cross=0.5e-21 * (i + 1)
        )
        buf.append(sample)

    assert buf._total_appended == 6
    assert buf.is_full

    with patch("numpy.roll", wraps=np.roll) as roll_spy:
        # First post-wrap request computes cache using np.roll
        t_wrap1, hp_wrap1, hc_wrap1, fgw_wrap1 = buf.get_chronological()
        assert len(t_wrap1) == 5
        assert np.allclose(t_wrap1, [0.1, 0.2, 0.3, 0.4, 0.5])
        roll_spy.assert_called_once()
        assert buf._cached_appended_count == 6

        # Second post-wrap request WITHOUT new appends MUST REUSE CACHE (zero np.roll calls)
        roll_spy.reset_mock()
        t_wrap2, hp_wrap2, hc_wrap2, fgw_wrap2 = buf.get_chronological()
        assert np.array_equal(t_wrap1, t_wrap2)
        assert np.array_equal(hp_wrap1, hp_wrap2)
        roll_spy.assert_not_called()  # Proves np.roll was bypassed!

    # Verify returned array is defensive copy (modifying returned array does not corrupt cache)
    t_wrap2[0] = 999.0
    t_wrap3, _, _, _ = buf.get_chronological()
    assert abs(t_wrap3[0] - 0.1) < 1e-6, "Defensive copy must prevent cache corruption"
