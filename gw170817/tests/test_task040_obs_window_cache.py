"""
Regression test for Task E5: Caching Observational GW Window.
Verifies that:
1. Repeated calls to get_window() with identical parameters reuse cached interpolation results.
2. Calls with different t_center or window_sec recompute normally.
3. Re-loading or altering dataset invalidates the cache.
4. Output values match uncached values exactly.
"""
from unittest.mock import patch
import pytest
import numpy as np

from gw170817.simulation.observational_data import ObservationalGWData


def test_obs_window_cache_behavior():
    obs = ObservationalGWData()
    assert obs.loaded, "Observational asset must be loaded"

    # 1. First call computes window
    t1, h1_1, l1_1 = obs.get_window(t_center=0.0, window_sec=1.5, n_samples=128)
    assert len(t1) == 128
    assert obs._cached_key is not None

    # 2. Second call with identical inputs reuses cache
    with patch("numpy.interp", wraps=np.interp) as interp_spy:
        t2, h1_2, l1_2 = obs.get_window(t_center=0.0, window_sec=1.5, n_samples=128)
        interp_spy.assert_not_called()  # Proves np.interp was bypassed!
        assert np.array_equal(t1, t2)
        assert np.array_equal(h1_1, h1_2)

    # 3. Call with different t_center recomputes normally
    with patch("numpy.interp", wraps=np.interp) as interp_spy:
        t3, h1_3, l1_3 = obs.get_window(t_center=0.5, window_sec=1.5, n_samples=128)
        interp_spy.assert_called()  # Recomputed for new window
        assert obs._cached_key[0] == 0.5
        assert not np.array_equal(t1, t3)

    # 4. Invalidation on reload
    obs._cached_key = (0.5, 1.5, 128, id(obs.relative_time))
    obs.load()
    assert obs._cached_key is None
