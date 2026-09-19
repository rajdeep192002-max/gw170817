"""
Observational GW170817 Strain Data Loader & Runtime Provider.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Provides offline access to the preprocessed GWOSC GW170817 detector strain asset (.npz).
Enables the dashboard and renderer to display actual observed GW170817 strain (H1 & L1)
alongside the simulation's reduced-order quadrupole waveform prediction without per-frame network calls.

Data Source: Gravitational Wave Open Science Center (https://gwosc.org)
Event: GW170817 (GPS 1187008882.43 s)
"""
import os
from typing import Tuple, Optional, Dict
import numpy as np


class ObservationalGWData:
    """
    Offline provider of GWOSC GW170817 cleaned detector strain data (H1 & L1).
    """

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "gw170817_observed_strain.npz")

        self.data_path = data_path
        self.loaded = False

        self.relative_time = np.zeros(0, dtype=np.float32)
        self.h1_strain = np.zeros(0, dtype=np.float32)
        self.l1_strain = np.zeros(0, dtype=np.float32)
        self.event_gps = 1187008882.43
        self.sample_rate = 4096
        self.detectors = "H1,L1"
        self.conditioning_band = "20-2000 Hz"
        self.processing_notes = "GWOSC GW170817 cleaned strain data"

        self._cached_key: Optional[Tuple[float, float, int, int]] = None
        self._cached_window: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        self.load()

    def load(self) -> bool:
        """Load local compact .npz observational data asset."""
        if not os.path.exists(self.data_path):
            print(f"[ObservationalGWData] Warning: Asset file {self.data_path} not found.")
            return False

        try:
            data = np.load(self.data_path)
            self.relative_time = np.array(data["relative_time"], dtype=np.float32)
            self.h1_strain = np.array(data["observed_H1"], dtype=np.float32)
            self.l1_strain = np.array(data["observed_L1"], dtype=np.float32)

            if "event_gps" in data:
                self.event_gps = float(data["event_gps"])
            if "sample_rate" in data:
                self.sample_rate = int(data["sample_rate"])
            if "detectors" in data:
                self.detectors = str(data["detectors"])
            if "conditioning_band" in data:
                self.conditioning_band = str(data["conditioning_band"])
            if "processing_notes" in data:
                self.processing_notes = str(data["processing_notes"])

            self._cached_key = None
            self._cached_window = None

            self.loaded = True
            return True
        except Exception as e:
            print(f"[ObservationalGWData] Error loading asset {self.data_path}: {e}")
            return False

    def get_sample_at_time(self, t_event: float) -> Tuple[float, float]:
        """
        Return (h1, l1) strain samples at given relative event time t_event [s].
        t_event = 0.0 s corresponds to GW170817 merger peak.
        """
        if not self.loaded or len(self.relative_time) == 0:
            return 0.0, 0.0

        idx = np.searchsorted(self.relative_time, t_event)
        idx = int(np.clip(idx, 0, len(self.relative_time) - 1))
        return float(self.h1_strain[idx]), float(self.l1_strain[idx])

    def get_window(self, t_center: float, window_sec: float = 1.5, n_samples: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return sliding window (t_window, h1_window, l1_window) centered around t_center [s].
        Decimated to n_samples for low-overhead GPU line buffer visualization.
        Caches interpolation results for identical input parameters.
        """
        dataset_id = id(self.relative_time)
        cache_key = (float(t_center), float(window_sec), int(n_samples), dataset_id)

        if self._cached_window is not None and self._cached_key == cache_key:
            tw, h1w, l1w = self._cached_window
            return tw.copy(), h1w.copy(), l1w.copy()

        if not self.loaded or len(self.relative_time) == 0:
            t_dummy = np.linspace(t_center - window_sec * 0.5, t_center + window_sec * 0.5, n_samples, dtype=np.float32)
            zeros = np.zeros(n_samples, dtype=np.float32)
            res = (t_dummy, zeros, zeros)
            self._cached_key = cache_key
            self._cached_window = res
            return t_dummy.copy(), zeros.copy(), zeros.copy()

        half_w = window_sec * 0.5
        t_min = t_center - half_w
        t_max = t_center + half_w

        idx_start = np.searchsorted(self.relative_time, t_min)
        idx_end = np.searchsorted(self.relative_time, t_max)

        idx_start = max(0, min(len(self.relative_time) - 1, idx_start))
        idx_end = max(idx_start + 1, min(len(self.relative_time), idx_end))

        sub_t = self.relative_time[idx_start:idx_end]
        sub_h1 = self.h1_strain[idx_start:idx_end]
        sub_l1 = self.l1_strain[idx_start:idx_end]

        if len(sub_t) < 2:
            t_dummy = np.linspace(t_min, t_max, n_samples, dtype=np.float32)
            zeros = np.zeros(n_samples, dtype=np.float32)
            res = (t_dummy, zeros, zeros)
            self._cached_key = cache_key
            self._cached_window = res
            return t_dummy.copy(), zeros.copy(), zeros.copy()

        # Resample smoothly to n_samples using linear interpolation
        t_out = np.linspace(t_min, t_max, n_samples, dtype=np.float32)
        h1_out = np.interp(t_out, sub_t, sub_h1).astype(np.float32)
        l1_out = np.interp(t_out, sub_t, sub_l1).astype(np.float32)

        res = (t_out, h1_out, l1_out)
        self._cached_key = cache_key
        self._cached_window = res

        return t_out.copy(), h1_out.copy(), l1_out.copy()
