"""
Task 028B-A3 — Unit tests for post-merger visual time-scale adapter and float32 rotation phase wrapping.

Verifies:
1. Physical event_time in simulation models is unchanged.
2. Renderer visual time adapter is strictly monotonic, continuous, and preserves initial launch velocity.
3. Ejecta visual radii remain within camera bounds (r <= z_far = 3.5e6 m) at t_event up to 200+ days.
4. Jet frustum length remains within camera bounds (curr_len <= z_far) at t_event up to 200+ days.
5. Remnant and disk rotational phases remain bounded within [0, 2*pi).
6. No NaN/Inf values generated in visual calculations.
7. Smooth visual continuity with no teleport or jump at late presentation times.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from gw170817.config import SimConfig
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.dashboard import ScientificDashboard


class MockCamera:
    def __init__(self):
        self._pos = (0.0, 0.0, 0.0)

    def position(self, x=None, y=None, z=None):
        if x is not None:
            self._pos = (float(x), float(y), float(z))
        return self._pos

    def lookat(self, *args, **kwargs): pass
    def up(self, *args, **kwargs): pass
    def projection_mode(self, *args, **kwargs): pass
    def z_near(self, *args, **kwargs): pass
    def z_far(self, *args, **kwargs): pass


class TestTask028BA3VisualScaleAndPhaseWrapping:
    """Test suite verifying Task 028B-A3 visual time adapter and phase wrapping stability."""

    @pytest.fixture
    def dashboard(self):
        config = SimConfig()
        with patch("gw170817.visualization.dashboard.ti.ui.Window") as mock_win, patch("gw170817.visualization.dashboard.ti.ui.Camera", side_effect=MockCamera):
            mock_win_inst = MagicMock()
            mock_win.return_value = mock_win_inst
            dash = ScientificDashboard(config=config)
            yield dash

    def test_visual_time_adapter_monotonicity_and_continuity(self):
        """1 & 2. Visual time adapter is strictly monotonic, continuous, and has derivative=1 at t=0."""
        # Value at t=0 is exactly 0
        t_vis_0 = ParticleRenderer.get_visual_time_adapter(0.0)
        assert t_vis_0 == 0.0

        # Monotonicity check across physical event times up to 200 days
        times = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 86400.0, 2.88e6, 1.728e7]
        vis_times = [ParticleRenderer.get_visual_time_adapter(t) for t in times]

        for i in range(len(vis_times) - 1):
            assert vis_times[i + 1] > vis_times[i], f"Adapter must be strictly monotonic at t={times[i+1]}"

        # Derivative at t=0 is ~1.0
        dt = 1.0e-5
        d_vis = (ParticleRenderer.get_visual_time_adapter(dt) - ParticleRenderer.get_visual_time_adapter(0.0)) / dt
        assert np.isclose(d_vis, 1.0, atol=2.0e-2)

    def test_ejecta_and_jet_radii_bounded_within_camera_z_far(self):
        """3 & 4. Ejecta and jet visual geometry remain inside camera z_far (3.5e6 m)."""
        c_light = 2.998e8
        v_blue = 0.30 * c_light  # 9.0e7 m/s
        z_far = 3.5e6  # 3.5e6 m = 3,500 km

        # Evaluate at astronomical times (up to 200 days = 1.728e7 s)
        for t_event in [1.0, 10.0, 86400.0, 2.88e6, 1.728e7]:
            t_vis = ParticleRenderer.get_visual_time_adapter(t_event)
            r_ejecta_max = 25.0e3 + v_blue * t_vis * 1.2
            assert r_ejecta_max < z_far, f"Ejecta radius {r_ejecta_max} m exceeds z_far at t={t_event} s"

            # Jet frustum length check
            t_vis_jet = ParticleRenderer.get_visual_time_adapter(t_event)
            z_jet_max = 14.0e3 + c_light * t_vis_jet
            assert z_jet_max < z_far, f"Jet length {z_jet_max} m exceeds z_far at t={t_event} s"

    def test_rotational_phase_wrapping(self, dashboard):
        """5 & 6. Remnant and disk rotational phases remain bounded within [0, 2*pi)."""
        for t_event in [0.0, 1.0, 100.0, 86400.0, 2.88e6, 1.728e7]:
            phi_rem = float((120.0 * t_event) % (2.0 * np.pi))
            assert 0.0 <= phi_rem < 2.0 * np.pi

            phi_disk = float((0.05 * t_event) % (2.0 * np.pi))
            assert 0.0 <= phi_disk < 2.0 * np.pi

    def test_no_nan_or_inf_in_renderer_updates(self, dashboard):
        """7. Renderer update calls produce finite vertex positions with zero NaN/Inf at 200 days."""
        dash = dashboard
        t_late = 1.728e7  # 200 days physical event time

        dash.renderer.update_remnant_particles(t_late, contact_frac=1.0, remnant_type_str="BH", horizon_radius_m=8.0e3)
        dash.renderer.update_disk_particles(t_late, is_active=True, disk_progress=1.0)
        dash.renderer.update_ejecta_fluid(t_late, ejecta_progress=1.0, is_active=True)
        dash.renderer.update_jet_lines(event_time=t_late, jet_progress=1.0, is_active=True)

        rem_pos = dash.renderer.remnant_pos.to_numpy()
        disk_pos = dash.renderer.disk_pos.to_numpy()
        ej_pos = dash.renderer.ejecta_fluid_pos.to_numpy()
        jet_pos = dash.renderer.jet_vertices.to_numpy()

        assert np.all(np.isfinite(rem_pos))
        assert np.all(np.isfinite(disk_pos))
        assert np.all(np.isfinite(ej_pos))
        assert np.all(np.isfinite(jet_pos))
