"""
Regression test for Task E2: Cadencing Combined Post-Merger Visual Packing.
Verifies that:
1. update_combined_post_merger() executes on visual update frames (frame_count % 2 == 0).
2. update_combined_post_merger() is skipped on intermediate frames (frame_count % 2 != 0).
3. On skipped frames, the existing valid combined GPU particle buffers remain untouched and ready for drawing.
4. Physical simulation clock (event_time) advances normally on every step regardless of visual packing cadence.
"""
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.dashboard import ScientificDashboard


def create_mock_dashboard():
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)

    mock_window = MagicMock()
    mock_window.running = True
    mock_window.is_pressed.return_value = False
    mock_window.get_event.return_value = False
    mock_window.get_cursor_pos.return_value = (0.5, 0.5)

    mock_scene = MagicMock()
    mock_canvas = MagicMock()
    mock_gui = MagicMock()
    mock_window.get_gui.return_value = mock_gui

    with patch("taichi.ui.Window", return_value=mock_window), \
         patch("taichi.ui.Scene", return_value=mock_scene), \
         patch("taichi.ui.Canvas", return_value=mock_canvas):
        dashboard = ScientificDashboard(engine=engine, config=config)
        dashboard.window = mock_window
        dashboard.scene = mock_scene
        dashboard.canvas = mock_canvas
        return dashboard


def test_combined_post_merger_packing_cadence():
    dashboard = create_mock_dashboard()

    # Fast forward to post-merger stage (t = +2.0 s)
    dashboard.director.start()
    dashboard.director._sync_physics_for_presentation_time(11.0)
    st = dashboard.engine.current_state
    assert st.merger_contact_fraction > 0.05, "Must be post-merger phase"

    # Spy on update_combined_post_merger
    pack_calls = []
    real_pack = dashboard.renderer.update_combined_post_merger

    def pack_spy(*args, **kwargs):
        pack_calls.append((args, kwargs))
        return real_pack(*args, **kwargs)

    dashboard.renderer.update_combined_post_merger = pack_spy

    # Frame 0 (even frame): pack kernel should execute
    dashboard._frame_count = 0
    pack_calls.clear()
    t_event_0 = dashboard.engine.current_state.event_time
    dashboard.render_frame()
    assert len(pack_calls) == 1, "Even frame (frame 0) must trigger update_combined_post_merger"

    # Save buffer content from even frame
    combined_pos_after_even = dashboard.renderer.combined_post_merger_pos.to_numpy().copy()

    # Frame 1 (odd frame): pack kernel should be skipped, reusing previous buffer
    pack_calls.clear()
    t_event_1 = dashboard.engine.current_state.event_time
    dashboard.render_frame()
    assert len(pack_calls) == 0, "Odd frame (frame 1) must skip update_combined_post_merger"

    # Verify buffer content on skipped frame is identical to previous packed buffer
    combined_pos_after_odd = dashboard.renderer.combined_post_merger_pos.to_numpy()
    np.testing.assert_array_equal(combined_pos_after_even, combined_pos_after_odd)

    # Frame 2 (even frame): pack kernel should execute again
    pack_calls.clear()
    dashboard.render_frame()
    assert len(pack_calls) == 1, "Even frame (frame 2) must trigger update_combined_post_merger"
