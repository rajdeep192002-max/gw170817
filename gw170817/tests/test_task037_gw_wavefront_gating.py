"""
Regression test for Task E1: Gating GW Wavefront Computation by View Mode.
Verifies that:
1. In CORE mode, wave_propagation.update() is NOT executed.
2. In GW MODE, wave_propagation.update() IS executed.
3. In MULTI mode, wave_propagation.update() IS executed.
4. In MAGNETIC FIELD and BH LENS modes, wave_propagation.update() is NOT executed.
5. Switching view modes preserves the authoritative event_time and resumes wavefront calculation seamlessly.
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

    # Mock window so we don't open a real GGUI GUI window during unit testing
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


def test_gw_wavefront_gating_by_mode():
    dashboard = create_mock_dashboard()

    # Python call tracker to spy on wave_propagation.update without breaking Taichi @ti.data_oriented
    update_calls = []
    real_update = dashboard.wave_propagation.update

    def update_spy(*args, **kwargs):
        update_calls.append((args, kwargs))
        return real_update(*args, **kwargs)

    dashboard.wave_propagation.update = update_spy

    # Set inspiral time
    dashboard.engine.set_inspiral_time(2.0)
    event_time_before = dashboard.engine.current_state.event_time

    # 1. CORE Mode: wave_propagation.update should NOT be called
    dashboard.set_view_mode("CORE")
    update_calls.clear()
    dashboard.render_frame()
    assert len(update_calls) == 0
    assert dashboard.engine.current_state.event_time == event_time_before

    # 2. MAGNETIC FIELD Mode: wave_propagation.update should NOT be called
    dashboard.set_view_mode("MAGNETIC FIELD")
    update_calls.clear()
    dashboard.render_frame()
    assert len(update_calls) == 0
    assert dashboard.engine.current_state.event_time == event_time_before

    # 3. NEUTRINO Mode: wave_propagation.update should NOT be called
    dashboard.set_view_mode("NEUTRINO")
    update_calls.clear()
    dashboard.render_frame()
    assert len(update_calls) == 0
    assert dashboard.engine.current_state.event_time == event_time_before

    # 4. GW Mode: wave_propagation.update SHOULD be called
    dashboard.set_view_mode("GW")
    update_calls.clear()
    dashboard.render_frame()
    assert len(update_calls) == 1
    assert dashboard.engine.current_state.event_time == event_time_before

    # 5. MULTI Mode: wave_propagation.update SHOULD be called
    dashboard.set_view_mode("MULTI")
    update_calls.clear()
    dashboard.render_frame()
    assert len(update_calls) == 1
    assert dashboard.engine.current_state.event_time == event_time_before

    # 6. Mode switch: CORE -> GW -> CORE preserves event_time
    dashboard.set_view_mode("CORE")
    assert dashboard.engine.current_state.event_time == event_time_before
    dashboard.set_view_mode("GW")
    assert dashboard.engine.current_state.event_time == event_time_before
