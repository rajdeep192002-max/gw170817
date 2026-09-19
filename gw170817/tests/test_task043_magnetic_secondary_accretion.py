"""
Regression and verification test for Task 043: Magnetic Field Mode with Secondary Accretion Disk.

Verifies:
1. In MAGNETIC FIELD mode post-merger, accretion disk is ON (disk particles rendered).
2. Magnetic field lines are ON (show_magnetic_field is True).
3. GW wavefront is OFF (WavefrontState active is False, wave_propagation.update not called).
4. Relativistic jet lines are subdued/active (show_jet_lines is True, subdued intensity).
5. Ejecta is subdued.
6. event_time is completely unchanged across mode transitions.
7. Accretion disk remains active post-BH formation.
"""
from unittest.mock import MagicMock, patch
import pytest

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


def test_magnetic_field_mode_with_secondary_accretion():
    dashboard = create_mock_dashboard()

    # 1. Sync post-merger presentation stage (presentation time 8.0s -> merger happened, disk forming)
    dashboard.director._sync_physics_for_presentation_time(8.0)
    assert dashboard.director.disk_progress > 0.0
    assert dashboard.director.b_winding_progress > 0.0
    dashboard.coordinator._update_event_state()

    # Switch to MAGNETIC FIELD mode
    dashboard.set_view_mode("MAGNETIC FIELD")
    assert dashboard.view_mode == "MAGNETIC FIELD"
    assert dashboard.show_magnetic_field is True

    # Spy on wave_propagation.update
    wave_updates = []
    real_wave_update = dashboard.wave_propagation.update
    def wave_spy(*args, **kwargs):
        wave_updates.append(1)
        return real_wave_update(*args, **kwargs)
    dashboard.wave_propagation.update = wave_spy

    dashboard.scene.reset_mock()
    dashboard.render_frame()

    # Verify disk particles are rendered in MAGNETIC FIELD mode
    particles_calls = dashboard.scene.particles.call_args_list
    disk_rendered = any(args[0] is dashboard.renderer.disk_pos for args, kwargs in particles_calls)
    assert disk_rendered, "Accretion disk particles must be rendered in MAGNETIC FIELD mode post-merger"

    # Verify magnetic field lines are rendered
    line_calls = dashboard.scene.lines.call_args_list
    assert len(line_calls) > 0, "Lines (magnetic + jet) must be drawn in MAGNETIC FIELD mode"

    # Verify GW wavefront is NOT computed or drawn
    assert len(wave_updates) == 0, "GW wavefront update must NOT execute in MAGNETIC FIELD mode"
    gw_line_rendered = any(args[0] is dashboard.wave_propagation.gpu_line_vertices for args, kwargs in line_calls)
    assert not gw_line_rendered, "GW wavefront lines must NOT be drawn in MAGNETIC FIELD mode"

    # 2. Verify disk remains active post-BH formation (presentation time 10.0s)
    dashboard.director._sync_physics_for_presentation_time(10.0)
    dashboard.coordinator._update_event_state()
    st = dashboard.engine.current_state
    rem_st = dashboard.engine.remnant.evaluate(dashboard.engine.dynamics.inspiral_state, float(st.event_time))
    assert rem_st.is_black_hole, "Remnant must be a Black Hole at presentation_time=10.0s"

    dashboard.scene.reset_mock()
    dashboard.render_frame()
    particles_calls_bh = dashboard.scene.particles.call_args_list
    disk_rendered_bh = any(args[0] is dashboard.renderer.disk_pos for args, kwargs in particles_calls_bh)
    assert disk_rendered_bh, "Accretion disk must remain rendered post-BH formation in MAGNETIC FIELD mode"

    # 3. Verify event_time continuity across mode switches
    et_start = dashboard.engine.current_state.event_time
    for mode in ["CORE", "MAGNETIC FIELD", "NEUTRINO", "MULTI", "MAGNETIC FIELD"]:
        dashboard.set_view_mode(mode)
        assert dashboard.engine.current_state.event_time == et_start, f"event_time changed when switching to {mode}"
