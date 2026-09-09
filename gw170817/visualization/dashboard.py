"""
Scientific Dashboard v0.9 for GW170817 Multi-Messenger Simulation.

REDUCED-ORDER APPROXIMATION:
Integrates the reduced-order simulation engine with a Taichi GGUI visualization window.
Does NOT perform full numerical relativity, GRHD, or radiative transfer.
"""
import sys
from typing import Optional
import numpy as np
import taichi as ti
from gw170817.constants import day, Mpc, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation, SimulationState
from gw170817.visualization.renderer import ParticleRenderer


class ScientificDashboard:
    """
    Single-window GGUI Scientific Dashboard for GW170817.
    """

    def __init__(
        self,
        engine: Optional[GW170817Simulation] = None,
        config: Optional[SimConfig] = None,
        window_size: tuple = (1600, 900),
        fps_target: int = 60,
        substeps_per_frame: int = 5
    ):
        if config is None:
            config = SimConfig()
        self.config = config

        if engine is None:
            engine = GW170817Simulation(config=self.config)
        self.engine = engine

        self.window_size = window_size
        self.fps_target = fps_target
        self.substeps_per_frame = substeps_per_frame

        self.paused = False

        # Instantiate Particle & Waveform Renderer
        self.renderer = ParticleRenderer(self.engine.psys)

        # Initialize GGUI Window & Scene Components
        self.window = ti.ui.Window(
            "GW170817 Multi-Messenger Scientific Simulation (v0.9)",
            self.window_size,
            vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()

        self._setup_default_camera()

    def _setup_default_camera(self):
        """Set up standard 3D camera viewing orbital plane."""
        cam_dist = 500.0e3  # 500 km visual distance
        self.camera.position(0.0, -cam_dist * 0.85, cam_dist * 0.65)
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    def process_input(self):
        """Process keyboard events for simulation control."""
        if self.window.get_event(ti.ui.PRESS):
            key = self.window.event.key
            if key == ti.ui.SPACE:
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RESUMED"
                print(f"[Dashboard Control] Simulation {status}")
            elif key in ['r', 'R']:
                print("[Dashboard Control] Resetting simulation [R]")
                self.engine.reset()
                self._setup_default_camera()
            elif key in ['m', 'M']:
                print("[Dashboard Control] Jumping to accelerated near-merger demo state [M]")
                self.engine.jump_to_demo_phase(f_gw=1200.0, separation=30.0e3)
            elif key == ti.ui.ESCAPE:
                print("[Dashboard Control] Exiting simulation [ESC]")
                self.window.running = False

    def render_overlay(self):
        """Render GGUI scientific text overlay windows."""
        gui = self.window.get_gui()
        st = self.engine.current_state

        # 1. Primary Scientific Overview Panel
        gui.begin("GW170817 Overview", 0.01, 0.01, 0.32, 0.36)
        gui.text("GW170817 Multi-Messenger Model")
        gui.text("Physics-Informed Reduced-Order Simulation")
        gui.text("--------------------------------------------")
        gui.text(f"Evolution Phase: {st.phase}")
        gui.text(f"Event Time (rel to merger): {st.event_time:.4f} s")
        gui.text(f"Elapsed Sim Time: {st.elapsed_time:.4f} s")
        gui.text(f"GW Frequency f_gw: {st.gw_frequency:.2f} Hz")
        gui.text(f"Binary Separation a: {st.separation / 1e3:.2f} km")
        gui.text(f"Contact Fraction: {st.merger_contact_fraction:.4f}")
        gui.text(f"Effective Tidal Lambda~: {self.engine.tidal.lambda_tilde:.1f}")
        gui.text(f"Distance D_L: {self.config.distance / Mpc:.1f} Mpc")
        gui.text(f"Total Mass M_tot: {self.config.M_total / M_sun:.2f} M_sun")
        gui.end()

        # 2. Multi-Messenger Observables Status Panel
        gui.begin("Multi-Messenger Status", 0.01, 0.38, 0.32, 0.32)
        gui.text("Multi-Messenger Observables")
        gui.text("--------------------------------------------")

        gw_status = "DETECTED / ACTIVE" if st.gw_frequency > 0 else "INACTIVE"
        gui.text(f"GW Strain: {gw_status}")

        kn_status = "ACTIVE / MODELLED" if st.kilonova_luminosity > 0 else "PENDING"
        gui.text(f"Kilonova Lum: {kn_status} ({st.kilonova_luminosity:.3e} W)")

        grb_status = "ACTIVE / TRIGGERED" if st.grb_triggered else ("LAUNCHED" if st.grb_launched else "PENDING (+1.7s)")
        gui.text(f"GRB 170817A: {grb_status}")

        ag_status = "ACTIVE" if st.afterglow_flux > 1e-35 else "PENDING (~150d peak)"
        gui.text(f"Afterglow Flux: {ag_status}")
        gui.text(f"  F_nu (1 GHz): {st.afterglow_flux:.3e} W/m^2/Hz")
        gui.end()

        # 3. Multi-Messenger Event Timeline Panel
        gui.begin("Event Timeline", 0.01, 0.71, 0.32, 0.26)
        gui.text("GW170817 Event Timeline Reference")
        gui.text("--------------------------------------------")
        gui.text("  -12.4 s  | Inspiraling Binary (40 Hz -> Merger)")
        gui.text("   0.0 s  | BNS Merger Reference Event")
        gui.text("  +1.7 s  | GRB 170817A Prompt Emission Delay")
        gui.text("  +150 d  | Off-Axis Broadband Afterglow Peak")
        gui.text("--------------------------------------------")
        gui.text(f"Current System State: {st.phase}")
        gui.end()

        # 4. Interactive Controls Overlay
        gui.begin("Controls", 0.67, 0.01, 0.32, 0.18)
        gui.text("Interactive Controls")
        gui.text("--------------------------------------------")
        gui.text(" [SPACE] : Pause / Resume")
        gui.text(" [R]     : Reset Simulation & Camera")
        gui.text(" [M]     : Jump to Near-Merger Demo Phase")
        gui.text(" [ESC]   : Exit Simulation")
        gui.text(" Mouse RMB + Drag: Orbit / Rotate 3D View")
        gui.end()

        # 5. Live GW Waveform Overlay Header
        gui.begin("GW Waveform h+(t)", 0.65, 0.73, 0.34, 0.24)
        gui.text("Live GW Strain Waveform History")
        h_last = self.engine.waveform_buffer._h_plus[self.engine.waveform_buffer._head - 1] if self.engine.waveform_buffer.count > 0 else 0.0
        gui.text(f"Instantaneous Strain h+: {h_last:.3e}")
        gui.text(f"Buffer Capacity: {self.engine.buffer_capacity} samples")
        gui.end()

    def render_frame(self):
        """Render full 3D viewport, lighting, particles, waveform trace, and overlays."""
        # Track user mouse camera inputs (RMB drag)
        self.camera.track_user_inputs(self.window, movement_speed=10.0e3, hold_key=ti.ui.RMB)

        self.scene.set_camera(self.camera)
        self.scene.point_light(pos=(0.0, -500.0e3, 500.0e3), color=(1.0, 1.0, 1.0))
        self.scene.ambient_light((0.2, 0.2, 0.2))

        # Render 3D particles directly from GPU fields
        st = self.engine.current_state
        self.renderer.update_particle_colors(
            self.engine.psys.star_id,
            self.engine.psys.active,
            self.engine.psys.max_particles,
            float(st.merger_contact_fraction)
        )
        self.scene.particles(
            self.engine.psys.pos,
            radius=4.0e3,
            per_vertex_color=self.renderer.colors
        )

        self.canvas.scene(self.scene)

        # Draw 2D live GW strain waveform trace on canvas
        _, hp_arr, _, _ = self.engine.waveform_buffer.get_chronological()
        if len(hp_arr) > 0:
            self.renderer.update_waveform_buffer(hp_arr)
            self.canvas.lines(
                self.renderer.waveform_vertices,
                width=0.003,
                color=(0.0, 0.85, 1.0)
            )

        # Render text overlays
        self.render_overlay()

    def run(self, max_frames: Optional[int] = None):
        """
        Main dashboard execution loop.
        """
        frame_count = 0
        while self.window.running:
            self.process_input()

            if not self.paused:
                for _ in range(self.substeps_per_frame):
                    self.engine.step()

            self.render_frame()

            self.window.show()
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break


def launch_dashboard(max_frames: Optional[int] = None):
    """Entry point helper to launch Scientific Dashboard v0.9."""
    config = SimConfig(mode="DEV", seed=42)
    engine = GW170817Simulation(config=config)
    dashboard = ScientificDashboard(engine=engine, config=config)
    dashboard.run(max_frames=max_frames)


if __name__ == "__main__":
    launch_dashboard()
