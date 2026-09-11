"""
Scientific Presentation Dashboard for GW170817 Multi-Messenger Simulation.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Integrates the reduced-order simulation engine with a Taichi GGUI visualization window,
a 15-second continuous presentation timeline, dynamic 2D GPU starfield coordinate-warp lensing,
time-frequency chirp track, 3D multi-component ejecta, rotational dynamics, and a deterministic DemoDirector playback controller.

Does NOT perform full numerical relativity, GRHD, MHD, or radiation transport.
"""
import sys
import time
from typing import Optional
import numpy as np
import taichi as ti

from gw170817.constants import day, Mpc, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation, SimulationState
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector, DemoStage, PlaybackMode
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.lensing import RelativisticLensingModel
from gw170817.visualization.background import BackgroundStarfield
from gw170817.visualization.raytracer import SchwarzschildRaytracer
from gw170817.visualization.field_lines import MagneticFieldLines


class ScientificDashboard:
    """
    Single-window GGUI Scientific Presentation Dashboard for GW170817.
    """

    def __init__(
        self,
        engine: Optional[GW170817Simulation] = None,
        config: Optional[SimConfig] = None,
        window_size: tuple = (1600, 900),
        fps_target: int = 60,
        substeps_per_frame: int = 2
    ):
        if config is None:
            config = SimConfig()
        self.config = config

        if engine is None:
            engine = GW170817Simulation(config=self.config)
        self.engine = engine

        self.coordinator = MultiMessengerCoordinator(engine=self.engine, config=self.config)
        self.director = DemoDirector(coordinator=self.coordinator, config=self.config)
        self.lensing = RelativisticLensingModel(config=self.config)

        self.window_size = window_size
        self.fps_target = fps_target
        self.substeps_per_frame = substeps_per_frame

        self.paused = False
        self._last_time = time.time()
        self.fps = 60.0
        self.frame_time_ms = 16.7

        # Renderer & GPU Raytracer
        self.renderer = ParticleRenderer(self.engine.psys)
        self.bg = BackgroundStarfield(width=512, height=288)
        self.raytracer = SchwarzschildRaytracer(bg=self.bg, width=256, height=144)
        self.field_lines = MagneticFieldLines(n_lines=20)

        # Initialize GGUI Window & Scene
        self.window = ti.ui.Window(
            "GW170817 Multi-Messenger Scientific Presentation Dashboard (Task 025 Integration)",
            self.window_size,
            vsync=False
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._frame_count = 0

        self._setup_default_camera()

    def _setup_default_camera(self):
        """Set up fixed scientific 3D camera viewing orbital plane."""
        cam_dist = 280.0e3
        self.camera.position(0.0, -cam_dist * 0.85, cam_dist * 0.65)
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.camera.z_near(1.0e3)
        self.camera.z_far(3.0e6)

    def _update_camera_tracking(self):
        pass

    def process_input(self):
        """Process keyboard events for simulation, director, playback, and lensing control."""
        if self.window.get_event(ti.ui.PRESS):
            evt = getattr(self.window, "event", getattr(self.window, "current_event", None))
            key = getattr(evt, "key", None) if evt else None
            if key is None:
                return

            if key == ti.ui.SPACE:
                if not self.director.is_running or self.director.is_complete:
                    print("[Demo Director] Starting 15-second continuous presentation playback [SPACE]")
                    self.director.start()
                else:
                    self.director.toggle_pause()
                    status = "PAUSED" if self.director.is_paused else "RESUMED"
                    print(f"[Demo Director] Playback {status}")

            elif key in ['1']:
                self.director.set_real_time()
                print("[Playback Mode] Set to REAL TIME (1.00x)")

            elif key in ['2']:
                self.director.set_slow_motion()
                print("[Playback Mode] Set to SLOW MOTION (0.10x)")

            elif key in ['t', 'T']:
                self.director.toggle_playback_mode()
                print(f"[Playback Mode] Toggled to {self.director.playback_mode_str} ({self.director.speed_multiplier:.2f}x)")

            elif key in ['+', '=']:
                self.director.increase_speed()
                print(f"[Playback Speed] Increased to {self.director.speed_multiplier:.2f}x")

            elif key in ['-', '_']:
                self.director.decrease_speed()
                print(f"[Playback Speed] Decreased to {self.director.speed_multiplier:.2f}x")

            elif key in ['l', 'L']:
                if self.window.is_pressed(ti.ui.SHIFT):
                    enh = self.lensing.toggle_enhanced()
                    print(f"[Relativistic Lensing] Enhanced mode toggled ({enh})")
                else:
                    en = self.lensing.toggle_enabled()
                    print(f"[Relativistic Lensing] Toggled ({en})")

            elif key in ['n', 'N']:
                print("[Demo Director] Advancing to next continuous demonstration stage [N]")
                self.director.next_stage()

            elif key in ['b', 'B']:
                print("[Demo Director] Stepping backward to previous continuous demonstration stage [B]")
                self.director.previous_stage()

            elif key in ['r', 'R']:
                print("[Demo Director] Resetting demonstration timeline to t = -5.0s [R]")
                self.director.reset()

            elif key == ti.ui.ESCAPE:
                self.window.running = False

    def render_gui_overlays(self):
        """Render scientific telemetry HUD overlays with explicit [OBS], [REF], and [MOD] provenance labels."""
        gui = self.window.get_gui()
        st = self.engine.current_state
        evt_st = self.coordinator.current_state

        # 1. Left Telemetry, Observables & Compact Engine Panel
        gui.begin("GW170817 Scientific Telemetry & Engine", 0.01, 0.01, 0.35, 0.95)
        
        lens_active_str = "ACTIVE (Vulkan RK4)" if self.lensing.enabled else "OFF"
        kn_status = "ACTIVE (3D Multi-Comp Ejecta)" if st.kilonova_luminosity > 0 else "PENDING"
        grb_status = "TRIGGERED (+1.74s)" if st.grb_triggered else ("LAUNCHED" if st.grb_launched else "PENDING")
        ag_status = "ACTIVE (Off-Axis Synchrotron)" if st.afterglow_flux > 1e-35 else "PENDING (~150d peak)"
        j_state_str = "OUTFLOW" if evt_st.grb_triggered else ("COLLIMATING" if evt_st.grb_launched else "BUILDING")

        text_left = (
            f"GW170817 MULTI-MESSENGER SIMULATION\n"
            f"--------------------------------------------\n"
            f"Presentation Time: {self.director.presentation_time:5.2f} / 15.0 s\n"
            f"Physical Event Time: {st.event_time:+7.3f} s relative to merger\n"
            f"Primary NSM Phase: {self.director.current_stage_name}\n"
            f"Playback Speed: {self.director.speed_multiplier:4.2f}x ({self.director.playback_mode_str})\n"
            f"Binary Separation: {st.separation / 1.0e3:6.1f} km [OBS]\n"
            f"GW Frequency: {st.gw_frequency:6.1f} Hz [OBS]\n"
            f"Lensing: {lens_active_str} [MODEL] Schwarzschild/weak-field\n"
            f"--------------------------------------------\n"
            f"MULTI-MESSENGER OBSERVABLES:\n"
            f"GW Strain: {'[OBS] GW170817 Inspiral' if st.event_time <= 0.0 else '[MODEL] Post-Merger Ringdown'}\n"
            f"Kilonova Lum: {kn_status} ({st.kilonova_luminosity:.3e} W) [MODEL]\n"
            f"GRB 170817A Prompt: {grb_status} [OBS]\n"
            f"Broadband Afterglow: {ag_status} [OBS/REF]\n"
            f"  F_nu (1 GHz): {st.afterglow_flux:.3e} W/m^2/Hz [MODEL]\n"
            f"--------------------------------------------\n"
            f"ROTATIONAL DYNAMICS & ENGINE [MODEL]:\n"
            f"REMNANT: {evt_st.remnant_type} ({evt_st.remnant_scenario})\n"
            f"  Omega_core = {evt_st.omega_core:.1f} rad/s | Omega_outer = {evt_st.omega_outer:.1f} rad/s\n"
            f"  Delta Omega = {evt_st.differential_rotation:.1f} rad/s | J = {evt_st.angular_momentum:.2e}\n"
            f"  T/|W| = {evt_st.t_over_w:.4f} | Shear = {evt_st.shear:.1f} s^-1 | Redshift z = {evt_st.gravitational_redshift:.3f}\n"
            f"DISK [MODEL]: M = {evt_st.disk_mass_msun:.3f} Msun | thick Keplerian torus\n"
            f"MAGNETIC [MODEL]: B_p = {evt_st.b_poloidal:.1e} G | B_phi = {evt_st.b_toroidal:.1e} G\n"
            f"  B_phi/B_p = {evt_st.b_ratio:.2f} | MRI = {'ACTIVE' if evt_st.mri_active else 'OFF'}\n"
            f"  L_Poynting = {evt_st.poynting_luminosity:.2e} W\n"
            f"JET [MODEL]: Gamma = 100 | theta = 3.5 deg ({j_state_str})\n"
        )
        gui.text(text_left)
        gui.end()

        # 2. Right Controls, Waveform & Performance Panel
        gui.begin("GW Strain & Interactive Controls", 0.67, 0.01, 0.32, 0.95)
        
        h_last = self.engine.waveform_buffer._h_plus[self.engine.waveform_buffer._head - 1] if self.engine.waveform_buffer.count > 0 else 0.0
        h_disp = h_last / 1.0e-21

        gw_ind = "[GW] ACTIVE" if st.event_time <= 0.0 else "[GW] RINGDOWN [MODEL]"
        grb_ind = "[GRB] TRIGGERED (+1.7s)" if st.grb_triggered else ("[GRB] LAUNCHING" if st.grb_launched else "[GRB] PENDING")
        kn_ind = f"[KN] L={st.kilonova_luminosity:.1e}W" if st.kilonova_luminosity > 0 else "[KN] PENDING"
        ag_ind = f"[AG] F={st.afterglow_flux:.1e}" if st.afterglow_flux > 1e-35 else "[AG] PENDING (~150d peak)"

        prog_bar = int(self.director.progress * 30.0)
        bar_str = "[" + "=" * prog_bar + ">" + " " * (30 - prog_bar) + "]"

        text_right = (
            f"INTERACTIVE DIRECTOR CONTROLS:\n"
            f" [SPACE] : Start / Pause / Resume Demo\n"
            f" [A/D/W/S]: Camera Move / Rotate\n"
            f" [1]     : REAL TIME Playback (1.00x)\n"
            f" [2]     : SLOW MOTION Playback (0.10x)\n"
            f" [T]     : Toggle REAL TIME / SLOW MOTION\n"
            f" [+/-]   : Speed Control (0.05x to 4.0x)\n"
            f" [L]     : Toggle Relativistic Lensing\n"
            f" [N/B]   : Next / Previous Checkpoint\n"
            f" [R]     : Reset Simulation (-5.0s)\n"
            f" [ESC]   : Exit Presentation\n"
            f"--------------------------------------------\n"
            f"GW STRAIN WAVEFORM [OBS / MODEL]:\n"
            f"Scaled Strain h (1e-21): {h_disp:+.2f}\n"
            f"Status: {'Cyan Polyline: Inspiral Strain h(t) [OBS]' if st.event_time <= 0.0 else 'Damped Post-Merger Ringdown [MODEL]'}\n"
            f"--------------------------------------------\n"
            f"PERFORMANCE DIAGNOSTICS:\n"
            f"Render Backend: Taichi Vulkan GPU\n"
            f"Framerate: {self.fps:5.1f} FPS ({self.frame_time_ms:4.1f} ms/frame)\n"
            f"--------------------------------------------\n"
            f"CONTINUOUS PRESENTATION TIMELINE:\n"
            f"PHASE: 0-4s INSPIRAL -> 4-6s LATE -> 6-9s MERGER -> 9-15s RINGDOWN\n"
            f"MESSENGERS: {gw_ind} | {grb_ind}\n"
            f"            {kn_ind} | {ag_ind}\n"
            f"Progress: {bar_str} ({self.director.progress * 100.0:5.1f}%)\n"
        )
        gui.text(text_right)
        gui.end()

    def render_frame(self):
        """Render 2D background starfield image, 3D viewport, particles, accretion disk, field lines, jet, chirp track, waveform trace, and overlays."""
        self.camera.track_user_inputs(self.window, movement_speed=10.0e3, hold_key=ti.ui.RMB)

        self._update_camera_tracking()
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.85, 0.85, 0.85))

        st = self.engine.current_state
        evt_st = self.coordinator.current_state
        lens_st = self.lensing.evaluate()

        # 1. Update 2D GPU RK4 Schwarzschild Raytracer Background Image (when lensing enabled)
        if lens_st.enabled:
            if self._frame_count % 3 == 0 or not self.director.is_running:
                r1, r2 = self.engine.inspiral.orbital_positions(
                    self.engine.dynamics.inspiral_state,
                    self.config.m1, self.config.m2
                )
                lens_flag = bool(lens_st.enabled)
                enh_scale = 2.5 if lens_st.enhanced_mode else 1.0

                self.raytracer.render(
                    ns1_pos=r1,
                    ns2_pos=r2,
                    m1=self.config.m1,
                    m2=self.config.m2,
                    lensing_active=lens_flag,
                    enhanced_scale=enh_scale
                )
            self.canvas.set_image(self.raytracer.output_img)

        # 2. Render 3D Inspiral NS Particles (during inspiral/merger before post-merger ejection)
        post_merger_active = bool(st.merger_contact_fraction > 0.05 or self.director.ejecta_progress > 0.0)
        if not post_merger_active:
            if self._frame_count % 2 == 0 or not self.director.is_running:
                self.renderer.update_particle_colors(
                    self.engine.psys.pos,
                    self.engine.psys.star_id,
                    self.engine.psys.active,
                    self.engine.psys.ye,
                    self.engine.psys.max_particles,
                    float(st.merger_contact_fraction),
                    float(self.director.ejecta_progress)
                )
            self.scene.particles(
                self.renderer.render_pos,
                radius=1.8e3,
                per_vertex_color=self.renderer.colors
            )
        else:
            # 3. Post-Merger Visual Geometry Updates (cadenced at 30 Hz)
            eff_event_time = max(float(st.event_time), float(evt_st.event_time))
            ejecta_fluid_active = bool(self.director.ejecta_progress > 0.0 or st.event_time >= 0.0)
            remnant_active = bool(st.merger_contact_fraction > 0.05)
            disk_active = bool(self.director.disk_progress > 0.0)

            if self._frame_count % 2 == 0 or not self.director.is_running:
                if ejecta_fluid_active:
                    self.renderer.update_ejecta_fluid(
                        event_time=eff_event_time,
                        ejecta_progress=self.director.ejecta_progress,
                        is_active=ejecta_fluid_active
                    )
                if remnant_active:
                    self.renderer.update_remnant_particles(
                        event_time=eff_event_time,
                        contact_frac=float(st.merger_contact_fraction),
                        remnant_type_str=evt_st.remnant_type
                    )
                if disk_active:
                    self.renderer.update_disk_particles(
                        event_time=eff_event_time,
                        disk_progress=self.director.disk_progress,
                        is_active=disk_active
                    )

            # Single consolidated draw call for post-merger particles (remnant + disk + ejecta)
            if ejecta_fluid_active or remnant_active or disk_active:
                self.renderer.update_combined_post_merger()
                self.scene.particles(
                    self.renderer.combined_post_merger_pos,
                    radius=2.5e3,
                    per_vertex_color=self.renderer.combined_post_merger_colors
                )

        # 4. Helical Magnetic Field Lines & Relativistic Jet Lines (cadenced at 30 Hz)
        b_active = bool(st.event_time >= 0.0 or self.director.b_winding_progress > 0.0)
        jet_active = bool(st.grb_launched or self.director.jet_progress > 0.0)

        if self._frame_count % 2 == 0 or not self.director.is_running:
            if b_active:
                self.field_lines.update(
                    b_pol=evt_st.b_poloidal,
                    b_tor=evt_st.b_toroidal,
                    r_rem=14.0e3,
                    is_active=b_active,
                    winding_progress=self.director.b_winding_progress
                )
            if jet_active:
                _jet_delay = float(self.engine.jet.jet_delay)
                _jet_beta  = float(self.engine.jet.beta(self.engine.jet.lorentz_profile(0.0)))
                self.renderer.update_jet_lines(
                    event_time=st.event_time,
                    jet_progress=self.director.jet_progress,
                    is_active=jet_active,
                    jet_delay=_jet_delay,
                    beta_jet=_jet_beta
                )

        # Single consolidated draw call for lines (magnetic + jet)
        if b_active or jet_active:
            self.renderer.update_combined_lines(
                self.field_lines.line_vertices,
                self.field_lines.line_colors,
                self.field_lines.n_vertices
            )
            self.scene.lines(
                self.renderer.combined_line_vertices,
                width=3.0,
                per_vertex_color=self.renderer.combined_line_colors
            )

        self.canvas.scene(self.scene)
        self.render_gui_overlays()
        self.window.show()
        self._frame_count += 1

    def run_step(self):
        """Execute single frame step."""
        now = time.time()
        dt_wall = now - self._last_time
        self._last_time = now

        if dt_wall > 0.0:
            current_fps = 1.0 / dt_wall
            self.fps = 0.9 * self.fps + 0.1 * current_fps
            self.frame_time_ms = dt_wall * 1000.0

        self.process_input()

        if self.director and self.director.is_running:
            self.director.step(dt_wall)
        elif self.paused:
            pass
        else:
            # IDLE mode before SPACE: maintain initial presentation state (t = -5.0 s)
            self.director._sync_physics_for_presentation_time(0.0)

        self.render_frame()

    def run(self):
        """Main Dashboard execution loop."""
        print("[Dashboard] Launching GW170817 Multi-Messenger Presentation Dashboard...")
        print("[Dashboard] Press [SPACE] to start continuous 15-second demonstration playback")
        while self.window.running:
            self.run_step()


def launch_dashboard():
    """Entry point for launching the scientific dashboard."""
    dashboard = ScientificDashboard()
    dashboard.run()


if __name__ == "__main__":
    launch_dashboard()
