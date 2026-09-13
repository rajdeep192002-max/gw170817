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
from gw170817.visualization.wave_propagation import GWWavefrontPropagation
from gw170817.audio.chirp_sound import GWChirpSonification
from gw170817.simulation.observational_data import ObservationalGWData


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
        self.obs_data = ObservationalGWData()
        self.wave_propagation = GWWavefrontPropagation()
        self.chirp_audio = GWChirpSonification()

        self.window_size = window_size
        self.fps_target = fps_target
        self.substeps_per_frame = substeps_per_frame

        self.paused = False
        self._last_time = time.time()
        self.fps = 60.0
        self.frame_time_ms = 16.7

        # Renderer & GPU Raytracer
        self.renderer = ParticleRenderer(self.engine.psys)
        self.bg = BackgroundStarfield(width=512, height=288, n_stars=self.renderer.n_stars, seed=int(self.config.seed))
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

        self.cam_distance = 280.0e3
        self.cam_yaw = -np.pi / 2.0
        self.cam_pitch = 0.65

        self.view_mode = "CORE"
        self.show_magnetic_field = False
        self.bh_lens_waiting = False

        # Camera interpolation targets for smooth mode transitions
        self.target_cam_distance = 280.0e3
        self.target_cam_yaw = -np.pi / 2.0
        self.target_cam_pitch = 0.65
        self.is_interpolating_cam = False

        self._setup_default_camera()

    def set_view_mode(self, mode: str):
        """
        Switch presentation view mode (CORE, GW, BH LENS, MAGNETIC FIELD, MULTI) and apply safe camera presets smoothly.
        Does NOT alter or reset underlying simulation physics or event time.
        """
        mode = mode.upper()
        if mode not in ["CORE", "GW", "BH LENS", "MAGNETIC FIELD", "MULTI"]:
            return

        self.view_mode = mode
        self.show_magnetic_field = (mode in ["MAGNETIC FIELD", "CORE", "MULTI"])
        st = self.engine.current_state
        rem_st = self.engine.remnant.evaluate(self.engine.dynamics.inspiral_state, float(st.event_time))

        if mode == "CORE":
            self.target_cam_distance = 280.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.65
            self.bh_lens_waiting = False
            self.is_interpolating_cam = True
            print("[View Mode] Set to CORE — Binary Dynamics & Merger Evolution")

        elif mode == "GW":
            # Framed to make 3D world-space expanding wavefront shells clearly readable
            self.target_cam_distance = 350.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.70
            self.bh_lens_waiting = False
            self.is_interpolating_cam = True
            self.wave_propagation.trigger(float(st.event_time))
            print("[View Mode] Set to GW MODE — Quadrupole Radiation & 3D Wavefront Propagation")

        elif mode == "BH LENS":
            if rem_st.is_black_hole:
                self.target_cam_distance = 32.0e3
                self.target_cam_yaw = -np.pi / 2.0
                self.target_cam_pitch = 0.25
                self.lensing.enabled = True
                self.bh_lens_waiting = False
                self.is_interpolating_cam = True
                print("[View Mode] Set to BH LENS / ACCRETION — Dedicated Close-Up Preset (32 km)")
            else:
                self.bh_lens_waiting = True
                print("[View Mode] BH LENS requested — WAITING FOR BLACK-HOLE FORMATION")

        elif mode == "MAGNETIC FIELD":
            # Framing that clearly shows central remnant/BH, field-line topology, winding, and jet base
            self.target_cam_distance = 160.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.45
            self.bh_lens_waiting = False
            self.is_interpolating_cam = True
            print("[View Mode] Set to MAGNETIC FIELD — Differential Rotation & Toroidal Winding")

        elif mode == "MULTI":
            self.target_cam_distance = 280.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.65
            self.bh_lens_waiting = False
            self.is_interpolating_cam = True
            print("[View Mode] Set to MULTI — Clean Multi-Messenger Story")

    def _setup_default_camera(self):
        """Set up fixed scientific 3D camera viewing orbital plane."""
        self._update_camera_position()

    def _update_camera_position(self):
        """Update 3D camera position from spherical orbit parameters."""
        x = self.cam_distance * np.cos(self.cam_pitch) * np.cos(self.cam_yaw)
        y = self.cam_distance * np.cos(self.cam_pitch) * np.sin(self.cam_yaw)
        z = self.cam_distance * np.sin(self.cam_pitch)

        self.camera.position(float(x), float(y), float(z))
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.camera.z_near(1.0e3)
        self.camera.z_far(3.5e6)

    def _sync_manual_camera_target(self):
        """Cancel mode camera interpolation on manual user camera action."""
        self.target_cam_distance = self.cam_distance
        self.target_cam_yaw = self.cam_yaw
        self.target_cam_pitch = self.cam_pitch
        self.is_interpolating_cam = False

    def camera_orbit_left(self, step: float = 0.06):
        """Orbit camera left around merger target (yaw left)."""
        self._sync_manual_camera_target()
        self.cam_yaw -= step
        self._update_camera_position()

    def camera_orbit_right(self, step: float = 0.06):
        """Orbit camera right around merger target (yaw right)."""
        self._sync_manual_camera_target()
        self.cam_yaw += step
        self._update_camera_position()

    def camera_orbit_up(self, step: float = 0.05):
        """Orbit camera upward around merger target (pitch up with inversion protection)."""
        self._sync_manual_camera_target()
        self.cam_pitch = min(np.pi / 2.0 - 0.05, self.cam_pitch + step)
        self._update_camera_position()

    def camera_orbit_down(self, step: float = 0.05):
        """Orbit camera downward around merger target (pitch down with inversion protection)."""
        self._sync_manual_camera_target()
        self.cam_pitch = max(-np.pi / 2.0 + 0.05, self.cam_pitch - step)
        self._update_camera_position()

    def camera_zoom_in(self, factor: float = 0.90):
        """Zoom camera inward toward remnant center."""
        self._sync_manual_camera_target()
        self.cam_distance = max(40.0e3, self.cam_distance * factor)
        self._update_camera_position()

    def camera_zoom_out(self, factor: float = 1.10):
        """Zoom camera outward away from remnant center."""
        self._sync_manual_camera_target()
        self.cam_distance = min(1.5e6, self.cam_distance * factor)
        self._update_camera_position()

    def _update_camera_tracking(self):
        """Smoothly interpolate camera parameters toward mode preset targets if active."""
        if getattr(self, "is_interpolating_cam", False):
            d_dist = self.target_cam_distance - self.cam_distance
            d_yaw = self.target_cam_yaw - self.cam_yaw
            d_pitch = self.target_cam_pitch - self.cam_pitch

            if abs(d_dist) < 50.0 and abs(d_yaw) < 0.002 and abs(d_pitch) < 0.002:
                self.cam_distance = self.target_cam_distance
                self.cam_yaw = self.target_cam_yaw
                self.cam_pitch = self.target_cam_pitch
                self.is_interpolating_cam = False
            else:
                self.cam_distance += 0.20 * d_dist
                self.cam_yaw += 0.20 * d_yaw
                self.cam_pitch += 0.20 * d_pitch
            self._update_camera_position()

    def process_input(self):
        """Process keyboard & mouse events for camera, simulation, director, playback, and lensing control."""
        if self.window.get_event(ti.ui.PRESS):
            evt = getattr(self.window, "event", getattr(self.window, "current_event", None))
            key = getattr(evt, "key", None) if evt else None
            if key is None:
                return

            if key == ti.ui.SPACE:
                if self.director.stage_enum in [DemoStage.IDLE, DemoStage.COMPLETE]:
                    print("[Demo Director] Starting 15-second continuous presentation playback [SPACE]")
                    self.director.start()
                    st = self.engine.current_state
                    self.chirp_audio.start_playback(st.event_time, self.director.speed_multiplier)
                else:
                    self.director.toggle_pause()
                    st = self.engine.current_state
                    if self.director.is_paused:
                        self.chirp_audio.pause_playback()
                        status = "PAUSED"
                    else:
                        self.chirp_audio.resume_playback(st.event_time, self.director.speed_multiplier)
                        status = "RESUMED"
                    print(f"[Demo Director] Playback {status}")

            elif key in [ti.ui.UP, 'Up', 'Up Arrow']:
                self.camera_orbit_up()

            elif key in [ti.ui.DOWN, 'Down', 'Down Arrow']:
                self.camera_orbit_down()

            elif key in [ti.ui.LEFT, 'Left', 'Left Arrow']:
                self.camera_orbit_left()

            elif key in [ti.ui.RIGHT, 'Right', 'Right Arrow']:
                self.camera_orbit_right()

            elif key in ['1']:
                self.director.set_real_time()
                st = self.engine.current_state
                if self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.resume_playback(st.event_time, self.director.speed_multiplier)
                print("[Playback Mode] Set to REAL TIME (1.00x)")

            elif key in ['2']:
                self.director.set_slow_motion()
                self.chirp_audio.stop_playback()
                print("[Playback Mode] Set to SLOW MOTION (0.10x)")

            elif key in ['t', 'T']:
                self.director.toggle_playback_mode()
                st = self.engine.current_state
                if self.director.speed_multiplier == 1.00 and self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.resume_playback(st.event_time, 1.00)
                else:
                    self.chirp_audio.stop_playback()
                print(f"[Playback Mode] Toggled to {self.director.playback_mode_str} ({self.director.speed_multiplier:.2f}x)")

            elif key in ['+', '=']:
                self.director.increase_speed()
                st = self.engine.current_state
                if self.director.speed_multiplier == 1.00 and self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.resume_playback(st.event_time, 1.00)
                else:
                    self.chirp_audio.stop_playback()
                print(f"[Playback Speed] Increased to {self.director.speed_multiplier:.2f}x")

            elif key in ['-', '_']:
                self.director.decrease_speed()
                st = self.engine.current_state
                if self.director.speed_multiplier == 1.00 and self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.resume_playback(st.event_time, 1.00)
                else:
                    self.chirp_audio.stop_playback()
                print(f"[Playback Speed] Decreased to {self.director.speed_multiplier:.2f}x")

            elif key in ['m', 'M']:
                muted = self.chirp_audio.toggle_mute()
                status = "MUTED" if muted else "UNMUTED"
                print(f"[GW Audio] Toggled mute ({status}) [M]")
                if not muted and self.director.is_running and not self.director.is_paused:
                    st = self.engine.current_state
                    self.chirp_audio.start_playback(st.event_time, self.director.speed_multiplier)

            elif key in ['c', 'C']:
                self.set_view_mode("CORE")

            elif key in ['g', 'G']:
                self.set_view_mode("GW")

            elif key in ['v', 'V']:
                self.set_view_mode("BH LENS")

            elif key in ['f', 'F']:
                self.set_view_mode("MAGNETIC FIELD")

            elif key in ['u', 'U']:
                self.set_view_mode("MULTI")

            elif key in ['l', 'L']:
                if self.window.is_pressed(ti.ui.SHIFT):
                    enh = self.lensing.toggle_enhanced()
                    print(f"[Relativistic Lensing] Enhanced mode toggled ({enh})")
                else:
                    self.set_view_mode("BH LENS")

            elif key in ['k', 'K']:
                active = self.renderer.toggle_starfield_debug()
                status = "ON (64 Bright Grid Markers)" if active else "OFF (Normal 3D Catalog)"
                print(f"[Starfield Debug Diagnostic Mode] Toggled {status} [K]")

            elif key in ['e', 'E', 'w', 'W']:
                st = self.engine.current_state
                active = self.wave_propagation.toggle(st.event_time)
                status = "ENABLED" if active else "DISABLED"
                print(f"[3D GW Wavefront Propagation] Toggled {status} [E/W]")

            elif key in ['n', 'N']:
                print("[Demo Director] Advancing to next continuous demonstration stage [N]")
                self.director.next_stage()
                st = self.engine.current_state
                if self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.start_playback(st.event_time, self.director.speed_multiplier)

            elif key in ['b', 'B']:
                print("[Demo Director] Stepping backward to previous continuous demonstration stage [B]")
                self.director.previous_stage()
                st = self.engine.current_state
                if self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.start_playback(st.event_time, self.director.speed_multiplier)

            elif key in ['r', 'R']:
                print("[Demo Director] Resetting demonstration timeline to t = -5.0s [R]")
                self.director.reset()
                self.chirp_audio.stop_playback()

            elif key == ti.ui.ESCAPE:
                self.window.running = False

    def render_gui_overlays(self):
        """Render futuristic scientific HUD presentation overlays with scientific provenance labels."""
        gui = self.window.get_gui()
        st = self.engine.current_state
        evt_st = self.coordinator.current_state
        rem_st = self.engine.remnant.evaluate(self.engine.dynamics.inspiral_state, float(st.event_time))

        # Check if BH formed while waiting in BH LENS mode
        if self.view_mode == "BH LENS" and self.bh_lens_waiting:
            if rem_st.is_black_hole:
                self.target_cam_distance = 42.0e3
                self.target_cam_yaw = -np.pi / 2.0
                self.target_cam_pitch = 0.22
                self.is_interpolating_cam = True
                self.lensing.enabled = True
                self.bh_lens_waiting = False
                print("[View Mode] Black Hole formed! BH LENS camera preset applied (42 km)")

        is_cont = (self.director.presentation_time >= 15.0 or self.director.current_stage == DemoStage.CONTINUOUS_POST_MERGER.value)
        phase_str = "POST-MERGER" if is_cont else self.director.current_stage_name
        if self.director.is_running and not self.director.is_paused:
            if self.director.is_slow_motion:
                status_str = f"RUNNING [{self.director.speed_multiplier:.2f}x SLOW]"
            else:
                status_str = "RUNNING [REAL TIME]"
        elif self.director.is_paused:
            status_str = "PAUSED"
        else:
            status_str = "IDLE"

        # 1. TOP CENTER: Header & Mode Selector
        gui.begin("GW170817 Multi-Messenger Event", 0.12, 0.01, 0.76, 0.09)
        gui.text("GW170817 • BINARY NEUTRON-STAR MERGER   |   SIMULATION • LIVE   |   REDUCED-ORDER RELATIVISTIC MODEL")

        if gui.button(" [ CORE ] " if self.view_mode == "CORE" else " CORE "):
            self.set_view_mode("CORE")
        if gui.button(" [ GW MODE ] " if self.view_mode == "GW" else " GW MODE "):
            self.set_view_mode("GW")
        if gui.button(" [ BH LENS ] " if self.view_mode == "BH LENS" else " BH LENS "):
            self.set_view_mode("BH LENS")
        if gui.button(" [ MAGNETIC FIELD ] " if self.view_mode == "MAGNETIC FIELD" else " MAGNETIC FIELD "):
            self.set_view_mode("MAGNETIC FIELD")
        if gui.button(" [ MULTI ] " if self.view_mode == "MULTI" else " MULTI "):
            self.set_view_mode("MULTI")

        # Short mode-specific scientific explanation
        mode_header_info = {
            "CORE": ("CORE", "Binary dynamics and merger evolution"),
            "GW": ("GW PROPAGATION", "Quadrupole radiation from the accelerating binary"),
            "MAGNETIC FIELD": ("MAGNETIC FIELD", "Differential rotation winds the magnetic field"),
            "BH LENS": ("BH LENS / ACCRETION", "Disk material loses angular momentum and spirals inward"),
            "MULTI": ("MULTI-MESSENGER", "One neutron-star merger, observed across multiple channels")
        }
        title, desc = mode_header_info.get(self.view_mode, ("GW170817", "Relativistic Visualization"))
        gui.text(f" MODE: {title}  |  \"{desc}\"")
        gui.end()

        # 2. TOP LEFT: Event Telemetry Panel
        gui.begin("Event Telemetry", 0.01, 0.01, 0.18, 0.32)
        sep_km = st.separation / 1e3
        prog_bar = int(self.director.progress * 15.0)
        bar_str = "[" + "=" * prog_bar + ">" + " " * (15 - prog_bar) + "]"

        mode_specific_txt = ""
        post_contact = st.merger_contact_fraction > 0.05
        if self.view_mode == "CORE":
            if not post_contact:
                mode_specific_txt = "OBJECTS: [NEUTRON STAR A]\n        [NEUTRON STAR B]\n"
            else:
                mode_specific_txt = "OBJECTS: [REMNANT]\n        [EJECTA]\n"
        elif self.view_mode == "GW":
            mode_specific_txt = (
                f"f_GW:    {st.gw_frequency:6.1f} Hz\n"
                f"STRAIN:  h+(t) Quadrupole\n"
                f"FRONT:   [GRAVITATIONAL-WAVE FRONT]\n"
            )
        elif self.view_mode == "MAGNETIC FIELD":
            b_pol = evt_st.b_poloidal
            b_tor = evt_st.b_toroidal
            b_ratio = evt_st.b_ratio
            mode_specific_txt = (
                f"Bp (POLOIDAL): {b_pol:.1e} T\n"
                f"Bφ (TOROIDAL): {b_tor:.1e} T\n"
                f"Bφ/Bp WINDING: {b_ratio:5.2f}\n"
                f"[REDUCED-ORDER DYNAMO MODEL]\n"
            )
        elif self.view_mode == "BH LENS":
            m_bh = rem_st.mass / M_sun
            rs_km = rem_st.radius / 1.0e3
            disk_m = evt_st.disk_mass_msun
            if rem_st.is_black_hole:
                mode_specific_txt = (
                    f"M_BH:      {m_bh:5.2f} M☉\n"
                    f"r_s:       {rs_km:5.1f} km\n"
                    f"DISK MASS: {disk_m:5.2f} M☉\n"
                    f"LENSING:   GRAVITATIONAL LENSING\n"
                    f"FLOW:      ACCRETION FLOW\n"
                )
            else:
                mode_specific_txt = (
                    f"M_REM:     {m_bh:5.2f} M☉\n"
                    f"PRE-MERGER / HMNS STAGE\n"
                    f"[WAITING FOR BH COLLAPSE]\n"
                )
        elif self.view_mode == "MULTI":
            mode_specific_txt = (
                f"CHANNELS: GW + EM + JET\n"
                f"ONE EVENT -> MULTI MESSENGER\n"
                f"PROVENANCE: MODEL vs OBSERVED\n"
            )

        text_phase = (
            f"EVENT TELEMETRY\n"
            f"--------------------\n"
            f"PHASE: {phase_str}\n"
            f"EVENT TIME: t = {st.event_time:+7.2f} s\n"
            f"SEPARATION: a = {sep_km:6.1f} km\n"
            f"GW FREQ:    f = {st.gw_frequency:6.1f} Hz\n"
            f"NS MASSES:  1.36 + 1.36 M☉\n"
            f"STATUS:     {status_str}\n"
            f"--------------------\n"
            f"{mode_specific_txt}"
            f"--------------------\n"
            f"TIMELINE: {bar_str}\n"
        )
        gui.text(text_phase)
        gui.end()

        # 3. TOP RIGHT: Remnant & Multi-Messenger Panel
        gui.begin("Remnant & Multi-Messenger", 0.81, 0.01, 0.18, 0.48)
        
        j_state_str = "LAUNCHED" if evt_st.grb_triggered else ("COLLIMATING" if evt_st.grb_launched else "UPCOMING")
        lens_active_str = "ACTIVE" if self.lensing.enabled else "OFF"
        
        if not post_contact:
            state_info = (
                f"SYSTEM: BINARY NEUTRON STAR\n"
                f"NS1 (Cyan):  1.36 M☉ [Approaching]\n"
                f"NS2 (Gold):  1.36 M☉ [Receding]\n"
                f"TIDAL BULGE: Active [(R/a)³]\n"
                f"DOPPLER:     Active [v/c ~ 0.4]\n"
            )
        else:
            obj_type = "BLACK HOLE" if rem_st.is_black_hole else "HMNS REMNANT"
            state_info = (
                f"SYSTEM: {obj_type}\n"
                f"M_rem:      {rem_st.mass / M_sun:.2f} M☉\n"
                f"r_horizon:  {rem_st.radius / 1e3:.1f} km\n"
                f"ACCRETION:  Active ({evt_st.disk_mass_msun:.2f} M☉)\n"
            )

        text_diag = (
            f"REMNANT / EVENT STATE\n"
            f"--------------------\n"
            f"{state_info}"
            f"--------------------\n"
            f"MULTI-MESSENGER STRIP:\n"
            f" GW:       ACTIVE [MODEL]\n"
            f" EJECTA:   {'ACTIVE' if evt_st.ejecta_fraction > 0 else 'UPCOMING'}\n"
            f" KILONOVA: {'ACTIVE' if evt_st.ejecta_fraction > 0 else 'UPCOMING'}\n"
            f" GRB JET:  {j_state_str}\n"
            f" LENSING:  {lens_active_str} [MODEL]\n"
            f"--------------------\n"
            f"FPS: {self.fps:4.1f} ({self.frame_time_ms:4.1f} ms)\n"
        )
        gui.text(text_diag)

        if gui.button("[ AUDIO ON/OFF ]"):
            muted = self.chirp_audio.toggle_mute()
            if not muted and self.director.is_running and not self.director.is_paused:
                self.chirp_audio.start_playback(st.event_time, self.director.speed_multiplier)

        gui.text("CAMERA CONTROLS:")
        if gui.button(" [ UP ] "): self.camera_orbit_up()
        if gui.button(" [ DOWN ] "): self.camera_orbit_down()
        if gui.button(" [ LEFT ] "): self.camera_orbit_left()
        if gui.button(" [ RIGHT ] "): self.camera_orbit_right()
        if gui.button(" [ + ] "): self.camera_zoom_in()
        if gui.button(" [ - ] "): self.camera_zoom_out()

        gui.end()

        # 4. BOTTOM PANEL: Model Waveform Panel
        gui.begin("GW Model Waveform Panel", 0.20, 0.87, 0.60, 0.11)
        wf_label = "MODEL GRAVITATIONAL-WAVE STRAIN  h(t)   |   Time-Domain Quadrupole Strain Model h+(t)"
        if self.view_mode == "GW":
            wf_label = "GW MODE: MODEL GRAVITATIONAL-WAVE STRAIN h(t)  |  Quadrupole Wavefront Propagation"
        gui.text(f" {wf_label}")
        gui.end()

    def get_lensing_sources(self):
        """
        Return effective compact-object lensing sources (pos1, pos2, m1, m2)
        with smooth transition from binary NSs to adopted single compact remnant.
        """
        st = self.engine.current_state
        rem_st = self.engine.remnant.evaluate(
            self.engine.dynamics.inspiral_state,
            st.event_time
        )

        r1, r2 = self.engine.inspiral.orbital_positions(
            self.engine.dynamics.inspiral_state,
            self.config.m1, self.config.m2
        )

        contact_frac = float(st.merger_contact_fraction)

        if contact_frac <= 0.01 and not rem_st.is_black_hole:
            # Pre-merger inspiral: binary neutron star lensing sources
            return r1, r2, self.config.m1, self.config.m2
        elif rem_st.is_black_hole or contact_frac >= 0.95:
            # Single compact remnant (HMNS -> Adopted BH collapse): single lens at origin
            rem_pos = (0.0, 0.0, 0.0)
            return rem_pos, rem_pos, rem_st.mass, 0.0
        else:
            # Merger contact transition (smooth interpolation from BNS to single remnant)
            w = max(0.0, min(1.0, contact_frac))
            p1 = (float(r1[0]) * (1.0 - w), float(r1[1]) * (1.0 - w), float(r1[2]) * (1.0 - w))
            p2 = (float(r2[0]) * (1.0 - w), float(r2[1]) * (1.0 - w), float(r2[2]) * (1.0 - w))
            m1_eff = self.config.m1 * (1.0 - w) + rem_st.mass * w
            m2_eff = self.config.m2 * (1.0 - w)
            return p1, p2, m1_eff, m2_eff

    def render_frame(self):
        """Render 2D background starfield image, 3D viewport, particles, accretion disk, field lines, jet, chirp track, waveform trace, and overlays."""
        self.camera.track_user_inputs(self.window, movement_speed=10.0e3, hold_key=ti.ui.RMB)

        self._update_camera_tracking()
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.85, 0.85, 0.85))

        st = self.engine.current_state
        evt_st = self.coordinator.current_state
        lens_st = self.lensing.evaluate()

        # 1. Update 3D GPU Relativistic Gravitational Lensing Deflection on Astronomical Starfield
        p1, p2, m1_l, m2_l = self.get_lensing_sources()
        lens_flag = 1 if self.lensing.enabled else 0
        if self.view_mode == "BH LENS":
            enh_scale = 1.8 if not self.lensing.enhanced_mode else 2.5
        elif self.view_mode == "MULTI":
            enh_scale = 1.2
        elif self.view_mode in ["GW", "MAGNETIC FIELD"]:
            enh_scale = 1.0
        else:
            enh_scale = 2.5 if self.lensing.enhanced_mode else 1.5

        cam_x = float(self.cam_distance * np.cos(self.cam_pitch) * np.cos(self.cam_yaw))
        cam_y = float(self.cam_distance * np.cos(self.cam_pitch) * np.sin(self.cam_yaw))
        cam_z = float(self.cam_distance * np.sin(self.cam_pitch))
        rem_st = self.engine.remnant.evaluate(self.engine.dynamics.inspiral_state, float(st.event_time))
        rem_m = float(rem_st.mass)
        is_bh = 1 if rem_st.is_black_hole else 0

        if self._frame_count % 2 == 0 or not self.director.is_running:
            self.renderer.update_star_lensing_deflection_kernel(
                float(p1[0]), float(p1[1]), float(p1[2]),
                float(p2[0]), float(p2[1]), float(p2[2]),
                float(m1_l), float(m2_l),
                lens_flag,
                float(enh_scale),
                cam_x, cam_y, cam_z,
                is_bh,
                rem_m
            )


        # Render 3D Astronomical Starfield Particles into 3D Viewport (Background Layer)
        star_radius = 4.2e3 if self.view_mode == "BH LENS" else 3.8e3
        self.renderer.update_star_brightness_mode(self.view_mode)
        self.scene.particles(
            self.renderer.star_deflected_pos,
            radius=star_radius,
            per_vertex_color=self.renderer.star_colors
        )

        # 2. Render 3D Inspiral NS Particles (during inspiral/merger before post-merger ejection)
        post_merger_active = bool(st.merger_contact_fraction > 0.05)
        if not post_merger_active:
            if self._frame_count % 2 == 0 or not self.director.is_running:
                self.renderer.update_particle_colors(
                    self.engine.psys.pos,
                    self.engine.psys.star_id,
                    self.engine.psys.active,
                    self.engine.psys.ye,
                    self.engine.psys.max_particles,
                    float(st.merger_contact_fraction),
                    float(self.director.ejecta_progress),
                    float(st.separation),
                    float(st.event_time),
                    self.engine.psys.vel,
                    cam_x, cam_y, cam_z
                )
            self.scene.particles(
                self.renderer.render_pos,
                radius=1.8e3,
                per_vertex_color=self.renderer.colors
            )
        else:
            # 3. Post-Merger Visual Geometry Updates (cadenced at 30 Hz)
            eff_event_time = float(st.event_time)
            ejecta_fluid_active = bool(post_merger_active and self.director.ejecta_progress > 0.0 and self.view_mode != "BH LENS")
            remnant_active = bool(st.merger_contact_fraction > 0.05)
            # In CORE mode, disk appears ONLY after BH formation (0 particles pre-BH)
            if self.view_mode == "CORE":
                disk_active = bool(post_merger_active and rem_st.is_black_hole and self.director.disk_progress > 0.0)
            else:
                disk_active = bool(post_merger_active and self.director.disk_progress > 0.0)

            last_pm = getattr(self, "_last_pm_active", False)
            just_transitioned = bool(post_merger_active and not last_pm)
            self._last_pm_active = post_merger_active

            if self._frame_count % 2 == 0 or not self.director.is_running or just_transitioned:
                if ejecta_fluid_active:
                    self.renderer.update_ejecta_fluid(
                        event_time=eff_event_time,
                        ejecta_progress=self.director.ejecta_progress,
                        is_active=ejecta_fluid_active,
                        ejecta_mass_fraction=evt_st.ejecta_fraction,
                    )
                    self.renderer.update_ejecta_thermal_emission(
                        event_time=eff_event_time,
                        ejecta_progress=self.director.ejecta_progress,
                        radioactive_heating_rate=evt_st.ejecta_radioactive_heating_rate,
                        opacity_mean=evt_st.ejecta_opacity_mean,
                        mean_ye=evt_st.ejecta_mean_ye,
                        kilonova_luminosity=evt_st.kilonova_luminosity,
                        is_active=ejecta_fluid_active,
                    )
                if remnant_active or just_transitioned:
                    self.renderer.update_remnant_particles(
                        event_time=eff_event_time,
                        contact_frac=float(st.merger_contact_fraction),
                        remnant_type_str=evt_st.remnant_type,
                        horizon_radius_m=rem_st.radius
                    )
                if disk_active:
                    _dt_vis = 0.016 * self.director.speed_multiplier if (self.director and self.director.is_running and not self.director.is_paused) else 0.0
                    _disk_scale = 1.00 if self.view_mode == "BH LENS" else 0.72
                    _lens_disk = bool(self.lensing.enabled and self.view_mode == "BH LENS")
                    self.renderer.update_disk_particles(
                        event_time=eff_event_time,
                        dt_vis=_dt_vis,
                        disk_progress=self.director.disk_progress,
                        is_active=disk_active,
                        disk_mass_msun=evt_st.disk_mass_msun,
                        is_black_hole=rem_st.is_black_hole,
                        cam_x=cam_x, cam_y=cam_y, cam_z=cam_z,
                        lensing_enabled=_lens_disk,
                        intensity_scale=_disk_scale
                    )


            # Render post-merger particles in strict physical visual hierarchy:
            # Hierarchy: (1) Background ejecta envelope -> (2) Accretion disk plane -> (3) Central BH Remnant Shadow
            if ejecta_fluid_active or remnant_active or disk_active:
                self.renderer.update_combined_post_merger()

                # 1. Background Ejecta Fluid Envelope
                if ejecta_fluid_active:
                    _ej_radius = 0.75e3 if (rem_st.is_black_hole and self.view_mode == "CORE") else 0.85e3
                    self.scene.particles(
                        self.renderer.ejecta_fluid_pos,
                        radius=_ej_radius,
                        per_vertex_color=self.renderer.ejecta_fluid_colors
                    )
                # 2. Accretion Disk (clearly readable thin equatorial disk)
                if disk_active:
                    _disk_radius = 2.2e3 if self.view_mode == "BH LENS" else 1.45e3
                    self.scene.particles(
                        self.renderer.disk_pos,
                        radius=_disk_radius,
                        per_vertex_color=self.renderer.disk_colors
                    )
                # 3. Central Remnant / Black Hole Shadow (drawn dominantly on top in the center)
                if remnant_active or just_transitioned:
                    _rem_radius = 1.6e3
                    self.scene.particles(
                        self.renderer.remnant_pos,
                        radius=_rem_radius,
                        per_vertex_color=self.renderer.remnant_colors
                    )

        # 4. Helical Magnetic Field Lines & Relativistic Jet Lines (cadenced at 30 Hz)
        b_active = bool(post_merger_active and self.director.b_winding_progress > 0.0)
        jet_active = bool(st.grb_triggered and self.director.jet_progress > 0.0)

        show_b_lines = (self.view_mode == "MAGNETIC FIELD") or (self.show_magnetic_field and self.view_mode in ["CORE", "MULTI"])
        show_jet_lines = (self.view_mode not in ["BH LENS", "GW"])

        if self._frame_count % 2 == 0 or not self.director.is_running:
            # Update live GW strain waveform and chirp track GPU buffers
            t_arr, h_p_arr, h_c_arr, f_gws = self.engine.waveform_buffer.get_chronological()
            t_obs, obs_h1, obs_l1 = self.obs_data.get_window(t_center=st.event_time, window_sec=1.5, n_samples=128)
            if len(h_p_arr) > 0:
                self.renderer.update_waveform_buffer(
                    waveform_data=h_p_arr,
                    f_gw=st.gw_frequency,
                    event_time=st.event_time,
                    obs_h1_data=obs_h1,
                    obs_l1_data=obs_l1
                )

            if b_active and show_b_lines:
                self.field_lines.update(
                    b_pol=evt_st.b_poloidal,
                    b_tor=evt_st.b_toroidal,
                    r_rem=14.0e3,
                    is_active=b_active,
                    winding_progress=self.director.b_winding_progress,
                    delta_omega=evt_st.differential_rotation,
                    event_time=st.event_time,
                    omega_rot=evt_st.omega_core
                )
            if jet_active and show_jet_lines:
                _jet_delay = float(self.engine.jet.jet_delay)
                _jet_beta  = float(self.engine.jet.beta(self.engine.jet.lorentz_profile(0.0)))
                if self.view_mode == "MAGNETIC FIELD":
                    _jet_intensity = 0.30  # Subdued jet to highlight magnetic field lines centerpiece
                elif self.view_mode == "MULTI":
                    _jet_intensity = 0.50  # Medium jet in MULTI mode
                else:
                    _jet_intensity = 0.85  # Standard intensity in CORE mode
                self.renderer.update_jet_lines(
                    event_time=st.event_time,
                    jet_progress=self.director.jet_progress,
                    is_active=jet_active,
                    jet_delay=_jet_delay,
                    beta_jet=_jet_beta,
                    b_ratio=evt_st.b_ratio,
                    omega_rot=evt_st.omega_core,
                    intensity=_jet_intensity
                )

        # Single consolidated draw call for lines (magnetic + jet)
        n_b = self.field_lines.n_vertices if (b_active and show_b_lines) else 0
        if n_b > 0 or (jet_active and show_jet_lines):
            self.renderer.update_combined_lines(
                self.field_lines.line_vertices,
                self.field_lines.line_colors,
                n_b,
                show_jet=show_jet_lines
            )
            line_w = 1.0
            self.scene.lines(
                self.renderer.combined_line_vertices,
                width=line_w,
                per_vertex_color=self.renderer.combined_line_colors
            )

        # 5. 3D GW Wavefront Propagation Lines
        wf_st = self.wave_propagation.update(
            event_time=st.event_time,
            f_gw=st.gw_frequency,
            merger_active=(st.event_time >= 0.0),
            h_plus=evt_st.h_plus,
            h_cross=evt_st.h_cross
        )
        self.chirp_audio.evaluate(st.event_time)


        if wf_st.active and wf_st.n_vertices > 0 and self.view_mode != "BH LENS":
            if self._frame_count % 2 == 0 or not self.director.is_running:
                self.renderer.update_gw_wavefront_lines(
                    self.wave_propagation.gpu_line_vertices,
                    self.wave_propagation.gpu_line_colors,
                    wf_st.n_vertices
                )
            if self.view_mode == "GW":
                lw = 2.8
            elif self.view_mode == "MULTI":
                lw = 2.0
            elif self.view_mode == "MAGNETIC FIELD":
                lw = 1.0
            else:
                lw = 2.5
            self.scene.lines(
                self.renderer.gw_wave_vertices,
                width=lw,
                per_vertex_color=self.renderer.gw_wave_colors
            )

        self.canvas.scene(self.scene)

        # Render single clean 2D model strain trace on canvas
        if self.renderer.waveform_vertices is not None:
            self.canvas.lines(
                self.renderer.waveform_vertices,
                width=0.003,
                color=(1.0, 0.75, 0.2)  # Electric gold MODEL strain trace
            )

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
        elif self.director.stage_enum == DemoStage.IDLE:
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
