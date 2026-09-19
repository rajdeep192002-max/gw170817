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
import math
from typing import Optional
import numpy as np
import taichi as ti

from gw170817.constants import G, c, day, Mpc, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation, SimulationState
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector, DemoStage, PlaybackMode
from gw170817.visualization.renderer import ParticleRenderer
from gw170817.visualization.lensing import RelativisticLensingModel
from gw170817.visualization.background import BackgroundStarfield
from gw170817.visualization.raytracer import SchwarzschildRaytracer
from gw170817.visualization.field_lines import MagneticFieldLines
from gw170817.visualization.wave_propagation import GWWavefrontPropagation, WavefrontState
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

        # Camera interpolation targets for smooth mode transitions
        self.target_cam_distance = 280.0e3
        self.target_cam_yaw = -np.pi / 2.0
        self.target_cam_pitch = 0.65
        self.is_interpolating_cam = False

        # Smooth inertial camera controls (floating through space feel)
        self.cam_omega_yaw: float = 0.0
        self.cam_omega_pitch: float = 0.0
        self.cam_accel: float = 4.2
        self.cam_max_omega: float = 0.95
        self.cam_damping: float = 4.8
        self.cam_pitch_min: float = -np.pi / 2.0 + 0.05
        self.cam_pitch_max: float = np.pi / 2.0 - 0.05
        self._last_mouse_pos: tuple[float, float] | None = None
        self._smoothed_dt_cam: float = 0.016
        self.show_controls_overlay: bool = False

        self._setup_default_camera()

    def __getattr__(self, name: str):
        if "lens" in name:
            return False
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def set_view_mode(self, mode: str):
        """
        Switch presentation view mode (CORE, GW, MAGNETIC FIELD, NEUTRINO, MULTI) and apply safe camera presets smoothly.
        Does NOT alter or reset underlying simulation physics or event time.
        """
        mode = mode.upper()
        if mode not in ["CORE", "GW", "MAGNETIC FIELD", "NEUTRINO", "MULTI"]:
            return

        self.view_mode = mode
        self.show_magnetic_field = (mode == "MAGNETIC FIELD")
        self.cam_omega_yaw = 0.0
        self.cam_omega_pitch = 0.0
        st = self.engine.current_state

        if mode == "CORE":
            rem_st = self.engine.remnant.evaluate(self.engine.dynamics.inspiral_state, float(st.event_time))
            self.target_cam_distance = 150.0e3 if rem_st.is_black_hole else 280.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.65
            self.is_interpolating_cam = True
            print("[View Mode] Set to CORE -- Flagship Black Hole & Binary Dynamics Evolution")

        elif mode in ["GW", "GW MODE"]:
            self.view_mode = "GW"
            self.target_cam_distance = 260.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.58
            self.is_interpolating_cam = True
            self.wave_propagation.active = True
            self.wave_propagation.manual_trigger = False
            print("[View Mode] Set to GW MODE -- Quadrupole Radiation & 3D Wavefront Propagation")

        elif mode == "MAGNETIC FIELD":
            self.cam_distance = self.target_cam_distance = 160.0e3
            self.cam_pitch = self.target_cam_pitch = 0.45
            self.target_cam_yaw = -np.pi / 2.0
            self.is_interpolating_cam = True
            print("[View Mode] Set to MAGNETIC FIELD -- Differential Rotation & Toroidal Winding")

        elif mode == "NEUTRINO":
            self.target_cam_distance = 140.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.40
            self.is_interpolating_cam = True
            print("[View Mode] Set to NEUTRINO -- Reduced-Order Neutrino Emission / Transport Model")

        elif mode == "MULTI":
            self.target_cam_distance = 280.0e3
            self.target_cam_yaw = -np.pi / 2.0
            self.target_cam_pitch = 0.65
            self.is_interpolating_cam = True
            print("[View Mode] Set to MULTI -- Clean Multi-Messenger Story")

    def _setup_default_camera(self):
        """Set up fixed scientific 3D camera viewing orbital plane."""
        self._update_camera_position()

    def _update_camera_position(self):
        """Reconstruct 3D camera pose deterministically from explicit scalar state."""
        x = float(self.cam_distance * math.cos(self.cam_pitch) * math.cos(self.cam_yaw))
        y = float(self.cam_distance * math.cos(self.cam_pitch) * math.sin(self.cam_yaw))
        z = float(self.cam_distance * math.sin(self.cam_pitch))

        self.camera.position(x, y, z)
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.camera.z_near(1.0e3)
        self.camera.z_far(3.5e6)
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
        self.cam_omega_yaw = 0.0
        self.cam_omega_pitch = 0.0
        self._sync_manual_camera_target()
        self.cam_yaw -= step
        self._update_camera_position()

    def camera_orbit_right(self, step: float = 0.06):
        """Orbit camera right around merger target (yaw right)."""
        self.cam_omega_yaw = 0.0
        self.cam_omega_pitch = 0.0
        self._sync_manual_camera_target()
        self.cam_yaw += step
        self._update_camera_position()

    def camera_orbit_up(self, step: float = 0.05):
        """Orbit camera upward around merger target (pitch up with inversion protection)."""
        self.cam_omega_yaw = 0.0
        self.cam_omega_pitch = 0.0
        self._sync_manual_camera_target()
        self.cam_pitch = min(np.pi / 2.0 - 0.05, self.cam_pitch + step)
        self._update_camera_position()

    def camera_orbit_down(self, step: float = 0.05):
        """Orbit camera downward around merger target (pitch down with inversion protection)."""
        self.cam_omega_yaw = 0.0
        self.cam_omega_pitch = 0.0
        self._sync_manual_camera_target()
        self.cam_pitch = max(-np.pi / 2.0 + 0.05, self.cam_pitch - step)
        self._update_camera_position()

    def camera_zoom_in(self, factor: float = 0.90):
        """Zoom camera inward toward remnant center."""
        self.cam_omega_yaw = 0.0
        self.cam_omega_pitch = 0.0
        self._sync_manual_camera_target()
        self.cam_distance = max(40.0e3, self.cam_distance * factor)
        self.target_cam_distance = self.cam_distance
        self._update_camera_position()

    def camera_zoom_out(self, factor: float = 1.10):
        """Zoom camera outward away from remnant center."""
        self.cam_omega_yaw = 0.0
        self.cam_omega_pitch = 0.0
        self._sync_manual_camera_target()
        self.cam_distance = min(1.5e6, self.cam_distance * factor)
        self.target_cam_distance = self.cam_distance
        self._update_camera_position()

    def _update_camera_tracking(self, dt: float = 0.016):
        """Smoothly interpolate camera parameters toward mode preset targets if active."""
        if getattr(self, "is_interpolating_cam", False):
            d_dist = self.target_cam_distance - self.cam_distance
            d_yaw = (self.target_cam_yaw - self.cam_yaw + math.pi) % (2.0 * math.pi) - math.pi
            d_pitch = self.target_cam_pitch - self.cam_pitch

            if abs(d_dist) < 10.0 and abs(d_yaw) < 1.0e-4 and abs(d_pitch) < 1.0e-4:
                self.cam_distance = float(self.target_cam_distance)
                self.cam_yaw = float(self.target_cam_yaw)
                self.cam_pitch = float(self.target_cam_pitch)
                self.is_interpolating_cam = False
            else:
                smooth_factor = 1.0 - math.exp(-8.0 * max(0.001, min(0.05, float(dt))))
                self.cam_distance += smooth_factor * d_dist
                self.cam_yaw += smooth_factor * d_yaw
                self.cam_pitch += smooth_factor * d_pitch
            self._update_camera_position()

    def _is_key_pressed(self, *keys) -> bool:
        """Helper to check if any of the given keys is pressed on window, safe for mocked windows."""
        if not hasattr(self.window, "is_pressed"):
            return False
        for k in keys:
            try:
                val = self.window.is_pressed(k)
                # In unit tests with MagicMock, window.is_pressed returns MagicMock, which is not True
                if val is True:
                    return True
            except Exception:
                pass
        return False

    def _poll_arrow_inputs(self) -> tuple[float, float]:
        """Poll continuous arrow key inputs for inertial camera rotation.

        Returns:
            (input_yaw, input_pitch) in range [-1.0, 1.0].
            input_yaw: +1.0 for RIGHT, -1.0 for LEFT
            input_pitch: +1.0 for UP, -1.0 for DOWN
        """
        input_yaw = 0.0
        input_pitch = 0.0

        is_left = self._is_key_pressed(ti.ui.LEFT, 'Left', 'Left Arrow')
        is_right = self._is_key_pressed(ti.ui.RIGHT, 'Right', 'Right Arrow')
        is_up = self._is_key_pressed(ti.ui.UP, 'Up', 'Up Arrow')
        is_down = self._is_key_pressed(ti.ui.DOWN, 'Down', 'Down Arrow')

        if is_left and not is_right:
            input_yaw = -1.0
        elif is_right and not is_left:
            input_yaw = 1.0

        if is_up and not is_down:
            input_pitch = 1.0
        elif is_down and not is_up:
            input_pitch = -1.0

        return input_yaw, input_pitch

    def _poll_mouse_inputs(self) -> tuple[float, float]:
        """Poll RMB mouse drag inputs for smooth camera rotation with a sub-pixel dead-band filter.

        Returns:
            (mouse_yaw, mouse_pitch) deltas in normalized input scale.
        """
        if not hasattr(self.window, "get_cursor_pos"):
            return 0.0, 0.0

        is_rmb = self._is_key_pressed(ti.ui.RMB)
        if not is_rmb:
            self._last_mouse_pos = None
            return 0.0, 0.0

        try:
            curr_pos = self.window.get_cursor_pos()
            if not isinstance(curr_pos, (tuple, list)) or len(curr_pos) < 2:
                return 0.0, 0.0
            cx, cy = float(curr_pos[0]), float(curr_pos[1])
        except Exception:
            return 0.0, 0.0

        if self._last_mouse_pos is None:
            self._last_mouse_pos = (cx, cy)
            return 0.0, 0.0

        dx = cx - self._last_mouse_pos[0]
        dy = cy - self._last_mouse_pos[1]
        self._last_mouse_pos = (cx, cy)

        # Sub-pixel dead-band filter: ignore microscopic mouse sensor noise (< 0.0003 normalized units)
        if abs(dx) < 0.0003:
            dx = 0.0
        if abs(dy) < 0.0003:
            dy = 0.0

        if dx == 0.0 and dy == 0.0:
            return 0.0, 0.0

        mouse_yaw = float(np.clip(-dx * 12.0, -1.0, 1.0))
        mouse_pitch = float(np.clip(dy * 12.0, -1.0, 1.0))
        return mouse_yaw, mouse_pitch

    def update_inertial_camera(self, dt: float, input_yaw: float = 0.0, input_pitch: float = 0.0):
        """Update camera orientation using smooth inertial dynamics (acceleration, braking, coasting)."""
        dt_cam = max(0.001, min(0.05, float(dt)))

        # If user provides manual arrow input or camera has active angular velocity, cancel preset interpolation
        if input_yaw != 0.0 or input_pitch != 0.0 or abs(self.cam_omega_yaw) > 1e-4 or abs(self.cam_omega_pitch) > 1e-4:
            if getattr(self, "is_interpolating_cam", False):
                self._sync_manual_camera_target()

        if getattr(self, "is_interpolating_cam", False):
            self._update_camera_tracking(dt_cam)
            return

        # Update Yaw angular velocity
        if input_yaw != 0.0:
            # Responsive braking: boost acceleration if input opposes current velocity
            accel = self.cam_accel * 2.2 if (input_yaw * self.cam_omega_yaw < 0.0) else self.cam_accel
            target_omega = (accel * input_yaw) / self.cam_damping
            decay = math.exp(-self.cam_damping * dt_cam)
            self.cam_omega_yaw = (self.cam_omega_yaw - target_omega) * decay + target_omega
        else:
            decay = math.exp(-self.cam_damping * dt_cam)
            self.cam_omega_yaw *= decay
            if abs(self.cam_omega_yaw) < 2.0e-4:
                self.cam_omega_yaw = 0.0

        # Clamp Yaw velocity
        if self.cam_omega_yaw > self.cam_max_omega:
            self.cam_omega_yaw = self.cam_max_omega
        elif self.cam_omega_yaw < -self.cam_max_omega:
            self.cam_omega_yaw = -self.cam_max_omega

        # Update Pitch angular velocity
        if input_pitch != 0.0:
            accel = self.cam_accel * 2.2 if (input_pitch * self.cam_omega_pitch < 0.0) else self.cam_accel
            target_omega = (accel * input_pitch) / self.cam_damping
            decay = math.exp(-self.cam_damping * dt_cam)
            self.cam_omega_pitch = (self.cam_omega_pitch - target_omega) * decay + target_omega
        else:
            decay = math.exp(-self.cam_damping * dt_cam)
            self.cam_omega_pitch *= decay
            if abs(self.cam_omega_pitch) < 2.0e-4:
                self.cam_omega_pitch = 0.0

        # Clamp Pitch velocity
        if self.cam_omega_pitch > self.cam_max_omega:
            self.cam_omega_pitch = self.cam_max_omega
        elif self.cam_omega_pitch < -self.cam_max_omega:
            self.cam_omega_pitch = -self.cam_max_omega

        # Integrate angles if velocity is non-zero
        if abs(self.cam_omega_yaw) > 1.0e-6 or abs(self.cam_omega_pitch) > 1.0e-6:
            self.cam_yaw += self.cam_omega_yaw * dt_cam
            # Normalize yaw to [-pi, pi] to prevent numerical drift
            self.cam_yaw = (self.cam_yaw + math.pi) % (2.0 * math.pi) - math.pi

            new_pitch = self.cam_pitch + self.cam_omega_pitch * dt_cam
            if new_pitch >= self.cam_pitch_max:
                new_pitch = self.cam_pitch_max
                self.cam_omega_pitch = 0.0
            elif new_pitch <= self.cam_pitch_min:
                new_pitch = self.cam_pitch_min
                self.cam_omega_pitch = 0.0
            self.cam_pitch = new_pitch

            self._update_camera_position()

    def process_input(self, dt: float = 0.016):
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

            elif key in [ti.ui.UP, 'Up', 'Up Arrow', ti.ui.DOWN, 'Down', 'Down Arrow',
                         ti.ui.LEFT, 'Left', 'Left Arrow', ti.ui.RIGHT, 'Right', 'Right Arrow']:
                self._sync_manual_camera_target()

            elif key in ['1']:
                self.director.set_real_time()
                st = self.engine.current_state
                if self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.resume_playback(st.event_time, self.director.speed_multiplier)
                self.show_controls_overlay = False
                print("[Playback Mode] Set to REAL TIME (1.00x)")

            elif key in ['2']:
                self.director.set_slow_motion()
                self.chirp_audio.stop_playback()
                self.show_controls_overlay = True
                print("[Playback Mode] Set to SLOW MOTION (0.10x)")

            elif key in ['t', 'T']:
                self.director.toggle_playback_mode()
                st = self.engine.current_state
                if self.director.speed_multiplier == 1.00 and self.director.is_running and not self.director.is_paused:
                    self.chirp_audio.resume_playback(st.event_time, 1.00)
                else:
                    self.chirp_audio.stop_playback()
                self.show_controls_overlay = self.director.is_slow_motion
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

            elif key in ['f', 'F']:
                self.set_view_mode("MAGNETIC FIELD")

            elif key in ['i', 'I']:
                self.set_view_mode("NEUTRINO")

            elif key in ['u', 'U']:
                self.set_view_mode("MULTI")

            elif key in ['l', 'L']:
                enh = self.lensing.toggle_enhanced()
                print(f"[Relativistic Lensing] Enhanced mode toggled ({enh})")

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

        # Continuous inertial camera update (independent of simulation time/slow motion)
        arr_yaw, arr_pitch = self._poll_arrow_inputs()
        m_yaw, m_pitch = self._poll_mouse_inputs()

        tot_yaw = float(np.clip(arr_yaw + m_yaw, -1.0, 1.0))
        tot_pitch = float(np.clip(arr_pitch + m_pitch, -1.0, 1.0))
        self.update_inertial_camera(dt, tot_yaw, tot_pitch)

    def render_gui_overlays(self):
        """Render futuristic scientific HUD presentation overlays with scientific provenance labels."""
        gui = self.window.get_gui()
        st = self.engine.current_state
        evt_st = self.coordinator.current_state
        rem_st = self.engine.remnant.evaluate(self.engine.dynamics.inspiral_state, float(st.event_time))

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
        gui.text("GW170817 | BINARY NEUTRON-STAR MERGER   |   SIMULATION - LIVE   |   REDUCED-ORDER RELATIVISTIC MODEL")

        if gui.button(" [ CORE ] " if self.view_mode == "CORE" else " CORE "):
            self.set_view_mode("CORE")
        if gui.button(" [ GW MODE ] " if self.view_mode == "GW" else " GW MODE "):
            self.set_view_mode("GW")
        if gui.button(" [ MAGNETIC FIELD ] " if self.view_mode == "MAGNETIC FIELD" else " MAGNETIC FIELD "):
            self.set_view_mode("MAGNETIC FIELD")
        if gui.button(" [ NEUTRINO ] " if self.view_mode == "NEUTRINO" else " NEUTRINO "):
            self.set_view_mode("NEUTRINO")
        if gui.button(" [ MULTI ] " if self.view_mode == "MULTI" else " MULTI "):
            self.set_view_mode("MULTI")

        # Short mode-specific scientific explanation
        mode_header_info = {
            "CORE": ("CORE", "Binary dynamics, merger evolution, black hole lensing & polar jet"),
            "GW": ("GW PROPAGATION", "Quadrupole radiation from the accelerating binary"),
            "MAGNETIC FIELD": ("MAGNETIC FIELD", "Differential rotation winds the magnetic field"),
            "NEUTRINO": ("NEUTRINO EMISSION", "Reduced-order neutrino emission / transport model -- NO CONFIRMED DETECTION"),
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
                mode_specific_txt = (
                    "OBJECTS: [NEUTRON STAR A]\n"
                    "        [NEUTRON STAR B]\n"
                    f"TIDAL:   QUADRUPOLAR (R/a)^3\n"
                    f"Lambda_tilde: {self.engine.tidal.lambda_tilde:.0f} [MODEL]\n"
                )
            else:
                m_bh = rem_st.mass / M_sun
                rs_km = (2.0 * G * rem_st.mass / (c ** 2)) / 1.0e3
                disk_m = evt_st.disk_mass_msun
                disk_m_str = f"{disk_m:.2f} M_sun" if disk_m >= 0.01 else f"{disk_m:.2e} M_sun"
                obj_type = "BLACK HOLE" if rem_st.is_black_hole else "HMNS REMNANT"
                if rem_st.is_black_hole:
                    mode_specific_txt = (
                        f"OBJECT:  [{obj_type}]\n"
                        f"M_BH:    {m_bh:5.2f} M_sun\n"
                        f"r_s:     {rs_km:5.1f} km\n"
                        f"DISK:    {disk_m_str}\n"
                        f"LENSING: REDUCED-ORDER SCHWARZSCHILD\n"
                        f"FLOW:    VOLUMETRIC ACCRETION DISK\n"
                    )
                else:
                    mode_specific_txt = (
                        f"OBJECT:  [{obj_type}]\n"
                        f"M_REM:   {m_bh:5.2f} M_sun\n"
                        f"r_rem:   {rs_km:5.1f} km\n"
                        f"DISK:    {disk_m_str}\n"
                        f"EJECTA:  ACTIVE\n"
                    )
        elif self.view_mode == "GW":
            mode_specific_txt = (
                f"f_GW:    {st.gw_frequency:6.1f} Hz\n"
                f"STRAIN:  h+(t) Quadrupole [MODEL]\n"
                f"FRONT:   QUADRUPOLAR WAVEFRONT SHELLS [c_vis]\n"
            )
        elif self.view_mode == "MAGNETIC FIELD":
            b_pol = evt_st.b_poloidal
            b_tor = evt_st.b_toroidal
            b_ratio = evt_st.b_ratio
            disk_line = f"DISK:    {evt_st.disk_mass_msun:.2f} M_sun [SECONDARY]\n" if evt_st.disk_mass_msun > 0.0 else ""
            mode_specific_txt = (
                f"Bp (POLOIDAL):   {b_pol:.1e} G\n"
                f"B_phi (TOROIDAL): {b_tor:.1e} G\n"
                f"B_phi/B_p WINDING: {b_ratio:5.2f}\n"
                f"{disk_line}"
                f"[REDUCED-ORDER DYNAMO MODEL]\n"
            )
        elif self.view_mode == "MULTI":
            mode_specific_txt = (
                f"CHANNELS: GW + EM + JET\n"
                f"ONE EVENT -> MULTI MESSENGER\n"
                f"PROVENANCE: REDUCED-ORDER RELATIVISTIC MODEL\n"
            )
        elif self.view_mode == "NEUTRINO":
            nu_active_str = "ACTIVE" if evt_st.neutrino_is_active else "INACTIVE"
            nu_trans_str = "ACTIVE [c]" if evt_st.neutrino_transport_active else "OFF"
            nu_lum_erg = evt_st.neutrino_luminosity * 1.0e7  # W -> erg/s
            mode_specific_txt = (
                f"EMISSION:  {nu_active_str} [MODEL]\n"
                f"TRANSPORT: {nu_trans_str}\n"
                f"L_nu:      {nu_lum_erg:.1e} erg/s\n"
                f"<E_nu_e>:  {evt_st.neutrino_mean_energy_mev:.1f} MeV\n"
                f"[REDUCED-ORDER TRANSPORT]\n"
            )

        text_phase = (
            f"EVENT TELEMETRY\n"
            f"--------------------\n"
            f"PHASE: {phase_str}\n"
            f"EVENT TIME: t = {st.event_time:+7.2f} s\n"
            f"SEPARATION: a = {sep_km:6.1f} km\n"
            f"GW FREQ:    f = {st.gw_frequency:6.1f} Hz\n"
            f"NS MASSES:  {self.config.m1_solar:.2f} + {self.config.m2_solar:.2f} M_sun\n"
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
            lam1_val = self.engine.tidal.star1.lambda_dimensionless
            lam2_val = self.engine.tidal.star2.lambda_dimensionless
            lam_tilde = self.engine.tidal.lambda_tilde
            t_state = getattr(self.engine.dynamics, "tidal_state", None)
            resp_str = f"{t_state.relative_tidal_strength * 100.0:.0f}%" if t_state else "100%"
            state_info = (
                f"SYSTEM: BINARY NEUTRON STAR\n"
                f"NS1 (Cyan):  {self.config.m1_solar:.2f} M_sun [Approaching]\n"
                f"NS2 (Gold):  {self.config.m2_solar:.2f} M_sun [Receding]\n"
                f"--------------------\n"
                f"TIDAL DEFORMABILITY\n"
                f"Lambda_1 = {lam1_val:.0f} [ADOPTED MODEL]\n"
                f"Lambda_2 = {lam2_val:.0f} [ADOPTED MODEL]\n"
                f"Lambda_tilde = {lam_tilde:.0f} (GW170817 <= 800)\n"
                f"TIDAL RESPONSE = {resp_str}\n"
                f"[REDUCED-ORDER TIDAL DEFORMABILITY MODEL]\n"
                f"--------------------\n"
                f"DOPPLER:     Active [v/c ~ 0.4]\n"
            )
        else:
            obj_type = "BLACK HOLE" if rem_st.is_black_hole else "HMNS REMNANT"
            d_m = evt_st.disk_mass_msun
            d_m_str = f"{d_m:.2f} M_sun" if d_m >= 0.01 else f"{d_m:.2e} M_sun"
            state_info = (
                f"SYSTEM: {obj_type}\n"
                f"M_rem:      {rem_st.mass / M_sun:.2f} M_sun\n"
                f"r_horizon:  {rem_st.radius / 1e3:.1f} km\n"
                f"ACCRETION:  Active ({d_m_str})\n"
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

        # 4a. NEUTRINO MESSENGER PANEL (visible only in NEUTRINO mode)
        if self.view_mode == "NEUTRINO":
            gui.begin("Neutrino Transport Model", 0.01, 0.35, 0.18, 0.54)
            nu_active_str = "ACTIVE" if evt_st.neutrino_is_active else "INACTIVE"
            nu_trans_str = "PROPAGATING [c]" if evt_st.neutrino_transport_active else "OFF"
            remnant_str = "BLACK HOLE" if rem_st.is_black_hole else ("HMNS REMNANT" if post_contact else "PRE-MERGER")
            nu_lum_erg = evt_st.neutrino_luminosity * 1.0e7  # W -> erg/s
            nu_lum_nue_erg = evt_st.neutrino_luminosity_nue * 1.0e7
            nu_lum_nuebar_erg = evt_st.neutrino_luminosity_nue_bar * 1.0e7
            nu_lum_nux_erg = evt_st.neutrino_luminosity_nux * 1.0e7
            wind_mdot_msun_s = evt_st.neutrino_wind_mass_loss_rate / M_sun

            text_nu = (
                f"NEUTRINO MESSENGER\n"
                f"--------------------\n"
                f"EMISSION:   {nu_active_str}\n"
                f"TRANSPORT:  {nu_trans_str}\n"
                f"REMNANT:    {remnant_str}\n"
                f"L_nu:       {nu_lum_erg:.1e} erg/s\n"
                f"<E_nu_e>:   {evt_st.neutrino_mean_energy_mev:.1f} MeV\n"
                f"--------------------\n"
                f"SPECIES [MODEL]:\n"
                f" nu_e:     {nu_lum_nue_erg:.1e} erg/s (35%)\n"
                f"           [Cyan]\n"
                f" anti-nu_e:{nu_lum_nuebar_erg:.1e} erg/s (40%)\n"
                f"           [Teal-White]\n"
                f" nu_x:     {nu_lum_nux_erg:.1e} erg/s (25%)\n"
                f"           [Violet]\n"
                f"--------------------\n"
                f"WIND Mdot:  {wind_mdot_msun_s:.2e} M_sun/s\n"
                f"--------------------\n"
                f"REDUCED-ORDER NEUTRINO\n"
                f"EMISSION / TRANSPORT MODEL\n"
                f"--------------------\n"
                f"GW170817\n"
                f"NO CONFIRMED NEUTRINO\n"
                f"DETECTION\n"
            )
            gui.text(text_nu)
            gui.end()

        # 4b. BOTTOM PANEL: Model Waveform Panel
        if self.view_mode != "NEUTRINO":
            gui.begin("GW Model Waveform Panel", 0.20, 0.87, 0.60, 0.11)
            wf_label = "MODEL GRAVITATIONAL-WAVE STRAIN  h(t)   |   Time-Domain Quadrupole Strain Model h+(t)"
            if self.view_mode == "GW":
                wf_label = "GW MODE: MODEL GRAVITATIONAL-WAVE STRAIN h(t)  |  Quadrupole Wavefront Propagation (c_vis)"
            gui.text(f" {wf_label}")
            gui.end()
        else:
            gui.begin("Neutrino Transport Context", 0.25, 0.94, 0.50, 0.05)
            gui.text(" NEUTRINO TRANSPORT MODE  |  Outward Radially Expanding Tracers  [c]")
            gui.end()

        # 4c. SLOW-MOTION INDICATOR & CONTROLS OVERLAY (toggled by T)
        if self.director.is_slow_motion and not self.show_controls_overlay:
            gui.begin("Playback Mode", 0.81, 0.50, 0.18, 0.08)
            gui.text(f"[*] SLOW MOTION\nSimulation {self.director.speed_multiplier:.2f}x")
            gui.end()

        if self.show_controls_overlay:
            gui.begin("Simulation Controls", 0.81, 0.50, 0.18, 0.46)
            if self.director.is_slow_motion:
                gui.text(f"[*] SLOW MOTION\nSimulation {self.director.speed_multiplier:.2f}x\n--------------------")
            gui.text(
                "CONTROLS:\n"
                " UP/DN/L/R  Camera Orbit\n"
                " Mouse Drag Smooth Orbit\n"
                " Wheel/+/-  Zoom In/Out\n"
                " SPACE      Play / Pause\n"
                " T          Slow Motion\n"
                " C          CORE Mode\n"
                " G          GW Mode\n"
                " F          B-Field Mode\n"
                " I          Neutrino Mode\n"
                " U          Multi Mode\n"
                " N / B      Next/Prev Stage\n"
                " R          Reset Timeline\n"
                " M          Audio Mute\n"
                " ESC        Exit\n"
                "--------------------\n"
                " Press [T] to Toggle"
            )
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

        if self._is_key_pressed(ti.ui.RMB):
            cpos = getattr(self.camera, "curr_position", None)
            if cpos is not None:
                try:
                    cx, cy, cz = float(cpos[0]), float(cpos[1]), float(cpos[2])
                    dist = math.sqrt(cx * cx + cy * cy + cz * cz)
                    if dist > 1.0:
                        self.cam_distance = float(np.clip(dist, 40.0e3, 1.5e6))
                        self.cam_pitch = float(np.clip(math.asin(cz / dist), self.cam_pitch_min, self.cam_pitch_max))
                        self.cam_yaw = float(math.atan2(cy, cx))
                        self.cam_omega_yaw = 0.0
                        self.cam_omega_pitch = 0.0
                        self._sync_manual_camera_target()
                except Exception:
                    pass
        else:
            # Sync distance if mouse scroll wheel changed Taichi camera position without RMB drag
            cpos = getattr(self.camera, "curr_position", None)
            if cpos is not None and not getattr(self, "is_interpolating_cam", False):
                try:
                    cx, cy, cz = float(cpos[0]), float(cpos[1]), float(cpos[2])
                    dist = math.sqrt(cx * cx + cy * cy + cz * cz)
                    if dist > 1.0 and abs(dist - self.cam_distance) > 5.0:
                        self.cam_distance = float(np.clip(dist, 40.0e3, 1.5e6))
                        self.target_cam_distance = self.cam_distance
                        self.cam_omega_yaw = 0.0
                        self.cam_omega_pitch = 0.0
                except Exception:
                    pass

        self._update_camera_tracking()
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.85, 0.85, 0.85))

        st = self.engine.current_state
        evt_st = self.coordinator.current_state
        lens_st = self.lensing.evaluate()

        # 1. Update 3D GPU Relativistic Gravitational Lensing Deflection on Astronomical Starfield
        p1, p2, m1_l, m2_l = self.get_lensing_sources()
        lens_flag = 1 if self.lensing.enabled else 0
        if self.view_mode == "MULTI":
            enh_scale = 0.20
        elif self.view_mode in ["GW", "MAGNETIC FIELD", "NEUTRINO"]:
            enh_scale = 0.15
        else:
            # CORE mode: moderate/strong post-merger Schwarzschild lensing localized around BH
            enh_scale = 0.38 if self.lensing.enhanced_mode else 0.28

        cam_x = float(self.cam_distance * np.cos(self.cam_pitch) * np.cos(self.cam_yaw))
        cam_y = float(self.cam_distance * np.cos(self.cam_pitch) * np.sin(self.cam_yaw))
        cam_z = float(self.cam_distance * np.sin(self.cam_pitch))
        rem_st = self.engine.remnant.evaluate(self.engine.dynamics.inspiral_state, float(st.event_time))
        rem_m = float(rem_st.mass)
        is_bh = 1 if rem_st.is_black_hole else 0

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
        star_radius = 1.6e3 if self.view_mode == "MAGNETIC FIELD" else (1.8e3 if self.view_mode in ["GW", "GW MODE"] else (2.0e3 if self.view_mode == "NEUTRINO" else (2.8e3 if self.view_mode == "MULTI" else (2.2e3 if (rem_st.is_black_hole and self.view_mode == "CORE") else 3.8e3))))
        self.renderer.update_star_brightness_mode(self.view_mode)
        self.scene.particles(
            self.renderer.star_deflected_pos,
            radius=star_radius,
            per_vertex_color=self.renderer.star_colors
        )

        # 2. Render 3D Inspiral NS Particles (during inspiral/merger before post-merger ejection)
        post_merger_active = bool(st.merger_contact_fraction > 0.05)
        if not post_merger_active:
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
            _bns_radius = 1.2e3 if self.view_mode in ["GW", "GW MODE"] else 1.8e3
            self.scene.particles(
                self.renderer.render_pos,
                radius=_bns_radius,
                per_vertex_color=self.renderer.colors
            )
        else:
            # 3. Post-Merger Visual Geometry Updates (cadenced at 30 Hz)
            eff_event_time = float(st.event_time)
            ejecta_fluid_active = bool(post_merger_active and self.director.ejecta_progress > 0.0)
            remnant_active = bool(st.merger_contact_fraction > 0.05)
            # In CORE mode, disk appears ONLY after BH formation (0 particles pre-BH)
            # In other post-merger modes (including MAGNETIC FIELD as secondary context), disk activates post-merger
            if self.view_mode == "CORE":
                disk_active = bool(post_merger_active and rem_st.is_black_hole and self.director.disk_progress > 0.0)
            else:
                disk_active = bool(post_merger_active and self.director.disk_progress > 0.0)

            last_pm = getattr(self, "_last_pm_active", False)
            just_transitioned = bool(post_merger_active and not last_pm)
            self._last_pm_active = post_merger_active

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
                _disk_scale = 1.00 if self.view_mode == "CORE" else (0.20 if self.view_mode == "GW" else (0.30 if self.view_mode == "MULTI" else 0.35))
                _lens_disk = bool(self.lensing.enabled and self.view_mode == "CORE")
                self.renderer.update_disk_particles(
                    event_time=eff_event_time,
                    dt_vis=_dt_vis,
                    disk_progress=self.director.disk_progress,
                    is_active=disk_active,
                    disk_mass_msun=evt_st.disk_mass_msun,
                    is_black_hole=rem_st.is_black_hole,
                    cam_x=cam_x, cam_y=cam_y, cam_z=cam_z,
                    lensing_enabled=_lens_disk,
                    intensity_scale=_disk_scale,
                    remnant_mass_kg=rem_m
                )

            if self._frame_count % 2 == 0 or not self.director.is_running or just_transitioned:
                if ejecta_fluid_active or remnant_active or disk_active:
                    self.renderer.update_combined_post_merger()

            # Render post-merger particles in strict physical visual hierarchy:
            # Hierarchy: (1) Background ejecta envelope -> (2) Accretion disk plane -> (3) Central BH Remnant Shadow
            if ejecta_fluid_active or remnant_active or disk_active:

                # 1. Background Ejecta Fluid Envelope (subdued in GW mode to prioritize wavefront; micro-dots in NEUTRINO)
                if ejecta_fluid_active:
                    _ej_radius = 0.28e3 if self.view_mode == "GW" else (0.40e3 if self.view_mode == "MAGNETIC FIELD" else (0.08e3 if self.view_mode == "NEUTRINO" else (0.75e3 if (rem_st.is_black_hole and self.view_mode == "CORE") else 0.85e3)))
                    self.scene.particles(
                        self.renderer.ejecta_fluid_pos,
                        radius=_ej_radius,
                        per_vertex_color=self.renderer.ejecta_fluid_colors
                    )
                # 2. Continuous Accretion Disk (Interstellar-inspired continuous luminous plasma surface + lensed arches)
                if disk_active:
                    if hasattr(self.renderer, "disk_mesh_pos"):
                        self.scene.mesh(
                            self.renderer.disk_mesh_pos,
                            indices=self.renderer.disk_mesh_indices,
                            per_vertex_color=self.renderer.disk_mesh_colors,
                            two_sided=True
                        )
                    _disk_radius = 0.95e3 if (rem_st.is_black_hole and self.view_mode == "CORE") else (0.12e3 if self.view_mode in ["GW", "NEUTRINO"] else 0.35e3)
                    self.scene.particles(
                        self.renderer.disk_pos,
                        radius=_disk_radius,
                        per_vertex_color=self.renderer.disk_colors
                    )
                # 3. Central Remnant / Black Hole Shadow (drawn dominantly on top in the center)
                if remnant_active or just_transitioned:
                    _rem_radius = 2.0e3 if self.view_mode == "NEUTRINO" else (2.2e3 if self.view_mode == "GW" else (3.8e3 if (rem_st.is_black_hole and self.view_mode == "CORE") else 1.6e3))
                    self.scene.particles(
                        self.renderer.remnant_pos,
                        radius=_rem_radius,
                        per_vertex_color=self.renderer.remnant_colors
                    )

        # 4a. Neutrino Radiation Transport Particles (NEUTRINO and MULTI modes)
        show_neutrino = (self.view_mode in ["NEUTRINO", "MULTI"])
        if show_neutrino and post_merger_active:
            self.renderer.update_nu_particles(
                event_time=float(st.event_time),
                nu_luminosity_w=float(evt_st.neutrino_luminosity),
                is_active=bool(evt_st.neutrino_transport_active)
            )
            _nu_radius = 0.35e3 if self.view_mode == "NEUTRINO" else 0.50e3
            self.scene.particles(
                self.renderer.nu_pos,
                radius=_nu_radius,
                per_vertex_color=self.renderer.nu_colors
            )

        # 4b. Helical Magnetic Field Lines & Relativistic Jet Lines (cadenced at 30 Hz)
        b_active = bool(post_merger_active and self.director.b_winding_progress > 0.0)
        jet_active = bool(st.grb_triggered and self.director.jet_progress > 0.0)

        show_b_lines = (self.view_mode in ["MAGNETIC FIELD", "MULTI"]) and (self.view_mode != "CORE")
        show_jet_lines = (self.view_mode not in ["GW", "GW MODE", "NEUTRINO"])

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
                _b_scale = 1.65 if self.view_mode == "MAGNETIC FIELD" else (0.50 if self.view_mode == "MULTI" else 1.00)
                self.field_lines.update(
                    b_pol=evt_st.b_poloidal,
                    b_tor=evt_st.b_toroidal,
                    r_rem=14.0e3,
                    is_active=b_active,
                    winding_progress=self.director.b_winding_progress,
                    delta_omega=evt_st.differential_rotation,
                    event_time=st.event_time,
                    omega_rot=evt_st.omega_core,
                    intensity_scale=_b_scale
                )

        if jet_active and show_jet_lines:
            _jet_delay = float(self.engine.jet.jet_delay)
            _jet_beta  = float(self.engine.jet.beta(self.engine.jet.lorentz_profile(0.0)))
            if self.view_mode == "MAGNETIC FIELD":
                _jet_intensity = 0.12  # Subdued jet to highlight magnetic field lines centerpiece
            elif self.view_mode == "MULTI":
                _jet_intensity = 0.35  # Restrained multi-messenger jet
            else:
                _jet_intensity = 0.08  # Restrained polar outflow in CORE mode to prioritize BH & accretion disk visual hierarchy
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
            line_w = 2.0 if self.view_mode == "MAGNETIC FIELD" else 1.0
            self.scene.lines(
                self.renderer.combined_line_vertices,
                width=line_w,
                per_vertex_color=self.renderer.combined_line_colors
            )

        # 5. 3D GW Wavefront Propagation Lines
        if self.view_mode in ["GW", "GW MODE", "MULTI"]:
            _gw_intensity = 0.30 if self.view_mode == "MULTI" else 2.10
            wf_st = self.wave_propagation.update(
                event_time=st.event_time,
                f_gw=st.gw_frequency,
                merger_active=(st.event_time >= 0.0 or self.view_mode in ["GW", "GW MODE", "MULTI"]),
                h_plus=evt_st.h_plus,
                h_cross=evt_st.h_cross,
                intensity_scale=_gw_intensity
            )
        else:
            wf_st = WavefrontState(
                active=False,
                launch_time=0.0,
                current_time=st.event_time,
                radius=0.0,
                amplitude=0.0,
                f_gw=st.gw_frequency,
                n_vertices=0
            )

        self.chirp_audio.evaluate(st.event_time)

        if wf_st.active and wf_st.n_vertices > 0:
            if self.view_mode in ["GW", "GW MODE"]:
                lw = 1.4
            elif self.view_mode == "MULTI":
                lw = 2.0  # Clean, restrained wavefront line width in MULTI
            else:
                lw = 3.0
            self.scene.lines(
                self.wave_propagation.gpu_line_vertices,
                width=lw,
                per_vertex_color=self.wave_propagation.gpu_line_colors
            )

        self.canvas.scene(self.scene)

        # Render single clean 2D model strain trace on canvas (suppressed in NEUTRINO mode)
        if self.renderer.waveform_vertices is not None and self.view_mode != "NEUTRINO":
            wf_trace_w = 0.0042 if self.view_mode in ["GW", "GW MODE"] else 0.003
            wf_trace_col = (1.0, 0.82, 0.22) if self.view_mode in ["GW", "GW MODE"] else (1.0, 0.75, 0.2)
            self.canvas.lines(
                self.renderer.waveform_vertices,
                width=wf_trace_w,
                color=wf_trace_col  # Electric gold MODEL strain trace
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

        dt_raw = min(0.05, max(0.001, dt_wall)) if dt_wall > 0.0 else 0.016
        if not hasattr(self, "_smoothed_dt_cam") or self._smoothed_dt_cam is None:
            self._smoothed_dt_cam = dt_raw
        else:
            self._smoothed_dt_cam = 0.85 * self._smoothed_dt_cam + 0.15 * dt_raw

        dt_cam = max(0.005, min(0.05, self._smoothed_dt_cam))
        self.process_input(dt_cam)

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
