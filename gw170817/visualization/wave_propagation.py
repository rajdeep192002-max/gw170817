from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import taichi as ti


@dataclass
class WavefrontState:
    """State of the propagating 3D GW wavefront."""
    active: bool             # True if wave propagation is currently active
    launch_time: float       # Physical event time when wave launched [s]
    current_time: float      # Current physical event time [s]
    radius: float            # Expanding wavefront radius [m]
    amplitude: float         # Current peak wave strain amplitude [dimensionless]
    f_gw: float              # GW frequency [Hz]
    n_vertices: int          # Total number of 3D line vertices generated


@ti.func
def _calc_shell_point(
    r_shell: ti.f32,
    theta: ti.f32,
    phi: ti.f32,
    phase_shell: ti.f32,
    deflect_scale: ti.f32
) -> ti.types.vector(4, ti.f32):
    """Calculate 3D position [x, y, z] and local quadrupolar strain q for a point on a 3D wave shell."""
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)

    # 3D Quadrupolar m=2 strain pattern: Q(theta, phi) = 0.5 * (1 + cos^2 theta) * cos(2*phi - phase)
    q = 0.5 * (1.0 + cos_theta * cos_theta) * ti.cos(2.0 * phi - phase_shell)
    r = r_shell * (1.0 + deflect_scale * q)

    x = r * cos_theta * ti.cos(phi)
    y = r * cos_theta * ti.sin(phi)
    z = r * sin_theta
    return ti.Vector([x, y, z, q])


@ti.func
def _calc_vertex_color(q: ti.f32, col_intensity: ti.f32) -> ti.types.vector(3, ti.f32):
    """Calculate radiant electric blue/cyan RGB color for 3D GW wave shell vertices."""
    crest_pos = ti.max(0.0, q)
    # Luminous electric cyan base [0.25, 0.80, 1.00] with brilliant white-cyan peaks [1.00, 1.00, 1.00]
    c_r = (0.25 + 0.75 * crest_pos) * col_intensity
    c_g = (0.80 + 0.20 * crest_pos) * col_intensity
    c_b = 1.00 * col_intensity
    return ti.Vector([c_r, c_g, c_b])


@ti.data_oriented
class GWWavefrontPropagation:
    """
    3D Quadrupolar Gravitational Wavefront Generator.
    Produces 3D world-space closed quasi-spherical wave shell polyline vertices for rendering expanding GW strain wavefronts on GPU.
    Implements finite wavefront lifetime (fronts expand at constant c_vis, fade out, and exit scene), inspiral chirp acceleration, 8-shell merger burst, and post-merger emission collapse.
    """

    def __init__(self, c_vis: float = 110.0e3, n_rings: int = 8, pts_per_ring: int = 48):
        self.c_vis = c_vis            # Visual expansion speed [m/s] (constant 110 km/s)
        self.n_rings = n_rings        # Fixed pool of 8 radial 3D wavefront shells
        self.pts_per_ring = pts_per_ring # Angular resolution per latitude ring

        self.active = False
        self.manual_trigger = False
        self.launch_event_time = 0.0

        # Preallocated line vertices (2 vertices per line segment)
        # 3D shell structure per shell:
        # 3 latitude rings (upper +40 deg, equator 0 deg, lower -40 deg) of pts_per_ring (48) segments
        # Total per shell = 3 * 48 = 144 line segments = 288 line vertices
        self.n_lat_rings = 3
        self.lines_per_shell = self.n_lat_rings * self.pts_per_ring
        self.total_vertices = self.n_rings * self.lines_per_shell * 2

        self.gpu_line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.total_vertices)
        self.gpu_line_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.total_vertices)
        self.gpu_shell_radii = ti.field(dtype=ti.f32, shape=self.n_rings)
        self.gpu_shell_active = ti.field(dtype=ti.i32, shape=self.n_rings)
        self.gpu_shell_amplitudes = ti.field(dtype=ti.f32, shape=self.n_rings)

    @property
    def line_vertices(self) -> np.ndarray:
        """NumPy array view of line vertices for testing & compatibility."""
        return self.gpu_line_vertices.to_numpy()

    @property
    def line_colors(self) -> np.ndarray:
        """NumPy array view of line colors for testing & compatibility."""
        return self.gpu_line_colors.to_numpy()

    def trigger(self, event_time: float = 0.0):
        """Manually trigger GW wavefront propagation demo."""
        self.active = True
        self.manual_trigger = True
        self.launch_event_time = event_time

    def toggle(self, event_time: float = 0.0) -> bool:
        """Toggle wavefront propagation ON / OFF."""
        self.active = not self.active
        if self.active:
            self.launch_event_time = event_time
        return self.active

    @ti.kernel
    def _clear_gpu_fields(self):
        for i in range(self.total_vertices):
            self.gpu_line_vertices[i] = ti.Vector([0.0, 0.0, 0.0])
            self.gpu_line_colors[i] = ti.Vector([0.0, 0.0, 0.0])
        for k in range(self.n_rings):
            self.gpu_shell_radii[k] = 0.0
            self.gpu_shell_active[k] = 0
            self.gpu_shell_amplitudes[k] = 0.0

    @ti.kernel
    def _update_wavefront_gpu_kernel(
        self,
        active_flag: ti.i32,
        h_strain_norm: ti.f32,
        intensity_scale: ti.f32,
    ):
        two_pi = 6.283185307179586
        pi = 3.141592653589793

        r_min = 32.0e3
        r_max = 300.0e3
        spacing = 55.0e3

        lines_in_shell = self.n_lat_rings * self.pts_per_ring

        for shell_i in range(self.n_rings):
            shell_base_seg = shell_i * lines_in_shell

            is_active = self.gpu_shell_active[shell_i]
            r_shell = self.gpu_shell_radii[shell_i]
            shell_amp = self.gpu_shell_amplitudes[shell_i]

            if active_flag == 0 or is_active == 0 or r_shell < r_min or r_shell > r_max:
                for seg in range(lines_in_shell):
                    idx = 2 * (shell_base_seg + seg)
                    self.gpu_line_vertices[idx] = ti.Vector([0.0, 0.0, 0.0])
                    self.gpu_line_vertices[idx + 1] = ti.Vector([0.0, 0.0, 0.0])
                    self.gpu_line_colors[idx] = ti.Vector([0.0, 0.0, 0.0])
                    self.gpu_line_colors[idx + 1] = ti.Vector([0.0, 0.0, 0.0])
            else:
                # Physical 1/r amplitude attenuation + smooth boundary fade
                amp_att = 1.0 / (1.0 + r_shell / 80.0e3)
                fade_inner = ti.min(1.0, (r_shell - r_min) / 8.0e3)
                fade_outer = ti.min(1.0, (r_max - r_shell) / 30.0e3)
                shell_fade = ti.max(0.0, fade_inner * fade_outer)

                # Brightness hierarchy: boost intensity multiplier from 2.2 to 2.8 for vibrant, clear wavefront shells
                col_intensity = ti.max(0.08, ti.min(1.0, shell_amp * amp_att * 2.8 * h_strain_norm * shell_fade * intensity_scale))

                # Dynamic quadrupolar phase advances outward with radius
                phase_shell = (r_shell / spacing) * two_pi
                # Prominent 28% quadrupolar m=2 distortion for clear tidal lobes
                deflect_scale = 0.28 * shell_amp * ti.min(1.5, h_strain_norm)

                # Render 3D Latitude Rings: lower (-40 deg), equator (0 deg), upper (+40 deg)
                for lat_i in range(self.n_lat_rings):
                    theta = (-pi / 4.5) + float(lat_i) * (pi / 4.5) # -40 deg, 0 deg, +40 deg
                    for p_i in range(self.pts_per_ring):
                        seg_idx = shell_base_seg + lat_i * self.pts_per_ring + p_i
                        idx = 2 * seg_idx

                        phi1 = (float(p_i) / float(self.pts_per_ring)) * two_pi
                        phi2 = (float((p_i + 1) % self.pts_per_ring) / float(self.pts_per_ring)) * two_pi

                        pt1 = _calc_shell_point(r_shell, theta, phi1, phase_shell, deflect_scale)
                        pt2 = _calc_shell_point(r_shell, theta, phi2, phase_shell, deflect_scale)

                        v1 = ti.Vector([pt1[0], pt1[1], pt1[2]])
                        v2 = ti.Vector([pt2[0], pt2[1], pt2[2]])

                        c1 = _calc_vertex_color(pt1[3], col_intensity)
                        c2 = _calc_vertex_color(pt2[3], col_intensity)

                        self.gpu_line_vertices[idx] = v1
                        self.gpu_line_vertices[idx + 1] = v2
                        self.gpu_line_colors[idx] = c1
                        self.gpu_line_colors[idx + 1] = c2

    def update(
        self,
        event_time: float,
        f_gw: float = 100.0,
        merger_active: bool = False,
        h_plus: float = 1.0e-21,
        h_cross: float = 0.0,
        intensity_scale: float = 1.0
    ) -> WavefrontState:
        """
        Update 3D expanding quadrupolar wavefront shell line geometry on GPU centered at world origin [0,0,0].
        Models chirp acceleration, 8-shell merger burst, constant-speed (c_vis) propagation, and finite wavefront lifetime.
        """
        if event_time < 0.0 and not self.manual_trigger and not merger_active:
            self.active = False
            self._clear_gpu_fields()
            return WavefrontState(
                active=False,
                launch_time=0.0,
                current_time=event_time,
                radius=0.0,
                amplitude=0.0,
                f_gw=f_gw,
                n_vertices=0
            )

        if (merger_active or event_time >= 0.0) and not self.manual_trigger:
            self.active = True
            self.launch_event_time = 0.0

        if not self.active:
            self._clear_gpu_fields()
            return WavefrontState(
                active=False,
                launch_time=0.0,
                current_time=event_time,
                radius=0.0,
                amplitude=0.0,
                f_gw=f_gw,
                n_vertices=0
            )

        dt = max(0.0, event_time - self.launch_event_time)

        # Representative leading wavefront radius for physics/telemetry tests (expands at constant speed c_vis)
        lead_r = 15.0e3 + self.c_vis * dt

        # 8 emission offsets (t_emit <= 0) relative to peak merger (t_event = 0).
        # Accelerating emission spacing (chirp) during inspiral leading to peak merger burst.
        t_emit_offsets = [0.00, -0.06, -0.14, -0.24, -0.38, -0.56, -0.80, -1.15]
        # Visual emission amplitude hierarchy: peak at merger (1.0), late inspiral moderate (0.7-0.9), early subtle (0.35-0.5)
        shell_amps = [1.00, 0.95, 0.88, 0.78, 0.65, 0.52, 0.42, 0.35]

        r_min = 32.0e3
        r_max = 300.0e3

        shell_radii = np.zeros(self.n_rings, dtype=np.float32)
        shell_active = np.zeros(self.n_rings, dtype=np.int32)
        shell_amplitudes = np.zeros(self.n_rings, dtype=np.float32)

        for k, t_emit in enumerate(t_emit_offsets):
            if k >= self.n_rings:
                break
            if self.manual_trigger:
                age = dt - t_emit
            else:
                age = event_time - t_emit

            # Front is born at age >= 0 and expands at constant physical speed c_vis
            if age >= 0.0:
                r_k = r_min + self.c_vis * age
                # Finite Lifetime: front is active ONLY while inside local visual volume [r_min, r_max]
                if r_min <= r_k <= r_max:
                    shell_radii[k] = float(r_k)
                    shell_active[k] = 1
                    shell_amplitudes[k] = float(shell_amps[k])

        self.gpu_shell_radii.from_numpy(shell_radii)
        self.gpu_shell_active.from_numpy(shell_active)
        self.gpu_shell_amplitudes.from_numpy(shell_amplitudes)

        # Strain magnitude modulation driver
        h_mag = float(np.sqrt(h_plus**2 + h_cross**2))
        h_strain_norm = float(np.clip(h_mag / 1.0e-21, 0.25, 3.0)) if h_mag > 0.0 else 1.0

        n_active_vertices = self.total_vertices

        self._update_wavefront_gpu_kernel(
            1 if self.active else 0,
            float(h_strain_norm),
            float(intensity_scale)
        )

        return WavefrontState(
            active=True,
            launch_time=self.launch_event_time,
            current_time=event_time,
            radius=lead_r,
            amplitude=float(1.0 / (1.0 + lead_r / 80.0e3)),
            f_gw=f_gw,
            n_vertices=n_active_vertices
        )
