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


@ti.data_oriented
class GWWavefrontPropagation:
    """
    3D Quadrupolar Gravitational Wavefront Generator.
    Produces 3D world-space polyline vertices for rendering expanding GW strain wavefronts on GPU.
    """

    def __init__(self, c_vis: float = 80.0e3, n_rings: int = 12, pts_per_ring: int = 48):
        self.c_vis = c_vis            # Visual expansion speed [m/s]
        self.n_rings = n_rings        # Number of radial wavefront shells
        self.pts_per_ring = pts_per_ring # Angular resolution per ring

        self.active = False
        self.manual_trigger = False
        self.launch_event_time = 0.0

        # Preallocated line vertices (2 vertices per line segment)
        # For each ring: n_lat (6) * pts_per_ring (48) segments = 288 lines = 576 vertices
        self.n_lat_rings = 6
        self.lines_per_shell = self.n_lat_rings * pts_per_ring
        self.total_vertices = self.n_rings * self.lines_per_shell * 2

        self.gpu_line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.total_vertices)
        self.gpu_line_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.total_vertices)

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

    @ti.kernel
    def _update_wavefront_gpu_kernel(
        self,
        active_flag: ti.i32,
        base_r: ti.f32,
        wavelength: ti.f32,
        h_strain_norm: ti.f32,
    ):
        pi = 3.141592653589793
        two_pi = 6.283185307179586

        for shell_i, lat_i, p_i in ti.ndrange(self.n_rings, self.n_lat_rings, self.pts_per_ring):
            segment_idx = shell_i * (self.n_lat_rings * self.pts_per_ring) + lat_i * self.pts_per_ring + p_i
            idx = 2 * segment_idx

            r_shell = base_r - float(shell_i) * wavelength * 2.2

            if active_flag == 0 or r_shell < 5.0e3:
                self.gpu_line_vertices[idx] = ti.Vector([0.0, 0.0, 0.0])
                self.gpu_line_vertices[idx + 1] = ti.Vector([0.0, 0.0, 0.0])
                self.gpu_line_colors[idx] = ti.Vector([0.0, 0.0, 0.0])
                self.gpu_line_colors[idx + 1] = ti.Vector([0.0, 0.0, 0.0])
            else:
                amp_att = 1.0 / (1.0 + r_shell / 80.0e3)
                col_intensity = ti.max(0.05, ti.min(1.0, amp_att * 1.5 * h_strain_norm))
                phase_shell = (r_shell / wavelength) * two_pi

                lat = -pi * 0.4 + float(lat_i) * (pi * 0.8 / float(self.n_lat_rings - 1)) if self.n_lat_rings > 1 else 0.0
                cos_lat = ti.cos(lat)
                sin_lat = ti.sin(lat)
                plus_fac = 0.5 * (1.0 + sin_lat * sin_lat)
                cross_fac = sin_lat

                phi1 = (float(p_i) / float(self.pts_per_ring)) * two_pi
                p_next = (p_i + 1) % self.pts_per_ring
                phi2 = (float(p_next) / float(self.pts_per_ring)) * two_pi

                quad1 = plus_fac * ti.cos(2.0 * phi1 - phase_shell) + cross_fac * ti.sin(2.0 * phi1 - phase_shell)
                quad2 = plus_fac * ti.cos(2.0 * phi2 - phase_shell) + cross_fac * ti.sin(2.0 * phi2 - phase_shell)

                deflect_scale = 0.12 * h_strain_norm
                r1 = r_shell * (1.0 + deflect_scale * quad1)
                r2 = r_shell * (1.0 + deflect_scale * quad2)

                x1 = r1 * cos_lat * ti.cos(phi1)
                y1 = r1 * cos_lat * ti.sin(phi1)
                z1 = r1 * sin_lat

                x2 = r2 * cos_lat * ti.cos(phi2)
                y2 = r2 * cos_lat * ti.sin(phi2)
                z2 = r2 * sin_lat

                self.gpu_line_vertices[idx] = ti.Vector([x1, y1, z1])
                self.gpu_line_vertices[idx + 1] = ti.Vector([x2, y2, z2])

                c_r = 0.2 + 0.8 * float(ti.max(0.0, quad1)) * col_intensity
                c_g = 0.85 * col_intensity
                c_b = 1.0 * col_intensity

                self.gpu_line_colors[idx] = ti.Vector([c_r, c_g, c_b])
                self.gpu_line_colors[idx + 1] = ti.Vector([c_r, c_g, c_b])

    def update(
        self,
        event_time: float,
        f_gw: float = 100.0,
        merger_active: bool = False,
        h_plus: float = 1.0e-21,
        h_cross: float = 0.0
    ) -> WavefrontState:
        """
        Update 3D expanding quadrupolar wavefront line geometry on GPU centered at world origin [0,0,0].
        """
        # Auto-trigger wavefront emission when merger occurs (t_event >= 0.0)
        if event_time < 0.0 and not self.manual_trigger:
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
        # Visual-time adapter for smooth 3D world-space expanding wavefront shells
        t_scale = 2.0
        dt_vis = float(t_scale * np.log(1.0 + dt / t_scale)) if dt > 0.0 else 0.0
        base_r = 15.0e3 + self.c_vis * dt_vis

        # Strain magnitude modulation driver
        h_mag = float(np.sqrt(h_plus**2 + h_cross**2))
        h_strain_norm = float(np.clip(h_mag / 1.0e-21, 0.25, 3.0)) if h_mag > 0.0 else 1.0

        wavelength = max(18.0e3, 3.0e8 / max(20.0, f_gw) * 0.0008)  # Visual wavelength

        active_shells = 0
        for shell_i in range(self.n_rings):
            r_shell = base_r - shell_i * wavelength * 2.2
            if r_shell >= 5.0e3:
                active_shells += 1

        n_active_vertices = active_shells * self.n_lat_rings * self.pts_per_ring * 2

        self._update_wavefront_gpu_kernel(
            1 if self.active else 0,
            float(base_r),
            float(wavelength),
            float(h_strain_norm)
        )

        return WavefrontState(
            active=True,
            launch_time=self.launch_event_time,
            current_time=event_time,
            radius=base_r,
            amplitude=float(1.0 / (1.0 + base_r / 80.0e3)),
            f_gw=f_gw,
            n_vertices=n_active_vertices
        )

