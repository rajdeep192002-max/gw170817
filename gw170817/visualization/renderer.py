"""
Taichi GGUI GPU Particle & Waveform Renderer for GW170817.

REDUCED-ORDER APPROXIMATION:
This renderer visualizes preallocated GPU particle fields and strain waveforms.
It does NOT alter physics or perform per-frame CPU data copies.
"""
import taichi as ti
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem


@ti.data_oriented
class ParticleRenderer:
    """
    GPU-accelerated particle color and GGUI waveform trace renderer.
    """

    def __init__(self, psys: ParticleSystem):
        self.psys = psys
        self.max_particles = psys.max_particles

        # Preallocated GPU vector field for particle rendering colors
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)

        # 2D Waveform line trace (512 points = 511 line segments = 1022 vertices)
        self.n_wave_samples = 512
        self.n_wave_vertices = 2 * (self.n_wave_samples - 1)
        self.waveform_vertices = ti.Vector.field(2, dtype=ti.f32, shape=self.n_wave_vertices)
        self.h_buf_gpu = ti.field(dtype=ti.f32, shape=self.n_wave_samples)

        # Initialize default particle colors
        self._init_colors_kernel(psys.n_particles_1, psys.max_particles)

    @ti.kernel
    def _init_colors_kernel(self, n1: ti.i32, n_total: ti.i32):
        # NS1: Blue / Cyan
        for i in range(n1):
            self.colors[i] = ti.Vector([0.25, 0.70, 1.00])
        # NS2: Orange / Gold
        for i in range(n1, n_total):
            self.colors[i] = ti.Vector([1.00, 0.48, 0.12])

    @ti.kernel
    def update_particle_colors(
        self,
        star_id: ti.template(),
        active: ti.template(),
        n_total: ti.i32,
        contact_frac: ti.f32
    ):
        """
        Dynamically update particle colors based on star ID, merger contact fraction,
        and ejecta/remnant state on GPU.
        """
        for i in range(n_total):
            sid = star_id[i]
            act = active[i]
            if act == 0:
                # Inactive / unbound high-velocity ejecta: bright yellow
                self.colors[i] = ti.Vector([1.0, 0.9, 0.3])
            elif contact_frac > 0.5:
                # Merger/remnant phase: blended thermal glow
                if sid == 0:
                    self.colors[i] = ti.Vector([0.4 + 0.5 * contact_frac, 0.6 - 0.2 * contact_frac, 1.0])
                else:
                    self.colors[i] = ti.Vector([1.0, 0.5 + 0.3 * contact_frac, 0.2])
            else:
                # Inspiral phase
                if sid == 0:
                    self.colors[i] = ti.Vector([0.25, 0.70, 1.00])
                else:
                    self.colors[i] = ti.Vector([1.00, 0.48, 0.12])

    @ti.kernel
    def update_waveform_lines_kernel(
        self,
        n_pts: ti.i32,
        x_start: ti.f32,
        w_panel: ti.f32,
        y_center: ti.f32,
        h_scale: ti.f32
    ):
        """Map GPU waveform samples to 2D screen line segment vertices."""
        for i in range(n_pts - 1):
            x0 = x_start + (float(i) / float(n_pts - 1)) * w_panel
            x1 = x_start + (float(i + 1) / float(n_pts - 1)) * w_panel

            h0 = self.h_buf_gpu[i]
            h1 = self.h_buf_gpu[i + 1]

            # Clamp strain visualization to panel bounds
            y0 = y_center + ti.max(-0.08, ti.min(0.08, h0 * h_scale))
            y1 = y_center + ti.max(-0.08, ti.min(0.08, h1 * h_scale))

            self.waveform_vertices[2 * i]     = ti.Vector([x0, y0])
            self.waveform_vertices[2 * i + 1] = ti.Vector([x1, y1])

    def update_waveform_buffer(self, waveform_data: np.ndarray):
        """Update waveform buffer GPU field from numpy waveform array."""
        if waveform_data is None or len(waveform_data) == 0:
            return
        n = min(len(waveform_data), self.n_wave_samples)
        data_slice = waveform_data[-n:]

        max_amp = float(np.max(np.abs(data_slice))) if np.max(np.abs(data_slice)) > 0 else 1.0e-21
        scale_factor = 0.06 / max(max_amp, 1.0e-23)

        padded = np.zeros(self.n_wave_samples, dtype=np.float32)
        padded[-n:] = data_slice.astype(np.float32)
        self.h_buf_gpu.from_numpy(padded)

        self.update_waveform_lines_kernel(
            self.n_wave_samples,
            0.66,   # x_start
            0.32,   # w_panel
            0.82,   # y_center
            scale_factor
        )
