"""
Taichi GGUI GPU Particle & Waveform Renderer for GW170817.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Renders preallocated GPU particle fields, strain waveforms, relativistic jet outflows,
and lensing visual effects using a luminous density/temperature & Ye proxy.
Does NOT perform per-frame CPU particle copies or alter underlying physics equations.
"""
import taichi as ti
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem


@ti.data_oriented
class ParticleRenderer:
    """
    GPU-accelerated particle color, relativistic jet, and GGUI waveform trace renderer.
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

        # Relativistic Jet visual line vertices (64 lines = 128 vertices along polar z-axis)
        self.n_jet_lines = 64
        self.n_jet_vertices = 2 * self.n_jet_lines
        self.jet_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_jet_vertices)
        self.jet_active = False

        # Initialize default luminous particle colors
        self._init_luminous_colors_kernel(psys.pos, psys.star_id, psys.n_particles_1, psys.max_particles)

    @ti.kernel
    def _init_luminous_colors_kernel(
        self,
        pos: ti.template(),
        star_id: ti.template(),
        n1: ti.i32,
        n_total: ti.i32
    ):
        """
        Seed initial luminous density/temperature proxy colors on GPU.
        Emissive core (white-hot) fading outward to thermal plasma gold/cyan-gold.
        """
        for i in range(n_total):
            p = pos[i]
            r_dist = ti.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
            core_factor = ti.max(0.0, 1.0 - (r_dist / 300.0e3))

            if star_id[i] == 0:
                # NS1: White-hot cyan-gold plasma core
                r_col = 0.65 + 0.70 * core_factor
                g_col = 0.85 + 0.50 * core_factor
                b_col = 1.00 + 0.30 * core_factor
                self.colors[i] = ti.Vector([r_col, g_col, b_col])
            else:
                # NS2: White-hot gold-orange plasma core
                r_col = 1.00 + 0.40 * core_factor
                g_col = 0.65 + 0.60 * core_factor
                b_col = 0.30 + 0.70 * core_factor
                self.colors[i] = ti.Vector([r_col, g_col, b_col])

    @ti.kernel
    def update_particle_colors(
        self,
        pos: ti.template(),
        star_id: ti.template(),
        active: ti.template(),
        n_total: ti.i32,
        contact_frac: ti.f32,
        lensing_enhanced: ti.i32
    ):
        """
        Dynamically update particle colors on GPU using a luminous density/temperature
        and electron fraction (Ye) proxy. Replaces generic red/blue ball appearance.
        """
        lens_boost = 0.25 if lensing_enhanced == 1 else 0.0

        for i in range(n_total):
            sid = star_id[i]
            act = active[i]
            p = pos[i]
            r_dist = ti.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
            core_factor = ti.max(0.0, 1.0 - (r_dist / 250.0e3))

            if act == 0:
                # Unbound ejecta: Ye composition proxy (high Ye blue-cyan, low Ye red-orange)
                if i % 2 == 0:
                    # High Ye lanthanide-poor component
                    self.colors[i] = ti.Vector([0.35, 0.80 + lens_boost, 1.00])
                else:
                    # Low Ye lanthanide-rich component
                    self.colors[i] = ti.Vector([1.00, 0.40, 0.15 + lens_boost])

            elif contact_frac > 0.1:
                # Merger/remnant phase: intense thermal interaction glow along merger interface
                inter_glow = contact_frac * (1.0 + core_factor)
                r_col = ti.min(1.8, 0.8 + 0.9 * inter_glow + lens_boost)
                g_col = ti.min(1.8, 0.7 + 0.6 * inter_glow)
                b_col = ti.min(1.8, 0.5 + 0.8 * inter_glow)
                self.colors[i] = ti.Vector([r_col, g_col, b_col])

            else:
                # Inspiral phase: Luminous density/temperature proxy
                if sid == 0:
                    r_col = 0.50 + 0.80 * core_factor + lens_boost
                    g_col = 0.80 + 0.50 * core_factor
                    b_col = 1.00 + 0.30 * core_factor
                    self.colors[i] = ti.Vector([r_col, g_col, b_col])
                else:
                    r_col = 1.00 + 0.40 * core_factor
                    g_col = 0.65 + 0.65 * core_factor + lens_boost
                    b_col = 0.35 + 0.65 * core_factor
                    self.colors[i] = ti.Vector([r_col, g_col, b_col])

    @ti.kernel
    def update_jet_lines_kernel(self, active_flag: ti.i32, z_max: ti.f32):
        """Map relativistic jet outflow geometry along rotational z-axis to GPU vertices."""
        two_pi = 6.283185307179586
        for i in range(self.n_jet_lines):
            if active_flag == 0:
                self.jet_vertices[2 * i]     = ti.Vector([0.0, 0.0, 0.0])
                self.jet_vertices[2 * i + 1] = ti.Vector([0.0, 0.0, 0.0])
            else:
                phi = (float(i) / float(self.n_jet_lines)) * two_pi
                theta_core = 0.08  # ~4.5 deg core opening angle
                dir_sign = 1.0 if (i % 2 == 0) else -1.0

                z_end = dir_sign * z_max
                x_end = z_max * theta_core * ti.cos(phi)
                y_end = z_max * theta_core * ti.sin(phi)

                self.jet_vertices[2 * i]     = ti.Vector([0.0, 0.0, 0.0])
                self.jet_vertices[2 * i + 1] = ti.Vector([x_end, y_end, z_end])

    def update_jet_geometry(self, is_active: bool, max_extent_m: float = 400.0e3):
        """Update structured relativistic jet line vertices on GPU."""
        self.jet_active = is_active
        flag = 1 if is_active else 0
        self.update_jet_lines_kernel(flag, float(max_extent_m))

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
