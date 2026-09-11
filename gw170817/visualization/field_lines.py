"""
3D Dynamic Magnetic Field Line Geometry Renderer for BNS Remnants.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Generates 3D helical magnetic field lines on GPU (poloidal & toroidal field coils)
driven by B_poloidal and B_toroidal from MagneticFieldModel.
Animates field line winding, MRI saturation, and magnetosphere funnel expansion.
"""
import taichi as ti
import numpy as np


@ti.data_oriented
class MagneticFieldLines:
    """
    GPU dynamic 3D magnetic field lines renderer (16–24 GPU lines, multi-segment helices).
    """

    def __init__(self, n_lines: int = 20, segments_per_line: int = 16):
        self.n_lines = n_lines
        self.n_segments = segments_per_line
        self.n_vertices = 2 * (self.n_segments - 1) * self.n_lines
        self.line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.line_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)

    @ti.kernel
    def update_field_lines_kernel(
        self,
        b_pol: ti.f32,
        b_tor: ti.f32,
        r_rem: ti.f32,
        active_flag: ti.i32,
        winding_progress: ti.f32
    ):
        """
        Taichi kernel updating 3D helical magnetic field line vertices on GPU.
        """
        two_pi = 6.283185307179586
        pi = 3.141592653589793

        for l in range(self.n_lines):
            phi0 = (float(l) / float(self.n_lines)) * two_pi

            # Winding pitch depends on B_tor / B_pol ratio (differential rotation winding)
            b_ratio = b_tor / ti.max(1.0e10, b_pol)
            pitch = ti.min(8.0, 3.0 * b_ratio)

            # Scale height and intensity progressively as differential rotation winds field lines
            w_growth = ti.max(0.12, ti.min(1.0, winding_progress))

            # Intensity scales with poloidal field strength & winding progress
            intensity = ti.min(1.5, (0.2 + 0.8 * (b_pol / 1.0e14)) * w_growth)

            for s in range(self.n_segments - 1):
                idx = 2 * (l * (self.n_segments - 1) + s)

                if active_flag == 0 or (b_pol <= 0.0 and b_tor <= 0.0):
                    self.line_vertices[idx]     = ti.Vector([0.0, 0.0, 0.0])
                    self.line_vertices[idx + 1] = ti.Vector([0.0, 0.0, 0.0])
                    self.line_colors[idx]       = ti.Vector([0.0, 0.0, 0.0])
                    self.line_colors[idx + 1]   = ti.Vector([0.0, 0.0, 0.0])
                else:
                    # Parameter t in [0, 1] along poloidal dipole loop from north to south
                    t0 = float(s) / float(self.n_segments - 1)
                    t1 = float(s + 1) / float(self.n_segments - 1)

                    theta0 = 0.15 * pi + t0 * 0.70 * pi
                    theta1 = 0.15 * pi + t1 * 0.70 * pi

                    # Radius grows toward equatorial plane (theta = pi/2), shrinks toward poles
                    sin_th0 = ti.sin(theta0)
                    cos_th0 = ti.cos(theta0)
                    sin_th1 = ti.sin(theta1)
                    cos_th1 = ti.cos(theta1)

                    r0 = r_rem * (1.3 + 3.2 * sin_th0 * sin_th0) * w_growth
                    r1 = r_rem * (1.3 + 3.2 * sin_th1 * sin_th1) * w_growth

                    z0 = 160.0e3 * w_growth * cos_th0 / (sin_th0 + 0.1)
                    z1 = 160.0e3 * w_growth * cos_th1 / (sin_th1 + 0.1)

                    # Azimuthal helical twist
                    tw0 = phi0 + pitch * cos_th0
                    tw1 = phi0 + pitch * cos_th1

                    x0 = r0 * ti.cos(tw0)
                    y0 = r0 * ti.sin(tw0)

                    x1 = r1 * ti.cos(tw1)
                    y1 = r1 * ti.sin(tw1)

                    self.line_vertices[idx]     = ti.Vector([x0, y0, z0])
                    self.line_vertices[idx + 1] = ti.Vector([x1, y1, z1])

                    # Color palette: Poloidal dipole core = electric cyan; Toroidal winding = gold/magenta
                    w_factor = ti.min(1.0, pitch / 6.0)
                    r_c = (0.2 * (1.0 - w_factor) + 1.0 * w_factor) * intensity
                    g_c = (0.85 * (1.0 - w_factor) + 0.5 * w_factor) * intensity
                    b_c = (1.0 * (1.0 - w_factor) + 0.2 * w_factor) * intensity

                    col = ti.Vector([r_c, g_c, b_c])
                    self.line_colors[idx]     = col
                    self.line_colors[idx + 1] = col

    def update(self, b_pol: float, b_tor: float, r_rem: float, is_active: bool, winding_progress: float = 1.0):
        """Update 3D helical magnetic field lines on GPU."""
        flag = 1 if is_active else 0
        self.update_field_lines_kernel(float(b_pol), float(b_tor), float(r_rem), flag, float(winding_progress))

