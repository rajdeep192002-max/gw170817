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
        winding_progress: ti.f32,
        delta_omega: ti.f32,
        event_time: ti.f32,
        phi_rot_wrapped: ti.f32,
        intensity_scale: ti.f32,
    ):
        """
        Taichi kernel updating 3D helical magnetic field line vertices on GPU.
        Accumulates differential rotation phase (dphi/dt ~ delta_omega) with radius dependence
        and continuous 3D magnetosphere rotation (phi_rot_wrapped).
        """
        two_pi = 6.283185307179586
        pi = 3.141592653589793

        for l in range(self.n_lines):
            phi0 = (float(l) / float(self.n_lines)) * two_pi

            # Winding pitch depends on B_tor / B_pol ratio (differential rotation winding)
            b_ratio = b_tor / ti.max(1.0e10, b_pol)
            pitch = ti.min(5.0, 1.8 * b_ratio)

            # Growth factor smoothstep bounded [0.15, 1.0]
            w_growth = ti.max(0.15, ti.min(1.0, winding_progress))

            # Restrained luminous intensity scales with field strength and winding
            intensity = ti.min(0.42, (0.15 + 0.25 * (b_pol / 1.0e14)) * w_growth) * intensity_scale

            # Distributed multi-shell equatorial radii spanning 20 km to 43 km around remnant
            shell_mod = float(l % 4)
            r_eq = (r_rem * 1.30 + (7.5e3 + shell_mod * 5.5e3) * w_growth)
            z_scale = 0.55 * r_eq

            # Accumulated differential rotation shear phase (dphi/dt ~ delta_omega)
            t_eff = ti.max(0.0, ti.min(10.0, event_time))
            shear_accum = (delta_omega / 1.0e3) * t_eff * 6.0 * w_growth

            for s in range(self.n_segments - 1):
                idx = 2 * (l * (self.n_segments - 1) + s)

                if active_flag == 0 or (b_pol <= 0.0 and b_tor <= 0.0):
                    self.line_vertices[idx]     = ti.Vector([0.0, 0.0, 0.0])
                    self.line_vertices[idx + 1] = ti.Vector([0.0, 0.0, 0.0])
                    self.line_colors[idx]       = ti.Vector([0.0, 0.0, 0.0])
                    self.line_colors[idx + 1]   = ti.Vector([0.0, 0.0, 0.0])
                else:
                    # Normalized parameter u in [-1.0, +1.0] along loop from north to south
                    u0 = -1.0 + 2.0 * (float(s) / float(self.n_segments - 1))
                    u1 = -1.0 + 2.0 * (float(s + 1) / float(self.n_segments - 1))

                    theta0 = 0.50 * pi + u0 * (0.36 * pi)
                    theta1 = 0.50 * pi + u1 * (0.36 * pi)

                    sin_th0 = ti.sin(theta0)
                    cos_th0 = ti.cos(theta0)
                    sin_th1 = ti.sin(theta1)
                    cos_th1 = ti.cos(theta1)

                    # Smooth dipole radial profile: compact at footpoints, reaching r_eq at equator
                    r0 = r_rem * 1.05 + (r_eq - r_rem * 1.05) * (sin_th0 * sin_th0)
                    r1 = r_rem * 1.05 + (r_eq - r_rem * 1.05) * (sin_th1 * sin_th1)

                    # Arched oblate loop height (strictly bounded |z| <= 13 km)
                    z0 = -z_scale * cos_th0 * sin_th0
                    z1 = -z_scale * cos_th1 * sin_th1

                    # Radius-dependent differential winding (inner loops wind faster than outer)
                    w_rad0 = ti.pow(r_rem / ti.max(1.0, r0), 1.2)
                    w_rad1 = ti.pow(r_rem / ti.max(1.0, r1), 1.2)

                    # Azimuthal twist: base phi0 + toroidal pitch shear + accumulated differential winding + 3D core rotation
                    tw0 = (phi0 + pitch * cos_th0 + shear_accum * w_rad0 + phi_rot_wrapped) % two_pi
                    tw1 = (phi0 + pitch * cos_th1 + shear_accum * w_rad1 + phi_rot_wrapped) % two_pi

                    x0 = r0 * ti.cos(tw0)
                    y0 = r0 * ti.sin(tw0)

                    x1 = r1 * ti.cos(tw1)
                    y1 = r1 * ti.sin(tw1)

                    self.line_vertices[idx]     = ti.Vector([x0, y0, z0])
                    self.line_vertices[idx + 1] = ti.Vector([x1, y1, z1])

                    # Color palette: Poloidal electric cyan -> Toroidal golden amber with winding
                    w_factor = ti.min(1.0, pitch / 4.0)
                    r_c = (0.15 * (1.0 - w_factor) + 0.95 * w_factor) * intensity
                    g_c = (0.82 * (1.0 - w_factor) + 0.72 * w_factor) * intensity
                    b_c = (0.95 * (1.0 - w_factor) + 0.22 * w_factor) * intensity

                    col = ti.Vector([r_c, g_c, b_c])
                    self.line_colors[idx]     = col
                    self.line_colors[idx + 1] = col

    def update(
        self,
        b_pol: float,
        b_tor: float,
        r_rem: float,
        is_active: bool,
        winding_progress: float = 1.0,
        delta_omega: float = 0.0,
        event_time: float = 0.0,
        omega_rot: float = 120.0,
        intensity_scale: float = 1.0,
    ):
        """Update 3D helical magnetic field lines on GPU with radius-dependent differential winding and continuous 3D rotation."""
        flag = 1 if is_active else 0
        phi_rot_wrapped = float((omega_rot * 0.05 * max(0.0, event_time)) % (2.0 * np.pi))
        self.update_field_lines_kernel(
            float(b_pol), float(b_tor), float(r_rem), flag,
            float(winding_progress), float(delta_omega), float(event_time),
            float(phi_rot_wrapped), float(intensity_scale)
        )

