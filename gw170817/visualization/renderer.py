"""
Taichi GGUI GPU Particle, 2D Starfield Image Lensing, Jet, Chirp & Waveform Renderer for GW170817.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Renders preallocated GPU particle fields, 2D background image lensing warp,
time-frequency chirp spectrogram track, decimated strain waveform envelope,
and structured relativistic jet outflows.
Does NOT perform per-frame CPU particle copies or alter underlying physics equations.
"""
import taichi as ti
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem


@ti.data_oriented
class ParticleRenderer:
    """
    GPU-accelerated particle color, 2D starfield coordinate-warp lensing, jet, chirp, and waveform renderer.
    """

    def __init__(self, psys: ParticleSystem):
        self.psys = psys
        self.max_particles = psys.max_particles

        # Preallocated GPU vector fields for particle rendering positions and colors
        self.render_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)

        # 3D Central Remnant Compact Object Particles (120 particles)
        self.n_remnant_particles = 120
        self.remnant_local_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_remnant_particles)
        self.remnant_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_remnant_particles)
        self.remnant_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_remnant_particles)

        # 2D Waveform strain envelope (128 display segments = 256 vertices)
        self.n_wave_samples = 128
        self.n_wave_vertices = 2 * (self.n_wave_samples - 1)
        self.waveform_vertices = ti.Vector.field(2, dtype=ti.f32, shape=self.n_wave_vertices)
        self.h_buf_gpu = ti.field(dtype=ti.f32, shape=self.n_wave_samples)

        # 2D Time-Frequency Chirp track (64 line segments = 128 vertices)
        self.n_chirp_pts = 64
        self.n_chirp_vertices = 2 * (self.n_chirp_pts - 1)
        self.chirp_vertices = ti.Vector.field(2, dtype=ti.f32, shape=self.n_chirp_vertices)

        # Relativistic Jet visual line vertices (64 lines = 128 vertices along polar z-axis)
        self.n_jet_lines = 64
        self.n_jet_vertices = 2 * self.n_jet_lines
        self.jet_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_jet_vertices)
        self.jet_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_jet_vertices)
        self.jet_active = False

        # GPU Background 2D Image Field (512x512) for Coordinate-Warp Lensing
        self.img_w = 512
        self.img_h = 512
        self.star_texture = ti.Vector.field(3, dtype=ti.f32, shape=(self.img_h, self.img_w))
        self.star_bg_img = ti.Vector.field(3, dtype=ti.f32, shape=(self.img_h, self.img_w))

        # GPU Background Star Field (2,000 stars vector field for unit test compatibility)
        self.n_stars = 2000
        self.star_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_stars)
        self.star_deflected_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_stars)
        self.star_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_stars)
        self.star_brightness = ti.field(dtype=ti.f32, shape=self.n_stars)

        # 3D Accretion Disk Torus Particles (2,500 particles)
        self.n_disk_particles = 2500
        self.disk_local_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_disk_particles)
        self.disk_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_disk_particles)
        self.disk_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_disk_particles)
        self.disk_radius_rel = ti.field(dtype=ti.f32, shape=self.n_disk_particles)
        self.disk_omega = ti.field(dtype=ti.f32, shape=self.n_disk_particles)

        # 3D Neutrino Cooling Wind Halo Particles (400 particles)
        self.n_nu_particles = 400
        self.nu_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_nu_particles)
        self.nu_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_nu_particles)

        # 3D Continuous Multi-Component Ejecta Fluid Plasma Particles (2,500 particles)
        self.n_ejecta_fluid_particles = 2500
        self.ejecta_fluid_dir = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_comp = ti.field(dtype=ti.i32, shape=self.n_ejecta_fluid_particles)

        # Preallocated combined post-merger particle fields (5,120 particles total)
        self.n_combined_post_merger = self.n_remnant_particles + self.n_disk_particles + self.n_ejecta_fluid_particles
        self.combined_post_merger_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_post_merger)
        self.combined_post_merger_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_post_merger)

        # Preallocated combined lines field (600 magnetic lines + 128 jet lines = 728 vertices)
        self.n_combined_lines = 600 + self.n_jet_vertices
        self.combined_line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_lines)
        self.combined_line_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_lines)

        # Initialize GPU Starfield, 2D Star Texture, Central Remnant, Accretion Disk, Neutrinos, and Ejecta Fluid
        self._init_starfield_kernel(self.n_stars)
        self._init_star_texture_kernel(self.img_w, self.img_h)
        self._init_remnant_particles_kernel(self.n_remnant_particles)
        self._init_disk_particles_kernel(self.n_disk_particles)
        self._init_nu_particles_kernel(self.n_nu_particles)
        self._init_ejecta_fluid_kernel(self.n_ejecta_fluid_particles)

    @ti.kernel
    def _init_ejecta_fluid_kernel(self, n_ej: ti.i32):
        """
        Seed 2,500 continuous 3D multi-component ejecta fluid particles across angular sphere.
        - Polar Blue component (lanthanide-poor, fast, polar cone)
        - Purple Intermediate component
        - Equatorial Red component (lanthanide-rich, slower, equatorial torus)
        """
        two_pi = 6.283185307179586
        for i in range(n_ej):
            u = ti.random(ti.f32)
            v = ti.random(ti.f32)
            cos_th = 1.0 - 2.0 * u
            sin_th = ti.sqrt(ti.max(0.0, 1.0 - cos_th * cos_th))
            phi = v * two_pi

            dir_vec = ti.Vector([sin_th * ti.cos(phi), sin_th * ti.sin(phi), cos_th])
            self.ejecta_fluid_dir[i] = dir_vec
            self.ejecta_fluid_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
            self.ejecta_fluid_colors[i] = ti.Vector([0.0, 0.0, 0.0])

            abs_cos = ti.abs(cos_th)
            if abs_cos >= 0.65:
                self.ejecta_fluid_comp[i] = 0  # Blue polar
            elif abs_cos >= 0.35:
                self.ejecta_fluid_comp[i] = 1  # Purple intermediate
            else:
                self.ejecta_fluid_comp[i] = 2  # Red equatorial

    @ti.kernel
    def _init_starfield_kernel(self, n_stars: ti.i32):
        """
        Seed 2,000 background stars deterministically across 3D background sphere/plane.
        """
        two_pi = 6.283185307179586
        R_star_bg = 1.2e6  # Background distance [m]

        for i in range(n_stars):
            u = ti.random(ti.f32)
            v = ti.random(ti.f32)

            theta = ti.acos(1.0 - 2.0 * u)
            phi = v * two_pi

            if i % 5 == 0:
                theta = 0.5 * 3.1415926 + (ti.random(ti.f32) - 0.5) * 0.35

            x = R_star_bg * ti.sin(theta) * ti.cos(phi)
            y = R_star_bg * ti.sin(theta) * ti.sin(phi)
            z = R_star_bg * ti.cos(theta)

            self.star_pos[i] = ti.Vector([x, y, z])
            self.star_deflected_pos[i] = ti.Vector([x, y, z])

            c_type = i % 4
            if c_type == 0:
                self.star_colors[i] = ti.Vector([0.95, 0.95, 1.00])
            elif c_type == 1:
                self.star_colors[i] = ti.Vector([0.60, 0.85, 1.00])
            elif c_type == 2:
                self.star_colors[i] = ti.Vector([1.00, 0.90, 0.65])
            else:
                self.star_colors[i] = ti.Vector([0.70, 0.70, 0.80])

            self.star_brightness[i] = 0.3 + 0.9 * ti.random(ti.f32)

    @ti.kernel
    def _init_star_texture_kernel(self, w: ti.i32, h: ti.i32):
        """
        Render deterministic 2D background starfield texture on GPU into self.star_texture.
        """
        for i, j in ti.ndrange(h, w):
            u = float(j) / float(w)
            v = float(i) / float(h)

            # Deep space dark background gradient with faint Milky Way diffuse band
            mw_band = ti.exp(-((v - 0.5) / 0.15) ** 2) * 0.08
            r_bg = 0.01 + mw_band * 0.5
            g_bg = 0.01 + mw_band * 0.6
            b_bg = 0.03 + mw_band * 0.9

            col = ti.Vector([r_bg, g_bg, b_bg])

            # Grid-based pseudo-random star dots
            gx = ti.floor(u * 64.0)
            gy = ti.floor(v * 64.0)
            rnd = ti.sin(gx * 12.9898 + gy * 78.233) * 43758.5453
            rnd -= ti.floor(rnd)

            if rnd > 0.915:
                # Star center inside grid cell
                cx = (gx + 0.5 + 0.3 * ti.sin(rnd * 17.0)) / 64.0
                cy = (gy + 0.5 + 0.3 * ti.cos(rnd * 31.0)) / 64.0

                dist = ti.sqrt((u - cx) ** 2 + (v - cy) ** 2)
                if dist < 0.006:
                    star_intensity = ti.pow(1.0 - (dist / 0.006), 2.0) * (0.6 + 0.8 * rnd)

                    # Multi-spectral stellar color palette
                    c_idx = int(rnd * 100.0) % 3
                    s_col = ti.Vector([1.0, 1.0, 1.0])
                    if c_idx == 0:
                        s_col = ti.Vector([0.7, 0.85, 1.0])  # Hot blue star
                    elif c_idx == 1:
                        s_col = ti.Vector([1.0, 0.9, 0.6])   # Solar gold star

                    col += s_col * star_intensity

            self.star_texture[i, j] = col
            self.star_bg_img[i, j] = col

    @ti.kernel
    def update_star_lensing_deflection_kernel(
        self,
        ns1_x: ti.f32, ns1_y: ti.f32, ns1_z: ti.f32,
        ns2_x: ti.f32, ns2_y: ti.f32, ns2_z: ti.f32,
        m1_kg: ti.f32,
        m2_kg: ti.f32,
        lensing_active: ti.i32,
        enhanced_scale: ti.f32
    ):
        """
        2D Coordinate-Warp Image Lensing Kernel:
        For each pixel (i, j): source_coord = output_coord - deflection(output_coord).
        Warps the background 2D star texture field dynamically around compact objects when lensing_active == 1.
        """
        # 1. Update 3D vector star field for unit tests compatibility
        for k in range(self.n_stars):
            orig_p = self.star_pos[k]
            if lensing_active == 0:
                self.star_deflected_pos[k] = orig_p
            else:
                p1 = ti.Vector([ns1_x, ns1_y, ns1_z])
                p2 = ti.Vector([ns2_x, ns2_y, ns2_z])
                d1 = orig_p - p1
                d2 = orig_p - p2
                b1 = ti.max(15.0e3, ti.sqrt(d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]))
                b2 = ti.max(15.0e3, ti.sqrt(d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2]))
                c2 = 8.98755e16
                G_val = 6.6743e-11
                deflect1_mag = (4.0 * G_val * m1_kg / (c2 * b1)) * 3.5e5 * enhanced_scale
                deflect2_mag = (4.0 * G_val * m2_kg / (c2 * b2)) * 3.5e5 * enhanced_scale
                dir1 = d1 / b1
                dir2 = d2 / b2
                self.star_deflected_pos[k] = orig_p + dir1 * deflect1_mag + dir2 * deflect2_mag

        # 2. 2D Coordinate-Warp Image Lookup
        scale_view = 500.0e3  # Screen viewport half-width [m]
        u1 = 0.5 + (ns1_x / (2.0 * scale_view))
        v1 = 0.5 + (ns1_y / (2.0 * scale_view))
        u2 = 0.5 + (ns2_x / (2.0 * scale_view))
        v2 = 0.5 + (ns2_y / (2.0 * scale_view))

        for i, j in ti.ndrange(self.img_h, self.img_w):
            u_out = float(j) / float(self.img_w)
            v_out = float(i) / float(self.img_h)

            if lensing_active == 0:
                self.star_bg_img[i, j] = self.star_texture[i, j]
            else:
                # 2D Screen-space impact parameters
                du1 = u_out - u1
                dv1 = v_out - v1
                b1_sq = du1 * du1 + dv1 * dv1 + 0.0004
                b1 = ti.sqrt(b1_sq)

                du2 = u_out - u2
                dv2 = v_out - v2
                b2_sq = du2 * du2 + dv2 * dv2 + 0.0004
                b2 = ti.sqrt(b2_sq)

                # Weak-field Einstein deflection magnitude
                alpha_mag1 = (0.008 * enhanced_scale) / (b1 + 0.01)
                alpha_mag2 = (0.008 * enhanced_scale) / (b2 + 0.01)

                u_src = u_out - (alpha_mag1 * (du1 / b1) + alpha_mag2 * (du2 / b2))
                v_src = v_out - (alpha_mag1 * (dv1 / b1) + alpha_mag2 * (dv2 / b2))

                # Clamp to texture bounds
                u_src_c = ti.max(0.0, ti.min(1.0, u_src))
                v_src_c = ti.max(0.0, ti.min(1.0, v_src))

                src_j = int(u_src_c * float(self.img_w - 1))
                src_i = int(v_src_c * float(self.img_h - 1))

                # Add Einstein-ring circular distortion glow around compact objects
                ring_glow = 0.0
                if b1 < 0.03:
                    ring_glow += ti.pow(1.0 - (b1 / 0.03), 2.0) * 0.4
                if b2 < 0.03:
                    ring_glow += ti.pow(1.0 - (b2 / 0.03), 2.0) * 0.4

                warped_col = self.star_texture[src_i, src_j] + ti.Vector([0.3 * ring_glow, 0.6 * ring_glow, 1.0 * ring_glow])
                self.star_bg_img[i, j] = warped_col

    @ti.kernel
    def _init_remnant_particles_kernel(self, n_rem: ti.i32):
        """
        Seed 120 particles on a spherical shell of radius R_rem = 14 km for the central remnant object.
        """
        two_pi = 6.283185307179586
        R_rem = 14.0e3  # Central remnant radius [m]

        for i in range(n_rem):
            u = ti.random(ti.f32)
            v = ti.random(ti.f32)
            cos_th = 1.0 - 2.0 * u
            sin_th = ti.sqrt(ti.max(0.0, 1.0 - cos_th * cos_th))
            phi = v * two_pi

            r_shell = (0.3 + 0.7 * ti.pow(ti.random(ti.f32), 0.333)) * R_rem
            x = r_shell * sin_th * ti.cos(phi)
            y = r_shell * sin_th * ti.sin(phi)
            z = r_shell * cos_th * 0.85

            self.remnant_local_pos[i] = ti.Vector([x, y, z])
            self.remnant_pos[i] = ti.Vector([x, y, z])
            self.remnant_colors[i] = ti.Vector([2.0, 1.8, 1.3])

    @ti.kernel
    def update_remnant_particles_kernel(
        self,
        event_time: ti.f32,
        contact_frac: ti.f32,
        remnant_type: ti.i32
    ):
        """
        Animate bright incandescent central remnant compact object continuously rotating on GPU.
        """
        om_spin = 120.0  # High spin omega [rad/s]

        for i in range(self.n_remnant_particles):
            if contact_frac <= 0.05:
                self.remnant_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
                self.remnant_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                lp = self.remnant_local_pos[i]
                phi = om_spin * event_time

                cos_p = ti.cos(phi)
                sin_p = ti.sin(phi)

                rx = lp[0] * cos_p - lp[1] * sin_p
                ry = lp[0] * sin_p + lp[1] * cos_p
                rz = lp[2]

                self.remnant_pos[i] = ti.Vector([rx, ry, rz])

                pulse = 0.8 + 0.2 * ti.sin(event_time * 30.0 + float(i))
                if remnant_type == 1:
                    # HMNS: brilliant white-gold incandescent core
                    self.remnant_colors[i] = ti.Vector([
                        ti.min(1.0, 1.0 * pulse),
                        ti.min(1.0, 0.95 * pulse),
                        ti.min(1.0, 0.70 * pulse)
                    ])
                else:
                    # BH: intense accretion horizon white-blue
                    self.remnant_colors[i] = ti.Vector([
                        ti.min(1.0, 0.3 * pulse),
                        ti.min(1.0, 0.7 * pulse),
                        ti.min(1.0, 1.0 * pulse)
                    ])

    def update_remnant_particles(self, event_time: float, contact_frac: float = 1.0, remnant_type_str: str = "HMNS"):
        """Update 3D central remnant compact object particles on GPU."""
        r_type = 1 if remnant_type_str == "HMNS" else 0
        self.update_remnant_particles_kernel(float(event_time), float(contact_frac), r_type)

    @ti.kernel
    def update_particle_render_data(
        self,
        pos: ti.template(),
        star_id: ti.template(),
        active: ti.template(),
        ye_field: ti.template(),
        n_total: ti.i32,
        contact_frac: ti.f32,
        ejecta_progress: ti.f32
    ):
        """
        Dynamically update particle render positions and colors on GPU.
        - Inspiral phase: renders NS1 and NS2 particle clouds with surface colors.
        - Post-merger phase: culls core/bound particles (r < 150 km) by placing render_pos at [0, 0, -1e9],
          completely eliminating black particle occlusion and fragment overdraw around remnant/torus/jet.
        - Ejecta phase (ejecta_progress > 0.1): renders expanding unbound ejecta particles (r >= 150 km)
          with Ye-dependent composition colors.
        """
        for i in range(n_total):
            sid = star_id[i]
            act = active[i]
            p = pos[i]
            ye = ye_field[i]

            r_dist = ti.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

            if contact_frac > 0.05 or ejecta_progress > 0.0:
                # Post-merger: cull all original inspiral BNS particles (move out of camera view).
                # Eliminates 100% of black particle occlusion, overdraw, and dark dot artifacts!
                # Post-merger ejecta & kilonova are rendered via continuous ejecta_fluid_pos plasma volume.
                self.render_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
                self.colors[i] = ti.Vector([0.0, 0.0, 0.0])

            else:
                # Inspiral phase: NS surface colors (strictly clamped [0, 1] for Vulkan UNORM compatibility)
                self.render_pos[i] = p
                core_factor = ti.max(0.0, 1.0 - (r_dist / 250.0e3))
                if sid == 0:
                    r_col = ti.min(1.0, 0.30 + 0.50 * core_factor)
                    g_col = ti.min(1.0, 0.60 + 0.40 * core_factor)
                    b_col = ti.min(1.0, 0.90 + 0.10 * core_factor)
                    self.colors[i] = ti.Vector([r_col, g_col, b_col])
                else:
                    r_col = ti.min(1.0, 0.95 + 0.05 * core_factor)
                    g_col = ti.min(1.0, 0.60 + 0.35 * core_factor)
                    b_col = ti.min(1.0, 0.20 + 0.30 * core_factor)
                    self.colors[i] = ti.Vector([r_col, g_col, b_col])

    def update_particle_colors(
        self,
        pos: ti.template(),
        star_id: ti.template(),
        active: ti.template(),
        ye_field: ti.template(),
        n_total: int,
        contact_frac: float,
        ejecta_progress: float
    ):
        """Update particle render positions and colors on GPU."""
        self.update_particle_render_data(
            pos, star_id, active, ye_field, n_total,
            float(contact_frac), float(ejecta_progress)
        )

    @ti.kernel
    def _init_disk_particles_kernel(self, n_disk: ti.i32):
        """
        Seed thick rotating accretion torus geometry centered on remnant in equatorial plane (H/R ≈ 0.22).
        """
        two_pi = 6.283185307179586
        R_in = 18.0e3   # Inner disk radius [m]
        R_out = 120.0e3 # Outer disk radius [m]
        H_over_R = 0.22 # Thick torus aspect ratio

        for i in range(n_disk):
            u_r = ti.random(ti.f32)
            u_phi = ti.random(ti.f32)
            u_z = ti.random(ti.f32)

            r = R_in + u_r * (R_out - R_in)
            phi = u_phi * two_pi
            h_max = r * H_over_R
            z = (u_z - 0.5) * 2.0 * h_max

            self.disk_local_pos[i] = ti.Vector([r, phi, z])
            self.disk_radius_rel[i] = (r - R_in) / (R_out - R_in)
            # Keplerian angular velocity Omega ~ sqrt(G M / r^3)
            self.disk_omega[i] = ti.sqrt(6.6743e-11 * 2.7 * 1.989e30 / (r * r * r))

    @ti.kernel
    def update_disk_particles_kernel(
        self,
        event_time: ti.f32,
        is_active: ti.i32,
        disk_mass_frac: ti.f32
    ):
        """
        Animate thick rotating accretion torus driven by simulation event time.
        """
        for i in range(self.n_disk_particles):
            if is_active == 0 or event_time < 0.0 or disk_mass_frac <= 0.0:
                self.disk_pos[i] = ti.Vector([0.0, 0.0, 0.0])
                self.disk_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                lp = self.disk_local_pos[i]
                r = lp[0]
                phi0 = lp[1]
                z0 = lp[2]

                om = self.disk_omega[i]
                phi = phi0 + om * event_time * 0.05

                x = r * ti.cos(phi)
                y = r * ti.sin(phi)
                z = z0

                self.disk_pos[i] = ti.Vector([x, y, z])

                # Radial temperature gradient: inner region incandescent white-gold, outer maroon
                r_rel = self.disk_radius_rel[i]
                heat = (1.0 - r_rel) * disk_mass_frac

                r_c = ti.min(1.0, 0.5 + 0.5 * heat)
                g_c = ti.min(1.0, 0.2 + 0.8 * heat)
                b_c = ti.min(1.0, 0.05 + 0.95 * ti.pow(heat, 2.0))

                self.disk_colors[i] = ti.Vector([r_c, g_c, b_c])

    def update_disk_particles(self, event_time: float, is_active: bool = True, disk_progress: float = 1.0, disk_mass_msun: float = 0.06):
        """Update 3D thick accretion disk torus particles on GPU."""
        flag = 1 if is_active else 0
        frac = min(1.5, max(0.0, (disk_mass_msun / 0.06) * max(0.0, float(disk_progress))))
        self.update_disk_particles_kernel(float(event_time), flag, float(frac))

    @ti.kernel
    def _init_nu_particles_kernel(self, n_nu: ti.i32):
        """
        Seed subtle neutrino cooling wind halo particles around remnant/disk.
        """
        two_pi = 6.283185307179586
        for i in range(n_nu):
            u_r = ti.random(ti.f32)
            u_th = ti.random(ti.f32)
            u_ph = ti.random(ti.f32)
            r = 20.0e3 + u_r * 180.0e3
            th = ti.acos(1.0 - 2.0 * u_th)
            ph = u_ph * two_pi
            self.nu_pos[i] = ti.Vector([r * ti.sin(th) * ti.cos(ph), r * ti.sin(th) * ti.sin(ph), r * ti.cos(th)])
            self.nu_colors[i] = ti.Vector([0.4, 0.2, 0.8])

    @ti.kernel
    def update_nu_particles_kernel(
        self,
        event_time: ti.f32,
        nu_lum_norm: ti.f32
    ):
        """
        Animate subtle neutrino wind halo cooling indicator around remnant/disk.
        """
        for i in range(self.n_nu_particles):
            if event_time < 0.0 or nu_lum_norm <= 0.0:
                self.nu_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                intensity = ti.min(0.6, 0.15 * nu_lum_norm * (1.0 + 0.3 * ti.sin(event_time * 15.0 + float(i))))
                self.nu_colors[i] = ti.Vector([0.3 * intensity, 0.15 * intensity, 0.6 * intensity])

    def update_nu_particles(self, event_time: float, nu_luminosity_w: float = 1.0e45):
        """Update neutrino wind halo particles on GPU."""
        norm = min(2.0, max(0.0, nu_luminosity_w / 1.0e45))
        self.update_nu_particles_kernel(float(event_time), float(norm))

    @ti.kernel
    def update_ejecta_fluid_kernel(
        self,
        event_time: ti.f32,
        ejecta_progress: ti.f32,
        is_active: ti.i32
    ):
        """
        Animate 3D continuous expanding multi-component ejecta plasma volume on GPU.

        Radius is driven by PHYSICAL EVENT TIME via ballistic propagation:
            r_component = r_launch + v_component * event_time
        Component velocities (reduced-order):
            Blue polar:      v ≈ 0.30 c  (lanthanide-poor, fast)
            Purple mid-lat:  v ≈ 0.20 c
            Red equatorial:  v ≈ 0.10 c  (lanthanide-rich, slow)

        ejecta_progress is retained only as a visibility/fade-in gate (> 0.01).
        """
        # Reduced-order component velocities [m/s]
        c_light = 2.998e8
        v_blue   = 0.30 * c_light  # [MOD] blue polar velocity
        v_purple = 0.20 * c_light  # [MOD] intermediate velocity
        v_red    = 0.10 * c_light  # [MOD] red equatorial velocity

        for i in range(self.n_ejecta_fluid_particles):
            if is_active == 0 or event_time < 0.0:
                self.ejecta_fluid_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
                self.ejecta_fluid_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                d = self.ejecta_fluid_dir[i]
                comp = self.ejecta_fluid_comp[i]

                # Thermal cooling: temperature falls as t^{-0.3} proxy
                # At t=1s: heat≈1.0; at t=1day (86400s): heat≈1.0*(1/86400)^0.3≈0.015 → clamp to 0.25
                t_norm = ti.max(1.0, event_time)  # avoid log(0)
                heat = ti.max(0.25, 1.0 - 0.25 * ti.log(t_norm))

                # Smooth initial launch emission buildup (0 to 20 ms physical scale)
                growth = ti.min(1.0, event_time / 0.020) if event_time < 0.020 else 1.0
                heat_eff = heat * (0.35 + 0.65 * growth)

                # Pulsating fluid oscillation (physical wobble, NOT phase)
                wobble = 1.0 + 0.06 * ti.sin(event_time * 8.0 + float(i % 17))

                if comp == 0:
                    # Blue polar plume (fast, v≈0.30c, polar cone)
                    r_shell = (25.0e3 + v_blue * event_time) * wobble
                    x = d[0] * r_shell * 0.75
                    y = d[1] * r_shell * 0.75
                    z = d[2] * r_shell * 1.25
                    self.ejecta_fluid_pos[i] = ti.Vector([x, y, z])
                    self.ejecta_fluid_colors[i] = ti.Vector([
                        ti.min(1.0, 0.35 * heat_eff),
                        ti.min(1.0, 0.75 * heat_eff),
                        ti.min(1.0, 1.00 * heat_eff)
                    ])

                elif comp == 1:
                    # Purple intermediate layer (v≈0.20c, mid-latitudes)
                    r_shell = (22.0e3 + v_purple * event_time) * wobble
                    x = d[0] * r_shell
                    y = d[1] * r_shell
                    z = d[2] * r_shell
                    self.ejecta_fluid_pos[i] = ti.Vector([x, y, z])
                    self.ejecta_fluid_colors[i] = ti.Vector([
                        ti.min(1.0, 0.75 * heat_eff),
                        ti.min(1.0, 0.35 * heat_eff),
                        ti.min(1.0, 0.95 * heat_eff)
                    ])

                else:
                    # Red equatorial shell (slower v≈0.10c, dense lanthanide-rich torus)
                    r_shell = (18.0e3 + v_red * event_time) * wobble
                    x = d[0] * r_shell * 1.30
                    y = d[1] * r_shell * 1.30
                    z = d[2] * r_shell * 0.55
                    self.ejecta_fluid_pos[i] = ti.Vector([x, y, z])
                    self.ejecta_fluid_colors[i] = ti.Vector([
                        ti.min(1.0, 1.00 * heat_eff),
                        ti.min(1.0, 0.35 * heat_eff),
                        ti.min(1.0, 0.10 * heat_eff)
                    ])

    def update_ejecta_fluid(self, event_time: float, ejecta_progress: float, is_active: bool = True):
        """Update 3D continuous ejecta fluid plasma volume on GPU."""
        flag = 1 if is_active else 0
        self.update_ejecta_fluid_kernel(float(event_time), float(ejecta_progress), flag)

    @ti.kernel
    def update_dynamic_jet_kernel(
        self,
        active_flag: ti.i32,
        opening_angle_rad: ti.f32,
        intensity: ti.f32,
        event_time: ti.f32,
        jet_progress: ti.f32,
        jet_delay: ti.f32,
        beta_jet: ti.f32
    ):
        """
        Animate dynamic structured jet length expansion using PHYSICAL propagation time.

        Jet length is driven by:
            t_since_launch = max(0, event_time - jet_delay)
            jet_length = 14 km + beta * c * t_since_launch

        where beta_jet ≈ sqrt(1 - 1/Γ²) for Γ=100 (≈ 0.9999).
        jet_progress is retained only as a visibility/fade-in gate (> 0.01).
        """
        two_pi = 6.283185307179586
        c_light = 2.998e8

        for i in range(self.n_jet_lines):
            if active_flag == 0 or event_time < 0.0 or jet_progress <= 0.01:
                self.jet_vertices[2 * i]     = ti.Vector([0.0, 0.0, 0.0])
                self.jet_vertices[2 * i + 1] = ti.Vector([0.0, 0.0, 0.0])
                self.jet_colors[2 * i]       = ti.Vector([0.0, 0.0, 0.0])
                self.jet_colors[2 * i + 1]   = ti.Vector([0.0, 0.0, 0.0])
            else:
                phi = (float(i) / float(self.n_jet_lines)) * two_pi
                dir_sign = 1.0 if (i % 2 == 0) else -1.0

                # Physical jet propagation: r_jet = r_launch + beta*c*(t - t_launch)
                t_since_launch = ti.max(0.0, event_time - jet_delay)
                curr_len = 14.0e3 + beta_jet * c_light * t_since_launch

                z_start = dir_sign * 14.0e3
                z_end = dir_sign * curr_len

                layer = float(i % 3)
                layer_angle = opening_angle_rad * (0.4 + 0.3 * layer)

                x_start = 14.0e3 * layer_angle * ti.cos(phi)
                y_start = 14.0e3 * layer_angle * ti.sin(phi)

                x_end = curr_len * layer_angle * ti.cos(phi)
                y_end = curr_len * layer_angle * ti.sin(phi)

                self.jet_vertices[2 * i]     = ti.Vector([x_start, y_start, z_start])
                self.jet_vertices[2 * i + 1] = ti.Vector([x_end, y_end, z_end])

                # Outward relativistic pulse phase along jet frustum
                pulse = 0.5 + 0.5 * ti.sin(event_time * 25.0 - float(i) * 0.5)
                core_intensity = intensity * (0.8 + 0.6 * pulse)

                col = ti.Vector([0.0, 0.0, 0.0])
                if layer == 0.0:
                    # Ultra-relativistic core: brilliant electric cyan-blue
                    col = ti.Vector([0.4 * core_intensity, 1.1 * core_intensity, 1.6 * core_intensity])
                elif layer == 1.0:
                    # Shear layer: vivid purple-violet
                    col = ti.Vector([0.8 * core_intensity, 0.5 * core_intensity, 1.2 * core_intensity])
                else:
                    # Outer envelope: deep magenta
                    col = ti.Vector([1.1 * core_intensity, 0.3 * core_intensity, 0.8 * core_intensity])

                self.jet_colors[2 * i]     = col
                self.jet_colors[2 * i + 1] = col

    def update_jet_geometry(
        self,
        is_active: bool,
        jet_length_m: float = 400.0e3,
        opening_angle_rad: float = 0.08,
        intensity: float = 1.0,
        event_time: float = 0.0,
        jet_progress: float = 1.0,
        jet_delay: float = 1.7,
        beta_jet: float = 0.9999
    ):
        """Update dynamic relativistic jet line vertices and emissive intensity on GPU."""
        self.jet_active = is_active
        flag = 1 if is_active else 0
        self.update_dynamic_jet_kernel(
            flag, float(opening_angle_rad),
            float(intensity), float(event_time), float(jet_progress),
            float(jet_delay), float(beta_jet)
        )

    def update_jet_lines(
        self,
        event_time: float = 0.0,
        jet_progress: float = 1.0,
        is_active: bool = True,
        jet_delay: float = 1.7,
        beta_jet: float = 0.9999
    ):
        """Alias for update_jet_geometry to maintain backward compatibility."""
        self.update_jet_geometry(
            is_active=is_active,
            event_time=event_time,
            jet_progress=jet_progress,
            jet_delay=jet_delay,
            beta_jet=beta_jet
        )

    @ti.kernel
    def update_chirp_spectrogram_kernel(
        self,
        n_pts: ti.i32,
        x_start: ti.f32,
        w_panel: ti.f32,
        y_bottom: ti.f32,
        h_panel: ti.f32,
        f_current: ti.f32,
        is_inspiral: ti.i32
    ):
        """
        Draw time-frequency chirp track line showing f_gw(t) sweeping upward toward merger during inspiral.
        """
        for i in range(n_pts - 1):
            if is_inspiral == 0:
                self.chirp_vertices[2 * i]     = ti.Vector([0.0, 0.0])
                self.chirp_vertices[2 * i + 1] = ti.Vector([0.0, 0.0])
            else:
                t0 = float(i) / float(n_pts - 1)
                t1 = float(i + 1) / float(n_pts - 1)

                x0 = x_start + t0 * w_panel
                x1 = x_start + t1 * w_panel

                f0_norm = ti.min(1.0, 0.15 + 0.85 * ti.pow(t0, 2.5) * (f_current / 1500.0))
                f1_norm = ti.min(1.0, 0.15 + 0.85 * ti.pow(t1, 2.5) * (f_current / 1500.0))

                y0 = y_bottom + f0_norm * h_panel
                y1 = y_bottom + f1_norm * h_panel

                self.chirp_vertices[2 * i]     = ti.Vector([x0, y0])
                self.chirp_vertices[2 * i + 1] = ti.Vector([x1, y1])

    @ti.kernel
    def update_waveform_envelope_kernel(
        self,
        n_pts: ti.i32,
        x_start: ti.f32,
        w_panel: ti.f32,
        y_center: ti.f32,
        h_scale: ti.f32,
        is_inspiral: ti.i32
    ):
        """
        Draw decimated strain amplitude envelope h(t) polyline bounds on GPU during inspiral/merger.
        """
        for i in range(n_pts - 1):
            if is_inspiral == 0:
                self.waveform_vertices[2 * i]     = ti.Vector([0.0, 0.0])
                self.waveform_vertices[2 * i + 1] = ti.Vector([0.0, 0.0])
            else:
                x0 = x_start + (float(i) / float(n_pts - 1)) * w_panel
                x1 = x_start + (float(i + 1) / float(n_pts - 1)) * w_panel

                h0 = self.h_buf_gpu[i]
                h1 = self.h_buf_gpu[i + 1]

                y0 = y_center + ti.max(-0.06, ti.min(0.06, h0 * h_scale))
                y1 = y_center + ti.max(-0.06, ti.min(0.06, h1 * h_scale))

                self.waveform_vertices[2 * i]     = ti.Vector([x0, y0])
                self.waveform_vertices[2 * i + 1] = ti.Vector([x1, y1])

    def update_waveform_buffer(self, waveform_data: np.ndarray, f_gw: float = 72.4, event_time: float = -5.0):
        """Update waveform strain envelope and chirp track lines from waveform array and f_gw."""
        if waveform_data is None or len(waveform_data) == 0:
            return
        n = min(len(waveform_data), self.n_wave_samples)
        data_slice = waveform_data[-n:]

        max_amp = float(np.max(np.abs(data_slice))) if np.max(np.abs(data_slice)) > 0 else 1.0e-21
        scale_factor = 0.05 / max(max_amp, 1.0e-23)

        padded = np.zeros(self.n_wave_samples, dtype=np.float32)
        padded[-n:] = data_slice.astype(np.float32)
        self.h_buf_gpu.from_numpy(padded)

        is_insp = 1 if event_time <= 0.0 else 0

        self.update_waveform_envelope_kernel(
            self.n_wave_samples,
            0.66,   # x_start
            0.32,   # w_panel
            0.75,   # y_center
            scale_factor,
            is_insp
        )

        self.update_chirp_spectrogram_kernel(
            self.n_chirp_pts,
            0.66,   # x_start
            0.32,   # w_panel
            0.82,   # y_bottom
            0.12,   # h_panel
            float(f_gw),
            is_insp
        )

    @ti.kernel
    def pack_combined_post_merger_kernel(self, n_rem: ti.i32, n_disk: ti.i32, n_ej: ti.i32):
        for i in range(n_rem):
            self.combined_post_merger_pos[i] = self.remnant_pos[i]
            self.combined_post_merger_colors[i] = self.remnant_colors[i]
        for i in range(n_disk):
            self.combined_post_merger_pos[n_rem + i] = self.disk_pos[i]
            self.combined_post_merger_colors[n_rem + i] = self.disk_colors[i]
        for i in range(n_ej):
            self.combined_post_merger_pos[n_rem + n_disk + i] = self.ejecta_fluid_pos[i]
            self.combined_post_merger_colors[n_rem + n_disk + i] = self.ejecta_fluid_colors[i]

    def update_combined_post_merger(self):
        """Pack remnant, disk, and ejecta fluid particles into a single GPU field."""
        self.pack_combined_post_merger_kernel(
            self.n_remnant_particles,
            self.n_disk_particles,
            self.n_ejecta_fluid_particles
        )

    @ti.kernel
    def pack_combined_lines_kernel(
        self,
        b_line_verts: ti.template(),
        b_line_cols: ti.template(),
        n_b_verts: ti.i32,
        n_j_verts: ti.i32
    ):
        for i in range(n_b_verts):
            self.combined_line_vertices[i] = b_line_verts[i]
            self.combined_line_colors[i] = b_line_cols[i]
        for i in range(n_j_verts):
            self.combined_line_vertices[n_b_verts + i] = self.jet_vertices[i]
            self.combined_line_colors[n_b_verts + i] = self.jet_colors[i]

    def update_combined_lines(self, b_line_verts, b_line_cols, n_b_verts):
        """Pack magnetic field lines and jet lines into a single GPU line field."""
        self.pack_combined_lines_kernel(
            b_line_verts,
            b_line_cols,
            n_b_verts,
            self.n_jet_vertices
        )
