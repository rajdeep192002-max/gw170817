"""
Taichi GGUI GPU Particle, 2D Starfield Image Lensing, Jet, Chirp & Waveform Renderer for GW170817.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Renders preallocated GPU particle fields, 2D background image lensing warp,
time-frequency chirp spectrogram track, decimated strain waveform envelope,
and structured relativistic jet outflows.
Does NOT perform per-frame CPU particle copies or alter underlying physics equations.
"""
from typing import Optional, Tuple, Dict
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

    VISUAL_EJECTA_SAMPLE_COUNTS = {
        "DEV": 6_000,
        "NORMAL": 16_000,
        "HIGH": 32_000,
    }
    THERMAL_EJECTA_SAMPLE_DIVISOR = 4

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

        # 2D Observational GW170817 strain fields (H1 & L1)
        self.obs_h1_vertices = ti.Vector.field(2, dtype=ti.f32, shape=self.n_wave_vertices)
        self.obs_l1_vertices = ti.Vector.field(2, dtype=ti.f32, shape=self.n_wave_vertices)
        self.obs_h1_buf_gpu = ti.field(dtype=ti.f32, shape=self.n_wave_samples)
        self.obs_l1_buf_gpu = ti.field(dtype=ti.f32, shape=self.n_wave_samples)

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


        # GPU Background Star Field (allocates based on quality mode: DEV 2000, NORMAL 3500, HIGH 7000)
        mode = psys.config.mode.upper()
        self.VISUAL_STAR_COUNTS = {"DEV": 2000, "NORMAL": 3500, "HIGH": 7000}
        self.n_stars = self.VISUAL_STAR_COUNTS.get(mode, self.VISUAL_STAR_COUNTS["DEV"])
        self.star_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_stars)
        self.star_deflected_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_stars)
        self.star_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_stars)
        self.star_brightness = ti.field(dtype=ti.f32, shape=self.n_stars)
        self.starfield_debug = False

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

        # GPU-only granular ejecta samples.  These are deliberately separate
        # from the physical ParticleSystem resolution and are initialized once.
        mode = psys.config.mode.upper()
        self.n_ejecta_fluid_particles = self.VISUAL_EJECTA_SAMPLE_COUNTS.get(
            mode, self.VISUAL_EJECTA_SAMPLE_COUNTS["DEV"]
        )
        self.ejecta_fluid_dir = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_comp = ti.field(dtype=ti.i32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_radial_fill = ti.field(dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_brightness = ti.field(dtype=ti.f32, shape=self.n_ejecta_fluid_particles)
        self.ejecta_fluid_radius = ti.field(dtype=ti.f32, shape=self.n_ejecta_fluid_particles)

        # A lightweight, GPU-only thermal-emission subset reuses the granular
        # ejecta morphology. It is a visual proxy, not a second simulation.
        self.n_ejecta_thermal_particles = self.n_ejecta_fluid_particles // self.THERMAL_EJECTA_SAMPLE_DIVISOR
        self.ejecta_thermal_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_thermal_particles)
        self.ejecta_thermal_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ejecta_thermal_particles)

        # Preallocated combined post-merger particle fields.
        self.n_combined_post_merger = (
            self.n_remnant_particles + self.n_disk_particles + self.n_ejecta_fluid_particles
            + self.n_ejecta_thermal_particles
        )
        self.combined_post_merger_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_post_merger)
        self.combined_post_merger_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_post_merger)

        # Preallocated combined lines field (600 magnetic lines + 128 jet lines = 728 vertices)
        self.n_combined_lines = 600 + self.n_jet_vertices
        self.combined_line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_lines)
        self.combined_line_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_combined_lines)

        # Preallocated 3D GW propagation wavefront line fields (7000 vertices)
        self.n_gw_wave_vertices = 7000
        self.gw_wave_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_gw_wave_vertices)
        self.gw_wave_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_gw_wave_vertices)

        # Cached waveform buffers to eliminate redundant .from_numpy uploads
        self._last_uploaded_h_buf: Optional[np.ndarray] = None
        self._last_uploaded_obs_h1: Optional[np.ndarray] = None
        self._last_uploaded_obs_l1: Optional[np.ndarray] = None

        # Initialize GPU Starfield, Central Remnant, Accretion Disk, Neutrinos, and Ejecta Fluid
        self._init_starfield(self.n_stars, seed=int(psys.config.seed))
        self._init_remnant_particles_kernel(self.n_remnant_particles)
        self._init_disk_particles_kernel(self.n_disk_particles)
        self._init_nu_particles_kernel(self.n_nu_particles)
        self._init_ejecta_fluid_kernel(self.n_ejecta_fluid_particles, int(psys.config.seed))
        self._init_ejecta_thermal_emission_kernel(self.n_ejecta_thermal_particles)

    @ti.kernel
    def _init_ejecta_fluid_kernel(self, n_ej: ti.i32, seed: ti.i32):
        """
        Deterministically seed granular multi-component ejecta samples on GPU.
        - Polar Blue component (lanthanide-poor, fast, polar cone)
        - Purple Intermediate component
        - Equatorial Red component (lanthanide-rich, slower, equatorial torus)
        """
        two_pi = 6.283185307179586
        for i in range(n_ej):
            # Index/seed hashes are stable across renderer construction and do
            # not depend on Taichi's global random-number stream.
            h0 = ti.sin(float(i + 17 * seed) * 12.9898) * 43758.5453
            h1 = ti.sin(float(i + 31 * seed) * 78.233) * 24634.6345
            h2 = ti.sin(float(i + 47 * seed) * 39.425) * 13579.1357
            u = h0 - ti.floor(h0)
            v = h1 - ti.floor(h1)
            w = h2 - ti.floor(h2)
            cos_th = 1.0 - 2.0 * u
            sin_th = ti.sqrt(ti.max(0.0, 1.0 - cos_th * cos_th))
            phi = v * two_pi

            dir_vec = ti.Vector([sin_th * ti.cos(phi), sin_th * ti.sin(phi), cos_th])
            self.ejecta_fluid_dir[i] = dir_vec
            self.ejecta_fluid_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
            self.ejecta_fluid_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            self.ejecta_fluid_radial_fill[i] = 0.32 + 0.68 * w
            self.ejecta_fluid_brightness[i] = 0.60 + 0.55 * (u * 0.45 + v * 0.55)
            # GGUI currently accepts one radius per draw call.  This metadata
            # preserves deterministic granular size variation for a future
            # per-sample-radius backend without changing the consolidated draw.
            self.ejecta_fluid_radius[i] = 0.45e3 + 0.35e3 * w

            abs_cos = ti.abs(cos_th)
            if abs_cos >= 0.65:
                self.ejecta_fluid_comp[i] = 0  # Blue polar
            elif abs_cos >= 0.35:
                self.ejecta_fluid_comp[i] = 1  # Purple intermediate
            else:
                self.ejecta_fluid_comp[i] = 2  # Red equatorial

    @ti.kernel
    def _init_ejecta_thermal_emission_kernel(self, n_thermal: ti.i32):
        """Hide preallocated thermal samples until the existing ejecta gate opens."""
        for i in range(n_thermal):
            self.ejecta_thermal_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
            self.ejecta_thermal_colors[i] = ti.Vector([0.0, 0.0, 0.0])

    def _init_starfield(self, n_stars: int, seed: int = 42):
        """
        Initialize 3D astronomical starfield deterministically from StellarCatalog.
        Implements natural tiered stellar brightness distribution:
        - Majority (~81%): small, clearly visible stars (brightness 0.85 - 1.37)
        - Minority (~15.5%): brighter reference stars (brightness 1.23 - 1.97)
        - A few (~3.5%): distinctly bright beacon stars (brightness 1.87 - 2.88)
        Beacon and reference stars are spatially distributed across the full sky.
        """
        from gw170817.visualization.background import StellarCatalog
        catalog = StellarCatalog(n_stars=n_stars, seed=seed)
        pos_np = np.array([s.position_3d for s in catalog.stars], dtype=np.float32)
        colors_np = np.array([s.color_rgb for s in catalog.stars], dtype=np.float32)
        mags_np = np.array([s.magnitude for s in catalog.stars], dtype=np.float32)

        # 1. Natural astronomical magnitude-to-brightness scaling
        u_mag = np.clip((11.0 - mags_np) / 9.0, 0.0, 1.0)
        base_b = (0.85 + 0.65 * (u_mag ** 1.5)).astype(np.float32)

        # 2. Deterministic spatial reference and beacon star selection
        n = len(mags_np)
        indices = np.arange(n)
        is_beacon = ((indices * 19 + 23) % 28 == 0)
        is_ref = ((indices * 7 + 13) % 6 == 0) & (~is_beacon)
        is_regular = (~is_beacon) & (~is_ref)

        mult = np.ones(n, dtype=np.float32)
        mult[is_regular] = 1.0
        mult[is_ref] = 1.45
        mult[is_beacon] = 2.20

        star_brightness_np = (base_b * mult).astype(np.float32)
        eff_colors_np = (colors_np * star_brightness_np[:, None]).astype(np.float32)

        self._star_base_colors_np = eff_colors_np.copy()
        self._current_star_mode = None

        self.star_pos.from_numpy(pos_np)
        self.star_deflected_pos.from_numpy(pos_np)
        self.star_brightness.from_numpy(star_brightness_np)
        self.update_star_brightness_mode("CORE")

    def update_star_brightness_mode(self, view_mode: str):
        """
        Mode-specific star brightness scaling for gravitational-lensing readability:
        - CORE: moderately bright (1.05x)
        - GW MODE: subdued (0.75x) to maintain waveform dominance
        - MAGNETIC FIELD: moderate (0.85x) to maintain field line dominance
        - BH LENS: brightest visibility (1.45x) so distorted stars stand out around the hole
        - MULTI: moderate cinematic background (1.00x)
        """
        if getattr(self, "starfield_debug", False):
            return
        if getattr(self, "_current_star_mode", None) == view_mode:
            return
        if not hasattr(self, "_star_base_colors_np") or self._star_base_colors_np is None:
            return

        self._current_star_mode = view_mode
        mode_multipliers = {
            "CORE": 1.05,
            "GW MODE": 0.75,
            "MAGNETIC FIELD": 0.85,
            "BH LENS": 1.45,
            "MULTI": 1.00,
        }
        scale = mode_multipliers.get(view_mode, 1.0)
        eff_colors = (self._star_base_colors_np * scale).astype(np.float32)
        self.star_colors.from_numpy(eff_colors)

    def toggle_starfield_debug(self) -> bool:
        """Toggle starfield debug diagnostic mode (64 bright grid markers vs catalog)."""
        self.starfield_debug = not self.starfield_debug
        if self.starfield_debug:
            pts = []
            cols = []
            rng = np.random.RandomState(999)
            phi_grid = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            lat_grid = np.linspace(-np.pi * 0.4, np.pi * 0.4, 4)
            for lat in lat_grid:
                for phi in phi_grid:
                    r = 1.2e6 + float(rng.uniform(-0.2e6, 0.8e6))
                    x = r * np.cos(lat) * np.cos(phi)
                    y = r * np.cos(lat) * np.sin(phi)
                    z = r * np.sin(lat)
                    pts.append([x, y, z])
                    cols.append([0.2, 0.95, 1.0])  # Electric cyan diagnostic test markers

            pts_np = np.zeros((self.n_stars, 3), dtype=np.float32)
            cols_np = np.zeros((self.n_stars, 3), dtype=np.float32)
            n_pts = min(len(pts), self.n_stars)
            pts_np[:n_pts] = np.array(pts[:n_pts], dtype=np.float32)
            cols_np[:n_pts] = np.array(cols[:n_pts], dtype=np.float32)

            self.star_pos.from_numpy(pts_np)
            self.star_deflected_pos.from_numpy(pts_np)
            self.star_colors.from_numpy(cols_np)
        else:
            self._init_starfield(self.n_stars, seed=int(self.psys.config.seed))
        return self.starfield_debug

    def update_star_lensing_deflection_kernel(
        self,
        ns1_x: float, ns1_y: float, ns1_z: float,
        ns2_x: float, ns2_y: float, ns2_z: float,
        m1_kg: float,
        m2_kg: float,
        lensing_active: int,
        enhanced_scale: float,
        cam_x: float = 0.0,
        cam_y: float = -280.0e3,
        cam_z: float = 180.0e3,
        is_black_hole: int = 0,
        remnant_mass: float = 5.45e30
    ):
        """
        Python wrapper method for GPU Gravitational Lensing Kernel.
        Supports both pre-merger binary lensing and post-merger camera-dependent Schwarzschild BH lensing.
        """
        self._update_star_lensing_deflection_gpu_kernel(
            float(ns1_x), float(ns1_y), float(ns1_z),
            float(ns2_x), float(ns2_y), float(ns2_z),
            float(m1_kg), float(m2_kg),
            int(lensing_active), float(enhanced_scale),
            float(cam_x), float(cam_y), float(cam_z),
            int(is_black_hole), float(remnant_mass)
        )

    @ti.kernel
    def _update_star_lensing_deflection_gpu_kernel(
        self,
        ns1_x: ti.f32, ns1_y: ti.f32, ns1_z: ti.f32,
        ns2_x: ti.f32, ns2_y: ti.f32, ns2_z: ti.f32,
        m1_kg: ti.f32,
        m2_kg: ti.f32,
        lensing_active: ti.i32,
        enhanced_scale: ti.f32,
        cam_x: ti.f32,
        cam_y: ti.f32,
        cam_z: ti.f32,
        is_black_hole: ti.i32,
        remnant_mass: ti.f32
    ):
        """
        Taichi GPU parallel Gravitational Lensing Kernel.
        - LENSING OFF (lensing_active == 0): Undeflected background.
        - PRE-MERGER (is_black_hole == 0): Dual-lens weak-field BNS deflection.
        - POST-MERGER (is_black_hole == 1): Schwarzschild BH ray tracing with photon-sphere capture shadow (b <= b_crit) and Einstein ring glow.
        """
        c2 = 8.98755e16
        G_val = 6.6743e-11

        cam_pos = ti.Vector([cam_x, cam_y, cam_z])

        # Critical photon-sphere capture impact parameter for Schwarzschild BH: b_crit = (3*sqrt(3)/2) * r_s
        rs_bh = (2.0 * G_val * remnant_mass) / c2
        bcrit_bh = 2.598076211353316 * rs_bh

        # Update 3D Astronomical Starfield Positions
        for k in range(self.n_stars):
            orig_p = self.star_pos[k]
            if lensing_active == 0:
                self.star_deflected_pos[k] = orig_p
            else:
                if is_black_hole == 0:
                    p1 = ti.Vector([ns1_x, ns1_y, ns1_z])
                    p2 = ti.Vector([ns2_x, ns2_y, ns2_z])

                    ray_dir = orig_p - cam_pos
                    ray_dist = ti.max(1.0, ti.sqrt(ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2]))
                    ray_hat = ray_dir / ray_dist

                    # Sightline projections from camera to NS1 and NS2 mass centers
                    v_cam1 = p1 - cam_pos
                    v_cam2 = p2 - cam_pos
                    t1 = v_cam1.dot(ray_hat)
                    t2 = v_cam2.dot(ray_hat)

                    # Transverse displacement vectors from sightline ray to NS centers
                    perp1 = v_cam1 - t1 * ray_hat
                    perp2 = v_cam2 - t2 * ray_hat

                    b1_raw = ti.sqrt(perp1[0]*perp1[0] + perp1[1]*perp1[1] + perp1[2]*perp1[2])
                    b2_raw = ti.sqrt(perp2[0]*perp2[0] + perp2[1]*perp2[1] + perp2[2]*perp2[2])

                    b1 = ti.max(12.0e3, b1_raw)
                    b2 = ti.max(12.0e3, b2_raw)

                    # Schwarzschild radii: r_s = 2 G M / c^2
                    rs1 = (2.0 * G_val * m1_kg) / c2
                    rs2 = (2.0 * G_val * m2_kg) / c2

                    # Radial Gaussian distance attenuation: lensing drops off smoothly past ~120 km impact parameter
                    atten1 = ti.exp(- (b1 / 120.0e3) * (b1 / 120.0e3)) if t1 > 0.0 else 0.0
                    atten2 = ti.exp(- (b2 / 120.0e3) * (b2 / 120.0e3)) if t2 > 0.0 else 0.0

                    deflect1_mag = (2.0 * rs1 / b1) * 3.5e5 * enhanced_scale * atten1
                    deflect2_mag = (2.0 * rs2 / b2) * 3.5e5 * enhanced_scale * atten2

                    # Screen-space apparent displacement direction (outward away from NS center in plane perp to sightline)
                    n_out1 = ti.math.normalize(-perp1) if b1_raw > 1.0 else ti.Vector([0.0, 0.0, 0.0])
                    n_out2 = ti.math.normalize(-perp2) if b2_raw > 1.0 else ti.Vector([0.0, 0.0, 0.0])

                    # Deflected sightline direction
                    deflected_ray_hat = ti.math.normalize(ray_hat + (n_out1 * deflect1_mag + n_out2 * deflect2_mag) / ray_dist)
                    self.star_deflected_pos[k] = cam_pos + ray_dist * deflected_ray_hat
                else:
                    ray_dir = orig_p - cam_pos
                    ray_dist = ti.max(1.0, ti.sqrt(ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2]))
                    ray_hat = ray_dir / ray_dist

                    t_proj = -cam_pos.dot(ray_hat)
                    if t_proj > 0.0:
                        perp_vec = cam_pos + t_proj * ray_hat
                        b_imp = ti.sqrt(perp_vec[0]*perp_vec[0] + perp_vec[1]*perp_vec[1] + perp_vec[2]*perp_vec[2])

                        if b_imp <= bcrit_bh:
                            self.star_deflected_pos[k] = ti.Vector([0.0, 0.0, -1.0e9])
                        else:
                            deflect_angle = (4.0 * G_val * remnant_mass / (c2 * b_imp)) * 4.5e5 * enhanced_scale
                            n_to_bh = ti.math.normalize(-perp_vec) if b_imp > 1.0 else ti.Vector([0.0, 0.0, 0.0])
                            deflected_ray_hat = ti.math.normalize(ray_hat + deflect_angle * n_to_bh)
                            self.star_deflected_pos[k] = cam_pos + ray_dist * deflected_ray_hat
                    else:
                        self.star_deflected_pos[k] = orig_p


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

    @staticmethod
    def get_visual_time_adapter(t_event: float, t_scale: float = 0.0004) -> float:
        """
        Monotonic, smooth renderer-only visual time adapter.
        Maps astronomical physical event time [0, 200+ days] into bounded visual time [0, ~0.01s],
        keeping ejecta and jet geometry inside camera z_far (3,500 km).
        Preserves exact early launch velocity (derivative at t=0 is 1.0).
        """
        if t_event <= 0.0:
            return 0.0
        return float(t_scale * np.log(1.0 + t_event / t_scale))

    @staticmethod
    def compute_merger_flash_intensity(event_time: float, contact_frac: float) -> float:
        """
        Reduced-order temporal envelope for the transient GW170817 merger flash.
        Centered on merger contact (t_event = 0.0 s) with a smooth Gaussian profile F(t) = exp(-(t / sigma)^2).
        sigma = 0.035 s provides a rapid pre-merger rise and post-merger thermal decay (~0.10 s duration).
        """
        if -0.05 <= event_time <= 0.12 or (contact_frac > 0.05 and event_time < 0.12):
            sigma = 0.035
            t_eff = event_time if event_time >= 0.0 else 0.0
            if contact_frac > 0.05 and event_time < 0.0:
                t_eff = 0.0
            val = float(np.exp(- (t_eff / sigma) ** 2))
            return float(np.clip(val, 0.0, 1.0))
        return 0.0

    @ti.kernel
    def update_remnant_particles_kernel(
        self,
        phi_wrapped: ti.f32,
        pulse_wrapped: ti.f32,
        contact_frac: ti.f32,
        remnant_type: ti.i32,
        horizon_radius: ti.f32,
        flash_intensity: ti.f32
    ):
        """
        Animate central remnant compact object continuously rotating on GPU with float32 wrapped phase.
        - HMNS: brilliant white-gold incandescent core (R ~ 14 km) with transient merger flash peak.
        - BH (Adopted Delayed Collapse): compact accretion horizon proxy (R ~ R_horizon ~ 8 km)
          with a dark central horizon shadow surrounded by an electric cyan-blue accretion edge.
        """
        for i in range(self.n_remnant_particles):
            if contact_frac <= 0.001:
                self.remnant_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
                self.remnant_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                lp = self.remnant_local_pos[i]
                phi = phi_wrapped

                cos_p = ti.cos(phi)
                sin_p = ti.sin(phi)

                # Horizon radius scale factor (r_eff / 14 km)
                r_scale = ti.min(1.2, ti.max(0.35, horizon_radius / 14.0e3)) if remnant_type == 0 else 1.0

                rx = (lp[0] * cos_p - lp[1] * sin_p) * r_scale
                ry = (lp[0] * sin_p + lp[1] * cos_p) * r_scale
                rz = lp[2] * r_scale

                self.remnant_pos[i] = ti.Vector([rx, ry, rz])

                pulse = 0.8 + 0.2 * ti.sin(pulse_wrapped + float(i))
                if remnant_type == 1:
                    # HMNS: brilliant white-gold incandescent core with transient merger flash peak
                    r_core = ti.min(1.0, 1.0 * pulse + 0.35 * flash_intensity)
                    g_core = ti.min(1.0, 0.95 * pulse + 0.25 * flash_intensity)
                    b_core = ti.min(1.0, 0.70 * pulse + 0.30 * flash_intensity)
                    self.remnant_colors[i] = ti.Vector([r_core, g_core, b_core])
                else:
                    # BH: compact dark central horizon shadow proxy with subtle redshifted boundary
                    r_local_sq = lp[0]*lp[0] + lp[1]*lp[1] + lp[2]*lp[2]
                    shadow_val = ti.min(1.0, 0.002 + 0.95 * flash_intensity)
                    if r_local_sq < 30.0e6:
                        # Pitch dark central horizon shadow
                        self.remnant_colors[i] = ti.Vector([shadow_val, shadow_val, shadow_val])
                    else:
                        # Faint redshifted horizon boundary (not bright blue)
                        r_edge = ti.min(1.0, 0.08 * pulse + 0.85 * flash_intensity)
                        g_edge = ti.min(1.0, 0.04 * pulse + 0.65 * flash_intensity)
                        b_edge = ti.min(1.0, 0.02 * pulse + 0.40 * flash_intensity)
                        self.remnant_colors[i] = ti.Vector([r_edge, g_edge, b_edge])

    def update_remnant_particles(
        self,
        event_time: float,
        contact_frac: float = 1.0,
        remnant_type_str: str = "HMNS",
        horizon_radius_m: float = 14.0e3
    ):
        """Update 3D central remnant compact object particles on GPU."""
        r_type = 1 if remnant_type_str == "HMNS" else 0
        phi_wrapped = float((120.0 * event_time) % (2.0 * np.pi)) if event_time >= 0.0 else 0.0
        pulse_wrapped = float((30.0 * event_time) % (2.0 * np.pi)) if event_time >= 0.0 else 0.0
        flash_intensity = self.compute_merger_flash_intensity(event_time, contact_frac)
        self.flash_intensity = flash_intensity
        self.update_remnant_particles_kernel(
            phi_wrapped, pulse_wrapped, float(contact_frac), r_type, float(horizon_radius_m),
            float(flash_intensity)
        )

    @ti.kernel
    def update_particle_render_data(
        self,
        pos: ti.template(),
        vel: ti.template(),
        star_id: ti.template(),
        active: ti.template(),
        ye_field: ti.template(),
        n_total: ti.i32,
        contact_frac: ti.f32,
        ejecta_progress: ti.f32,
        separation: ti.f32,
        event_time: ti.f32,
        cam_x: ti.f32,
        cam_y: ti.f32,
        cam_z: ti.f32,
        has_vel: ti.i32
    ):
        """
        Dynamically update particle render positions and colors on GPU.
        - Inspiral phase: renders NS1 (electric cyan/blue) and NS2 (warm gold) particle clouds
          with separation-driven dynamic thermal/incandescent surface luminosity ramp and
          directional relativistic Doppler boosting/beaming asymmetry.
        - Post-merger phase: culls core/bound particles (r < 150 km) by placing render_pos at [0, 0, -1e9],
          completely eliminating black particle occlusion and fragment overdraw around remnant/torus/jet.
        """
        # Bounded separation range: 50 km (early inspiral baseline) -> 24 km (contact incandescence)
        sep_clamped = ti.max(24.0e3, ti.min(50.0e3, separation))
        # Normalized separation fraction (0.0 at >= 50 km, 1.0 at <= 24 km)
        x_norm = (50.0e3 - sep_clamped) / (50.0e3 - 24.0e3)
        # Smoothstep cubic S-curve: x^2 * (3 - 2x) for C1 continuous thermal brightness ramp
        lum_ramp = x_norm * x_norm * (3.0 - 2.0 * x_norm) if contact_frac <= 0.05 else 1.0

        # Transient GW170817 merger flash Gaussian envelope F(t) = exp(-(t / sigma)^2)
        # Peak at merger contact (t_event = 0.0s, sigma = 0.035s)
        flash_intensity = 0.0
        if -0.05 <= event_time <= 0.12 or (contact_frac > 0.05 and event_time < 0.12):
            t_eff = event_time if event_time >= 0.0 else 0.0
            if contact_frac > 0.05 and event_time < 0.0:
                t_eff = 0.0
            flash_intensity = ti.exp(- (t_eff / 0.035) * (t_eff / 0.035))

        for i in range(n_total):
            sid = star_id[i]
            act = active[i]
            p = pos[i]
            ye = ye_field[i]

            r_dist = ti.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

            if contact_frac > 0.05 or ejecta_progress > 0.0:
                # Post-merger: cull inspiral BNS particles cleanly
                self.render_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
                self.colors[i] = ti.Vector([0.0, 0.0, 0.0])

            else:
                # Inspiral phase: Luminous, compact neutron star emission with smoothstep thermal ramp, contact flash & relativistic Doppler beaming
                self.render_pos[i] = p
                core_factor = ti.max(0.0, 1.0 - (r_dist / 250.0e3))

                # Relativistic Doppler factor D = 1 / (gamma * (1 - beta_los))
                doppler_boost = 1.0
                if has_vel != 0:
                    v_p = vel[i]
                    c_light = 2.99792458e8
                    beta_v = v_p / c_light
                    speed_beta = ti.min(0.45, beta_v.norm())

                    cam_dir = ti.Vector([cam_x, cam_y, cam_z]) - p
                    cam_dist = ti.max(1.0, cam_dir.norm())
                    n_cam = cam_dir / cam_dist

                    beta_los = ti.max(-0.45, ti.min(0.45, beta_v.dot(n_cam)))
                    gamma_inv = ti.sqrt(ti.max(0.01, 1.0 - speed_beta * speed_beta))
                    doppler_D = ti.max(0.65, ti.min(1.35, gamma_inv / (1.0 - beta_los + 1.0e-5)))

                    # Restrained visual scaling: 30% modulation around neutral 1.0
                    doppler_boost = 1.0 + 0.30 * (doppler_D - 1.0)

                if sid == 0:
                    # NS1 (Electric Cyan/Blue star -> incandescent blue-white at contact/flash)
                    r_base = 0.22 + 0.30 * core_factor + 0.35 * lum_ramp * core_factor
                    g_base = 0.50 + 0.30 * core_factor + 0.15 * lum_ramp
                    b_base = 0.80 + 0.15 * core_factor + 0.05 * lum_ramp

                    r_col = ti.min(1.0, ti.max(0.0, r_base * doppler_boost + 0.35 * flash_intensity * core_factor))
                    g_col = ti.min(1.0, ti.max(0.0, g_base * doppler_boost + 0.30 * flash_intensity))
                    b_col = ti.min(1.0, ti.max(0.0, b_base * doppler_boost + 0.20 * flash_intensity))
                    self.colors[i] = ti.Vector([r_col, g_col, b_col])
                else:
                    # NS2 (Warm Gold/Amber star -> incandescent yellow-white at contact/flash)
                    r_base = 0.80 + 0.18 * core_factor
                    g_base = 0.45 + 0.30 * core_factor + 0.18 * lum_ramp
                    b_base = 0.12 + 0.20 * core_factor + 0.48 * lum_ramp * core_factor

                    r_col = ti.min(1.0, ti.max(0.0, r_base * doppler_boost + 0.20 * flash_intensity))
                    g_col = ti.min(1.0, ti.max(0.0, g_base * doppler_boost + 0.35 * flash_intensity))
                    b_col = ti.min(1.0, ti.max(0.0, b_base * doppler_boost + 0.40 * flash_intensity * core_factor))
                    self.colors[i] = ti.Vector([r_col, g_col, b_col])

    def update_particle_colors(
        self,
        pos: ti.template(),
        star_id: ti.template(),
        active: ti.template(),
        ye_field: ti.template(),
        n_total: int,
        contact_frac: float,
        ejecta_progress: float,
        separation: float = 45.0e3,
        event_time: float = -1.0,
        vel: ti.template() = None,
        cam_x: float = 0.0,
        cam_y: float = -280.0e3,
        cam_z: float = 180.0e3
    ):
        """Update particle render positions and colors on GPU with relativistic Doppler beaming."""
        has_v = 1 if vel is not None else 0
        v_arg = vel if vel is not None else pos
        self.update_particle_render_data(
            pos, v_arg, star_id, active, ye_field, n_total,
            float(contact_frac), float(ejecta_progress),
            float(separation), float(event_time),
            float(cam_x), float(cam_y), float(cam_z),
            int(has_v)
        )

    @ti.kernel
    def _init_disk_particles_kernel(self, n_disk: ti.i32):
        """
        Seed thin rotating accretion disk geometry centered on remnant in equatorial plane (H/R ≈ 0.04).
        """
        two_pi = 6.283185307179586
        R_in = 18.0e3   # Inner disk radius [m]
        R_out = 120.0e3 # Outer disk radius [m]
        H_over_R = 0.04 # Thin equatorial disk aspect ratio

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
        phase_wrapped: ti.f32,
        dt_vis: ti.f32,
        is_active: ti.i32,
        is_bh_accretion: ti.i32,
        disk_mass_frac: ti.f32,
        cam_x: ti.f32,
        cam_y: ti.f32,
        cam_z: ti.f32,
        lens_active: ti.i32,
        intensity_scale: ti.f32,
        remnant_mass_kg: ti.f32
    ):
        """
        Animate thin rotating accretion disk with post-ringdown BH accretion drift, horizon capture recycling,
        restrained relativistic Doppler boosting, and Schwarzschild gravitational light bending into Interstellar lensing arcs.
        """
        two_pi = 6.283185307179586
        c_light = 2.99792458e8
        G_val = 6.6743e-11
        M_rem = remnant_mass_kg

        R_capture = 14.5e3   # Safety capture radius outside 14.0 km BH horizon [m]
        R_in_base = 18.0e3   # Baseline inner disk radius [m]
        R_out = 120.0e3      # Outer disk radius [m]
        H_over_R = 0.04      # Thin equatorial disk aspect ratio

        cam_pos = ti.Vector([cam_x, cam_y, cam_z])
        cam_dist = ti.max(1.0, cam_pos.norm())

        # Schwarzschild radius r_s = 2 G M / c^2, b_crit = (3*sqrt(3)/2) * r_s
        rs_bh = (2.0 * G_val * M_rem) / (c_light * c_light)
        bcrit_bh = 2.598076211353316 * rs_bh

        for i in range(self.n_disk_particles):
            if is_active == 0 or disk_mass_frac <= 0.0:
                self.disk_pos[i] = ti.Vector([0.0, 0.0, 0.0])
                self.disk_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                lp = self.disk_local_pos[i]
                r = lp[0]
                phi0 = lp[1]
                z0 = lp[2]

                if is_bh_accretion == 1 and dt_vis > 0.0:
                    # Inward radial accretion drift: v_r(r) increases toward inner radius
                    v_r = 1500.0 * ti.pow(R_out / ti.max(R_capture, r), 1.2)
                    r = r - v_r * dt_vis

                    # Recycled at outer disk if particle reaches capture radius R_capture
                    if r <= R_capture:
                        u_r = ti.random(ti.f32)
                        u_phi = ti.random(ti.f32)
                        u_z = ti.random(ti.f32)

                        r = R_out - u_r * 30.0e3
                        phi0 = u_phi * two_pi
                        h_max = r * H_over_R
                        z0 = (u_z - 0.5) * 2.0 * h_max

                    # Save updated position, relative radius, and Keplerian angular velocity
                    self.disk_local_pos[i] = ti.Vector([r, phi0, z0])
                    self.disk_radius_rel[i] = ti.max(0.0, ti.min(1.0, (r - R_in_base) / (R_out - R_in_base)))
                    self.disk_omega[i] = ti.sqrt(G_val * M_rem / (r * r * r))

                # Azimuthal orbital motion: phi(t) = (phi0 + Omega_K * t_vis) mod 2pi
                om = self.disk_omega[i]
                phi = (phi0 + om * phase_wrapped) % two_pi

                x = r * ti.cos(phi)
                y = r * ti.sin(phi)
                z = z0
                orig_pos = ti.Vector([x, y, z])

                # Relativistic Keplerian velocity v_orb = sqrt(G M / r)
                v_orb_mag = ti.sqrt(G_val * M_rem / ti.max(1.0e3, r))
                beta_orb = ti.min(0.42, v_orb_mag / c_light)

                v_dir = ti.Vector([-ti.sin(phi), ti.cos(phi), 0.0])
                cam_ray_dir = cam_pos - orig_pos
                ray_len = ti.max(1.0, cam_ray_dir.norm())
                n_cam_ray = cam_ray_dir / ray_len

                # Line-of-sight velocity fraction beta_los = (v_dir . n_cam_ray) * beta_orb
                beta_los = ti.max(-0.35, ti.min(0.35, beta_orb * (v_dir.dot(n_cam_ray))))

                # Relativistic Doppler beaming factor: D = 1 / (gamma * (1 - beta_los))
                gamma_inv = ti.sqrt(ti.max(0.01, 1.0 - beta_orb * beta_orb))
                doppler_boost = ti.max(0.50, ti.min(2.20, gamma_inv / ti.max(0.1, 1.0 - beta_los)))

                # Gravitational redshift factor: g_grav = sqrt(1 - 2M/r) = sqrt(1 - r_s / r)
                g_grav = ti.sqrt(ti.max(1.0e-4, 1.0 - rs_bh / ti.max(rs_bh, r)))

                # Physically-motivated disk temperature scaling: T(r) ~ (r / R_ISCO)^(-3/4)
                r_isco = 3.0 * rs_bh  # ISCO at r = 6M = 3 r_s
                t_ratio = ti.max(1.0, r / r_isco)
                t_temp = ti.pow(t_ratio, -0.75)  # T ~ r^-3/4

                # Controlled visual emission multiplier for accretion disk readability (visualization only)
                vis_scale = ti.max(0.95, intensity_scale) * (0.60 + 0.40 * g_grav)

                # Base temperature color: inner hot (brilliant cyan/white-gold), outer warm amber-gold
                base_r = ti.min(1.0, 0.88 + 0.35 * t_temp)
                base_g = ti.min(1.0, 0.38 + 0.62 * ti.pow(t_temp, 1.2))
                base_b = ti.min(1.0, 0.08 + 0.92 * ti.pow(t_temp, 2.0))

                r_c = ti.min(1.0, base_r * doppler_boost * vis_scale)
                g_c = ti.min(1.0, base_g * doppler_boost * vis_scale)
                b_c = ti.min(1.0, base_b * doppler_boost * vis_scale)

                self.disk_colors[i] = ti.Vector([r_c, g_c, b_c])

                # Disk particle physical 3D position in equatorial plane
                self.disk_pos[i] = orig_pos

    def update_disk_particles(
        self,
        event_time: float,
        dt_vis: float = 0.016,
        is_active: bool = True,
        disk_progress: float = 1.0,
        disk_mass_msun: float = 0.06,
        is_black_hole: bool = False,
        cam_x: float = 0.0,
        cam_y: float = -280.0e3,
        cam_z: float = 180.0e3,
        lensing_enabled: bool = True,
        intensity_scale: float = 1.0,
        remnant_mass_kg: float = 2.7 * 1.989e30
    ):
        """Update 3D thin accretion disk particles on GPU with post-ringdown accretion drift, Doppler beaming, and Schwarzschild lensing arcs."""
        flag = 1 if is_active else 0
        is_bh_flag = 1 if (is_active and is_black_hole and event_time >= 0.0) else 0
        lens_flag = 1 if (lensing_enabled and is_black_hole) else 0
        if disk_mass_msun > 0.0 and disk_progress > 0.0:
            frac = min(1.5, max(0.001, (disk_mass_msun / 0.06) * float(disk_progress)))
        else:
            frac = 0.0
        phase_wrapped = float((0.05 * max(0.0, event_time)) % (2.0 * np.pi)) if event_time >= 0.0 else 0.0
        dt_v = max(0.0, float(dt_vis))
        self.update_disk_particles_kernel(
            phase_wrapped, dt_v, flag, is_bh_flag, float(frac),
            float(cam_x), float(cam_y), float(cam_z), lens_flag,
            float(intensity_scale), float(remnant_mass_kg)
        )

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
        pulse_wrapped: ti.f32,
        nu_lum_norm: ti.f32
    ):
        """
        Animate subtle neutrino wind halo cooling indicator around remnant/disk with wrapped phase.
        """
        for i in range(self.n_nu_particles):
            if event_time < 0.0 or nu_lum_norm <= 0.0:
                self.nu_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                intensity = ti.min(0.6, 0.15 * nu_lum_norm * (1.0 + 0.3 * ti.sin(pulse_wrapped + float(i))))
                self.nu_colors[i] = ti.Vector([0.3 * intensity, 0.15 * intensity, 0.6 * intensity])

    def update_nu_particles(self, event_time: float, nu_luminosity_w: float = 1.0e45):
        """Update neutrino wind halo particles on GPU."""
        norm = min(2.0, max(0.0, nu_luminosity_w / 1.0e45))
        pulse_wrapped = float((15.0 * event_time) % (2.0 * np.pi)) if event_time >= 0.0 else 0.0
        self.update_nu_particles_kernel(float(event_time), pulse_wrapped, float(norm))

    @ti.kernel
    def update_ejecta_fluid_kernel(
        self,
        vis_event_time: ti.f32,
        wobble_wrapped: ti.f32,
        event_time: ti.f32,
        ejecta_progress: ti.f32,
        ejecta_mass_fraction: ti.f32,
        is_active: ti.i32
    ):
        """
        Animate 3D continuous expanding multi-component ejecta plasma volume on GPU.
        Uses visual time adapter vis_event_time to bound radius within camera frustum.
        """
        # Reduced-order component velocities [m/s]
        c_light = 2.998e8
        v_blue   = 0.30 * c_light  # [MOD] blue polar velocity
        v_purple = 0.20 * c_light  # [MOD] intermediate velocity
        v_red    = 0.10 * c_light  # [MOD] red equatorial velocity

        for i in range(self.n_ejecta_fluid_particles):
            if is_active == 0 or event_time < 0.0 or ejecta_progress <= 0.01:
                self.ejecta_fluid_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
                self.ejecta_fluid_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                d = self.ejecta_fluid_dir[i]
                comp = self.ejecta_fluid_comp[i]
                radial_fill = self.ejecta_fluid_radial_fill[i]
                brightness = self.ejecta_fluid_brightness[i]

                # Thermal cooling: temperature falls as t^{-0.3} proxy
                heat = ti.max(0.25, 1.0 - 0.05 * ti.log(1.0 + event_time / 0.1))

                # Smooth initial launch emission buildup (0 to 20 ms physical scale)
                growth = ti.min(1.0, event_time / 0.020) if event_time < 0.020 else 1.0
                progress = ti.min(1.0, ti.max(0.0, ejecta_progress))
                visibility = progress * progress * (3.0 - 2.0 * progress)
                mass_visual = ti.min(1.0, ti.max(0.25, ejecta_mass_fraction * 40.0))
                heat_eff = heat * (0.35 + 0.65 * growth) * visibility * mass_visual * brightness

                # Pulsating fluid oscillation (physical wobble, NOT phase)
                wobble = 1.0 + 0.06 * ti.sin(wobble_wrapped + float(i % 17))

                if comp == 0:
                    # Blue polar plume (fast, v≈0.30c, polar cone)
                    r_shell = (25.0e3 + v_blue * vis_event_time * (0.72 + 0.45 * radial_fill)) * wobble * radial_fill
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
                    r_shell = (22.0e3 + v_purple * vis_event_time * (0.70 + 0.42 * radial_fill)) * wobble * radial_fill
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
                    r_shell = (18.0e3 + v_red * vis_event_time * (0.68 + 0.38 * radial_fill)) * wobble * radial_fill
                    x = d[0] * r_shell * 1.30
                    y = d[1] * r_shell * 1.30
                    z = d[2] * r_shell * 0.55
                    self.ejecta_fluid_pos[i] = ti.Vector([x, y, z])
                    self.ejecta_fluid_colors[i] = ti.Vector([
                        ti.min(1.0, 1.00 * heat_eff),
                        ti.min(1.0, 0.35 * heat_eff),
                        ti.min(1.0, 0.10 * heat_eff)
                    ])

    def update_ejecta_fluid(
        self,
        event_time: float,
        ejecta_progress: float,
        is_active: bool = True,
        ejecta_mass_fraction: float = 0.02,
    ):
        """Update 3D continuous ejecta fluid plasma volume on GPU."""
        flag = 1 if is_active else 0
        vis_event_time = ParticleRenderer.get_visual_time_adapter(event_time, t_scale=0.006)
        wobble_wrapped = float((8.0 * event_time) % (2.0 * np.pi)) if event_time >= 0.0 else 0.0
        self.update_ejecta_fluid_kernel(
            float(vis_event_time), wobble_wrapped, float(event_time), float(ejecta_progress),
            float(ejecta_mass_fraction), flag
        )

    @ti.kernel
    def update_ejecta_thermal_emission_kernel(
        self,
        vis_event_time: ti.f32,
        event_time: ti.f32,
        ejecta_progress: ti.f32,
        radioactive_heating_rate: ti.f32,
        opacity_mean: ti.f32,
        mean_ye: ti.f32,
        kilonova_luminosity: ti.f32,
        is_active: ti.i32,
    ):
        """
        Generate a GPU-efficient visual thermal and radiative emission layer
        driven by EjectaState and KilonovaState using bounded visual time.
        """
        c_light = 2.998e8
        v_blue   = 0.30 * c_light
        v_purple = 0.20 * c_light
        v_red    = 0.10 * c_light

        for i in range(self.n_ejecta_thermal_particles):
            if is_active == 0 or event_time < 0.0 or ejecta_progress <= 0.01:
                self.ejecta_thermal_pos[i] = ti.Vector([0.0, 0.0, -1.0e9])
                self.ejecta_thermal_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                src = (i * self.THERMAL_EJECTA_SAMPLE_DIVISOR) % self.n_ejecta_fluid_particles
                d = self.ejecta_fluid_dir[src]
                comp = self.ejecta_fluid_comp[src]
                fill = self.ejecta_fluid_radial_fill[src]
                brightness = self.ejecta_fluid_brightness[src]

                progress = ti.min(1.0, ti.max(0.0, ejecta_progress))
                visibility = progress * progress * (3.0 - 2.0 * progress)

                growth = ti.min(1.0, event_time / 0.015) if event_time < 0.015 else 1.0
                heat_log = ti.log(ti.max(1.0, radioactive_heating_rate))
                heat_norm = ti.min(1.0, ti.max(0.0, (heat_log - ti.log(1.0e6)) / ti.log(1.0e6)))
                opacity_trans = 1.0 / (1.0 + 0.60 * ti.max(0.0, opacity_mean))
                ye_norm = ti.min(1.0, ti.max(0.0, (mean_ye - 0.05) / 0.40))

                kn_norm = ti.min(1.0, ti.max(0.0, ti.log(1.0 + ti.max(0.0, kilonova_luminosity)) / ti.log(1.0e34)))

                blend = ti.min(1.0, ti.max(0.0, (event_time - 0.020) / 0.080))
                prompt_thermal = growth * (0.50 + 0.50 * heat_norm) * opacity_trans * brightness
                strength = visibility * ((1.0 - 0.60 * blend) * prompt_thermal + (0.40 + 0.60 * blend) * kn_norm)

                r_base = 25.0e3
                if comp == 0:
                    r_base += v_blue * vis_event_time * (0.72 + 0.45 * fill)
                elif comp == 1:
                    r_base += v_purple * vis_event_time * (0.70 + 0.42 * fill)
                else:
                    r_base += v_red * vis_event_time * (0.68 + 0.38 * fill)

                r_thermal = r_base * (0.85 + 0.33 * fill)
                dispersion = ti.Vector([
                    ti.sin(float(i * 17 % 31)) * 0.12 * r_base,
                    ti.cos(float(i * 23 % 37)) * 0.12 * r_base,
                    ti.sin(float(i * 13 % 41)) * 0.12 * r_base
                ])

                if comp == 0:
                    x = d[0] * r_thermal * 0.75 + dispersion[0]
                    y = d[1] * r_thermal * 0.75 + dispersion[1]
                    z = d[2] * r_thermal * 1.25 + dispersion[2]
                    self.ejecta_thermal_pos[i] = ti.Vector([x, y, z])
                    self.ejecta_thermal_colors[i] = ti.Vector([
                        ti.min(1.0, 0.40 * strength * (0.60 + 0.40 * ye_norm)),
                        ti.min(1.0, 0.85 * strength * (0.60 + 0.40 * ye_norm)),
                        ti.min(1.0, 1.00 * strength)
                    ])

                elif comp == 1:
                    x = d[0] * r_thermal + dispersion[0]
                    y = d[1] * r_thermal + dispersion[1]
                    z = d[2] * r_thermal + dispersion[2]
                    self.ejecta_thermal_pos[i] = ti.Vector([x, y, z])
                    self.ejecta_thermal_colors[i] = ti.Vector([
                        ti.min(1.0, 0.85 * strength),
                        ti.min(1.0, 0.42 * strength),
                        ti.min(1.0, 0.98 * strength)
                    ])

                else:
                    x = d[0] * r_thermal * 1.30 + dispersion[0]
                    y = d[1] * r_thermal * 1.30 + dispersion[1]
                    z = d[2] * r_thermal * 0.55 + dispersion[2]
                    self.ejecta_thermal_pos[i] = ti.Vector([x, y, z])
                    self.ejecta_thermal_colors[i] = ti.Vector([
                        ti.min(1.0, 1.00 * strength * (1.15 - 0.30 * ye_norm)),
                        ti.min(1.0, 0.40 * strength),
                        ti.min(1.0, 0.08 * strength)
                    ])

    def update_ejecta_thermal_emission(
        self,
        event_time: float,
        ejecta_progress: float,
        radioactive_heating_rate: float,
        opacity_mean: float,
        mean_ye: float,
        kilonova_luminosity: float,
        is_active: bool = True,
    ):
        """Update preallocated GPU thermal-emission samples from coordinator state."""
        vis_event_time = ParticleRenderer.get_visual_time_adapter(event_time, t_scale=0.006)
        self.update_ejecta_thermal_emission_kernel(
            float(vis_event_time), float(event_time), float(ejecta_progress), float(radioactive_heating_rate),
            float(opacity_mean), float(mean_ye), float(kilonova_luminosity),
            1 if is_active else 0,
        )

    @ti.kernel
    def update_dynamic_jet_kernel(
        self,
        active_flag: ti.i32,
        opening_angle_rad: ti.f32,
        intensity: ti.f32,
        vis_jet_time: ti.f32,
        pulse_wrapped: ti.f32,
        event_time: ti.f32,
        jet_progress: ti.f32,
        jet_delay: ti.f32,
        beta_jet: ti.f32,
        phi_jet_rot_wrapped: ti.f32,
        pitch_helical: ti.f32,
    ):
        """
        Animate dynamic structured jet length expansion using bounded visual time,
        3D continuous rotation around polar z-axis, and helical twist proportional to B_phi/B_p.
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
                phi0 = (float(i) / float(self.n_jet_lines)) * two_pi
                dir_sign = 1.0 if (i % 2 == 0) else -1.0

                curr_len = 14.0e3 + beta_jet * c_light * vis_jet_time

                z_start = dir_sign * 14.0e3
                z_end = dir_sign * curr_len

                layer = float(i % 3)
                layer_angle = opening_angle_rad * (0.4 + 0.3 * layer)

                phi_base = phi0 + phi_jet_rot_wrapped
                phi_tip = phi_base + dir_sign * pitch_helical

                x_start = 14.0e3 * layer_angle * ti.cos(phi_base)
                y_start = 14.0e3 * layer_angle * ti.sin(phi_base)

                x_end = curr_len * layer_angle * ti.cos(phi_tip)
                y_end = curr_len * layer_angle * ti.sin(phi_tip)

                self.jet_vertices[2 * i]     = ti.Vector([x_start, y_start, z_start])
                self.jet_vertices[2 * i + 1] = ti.Vector([x_end, y_end, z_end])

                # Outward relativistic pulse phase along jet frustum
                pulse = 0.5 + 0.5 * ti.sin(pulse_wrapped - float(i) * 0.5)
                core_intensity = ti.min(0.65, intensity * (0.6 + 0.4 * pulse))

                col = ti.Vector([0.0, 0.0, 0.0])
                if layer == 0.0:
                    col = ti.Vector([0.25 * core_intensity, 0.70 * core_intensity, 0.95 * core_intensity])
                elif layer == 1.0:
                    col = ti.Vector([0.55 * core_intensity, 0.30 * core_intensity, 0.80 * core_intensity])
                else:
                    col = ti.Vector([0.75 * core_intensity, 0.20 * core_intensity, 0.50 * core_intensity])

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
        beta_jet: float = 0.9999,
        b_ratio: float = 0.0,
        omega_rot: float = 120.0
    ):
        """Update dynamic relativistic jet line vertices and emissive intensity on GPU."""
        self.jet_active = is_active
        flag = 1 if is_active else 0
        t_since_launch = max(0.0, event_time - jet_delay)
        vis_jet_time = ParticleRenderer.get_visual_time_adapter(t_since_launch, t_scale=0.006)
        pulse_wrapped = float((25.0 * event_time) % (2.0 * np.pi)) if event_time >= 0.0 else 0.0
        phi_jet_rot_wrapped = float((0.05 * max(0.0, omega_rot) * max(0.0, event_time)) % (2.0 * np.pi)) if event_time >= 0.0 else 0.0
        pitch_helical = float(np.clip(0.25 * max(0.0, b_ratio), 0.0, 1.25))
        self.update_dynamic_jet_kernel(
            flag, float(opening_angle_rad),
            float(intensity), float(vis_jet_time), pulse_wrapped, float(event_time), float(jet_progress),
            float(jet_delay), float(beta_jet), float(phi_jet_rot_wrapped), float(pitch_helical)
        )

    def update_jet_lines(
        self,
        event_time: float = 0.0,
        jet_progress: float = 1.0,
        is_active: bool = True,
        jet_delay: float = 1.7,
        beta_jet: float = 0.9999,
        b_ratio: float = 0.0,
        omega_rot: float = 120.0,
        intensity: float = 1.0,
    ):
        """Alias for update_jet_geometry to maintain backward compatibility."""
        self.update_jet_geometry(
            is_active=is_active,
            intensity=intensity,
            event_time=event_time,
            jet_progress=jet_progress,
            jet_delay=jet_delay,
            beta_jet=beta_jet,
            b_ratio=b_ratio,
            omega_rot=omega_rot
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
        is_active: ti.i32
    ):
        """
        Draw time-frequency chirp track line showing f_gw(t) sweeping upward toward merger during inspiral.
        """
        for i in range(n_pts - 1):
            if is_active == 0:
                self.chirp_vertices[2 * i]     = ti.Vector([0.0, 0.0])
                self.chirp_vertices[2 * i + 1] = ti.Vector([0.0, 0.0])
            else:
                t0 = float(i) / float(n_pts - 1)
                t1 = float(i + 1) / float(n_pts - 1)

                x0 = x_start + t0 * w_panel
                x1 = x_start + t1 * w_panel

                f0_norm = ti.min(1.0, ti.max(0.05, 0.05 + 0.95 * ti.pow(t0, 2.2) * (f_current / 1500.0)))
                f1_norm = ti.min(1.0, ti.max(0.05, 0.05 + 0.95 * ti.pow(t1, 2.2) * (f_current / 1500.0)))

                y0 = y_bottom + f0_norm * h_panel
                y1 = y_bottom + f1_norm * h_panel

                self.chirp_vertices[2 * i]     = ti.Vector([x0, y0])
                self.chirp_vertices[2 * i + 1] = ti.Vector([x1, y1])

    @ti.kernel
    def update_observed_waveform_envelope_kernel(
        self,
        n_pts: ti.i32,
        x_start: ti.f32,
        w_panel: ti.f32,
        y_center_h1: ti.f32,
        y_center_l1: ti.f32,
        h_scale: ti.f32,
        is_active: ti.i32
    ):
        """
        Draw live observed GW170817 strain (H1 & L1) polyline traces on GPU.
        """
        for i in range(n_pts - 1):
            if is_active == 0:
                self.obs_h1_vertices[2 * i]     = ti.Vector([0.0, 0.0])
                self.obs_h1_vertices[2 * i + 1] = ti.Vector([0.0, 0.0])
                self.obs_l1_vertices[2 * i]     = ti.Vector([0.0, 0.0])
                self.obs_l1_vertices[2 * i + 1] = ti.Vector([0.0, 0.0])
            else:
                x0 = x_start + (float(i) / float(n_pts - 1)) * w_panel
                x1 = x_start + (float(i + 1) / float(n_pts - 1)) * w_panel

                h1_0 = self.obs_h1_buf_gpu[i]
                h1_1 = self.obs_h1_buf_gpu[i + 1]

                l1_0 = self.obs_l1_buf_gpu[i]
                l1_1 = self.obs_l1_buf_gpu[i + 1]

                y0_h1 = y_center_h1 + ti.max(-0.04, ti.min(0.04, h1_0 * h_scale))
                y1_h1 = y_center_h1 + ti.max(-0.04, ti.min(0.04, h1_1 * h_scale))

                y0_l1 = y_center_l1 + ti.max(-0.04, ti.min(0.04, l1_0 * h_scale))
                y1_l1 = y_center_l1 + ti.max(-0.04, ti.min(0.04, l1_1 * h_scale))

                self.obs_h1_vertices[2 * i]     = ti.Vector([x0, y0_h1])
                self.obs_h1_vertices[2 * i + 1] = ti.Vector([x1, y1_h1])

                self.obs_l1_vertices[2 * i]     = ti.Vector([x0, y0_l1])
                self.obs_l1_vertices[2 * i + 1] = ti.Vector([x1, y1_l1])

    @ti.kernel
    def update_waveform_envelope_kernel(
        self,
        n_pts: ti.i32,
        x_start: ti.f32,
        w_panel: ti.f32,
        y_center: ti.f32,
        h_scale: ti.f32,
        is_active: ti.i32
    ):
        """
        Draw live strain amplitude envelope h(t) polyline bounds on GPU during inspiral and post-merger ringdown.
        """
        for i in range(n_pts - 1):
            if is_active == 0:
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

    def update_waveform_buffer(
        self,
        waveform_data: np.ndarray,
        f_gw: float = 72.4,
        event_time: float = -5.0,
        obs_h1_data: Optional[np.ndarray] = None,
        obs_l1_data: Optional[np.ndarray] = None
    ):
        """Update live model waveform strain envelope, observed GW170817 strain traces, and chirp track lines."""
        if waveform_data is None or len(waveform_data) == 0:
            return
        n = min(len(waveform_data), self.n_wave_samples)
        data_slice = waveform_data[-n:]

        max_amp = float(np.max(np.abs(data_slice))) if np.max(np.abs(data_slice)) > 0 else 1.0e-21
        scale_factor = 0.04 / max(max_amp, 1.0e-23)

        padded = np.zeros(self.n_wave_samples, dtype=np.float32)
        padded[-n:] = data_slice.astype(np.float32)
        if self._last_uploaded_h_buf is None or not np.array_equal(self._last_uploaded_h_buf, padded):
            self.h_buf_gpu.from_numpy(padded)
            self._last_uploaded_h_buf = padded.copy()

        # Active throughout inspiral and post-merger
        is_act = 1 if event_time <= 15.0 else 0

        # Model strain trace at y_center = 0.08
        self.update_waveform_envelope_kernel(
            self.n_wave_samples,
            0.05,   # x_start
            0.58,   # w_panel
            0.08,   # y_center
            scale_factor,
            is_act
        )

        # Observed GW170817 strain traces (H1 at y=0.16, L1 at y=0.13)
        if obs_h1_data is not None and obs_l1_data is not None:
            n_obs = min(len(obs_h1_data), self.n_wave_samples)
            padded_h1 = np.zeros(self.n_wave_samples, dtype=np.float32)
            padded_l1 = np.zeros(self.n_wave_samples, dtype=np.float32)
            padded_h1[-n_obs:] = obs_h1_data[-n_obs:].astype(np.float32)
            padded_l1[-n_obs:] = obs_l1_data[-n_obs:].astype(np.float32)

            if self._last_uploaded_obs_h1 is None or not np.array_equal(self._last_uploaded_obs_h1, padded_h1):
                self.obs_h1_buf_gpu.from_numpy(padded_h1)
                self._last_uploaded_obs_h1 = padded_h1.copy()

            if self._last_uploaded_obs_l1 is None or not np.array_equal(self._last_uploaded_obs_l1, padded_l1):
                self.obs_l1_buf_gpu.from_numpy(padded_l1)
                self._last_uploaded_obs_l1 = padded_l1.copy()

            max_obs_h1 = float(np.max(np.abs(padded_h1))) if np.max(np.abs(padded_h1)) > 0 else 1.0e-20
            scale_obs = 0.03 / max(max_obs_h1, 1.0e-22)

            self.update_observed_waveform_envelope_kernel(
                self.n_wave_samples,
                0.05,   # x_start
                0.58,   # w_panel
                0.16,   # y_center_h1
                0.13,   # y_center_l1
                scale_obs,
                is_act
            )

        self.update_chirp_spectrogram_kernel(
            self.n_chirp_pts,
            0.05,   # x_start
            0.58,   # w_panel
            0.05,   # y_bottom
            0.14,   # h_panel
            float(f_gw),
            is_act
        )

    @ti.kernel
    def pack_combined_post_merger_kernel(self, n_rem: ti.i32, n_disk: ti.i32, n_ej: ti.i32, n_thermal: ti.i32):
        for i in range(n_rem):
            self.combined_post_merger_pos[i] = self.remnant_pos[i]
            self.combined_post_merger_colors[i] = self.remnant_colors[i]
        for i in range(n_disk):
            self.combined_post_merger_pos[n_rem + i] = self.disk_pos[i]
            self.combined_post_merger_colors[n_rem + i] = self.disk_colors[i]
        for i in range(n_ej):
            self.combined_post_merger_pos[n_rem + n_disk + i] = self.ejecta_fluid_pos[i]
            self.combined_post_merger_colors[n_rem + n_disk + i] = self.ejecta_fluid_colors[i]
        for i in range(n_thermal):
            self.combined_post_merger_pos[n_rem + n_disk + n_ej + i] = self.ejecta_thermal_pos[i]
            self.combined_post_merger_colors[n_rem + n_disk + n_ej + i] = self.ejecta_thermal_colors[i]

    def update_combined_post_merger(self):
        """Pack remnant, disk, ejecta fluid, and thermal proxy into one GPU field."""
        self.pack_combined_post_merger_kernel(
            self.n_remnant_particles,
            self.n_disk_particles,
            self.n_ejecta_fluid_particles,
            self.n_ejecta_thermal_particles,
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

    def update_combined_lines(self, b_line_verts, b_line_cols, n_b_verts: int, show_jet: bool = True):
        """Pack magnetic field lines and jet lines into a single GPU line field."""
        n_j = self.n_jet_vertices if (self.jet_active and show_jet) else 0
        self.pack_combined_lines_kernel(
            b_line_verts,
            b_line_cols,
            n_b_verts,
            n_j
        )

    @ti.kernel
    def pack_gw_wavefront_kernel(self, wf_verts: ti.template(), wf_cols: ti.template(), n_v: ti.i32):
        for i in range(self.n_gw_wave_vertices):
            if i < n_v:
                self.gw_wave_vertices[i] = wf_verts[i]
                self.gw_wave_colors[i] = wf_cols[i]
            else:
                self.gw_wave_vertices[i] = ti.Vector([0.0, 0.0, 0.0])
                self.gw_wave_colors[i] = ti.Vector([0.0, 0.0, 0.0])

    def update_gw_wavefront_lines(self, vertices, colors, n_vertices: int):
        """Update 3D GW wavefront line fields from Taichi GPU fields or NumPy arrays."""
        if n_vertices <= 0:
            return
        n = min(n_vertices, self.n_gw_wave_vertices)
        if isinstance(vertices, np.ndarray):
            padded_v = np.zeros((self.n_gw_wave_vertices, 3), dtype=np.float32)
            padded_c = np.zeros((self.n_gw_wave_vertices, 3), dtype=np.float32)
            padded_v[:n] = vertices[:n]
            padded_c[:n] = colors[:n]
            self.gw_wave_vertices.from_numpy(padded_v)
            self.gw_wave_colors.from_numpy(padded_c)
        else:
            self.pack_gw_wavefront_kernel(vertices, colors, n)

