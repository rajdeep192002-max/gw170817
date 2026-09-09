"""
Taichi particle system for GW170817 binary neutron star simulation.
Provides backend initialization with Vulkan preference and CPU fallback,
and preallocated particle state fields.

REDUCED-ORDER APPROXIMATION:
Particles represent discrete fluid mass elements of compact neutron stars.
Initial state sets up Keplerian binary orbit of two particle clouds.
"""
import taichi as ti
import numpy as np
from typing import Tuple
from gw170817.constants import G, M_sun
from gw170817.config import SimConfig


def initialize_taichi(preference: str = "vulkan") -> str:
    """
    Initialize Taichi with backend fallback.
    Tries Vulkan first if preferred, falling back to CPU on failure.

    Returns:
        str: Name of the active backend ("vulkan" or "cpu").
    """
    preference = preference.lower()
    selected_backend = "cpu"

    if preference == "vulkan":
        try:
            ti.init(arch=ti.vulkan, log_level=ti.WARN)
            selected_backend = "vulkan"
        except Exception as e:
            print(f"[Taichi Init] Vulkan initialization failed ({e}). Falling back to CPU.")
            ti.init(arch=ti.cpu, log_level=ti.WARN)
            selected_backend = "cpu"
    else:
        ti.init(arch=ti.cpu, log_level=ti.WARN)
        selected_backend = "cpu"

    return selected_backend


@ti.data_oriented
class ParticleSystem:
    """
    Preallocated Taichi particle fields for binary neutron star simulation.
    All memory is allocated once at construction.
    """

    def __init__(self, config: SimConfig = None):
        if config is None:
            config = SimConfig()
        self.config = config
        self.max_particles = config.particle_count
        self.n_particles_1 = self.max_particles // 2
        self.n_particles_2 = self.max_particles - self.n_particles_1

        # --- Preallocated Taichi Fields ---
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.mass = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.star_id = ti.field(dtype=ti.i32, shape=self.max_particles)  # 0: NS1, 1: NS2
        self.active = ti.field(dtype=ti.i32, shape=self.max_particles)   # 1: active, 0: inactive

        # Global scalars
        self.n_active = ti.field(dtype=ti.i32, shape=())
        self.n_active[None] = self.max_particles

    def init_bns_binary(
        self,
        r1_ns: float = 11.0e3,
        r2_ns: float = 11.0e3,
    ):
        """
        Initialize particle clouds for two neutron stars in a circular binary orbit.
        Uses physical parameters from SimConfig.
        """
        m1 = self.config.m1
        m2 = self.config.m2
        M_total = m1 + m2
        a0 = self.config.initial_separation

        # Center of mass offsets along x-axis
        x1_com = -a0 * (m2 / M_total)
        x2_com =  a0 * (m1 / M_total)

        # Keplerian orbital angular frequency & speeds
        omega = np.sqrt(G * M_total / (a0**3))
        v1_y = -omega * abs(x1_com)  # CCW orbit
        v2_y =  omega * abs(x2_com)

        # Particle mass allocations
        pmass1 = m1 / self.n_particles_1
        pmass2 = m2 / self.n_particles_2

        # Invoke Taichi kernel for particle population
        self._init_particles_kernel(
            self.n_particles_1,
            self.max_particles,
            float(x1_com), float(x2_com),
            float(v1_y), float(v2_y),
            float(r1_ns), float(r2_ns),
            float(pmass1), float(pmass2),
            self.config.seed
        )

    @ti.kernel
    def _init_particles_kernel(
        self,
        n1: ti.i32,
        n_total: ti.i32,
        x1_com: ti.f32,
        x2_com: ti.f32,
        v1_y: ti.f32,
        v2_y: ti.f32,
        r1_ns: ti.f32,
        r2_ns: ti.f32,
        pmass1: ti.f32,
        pmass2: ti.f32,
        seed: ti.i32
    ):
        """
        Taichi kernel to initialize particle positions, velocities, masses, and star IDs.
        No Python loops involved.
        """
        two_pi = 6.283185307179586

        # --- NS1 Particles ---
        for i in range(n1):
            # Uniform spherical distribution
            u = ti.random(ti.f32)
            r = ti.pow(u, 1.0 / 3.0) * r1_ns
            cos_theta = 1.0 - 2.0 * ti.random(ti.f32)
            sin_theta = ti.sqrt(ti.max(0.0, 1.0 - cos_theta * cos_theta))
            phi = ti.random(ti.f32) * two_pi

            dx = r * sin_theta * ti.cos(phi)
            dy = r * sin_theta * ti.sin(phi)
            dz = r * cos_theta * 0.3  # Slight z-flattening for orbital plane

            self.pos[i] = ti.Vector([x1_com + dx, dy, dz])
            self.vel[i] = ti.Vector([0.0, v1_y, 0.0])
            self.mass[i] = pmass1
            self.star_id[i] = 0
            self.active[i] = 1

        # --- NS2 Particles ---
        for i in range(n1, n_total):
            u = ti.random(ti.f32)
            r = ti.pow(u, 1.0 / 3.0) * r2_ns
            cos_theta = 1.0 - 2.0 * ti.random(ti.f32)
            sin_theta = ti.sqrt(ti.max(0.0, 1.0 - cos_theta * cos_theta))
            phi = ti.random(ti.f32) * two_pi

            dx = r * sin_theta * ti.cos(phi)
            dy = r * sin_theta * ti.sin(phi)
            dz = r * cos_theta * 0.3

            self.pos[i] = ti.Vector([x2_com + dx, dy, dz])
            self.vel[i] = ti.Vector([0.0, v2_y, 0.0])
            self.mass[i] = pmass2
            self.star_id[i] = 1
            self.active[i] = 1

    def compute_diagnostics_numpy(self) -> dict:
        """
        Compute summary diagnostics using NumPy arrays for validation.
        """
        pos_np = self.pos.to_numpy()
        vel_np = self.vel.to_numpy()
        mass_np = self.mass.to_numpy()
        star_id_np = self.star_id.to_numpy()

        total_mass = float(np.sum(mass_np))
        com_pos = np.sum(pos_np * mass_np[:, None], axis=0) / total_mass
        com_vel = np.sum(vel_np * mass_np[:, None], axis=0) / total_mass

        mask1 = (star_id_np == 0)
        mask2 = (star_id_np == 1)

        m1_calc = float(np.sum(mass_np[mask1]))
        m2_calc = float(np.sum(mass_np[mask2]))

        pos1_com = np.sum(pos_np[mask1] * mass_np[mask1, None], axis=0) / m1_calc
        pos2_com = np.sum(pos_np[mask2] * mass_np[mask2, None], axis=0) / m2_calc

        vel1_com = np.sum(vel_np[mask1] * mass_np[mask1, None], axis=0) / m1_calc
        vel2_com = np.sum(vel_np[mask2] * mass_np[mask2, None], axis=0) / m2_calc

        has_nans = bool(np.isnan(pos_np).any() or np.isnan(vel_np).any())
        has_infs = bool(np.isinf(pos_np).any() or np.isinf(vel_np).any())

        return {
            "total_mass": total_mass,
            "m1": m1_calc,
            "m2": m2_calc,
            "com_pos": com_pos,
            "com_vel": com_vel,
            "ns1_pos": pos1_com,
            "ns2_pos": pos2_com,
            "ns1_vel": vel1_com,
            "ns2_vel": vel2_com,
            "has_nans": has_nans,
            "has_infs": has_infs,
        }
