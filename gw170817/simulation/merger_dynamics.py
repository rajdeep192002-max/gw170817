"""
Taichi reduced-order BNS merger particle dynamics for GW170817.

REDUCED-ORDER APPROXIMATION:
This module provides GPU-accelerated reduced-order particle dynamics.
It does NOT solve Einstein field equations, GRHD, or MHD.
It transforms preallocated particle clouds according to orbital inspiral,
tidal deformation, and smooth contact/merger progression.
"""
import taichi as ti
import numpy as np
from typing import Dict, Any
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem
from gw170817.physics.inspiral import InspiralModel, InspiralState
from gw170817.physics.tidal import TidalModel, TidalState
from gw170817.physics.merger import MergerModel, MergerState


@ti.data_oriented
class MergerDynamics:
    """
    Taichi-based reduced-order dynamical evolution of neutron-star particles.
    Drives preallocated Taichi fields in ParticleSystem.
    """

    def __init__(
        self,
        particle_system: ParticleSystem,
        inspiral_model: InspiralModel = None,
        tidal_model: TidalModel = None,
        merger_model: MergerModel = None,
    ):
        self.psys = particle_system
        self.config = particle_system.config

        if inspiral_model is None:
            inspiral_model = InspiralModel(self.config)
        self.inspiral_model = inspiral_model

        if tidal_model is None:
            tidal_model = TidalModel(self.config)
        self.tidal_model = tidal_model

        if merger_model is None:
            merger_model = MergerModel(self.config, self.tidal_model)
        self.merger_model = merger_model

        # Preallocate local offset field relative to star CoM
        self.local_offset = ti.Vector.field(3, dtype=ti.f32, shape=self.psys.max_particles)

        # Velocity safety limit: 0.5 * c
        self.v_clamp = 0.5 * c

        # Initialise simulation state
        self.initialize()

    def initialize(self):
        """Reset inspiral model and re-seed particle positions and local offsets."""
        self.inspiral_state = self.inspiral_model.initial_state()
        self.tidal_state = self.tidal_model.evaluate(self.inspiral_state)
        self.merger_state = self.merger_model.evaluate(self.inspiral_state)

        # Re-initialize particle clouds in binary configuration
        self.psys.init_bns_binary()

        # Capture initial local offsets relative to respective star CoM positions
        r1_0, r2_0 = self.inspiral_model.orbital_positions(
            self.inspiral_state, self.config.m1, self.config.m2
        )
        self._capture_local_offsets_kernel(
            self.psys.n_particles_1,
            self.psys.max_particles,
            float(r1_0[0]), float(r1_0[1]), float(r1_0[2]),
            float(r2_0[0]), float(r2_0[1]), float(r2_0[2])
        )

    @ti.kernel
    def _capture_local_offsets_kernel(
        self,
        n1: ti.i32,
        n_total: ti.i32,
        x1: ti.f32, y1: ti.f32, z1: ti.f32,
        x2: ti.f32, y2: ti.f32, z2: ti.f32
    ):
        """Store initial particle position offsets relative to each star center of mass."""
        com1 = ti.Vector([x1, y1, z1])
        com2 = ti.Vector([x2, y2, z2])

        for i in range(n1):
            self.local_offset[i] = self.psys.pos[i] - com1

        for i in range(n1, n_total):
            self.local_offset[i] = self.psys.pos[i] - com2

    def update_particles(self):
        """Update particle positions and velocities from current inspiral and merger states."""
        r1, r2 = self.inspiral_model.orbital_positions(
            self.inspiral_state, self.config.m1, self.config.m2
        )
        v1, v2 = self.inspiral_model.orbital_velocities(
            self.inspiral_state, self.config.m1, self.config.m2
        )
        eps1 = min(0.5 * self.tidal_state.tidal_distortion_1, 0.4)
        eps2 = min(0.5 * self.tidal_state.tidal_distortion_2, 0.4)
        self._update_particles_kernel(
            self.psys.n_particles_1,
            self.psys.max_particles,
            float(r1[0]), float(r1[1]), float(r1[2]),
            float(r2[0]), float(r2[1]), float(r2[2]),
            float(v1[0]), float(v1[1]), float(v1[2]),
            float(v2[0]), float(v2[1]), float(v2[2]),
            float(self.inspiral_state.orbital_phase),
            float(self.inspiral_state.omega_orb),
            float(eps1), float(eps2),
            float(self.merger_state.contact_fraction),
            float(self.v_clamp)
        )

    def step(self, dt: float):
        """Advance simulation physics by dt seconds."""
        # 1. Advance inspiral model
        self.inspiral_state = self.inspiral_model.step(self.inspiral_state, dt)

        # 2. Evaluate tidal and merger states
        self.tidal_state = self.tidal_model.evaluate(self.inspiral_state)
        self.merger_state = self.merger_model.evaluate(self.inspiral_state)

        # 3. Update particle positions and velocities
        self.update_particles()

    @ti.kernel
    def _update_particles_kernel(
        self,
        n1: ti.i32,
        n_total: ti.i32,
        x1: ti.f32, y1: ti.f32, z1: ti.f32,
        x2: ti.f32, y2: ti.f32, z2: ti.f32,
        vx1: ti.f32, vy1: ti.f32, vz1: ti.f32,
        vx2: ti.f32, vy2: ti.f32, vz2: ti.f32,
        phi: ti.f32,
        omega: ti.f32,
        eps1: ti.f32,
        eps2: ti.f32,
        contact_frac: ti.f32,
        v_clamp: ti.f32
    ):
        """
        Taichi kernel for parallel particle position, velocity, tidal deformation,
        and merger remnant mapping update.
        """
        cos_phi = ti.cos(phi)
        sin_phi = ti.sin(phi)

        com1 = ti.Vector([x1, y1, z1])
        com2 = ti.Vector([x2, y2, z2])
        vbulk1 = ti.Vector([vx1, vy1, vz1])
        vbulk2 = ti.Vector([vx2, vy2, vz2])

        # Smooth merger contraction factor
        contract_scale = 1.0 - 0.65 * contact_frac

        # --- NS1 Particles ---
        for i in range(n1):
            off = self.local_offset[i]

            # Tidal deformation in local frame (prolate stretch along radial axis)
            lx = off[0] * (1.0 + eps1)
            ly = off[1] * (1.0 - 0.5 * eps1)
            lz = off[2] * (1.0 - 0.5 * eps1)

            # Rotate local offset by orbital phase
            rx = lx * cos_phi - ly * sin_phi
            ry = lx * sin_phi + ly * cos_phi
            rz = lz

            # Binary phase position
            pos_binary = com1 * contract_scale + ti.Vector([rx, ry, rz])
            vel_binary = vbulk1 + ti.Vector([-omega * ry, omega * rx, 0.0])

            # Post-merger remnant rotation around common COM (origin)
            remnant_v_tangent = ti.Vector([-omega * pos_binary[1], omega * pos_binary[0], 0.0])

            # Blend binary phase into remnant phase
            pos_final = (1.0 - contact_frac) * pos_binary + contact_frac * (pos_binary * 0.85)
            vel_final = (1.0 - contact_frac) * vel_binary + contact_frac * remnant_v_tangent

            # Safety clamp speed
            speed = vel_final.norm()
            if speed > v_clamp:
                vel_final = vel_final * (v_clamp / speed)

            self.psys.pos[i] = pos_final
            self.psys.vel[i] = vel_final

        # --- NS2 Particles ---
        for i in range(n1, n_total):
            off = self.local_offset[i]

            lx = off[0] * (1.0 + eps2)
            ly = off[1] * (1.0 - 0.5 * eps2)
            lz = off[2] * (1.0 - 0.5 * eps2)

            rx = lx * cos_phi - ly * sin_phi
            ry = lx * sin_phi + ly * cos_phi
            rz = lz

            pos_binary = com2 * contract_scale + ti.Vector([rx, ry, rz])
            vel_binary = vbulk2 + ti.Vector([-omega * ry, omega * rx, 0.0])

            remnant_v_tangent = ti.Vector([-omega * pos_binary[1], omega * pos_binary[0], 0.0])

            pos_final = (1.0 - contact_frac) * pos_binary + contact_frac * (pos_binary * 0.85)
            vel_final = (1.0 - contact_frac) * vel_binary + contact_frac * remnant_v_tangent

            speed = vel_final.norm()
            if speed > v_clamp:
                vel_final = vel_final * (v_clamp / speed)

            self.psys.pos[i] = pos_final
            self.psys.vel[i] = vel_final

    def compute_diagnostics(self) -> Dict[str, Any]:
        """Compute summary diagnostics for tracking and verification."""
        diag_np = self.psys.compute_diagnostics_numpy()
        pos_np = self.psys.pos.to_numpy()
        vel_np = self.psys.vel.to_numpy()
        mass_np = self.psys.mass.to_numpy().astype(np.float64)

        speeds = np.linalg.norm(vel_np, axis=1).astype(np.float64)
        radii = np.linalg.norm(pos_np - diag_np['com_pos'], axis=1)

        # Kinetic energy = 0.5 * sum(m_i * v_i^2)
        total_ke = float(0.5 * np.sum(mass_np * (speeds**2)))

        return {
            "total_mass": diag_np['total_mass'],
            "com_pos": diag_np['com_pos'],
            "com_vel": diag_np['com_vel'],
            "total_ke": total_ke,
            "m1": diag_np['m1'],
            "m2": diag_np['m2'],
            "active_particles": self.psys.max_particles,
            "separation_estimate": float(np.linalg.norm(diag_np['ns2_pos'] - diag_np['ns1_pos'])),
            "contact_fraction": self.merger_state.contact_fraction,
            "max_particle_speed": float(np.max(speeds)),
            "min_particle_radius": float(np.min(radii)),
            "max_particle_radius": float(np.max(radii)),
            "has_nans": diag_np['has_nans'],
            "has_infs": diag_np['has_infs'],
        }
