"""
Reduced-order dynamical ejecta model for GW170817.

REDUCED-ORDER APPROXIMATION:
This is a reduced-order ejecta model based on a Newtonian escape-energy proxy
and a phenomenological composition prescription.
It is NOT a full GR unboundness criterion and NOT a neutrino-transport calculation.
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, TYPE_CHECKING
import taichi as ti
import numpy as np
from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig

if TYPE_CHECKING:
    from gw170817.simulation.particles import ParticleSystem


@dataclass
class EjectaState:
    """Summary diagnostic state of dynamically unbound ejecta and R-process nucleosynthesis."""
    time: float                     # Physical simulation time [s]
    ejecta_mass: float              # Total mass of unbound ejecta [kg]
    ejecta_fraction: float          # Fraction of total mass unbound [0.0 to 1.0]
    ejecta_particle_count: int      # Number of unbound ejecta particles
    mean_velocity: float            # Mass-weighted mean ejecta speed [m/s]
    max_velocity: float             # Maximum ejecta particle speed [m/s]
    kinetic_energy: float           # Total kinetic energy of ejecta [J]
    angular_momentum_proxy: float   # Angular momentum magnitude proxy of ejecta [kg m^2/s]
    mean_Ye: float                  # Mass-weighted mean electron fraction Ye
    lanthanide_rich_fraction: float  # Mass fraction of ejecta with Ye <= 0.25 [0.0 to 1.0]
    lanthanide_poor_fraction: float = 0.0  # Mass fraction of ejecta with Ye > 0.25 [0.0 to 1.0]
    dynamical_ejecta_mass: float = 0.0  # Mass of dynamical ejecta [kg]
    disk_wind_ejecta_mass: float = 0.0 # Mass of post-merger disk wind ejecta [kg]
    radioactive_heating_rate: float = 0.0 # R-process decay heating rate eps_dot [W/kg]
    opacity_mean: float = 0.1       # Mass-weighted mean gray opacity kappa [m^2/kg]
    temperature_proxy: float = 0.0  # Mass-weighted mean temperature estimate [K]
    density_proxy: float = 0.0      # Mass-weighted mean density estimate [kg/m^3]
    r_process_active: bool = False  # True if r-process heating is active post-merger

    def __post_init__(self):
        if self.lanthanide_poor_fraction == 0.0 and self.lanthanide_rich_fraction < 1.0:
            self.lanthanide_poor_fraction = max(0.0, 1.0 - self.lanthanide_rich_fraction)


@ti.data_oriented
class EjectaModel:
    """
    Taichi-based reduced-order ejecta classifier, R-process proxy, and tracker.
    Uses preallocated Taichi fields for GPU particle classification.

    Diagnostic state is cached to avoid redundant GPU->CPU transfers.
    compute_state() only performs to_numpy() when the classification
    has actually been updated since the last call.
    """

    # Canonical zero state returned when no classification has been performed.
    _ZERO_STATE = EjectaState(
        time=0.0,
        ejecta_mass=0.0,
        ejecta_fraction=0.0,
        ejecta_particle_count=0,
        mean_velocity=0.0,
        max_velocity=0.0,
        kinetic_energy=0.0,
        angular_momentum_proxy=0.0,
        mean_Ye=0.05,
        lanthanide_rich_fraction=0.0,
        lanthanide_poor_fraction=1.0,
        dynamical_ejecta_mass=0.0,
        disk_wind_ejecta_mass=0.0,
        radioactive_heating_rate=0.0,
        opacity_mean=0.1,
        temperature_proxy=0.0,
        density_proxy=0.0,
        r_process_active=False
    )

    def __init__(self, config: SimConfig = None):
        if config is None:
            config = SimConfig()
        self.config = config
        self.max_particles = config.particle_count

        # Preallocated Taichi fields for classification
        self.ejecta_flag = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.Ye_field = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.lanthanide_rich = ti.field(dtype=ti.i32, shape=self.max_particles)

        # Diagnostic cache — avoids GPU->CPU transfers on reset / init / checkpoint nav
        self._cached_state: EjectaState = EjectaModel._ZERO_STATE
        self._classification_dirty: bool = False

        self.initialize()

    def initialize(self):
        """Reset ejecta classification fields to zero and clear cached state."""
        self._reset_kernel()
        self._cached_state = EjectaModel._ZERO_STATE
        self._classification_dirty = False

    @ti.kernel
    def _reset_kernel(self):
        for i in range(self.max_particles):
            self.ejecta_flag[i] = 0
            self.Ye_field[i] = 0.05
            self.lanthanide_rich[i] = 1

    def classify(
        self,
        psys: Any,
        com_pos: Any = None,
        com_vel: Any = None,
        M_rem: float = None
    ):
        """
        Classify particles into bound vs dynamically unbound ejecta using a
        Newtonian escape-energy proxy and compute phenomenological Ye composition.
        """
        if com_pos is None:
            com_pos = np.zeros(3, dtype=np.float32)
        if com_vel is None:
            com_vel = np.zeros(3, dtype=np.float32)
        if M_rem is None:
            M_rem = self.config.M_total

        if M_rem <= 0.0:
            raise ValueError(f"Remnant mass scale must be positive, got {M_rem}")

        cx, cy, cz = float(com_pos[0]), float(com_pos[1]), float(com_pos[2])
        cvx, cvy, cvz = float(com_vel[0]), float(com_vel[1]), float(com_vel[2])

        self._classify_kernel(
            psys.pos,
            psys.vel,
            psys.max_particles,
            cx, cy, cz,
            cvx, cvy, cvz,
            float(M_rem),
            float(c)
        )
        self._classification_dirty = True

    @ti.kernel
    def _classify_kernel(
        self,
        pos: ti.template(),
        vel: ti.template(),
        n_total: ti.i32,
        cx: ti.f32, cy: ti.f32, cz: ti.f32,
        cvx: ti.f32, cvy: ti.f32, cvz: ti.f32,
        M_rem: ti.f32,
        c_light: ti.f32
    ):
        """
        GPU kernel for particle ejecta classification:
        - Newtonian kinetic energy proxy: eps_kin = 0.5 * |v_rel|^2
        - Newtonian escape energy proxy: eps_esc = G * M_rem / r_rel
        - Dynamically unbound when: eps_kin > eps_esc AND dot(r_rel, v_rel) > 0
        """
        G_const = 6.67430e-11

        for i in range(n_total):
            p = pos[i]
            v = vel[i]

            rx = p[0] - cx
            ry = p[1] - cy
            rz = p[2] - cz

            vx = v[0] - cvx
            vy = v[1] - cvy
            vz = v[2] - cvz

            r_sq = rx * rx + ry * ry + rz * rz
            r = ti.sqrt(ti.max(r_sq, 1.0))  # Guard against r -> 0

            v_sq = vx * vx + vy * vy + vz * vz
            v_speed = ti.sqrt(v_sq)

            eps_kin = 0.5 * v_sq
            eps_esc = (G_const * M_rem) / r
            dot_rv = rx * vx + ry * vy + rz * vz

            if (eps_kin > eps_esc) and (dot_rv > 0.0):
                self.ejecta_flag[i] = 1

                # Phenomenological Ye calculation: polar angle / speed dependence
                abs_z_over_r = ti.abs(rz) / r
                speed_ratio = v_speed / c_light
                ye_calc = 0.05 + 0.30 * ti.min(1.0, abs_z_over_r + speed_ratio)

                # Clamp Ye to [0.05, 0.45]
                ye_clamped = ti.max(0.05, ti.min(0.45, ye_calc))
                self.Ye_field[i] = ye_clamped

                # Lanthanide-rich threshold (Ye <= 0.25)
                if ye_clamped <= 0.25:
                    self.lanthanide_rich[i] = 1
                else:
                    self.lanthanide_rich[i] = 0
            else:
                self.ejecta_flag[i] = 0
                self.Ye_field[i] = 0.05
                self.lanthanide_rich[i] = 1

    def compute_state(self, psys: Any, current_time: float) -> EjectaState:
        """
        Aggregate diagnostics and return an EjectaState dataclass.

        Uses a dirty-flag cache: GPU->CPU transfers (to_numpy) only occur
        when classify() has been called since the last compute_state().
        On reset / init / checkpoint navigation, returns the cached zero state
        without any GPU memory access.
        """
        if not self._classification_dirty:
            # Return cached state with updated timestamp — no GPU transfer needed
            return EjectaState(
                time=current_time,
                ejecta_mass=self._cached_state.ejecta_mass,
                ejecta_fraction=self._cached_state.ejecta_fraction,
                ejecta_particle_count=self._cached_state.ejecta_particle_count,
                mean_velocity=self._cached_state.mean_velocity,
                max_velocity=self._cached_state.max_velocity,
                kinetic_energy=self._cached_state.kinetic_energy,
                angular_momentum_proxy=self._cached_state.angular_momentum_proxy,
                mean_Ye=self._cached_state.mean_Ye,
                lanthanide_rich_fraction=self._cached_state.lanthanide_rich_fraction,
                lanthanide_poor_fraction=self._cached_state.lanthanide_poor_fraction,
                dynamical_ejecta_mass=self._cached_state.dynamical_ejecta_mass,
                disk_wind_ejecta_mass=self._cached_state.disk_wind_ejecta_mass,
                radioactive_heating_rate=self._cached_state.radioactive_heating_rate,
                opacity_mean=self._cached_state.opacity_mean,
                temperature_proxy=self._cached_state.temperature_proxy,
                density_proxy=self._cached_state.density_proxy,
                r_process_active=self._cached_state.r_process_active
            )

        # Classification has changed — perform GPU->CPU transfer
        ejecta_flags = self.ejecta_flag.to_numpy()
        ye_vals = self.Ye_field.to_numpy()
        lan_flags = self.lanthanide_rich.to_numpy()

        mass_np = psys.mass.to_numpy()
        pos_np = psys.pos.to_numpy()
        vel_np = psys.vel.to_numpy()

        total_mass = float(np.sum(mass_np))
        mask = (ejecta_flags == 1)
        count = int(np.sum(mask))

        if count == 0:
            state = EjectaState(
                time=current_time,
                ejecta_mass=0.0,
                ejecta_fraction=0.0,
                ejecta_particle_count=0,
                mean_velocity=0.0,
                max_velocity=0.0,
                kinetic_energy=0.0,
                angular_momentum_proxy=0.0,
                mean_Ye=0.05,
                lanthanide_rich_fraction=0.0,
                lanthanide_poor_fraction=1.0,
                dynamical_ejecta_mass=0.0,
                disk_wind_ejecta_mass=0.0,
                radioactive_heating_rate=0.0,
                opacity_mean=0.1,
                temperature_proxy=0.0,
                density_proxy=0.0,
                r_process_active=False
            )
            self._cached_state = state
            self._classification_dirty = False
            return state

        m_ej = mass_np[mask]
        pos_ej = pos_np[mask]
        vel_ej = vel_np[mask]
        ye_ej = ye_vals[mask]
        lan_ej = lan_flags[mask]

        dyn_mass = float(np.sum(m_ej))
        
        # Reduced-order post-merger disk-wind ejecta contribution [kg]
        t_pos = max(0.0, current_time)
        disk_wind_mass = float(0.15 * 0.06 * M_sun * (1.0 - np.exp(-t_pos / 0.3))) if current_time > 0.0 else 0.0
        
        ej_mass = dyn_mass + disk_wind_mass
        ej_frac = float(np.clip(ej_mass / total_mass, 0.0, 1.0))

        speeds = np.linalg.norm(vel_ej, axis=1)
        mean_v = float(np.sum(m_ej * speeds) / max(1.0e-10, dyn_mass))
        max_v = float(np.max(speeds))

        # Kinetic energy (float64 calculation for numerical safety)
        ke = float(0.5 * np.sum(m_ej.astype(np.float64) * (speeds.astype(np.float64)**2)))

        # Angular momentum proxy
        cross_prod = np.cross(pos_ej, vel_ej)
        L_vec = np.sum(m_ej[:, None] * cross_prod, axis=0)
        L_mag = float(np.linalg.norm(L_vec))

        mean_ye = float(np.sum(m_ej * ye_ej) / max(1.0e-10, dyn_mass))
        lan_mass = float(np.sum(m_ej[lan_ej == 1]))
        lan_frac = float(np.clip(lan_mass / max(1.0e-10, dyn_mass), 0.0, 1.0))
        lan_poor_frac = float(1.0 - lan_frac)

        # R-process radioactive decay heating rate eps_dot [W/kg]
        t0_heating = 10.0  # softening time [s]
        t_days = max(current_time + t0_heating, t0_heating) / 86400.0
        eps_dot = float(2.0e6 * (t_days ** (-1.3)))

        # Mean opacity kappa [m^2/kg] (1 cm^2/g = 0.1 m^2/kg)
        opacity_mean = float(0.1 * lan_poor_frac + 1.0 * lan_frac)

        # Temperature and density proxies
        r_eff = max(10.0e3, mean_v * t_pos)
        vol_eff = float((4.0 / 3.0) * np.pi * (r_eff ** 3))
        rho_proxy = float(ej_mass / vol_eff)
        temp_proxy = float(1.0e4 * ((t_pos + 1.0) ** (-0.35)))
        r_proc_active = bool(ej_mass > 0.0 and current_time > 0.0)

        state = EjectaState(
            time=current_time,
            ejecta_mass=ej_mass,
            ejecta_fraction=ej_frac,
            ejecta_particle_count=count,
            mean_velocity=mean_v,
            max_velocity=max_v,
            kinetic_energy=ke,
            angular_momentum_proxy=L_mag,
            mean_Ye=mean_ye,
            lanthanide_rich_fraction=lan_frac,
            lanthanide_poor_fraction=lan_poor_frac,
            dynamical_ejecta_mass=dyn_mass,
            disk_wind_ejecta_mass=disk_wind_mass,
            radioactive_heating_rate=eps_dot,
            opacity_mean=opacity_mean,
            temperature_proxy=temp_proxy,
            density_proxy=rho_proxy,
            r_process_active=r_proc_active
        )
        self._cached_state = state
        self._classification_dirty = False
        return state
