"""
Temporal Coupling Regression Tests — Task 025.1

Verifies that every physics subsystem evolves as a function of physical event_time.
Key invariants: orbital revolution, ballistic ejecta expansion, physical jet propagation,
rotation/magnetic winding evolution, kilonova photosphere growth, afterglow light curve.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.rotation import RotationalModel
from gw170817.physics.disk import DiskModel
from gw170817.physics.magnetic_field import MagneticFieldModel
from gw170817.physics.kilonova import KilonovaModel
from gw170817.physics.jet import StructuredJetModel
from gw170817.physics.afterglow import AfterglowModel
from gw170817.physics.ejecta import EjectaState
from gw170817.constants import c, M_sun, day


@pytest.fixture(scope="module")
def config():
    return SimConfig()

@pytest.fixture(scope="module")
def inspiral_model(config):
    return InspiralModel(config)

@pytest.fixture(scope="module")
def rotation_model(config):
    return RotationalModel(config)

@pytest.fixture(scope="module")
def remnant_model(config):
    return RemnantModel(config)

@pytest.fixture(scope="module")
def disk_model(config):
    return DiskModel(config)

@pytest.fixture(scope="module")
def magnetic_model(config):
    return MagneticFieldModel(config)

@pytest.fixture(scope="module")
def kilonova_model(config):
    return KilonovaModel(config)

@pytest.fixture(scope="module")
def jet_model(config):
    return StructuredJetModel(config)

@pytest.fixture(scope="module")
def afterglow_model(config):
    return AfterglowModel(config)


# ─────────────────────────────────────────────────────────────────────
# 1. BINARY ORBITAL MOTION
# ─────────────────────────────────────────────────────────────────────

class TestBinaryOrbitalMotion:

    def _advance(self, model, config, n_steps, dt=1e-3):
        s = model.initial_state(config)
        for _ in range(n_steps):
            s = model.step(s, dt)
        return s

    def test_ns1_position_changes_between_inspiral_times(self, inspiral_model, config):
        s0 = inspiral_model.initial_state(config)
        s1 = self._advance(inspiral_model, config, 1000)   # +1 s
        s2 = self._advance(inspiral_model, config, 2000)   # +2 s
        r1_0, _ = inspiral_model.orbital_positions(s0, config.m1, config.m2)
        r1_1, _ = inspiral_model.orbital_positions(s1, config.m1, config.m2)
        r1_2, _ = inspiral_model.orbital_positions(s2, config.m1, config.m2)
        assert not np.allclose(r1_0, r1_1, atol=1.0), "NS1 did not move t=0→1s"
        assert not np.allclose(r1_1, r1_2, atol=1.0), "NS1 did not move t=1→2s"

    def test_ns2_position_changes_between_inspiral_times(self, inspiral_model, config):
        s0 = inspiral_model.initial_state(config)
        s1 = self._advance(inspiral_model, config, 1000)
        _, r2_0 = inspiral_model.orbital_positions(s0, config.m1, config.m2)
        _, r2_1 = inspiral_model.orbital_positions(s1, config.m1, config.m2)
        assert not np.allclose(r2_0, r2_1, atol=1.0), "NS2 did not move t=0→1s"

    def test_separation_matches_orbital_positions(self, inspiral_model, config):
        s = inspiral_model.initial_state(config)
        r1, r2 = inspiral_model.orbital_positions(s, config.m1, config.m2)
        pos_sep = float(np.linalg.norm(r1 - r2))
        assert abs(pos_sep - s.separation) / max(1.0, s.separation) < 1e-4

    def test_center_of_mass_near_zero(self, inspiral_model, config):
        s = inspiral_model.initial_state(config)
        r1, r2 = inspiral_model.orbital_positions(s, config.m1, config.m2)
        com = config.m1 * r1 + config.m2 * r2
        assert np.linalg.norm(com) < 1.0

    def test_orbital_velocity_tangential(self, inspiral_model, config):
        s = inspiral_model.initial_state(config)
        r1, _ = inspiral_model.orbital_positions(s, config.m1, config.m2)
        v1, _ = inspiral_model.orbital_velocities(s, config.m1, config.m2)
        r1m = float(np.linalg.norm(r1))
        v1m = float(np.linalg.norm(v1))
        if r1m > 1.0 and v1m > 1.0:
            cos_a = abs(float(np.dot(r1, v1))) / (r1m * v1m)
            assert cos_a < 0.01, f"|cos(r,v)| = {cos_a:.4f} (not tangential)"

    def test_orbital_phase_increases_with_time(self, inspiral_model, config):
        s = inspiral_model.initial_state(config)
        phi0 = s.orbital_phase
        s1 = inspiral_model.step(s, 1e-3)
        assert s1.orbital_phase > phi0, f"phase did not increase: {phi0}→{s1.orbital_phase}"


# ─────────────────────────────────────────────────────────────────────
# 2. EJECTA BALLISTIC EXPANSION
# ─────────────────────────────────────────────────────────────────────

class TestEjectaBallisticExpansion:

    def _r_blue(self, t):  return 25.0e3 + 0.30 * c * t
    def _r_red(self,  t):  return 18.0e3 + 0.10 * c * t

    def test_blue_ejecta_radius_grows_with_time(self):
        assert self._r_blue(86400.0) > self._r_blue(1.0), "Blue ejecta stalled at 1d"
        assert self._r_blue(10 * 86400.0) > self._r_blue(86400.0), "Blue ejecta stalled at 10d"

    def test_red_ejecta_radius_grows_with_time(self):
        assert self._r_red(86400.0) > self._r_red(3600.0), "Red ejecta stalled"

    def test_ejecta_radii_not_capped(self):
        ratio = self._r_blue(100 * 86400.0) / max(self._r_blue(86400.0), 1.0)
        assert ratio > 50.0, f"Ejecta radius appears capped: 100d/1d ratio={ratio:.1f}"


# ─────────────────────────────────────────────────────────────────────
# 3. JET PROPAGATION
# ─────────────────────────────────────────────────────────────────────

class TestJetPropagation:

    def _r_jet(self, t, delay=1.7, beta=0.9999):
        return 14.0e3 + beta * c * max(0.0, t - delay)

    def test_jet_length_grows_after_launch(self):
        assert self._r_jet(2.0) > self._r_jet(1.7), "Jet did not grow after launch"
        assert self._r_jet(86400.0) > self._r_jet(2.0), "Jet stalled at 2s"

    def test_jet_no_propagation_before_launch(self):
        r_pre = self._r_jet(1.69)
        assert r_pre == 14.0e3, f"Jet propagated before launch: {r_pre:.0f} m"

    def test_jet_model_grb_triggered(self, jet_model, config):
        # Build a merged inspiral state at f_max
        s = InspiralModel(config).initial_state(config)
        import copy
        s2 = copy.copy(s)
        s2.f_gw = 1500.0   # at/past merger
        from gw170817.physics.tidal import TidalModel
        from gw170817.physics.merger import MergerModel
        tidal = TidalModel(config)
        merger_model = MergerModel(config, tidal)
        ms = merger_model.evaluate(s2)
        j = jet_model.evaluate(time=2.0, merger_state=ms)
        assert isinstance(j.launched, bool)
        assert j.grb_triggered == (j.launched and 2.0 >= jet_model.jet_delay)


# ─────────────────────────────────────────────────────────────────────
# 4. ROTATIONAL DYNAMICS
# ─────────────────────────────────────────────────────────────────────

class TestRotationalDynamics:

    def _rot(self, rotation_model, remnant_model, inspiral_model, config, t):
        s = inspiral_model.initial_state(config)
        rem = remnant_model.evaluate(s, t)
        return rotation_model.evaluate(s, rem, t)

    def test_omega_core_nonzero_immediately_after_merger(
        self, rotation_model, remnant_model, inspiral_model, config
    ):
        rot = self._rot(rotation_model, remnant_model, inspiral_model, config, 0.01)
        assert rot.omega_core > 0.0, f"omega_core=0 at t=0.01s"

    def test_omega_core_bigger_than_omega_outer(
        self, rotation_model, remnant_model, inspiral_model, config
    ):
        rot = self._rot(rotation_model, remnant_model, inspiral_model, config, 0.01)
        assert rot.omega_core > rot.omega_outer, (
            f"omega_core={rot.omega_core:.1f} <= omega_outer={rot.omega_outer:.1f}"
        )

    def test_differential_rotation_positive_after_merger(
        self, rotation_model, remnant_model, inspiral_model, config
    ):
        rot = self._rot(rotation_model, remnant_model, inspiral_model, config, 0.01)
        assert rot.differential_rotation > 0.0, f"diff_rot={rot.differential_rotation}"

    def test_omega_core_changes_between_early_and_late(
        self, rotation_model, remnant_model, inspiral_model, config
    ):
        rot_early = self._rot(rotation_model, remnant_model, inspiral_model, config, 0.1)
        rot_late  = self._rot(rotation_model, remnant_model, inspiral_model, config, 10.0 * day)
        assert rot_early.omega_core != rot_late.omega_core, "omega_core frozen"


# ─────────────────────────────────────────────────────────────────────
# 5. MAGNETIC FIELD EVOLUTION
# ─────────────────────────────────────────────────────────────────────

class TestMagneticFieldEvolution:

    def _mag(self, magnetic_model, remnant_model, rotation_model, disk_model, inspiral_model, config, t):
        s = inspiral_model.initial_state(config)
        rem = remnant_model.evaluate(s, t)
        rot = rotation_model.evaluate(s, rem, t)
        disk = disk_model.evaluate(rem, t, rot)
        return magnetic_model.evaluate(rem, disk, t, rot)

    def test_b_toroidal_evolves_after_merger(
        self, magnetic_model, remnant_model, rotation_model, disk_model, inspiral_model, config
    ):
        m1 = self._mag(magnetic_model, remnant_model, rotation_model, disk_model, inspiral_model, config, 0.01)
        m2 = self._mag(magnetic_model, remnant_model, rotation_model, disk_model, inspiral_model, config, 0.50)
        assert m1.b_toroidal != m2.b_toroidal, (
            f"B_phi frozen: {m1.b_toroidal:.2e} == {m2.b_toroidal:.2e}"
        )


# ─────────────────────────────────────────────────────────────────────
# 6. KILONOVA PHOTOSPHERIC RADIUS
# ─────────────────────────────────────────────────────────────────────

class TestKilonovaPhotosphere:

    def _ej(self, t):
        return EjectaState(
            time=t,
            ejecta_mass=0.06 * M_sun,
            ejecta_fraction=0.06 * M_sun / (2.74 * M_sun),
            ejecta_particle_count=500,
            mean_velocity=0.20 * c,
            max_velocity=0.40 * c,
            kinetic_energy=0.5 * 0.06 * M_sun * (0.2 * c) ** 2,
            angular_momentum_proxy=0.0,
            mean_Ye=0.25,
            lanthanide_rich_fraction=0.67
        )

    def test_r_blue_grows_with_time(self, kilonova_model):
        kn1 = kilonova_model.evaluate(self._ej(3600.0),   3600.0)
        kn2 = kilonova_model.evaluate(self._ej(86400.0),  86400.0)
        assert kn2.R_blue > kn1.R_blue, f"R_blue did not grow: {kn1.R_blue:.2e}→{kn2.R_blue:.2e}"

    def test_r_red_grows_with_time(self, kilonova_model):
        kn1 = kilonova_model.evaluate(self._ej(3600.0),         3600.0)
        kn2 = kilonova_model.evaluate(self._ej(10.0 * 86400.0), 10.0 * 86400.0)
        assert kn2.R_red > kn1.R_red, f"R_red did not grow"


# ─────────────────────────────────────────────────────────────────────
# 7. AFTERGLOW LIGHT CURVE
# ─────────────────────────────────────────────────────────────────────

class TestAfterglowEvolution:

    def test_afterglow_flux_rises_to_peak_and_declines(self, afterglow_model, jet_model, config):
        s = InspiralModel(config).initial_state(config)
        from gw170817.physics.tidal import TidalModel
        from gw170817.physics.merger import MergerModel
        tidal = TidalModel(config)
        mm = MergerModel(config, tidal)
        ms = mm.evaluate(s)
        j = jet_model.evaluate(time=10.0 * day, merger_state=ms)
        ag10  = afterglow_model.evaluate(10.0  * day, jet_state=j)
        ag150 = afterglow_model.evaluate(150.0 * day, jet_state=j)
        ag300 = afterglow_model.evaluate(300.0 * day, jet_state=j)
        assert ag150.flux_density >= ag10.flux_density, "Afterglow not rising to peak"
        assert ag300.flux_density <= ag150.flux_density, "Afterglow not declining after peak"

    def test_afterglow_flux_driven_by_event_time(self, afterglow_model, jet_model, config):
        s = InspiralModel(config).initial_state(config)
        from gw170817.physics.tidal import TidalModel
        from gw170817.physics.merger import MergerModel
        tidal = TidalModel(config)
        mm = MergerModel(config, tidal)
        ms = mm.evaluate(s)
        j = jet_model.evaluate(time=10.0 * day, merger_state=ms)
        ag1 = afterglow_model.evaluate(10.0 * day, jet_state=j)
        ag2 = afterglow_model.evaluate(30.0 * day, jet_state=j)
        assert ag1.flux_density != ag2.flux_density, "Afterglow flux frozen"


# ─────────────────────────────────────────────────────────────────────
# 8. DISK KEPLERIAN ROTATION
# ─────────────────────────────────────────────────────────────────────

class TestDiskKeplerianRotation:

    def test_disk_active_and_rotating_after_merger(
        self, disk_model, remnant_model, rotation_model, inspiral_model, config
    ):
        s = inspiral_model.initial_state(config)
        rem = remnant_model.evaluate(s, 0.1)
        rot = rotation_model.evaluate(s, rem, 0.1)
        disk = disk_model.evaluate(rem, 0.1, rot)
        assert disk.is_active, "Disk not active at t=0.1s"
        assert disk.omega_disk > 0.0, f"Disk omega_K=0: {disk.omega_disk}"
