"""
Regression test suite for adopted delayed-collapse Black Hole persistence:
Ensures the accreted black hole and surrounding accretion environment persist
throughout the entire continuous post-merger simulation (t = 15 s, 100 s, 1,000 s, 10,000 s, 1,000,000 s),
without reset, culling, or presentation-stage termination.

SCIENTIFIC CONTEXT:
ADOPTED DELAYED-COLLAPSE BH SCENARIO
- HMNS delayed-collapse phase (t ~ 0.08 s)
- Monotonic collapse: once is_black_hole == True, it persists for all subsequent event times
- Disk accretion continues with inward radial drift, Keplerian orbital motion, and capture recycling
- Pitch-dark central horizon shadow proxy with finite radius (R ~ 6.68 km)
"""
import pytest
import numpy as np
import taichi as ti

from gw170817.constants import G, c, M_sun
from gw170817.config import SimConfig
from gw170817.simulation.particles import ParticleSystem
from gw170817.simulation.engine import GW170817Simulation
from gw170817.simulation.multimessenger import MultiMessengerCoordinator
from gw170817.simulation.demo_director import DemoDirector, DemoStage
from gw170817.physics.remnant import RemnantModel, RemnantState
from gw170817.physics.disk import DiskModel, DiskState
from gw170817.visualization.renderer import ParticleRenderer


@pytest.fixture(scope="module")
def taichi_env():
    """Ensure Taichi is initialized."""
    try:
        ti.init(arch=ti.vulkan, implicit_prefer_if=True)
    except Exception:
        try:
            ti.init(arch=ti.cpu)
        except Exception:
            pass
    return True


@pytest.fixture
def test_setup(taichi_env):
    cfg = SimConfig(mode="DEV", seed=42)
    psys = ParticleSystem(cfg)
    renderer = ParticleRenderer(psys)
    coord = MultiMessengerCoordinator(config=cfg)
    director = DemoDirector(coordinator=coord, config=cfg)
    return {
        "config": cfg,
        "renderer": renderer,
        "coord": coord,
        "director": director
    }


class TestBlackHolePersistence:
    """Test suite covering the 12 requirements for adopted BH persistence."""

    def test_1_and_2_bh_formation_at_delayed_collapse_time(self, test_setup):
        """1 & 2. BH forms at delayed-collapse time (t = 0.08s) and is_black_hole becomes True."""
        rem_model = RemnantModel(config=test_setup["config"])
        insp_st = test_setup["coord"].engine.dynamics.inspiral_state

        # Pre-collapse (t = 0.04 s): HMNS stage
        st_pre = rem_model.evaluate(insp_st, event_time=0.04)
        assert not st_pre.is_black_hole, "Must not be BH before delayed-collapse time"
        assert st_pre.remnant_type == "HMNS"

        # At/after delayed collapse (t = 0.08 s): BH stage
        st_post = rem_model.evaluate(insp_st, event_time=0.08)
        assert st_post.is_black_hole, "Must collapse to BH at t_collapse_delay"
        assert st_post.remnant_type == "BH"
        assert st_post.scenario == "ADOPTED DELAYED-COLLAPSE BH SCENARIO" or "GW170817_LIKE" in st_post.scenario

    @pytest.mark.parametrize("t_event", [15.0, 100.0, 1_000.0, 10_000.0, 1_000_000.0])
    def test_3_to_7_bh_remains_true_at_large_event_times(self, test_setup, t_event):
        """3, 4, 5, 6, 7. BH remains True at +15s, +100s, +1,000s, +10,000s, +1,000,000s."""
        coord = test_setup["coord"]
        st = coord.evaluate_at_event_time(t_event)
        rem_st = coord.engine.remnant.evaluate(coord.engine.dynamics.inspiral_state, t_event)

        assert rem_st.is_black_hole is True, f"BH must persist at t = {t_event} s"
        assert rem_st.remnant_type == "BH", f"Remnant type must remain 'BH' at t = {t_event} s"
        assert st.bh_active is True, f"Coordinator bh_active must remain True at t = {t_event} s"
        assert st.is_black_hole is True, f"EventState is_black_hole must remain True at t = {t_event} s"

    @pytest.mark.parametrize("t_event", [15.0, 100.0, 1_000.0, 10_000.0, 1_000_000.0])
    def test_8_and_9_bh_mass_and_radius_remain_finite(self, test_setup, t_event):
        """8 & 9. BH mass and horizon radius remain finite and strictly positive."""
        coord = test_setup["coord"]
        rem_st = coord.engine.remnant.evaluate(coord.engine.dynamics.inspiral_state, t_event)

        # Mass: ~2.5 - 2.8 M_sun
        assert np.isfinite(rem_st.mass), f"Mass must be finite at t = {t_event} s"
        assert 2.0 * M_sun <= rem_st.mass <= 3.0 * M_sun, f"Mass must be physical BNS remnant mass: {rem_st.mass / M_sun:.2f} M_sun"

        # Horizon radius: R_H ~ 6 - 8 km
        assert np.isfinite(rem_st.radius), f"Radius must be finite at t = {t_event} s"
        assert 5.0e3 <= rem_st.radius <= 10.0e3, f"Horizon radius must be Kerr proxy ~6-8 km: {rem_st.radius / 1e3:.2f} km"

    @pytest.mark.parametrize("t_event", [15.0, 100.0, 1_000.0, 10_000.0, 1_000_000.0])
    def test_10_no_nan_inf_across_state(self, test_setup, t_event):
        """10. No NaN or Inf across physics models and GPU renderer buffers."""
        coord = test_setup["coord"]
        renderer = test_setup["renderer"]

        st = coord.evaluate_at_event_time(t_event)
        rem_st = coord.engine.remnant.evaluate(coord.engine.dynamics.inspiral_state, t_event)
        disk_st = coord.engine.disk.evaluate(rem_st, t_event)

        assert not np.isnan(rem_st.mass) and not np.isinf(rem_st.mass)
        assert not np.isnan(rem_st.radius) and not np.isinf(rem_st.radius)
        assert not np.isnan(rem_st.omega_rot) and not np.isinf(rem_st.omega_rot)
        assert not np.isnan(disk_st.disk_mass) and not np.isinf(disk_st.disk_mass)
        assert not np.isnan(disk_st.accretion_rate) and not np.isinf(disk_st.accretion_rate)

        # Update GPU particle fields
        renderer.update_remnant_particles(
            event_time=t_event,
            contact_frac=1.0,
            remnant_type_str=rem_st.remnant_type,
            horizon_radius_m=rem_st.radius
        )
        renderer.update_disk_particles(
            event_time=t_event,
            dt_vis=0.016,
            is_active=True,
            disk_progress=1.0,
            disk_mass_msun=disk_st.disk_mass_msun,
            is_black_hole=rem_st.is_black_hole
        )

        rem_pos = renderer.remnant_pos.to_numpy()
        rem_col = renderer.remnant_colors.to_numpy()
        disk_pos = renderer.disk_pos.to_numpy()
        disk_col = renderer.disk_colors.to_numpy()

        assert not np.isnan(rem_pos).any(), "Remnant positions must contain no NaN"
        assert not np.isinf(rem_pos).any(), "Remnant positions must contain no Inf"
        assert not np.isnan(rem_col).any(), "Remnant colors must contain no NaN"
        assert not np.isinf(rem_col).any(), "Remnant colors must contain no Inf"
        assert not np.isnan(disk_pos).any(), "Disk positions must contain no NaN"
        assert not np.isinf(disk_pos).any(), "Disk positions must contain no Inf"
        assert not np.isnan(disk_col).any(), "Disk colors must contain no NaN"
        assert not np.isinf(disk_col).any(), "Disk colors must contain no Inf"

    @pytest.mark.parametrize("t_event", [15.0, 100.0, 1_000.0, 10_000.0, 1_000_000.0])
    def test_11_accretion_state_remains_valid(self, test_setup, t_event):
        """11. Accretion state remains valid: disk mass > 0, mdot > 0, disk particles active."""
        coord = test_setup["coord"]
        renderer = test_setup["renderer"]

        rem_st = coord.engine.remnant.evaluate(coord.engine.dynamics.inspiral_state, t_event)
        disk_st = coord.engine.disk.evaluate(rem_st, t_event)

        assert disk_st.is_active is True
        assert disk_st.disk_mass > 0.0, f"Disk mass must remain strictly positive at t = {t_event} s"
        assert disk_st.disk_mass_msun > 0.0
        assert disk_st.accretion_rate > 0.0, f"Accretion rate must remain positive at t = {t_event} s"
        assert disk_st.inner_radius >= rem_st.radius

        # Step accretion on GPU renderer and verify active particle count and orbital motion
        renderer.update_disk_particles(
            event_time=t_event,
            dt_vis=0.016,
            is_active=True,
            disk_progress=1.0,
            disk_mass_msun=disk_st.disk_mass_msun,
            is_black_hole=rem_st.is_black_hole
        )
        pos1 = renderer.disk_pos.to_numpy().copy()

        # Step another visual dt
        renderer.update_disk_particles(
            event_time=t_event + 0.016,
            dt_vis=0.016,
            is_active=True,
            disk_progress=1.0,
            disk_mass_msun=disk_st.disk_mass_msun,
            is_black_hole=rem_st.is_black_hole
        )
        pos2 = renderer.disk_pos.to_numpy().copy()

        # All 2500 disk particles must be present and moving
        diff = np.linalg.norm(pos2 - pos1, axis=1)
        active_particles = np.sum(np.linalg.norm(pos2, axis=1) > 1.0)
        assert active_particles == renderer.n_disk_particles, f"All {renderer.n_disk_particles} disk particles must be active"
        assert np.all(diff > 0.0), "All disk particles must exhibit active orbital motion"

    def test_12_renderer_does_not_cull_bh_because_of_presentation_time(self, test_setup):
        """12. Renderer does not cull the BH or disk when presentation_time >= 15.0 s."""
        director = test_setup["director"]
        renderer = test_setup["renderer"]
        coord = test_setup["coord"]

        # Run director into CONTINUOUS_POST_MERGER stage (presentation time > 15.0 s)
        director.start()
        director.jump_to_stage_index(len(director.STAGE_SEQUENCE))
        director._presentation_time = 15.5
        director._sync_physics_for_presentation_time(15.5)

        st = coord.current_state
        rem_st = coord.engine.remnant.evaluate(coord.engine.dynamics.inspiral_state, st.event_time)
        disk_st = coord.engine.disk.evaluate(rem_st, st.event_time)

        assert director.current_stage == DemoStage.CONTINUOUS_POST_MERGER.value
        assert rem_st.is_black_hole is True
        assert st.bh_active is True

        renderer.update_remnant_particles(
            event_time=st.event_time,
            contact_frac=1.0,
            remnant_type_str=rem_st.remnant_type,
            horizon_radius_m=rem_st.radius
        )
        renderer.update_disk_particles(
            event_time=st.event_time,
            dt_vis=0.016,
            is_active=True,
            disk_progress=director.disk_progress,
            disk_mass_msun=disk_st.disk_mass_msun,
            is_black_hole=rem_st.is_black_hole
        )

        rem_pos = renderer.remnant_pos.to_numpy()
        disk_pos = renderer.disk_pos.to_numpy()

        # Remnant must NOT be placed offscreen [0, 0, -1e9]
        rem_dist = np.linalg.norm(rem_pos, axis=1)
        assert np.all(rem_dist < 1.0e6), "Remnant particles must be located at compact origin, not culled offscreen"
        assert np.mean(rem_dist) < 15.0e3, "Remnant horizon radius proxy must remain compact (~6.7 km)"

        # Disk must NOT be placed offscreen or collapsed to 0
        disk_dist = np.linalg.norm(disk_pos, axis=1)
        assert np.all(disk_dist > 10.0e3), "Disk particles must remain in physical orbit outside horizon"
        assert np.all(disk_dist < 200.0e3), "Disk particles must remain within outer torus radius"
