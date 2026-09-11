"""
Unit tests for magnetic field winding, MRI, and Poynting flux physics models.
"""
import pytest
from gw170817.config import SimConfig
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.disk import DiskModel
from gw170817.physics.magnetic_field import MagneticFieldModel
from gw170817.physics.inspiral import InspiralState


def test_magnetic_field_winding_and_poynting_flux():
    """Verify magnetic field winding, MRI growth, and Poynting flux calculation."""
    cfg = SimConfig()
    remnant = RemnantModel(config=cfg)
    disk = DiskModel(config=cfg)
    mag_model = MagneticFieldModel(config=cfg)

    dummy_insp = InspiralState(
        time=0.0, f_gw=1500.0, orbital_frequency=750.0, omega_orb=4712.0,
        separation=15.0e3, orbital_phase=0.0, df_dt=1e4, chirp_mass=1.188 * 1.989e30
    )

    rem_st = remnant.evaluate(dummy_insp, event_time=0.05)
    disk_st = disk.evaluate(rem_st, event_time=0.05)
    mag_st = mag_model.evaluate(rem_st, disk_st, event_time=0.05)

    assert mag_st.b_poloidal > mag_model.B0_pol
    assert mag_st.b_toroidal > 0.0
    assert mag_st.poynting_luminosity > 0.0
    assert 3.5 <= mag_st.collimation_angle_deg <= 25.0
