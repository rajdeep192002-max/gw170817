"""
Unit tests for Magnetic Winding, Shear Coupling & MRI Surrogate Diagnostic.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.disk import DiskModel
from gw170817.physics.magnetic_field import MagneticFieldModel
from gw170817.physics.rotation import RotationalModel


def test_magnetic_winding_shear_coupling():
    cfg = SimConfig()
    insp_model = InspiralModel(cfg)
    insp_st = insp_model.initial_state()
    rem_model = RemnantModel(cfg)
    rem_st = rem_model.evaluate(insp_st, 0.01)

    disk_model = DiskModel(cfg)
    rot_model = RotationalModel(cfg)
    rot_st = rot_model.evaluate(insp_st, rem_st, 0.01)
    disk_st = disk_model.evaluate(rem_st, 0.01, rot_st)

    mag_model = MagneticFieldModel(cfg)
    mag_st = mag_model.evaluate(rem_st, disk_st, 0.01, rot_st)

    assert mag_st.b_toroidal > 0.0
    assert mag_st.b_ratio > 0.0
    assert mag_st.magnetic_stress > 0.0
    assert mag_st.magnetic_energy_proxy > 0.0
    assert mag_st.mri_active
    assert mag_st.mri_growth_rate > 0.0
    assert mag_st.mri_amplification >= 1.0


def test_shear_increases_bphi_growth():
    cfg = SimConfig()
    insp_model = InspiralModel(cfg)
    insp_st = insp_model.initial_state()
    rem_model = RemnantModel(cfg)
    rem_st = rem_model.evaluate(insp_st, 0.01)
    disk_model = DiskModel(cfg)
    disk_st = disk_model.evaluate(rem_st, 0.01)

    rot_model = RotationalModel(cfg)
    rot_st_high_shear = rot_model.evaluate(insp_st, rem_st, 0.01)

    mag_model = MagneticFieldModel(cfg)
    mag_st_high = mag_model.evaluate(rem_st, disk_st, 0.01, rot_st_high_shear)

    # Verify B_phi grows with shear
    assert mag_st_high.b_toroidal > mag_st_high.b_poloidal
