"""
Unit tests for Disk Rotational Dynamics & Keplerian Profile.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.disk import DiskModel
from gw170817.physics.rotation import RotationalModel


def test_disk_keplerian_rotation():
    cfg = SimConfig()
    insp_model = InspiralModel(cfg)
    insp_st = insp_model.initial_state()
    rem_model = RemnantModel(cfg)
    rem_st = rem_model.evaluate(insp_st, 0.02)

    disk_model = DiskModel(cfg)
    rot_model = RotationalModel(cfg)
    rot_st = rot_model.evaluate(insp_st, rem_st, 0.02)
    disk_st = disk_model.evaluate(rem_st, 0.02, rot_st)

    assert disk_st.is_active
    assert disk_st.omega_disk > 0.0
    assert disk_st.v_phi > 0.0
    assert disk_st.specific_angular_momentum > 0.0

    # Keplerian angular velocity decreases with radius
    omega_in = disk_model.omega_keplerian(rem_st.mass, disk_st.inner_radius)
    omega_out = disk_model.omega_keplerian(rem_st.mass, disk_st.outer_radius)
    assert omega_in > omega_out
