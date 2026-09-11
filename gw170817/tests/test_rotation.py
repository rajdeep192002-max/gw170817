"""
Unit tests for Rotational Dynamics & Remnant Rotation Profile.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.rotation import RotationalModel, RotationalState


def test_rotational_model_inspiral():
    cfg = SimConfig()
    insp_model = InspiralModel(cfg)
    insp_st = insp_model.initial_state()
    rem_model = RemnantModel(cfg)
    rem_st = rem_model.evaluate(insp_st, -0.1)

    rot_model = RotationalModel(cfg)
    rot_st = rot_model.evaluate(insp_st, rem_st, -0.1)

    assert rot_st.l_orb > 0.0
    assert rot_st.differential_rotation == 0.0
    assert rot_st.compactness > 0.0
    assert rot_st.gravitational_redshift > 0.0
    assert not np.isnan(rot_st.rotational_energy)


def test_differential_rotation_post_merger():
    cfg = SimConfig()
    insp_model = InspiralModel(cfg)
    insp_st = insp_model.initial_state()
    rem_model = RemnantModel(cfg)
    rem_st = rem_model.evaluate(insp_st, 0.01)

    rot_model = RotationalModel(cfg)
    rot_st = rot_model.evaluate(insp_st, rem_st, 0.01)

    # Core angular velocity must exceed outer angular velocity
    assert rot_st.omega_core > rot_st.omega_outer
    assert rot_st.differential_rotation > 0.0
    assert rot_st.shear > 0.0
    assert rot_st.T_over_W > 0.0
    assert rot_st.gravitational_redshift > 0.0


def test_transport_reduces_differential_rotation():
    cfg = SimConfig()
    insp_model = InspiralModel(cfg)
    insp_st = insp_model.initial_state()
    rem_model = RemnantModel(cfg)

    rot_model = RotationalModel(cfg)

    rem_st_early = rem_model.evaluate(insp_st, 0.005)
    rot_st_early = rot_model.evaluate(insp_st, rem_st_early, 0.005)

    rem_st_late = rem_model.evaluate(insp_st, 0.10)
    rot_st_late = rot_model.evaluate(insp_st, rem_st_late, 0.10)

    # Transport reduces differential rotation over time
    assert rot_st_late.differential_rotation < rot_st_early.differential_rotation
