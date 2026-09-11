"""
Unit tests for Angular Momentum Conservation & Partitioning Budget.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.remnant import RemnantModel
from gw170817.physics.rotation import RotationalModel


def test_angular_momentum_budget_conservation():
    cfg = SimConfig()
    insp_model = InspiralModel(cfg)
    insp_st = insp_model.initial_state()
    rem_model = RemnantModel(cfg)
    rot_model = RotationalModel(cfg)

    for t_val in [0.001, 0.01, 0.05, 0.20, 1.0]:
        rem_st = rem_model.evaluate(insp_st, t_val)
        rot_st = rot_model.evaluate(insp_st, rem_st, t_val)

        j_sum = (
            rot_st.j_remnant
            + rot_st.j_disk
            + rot_st.j_ejecta
            + rot_st.j_gw_loss
            + rot_st.j_other_loss
        )

        assert rot_st.angular_momentum > 0.0
        # Check strict angular momentum sum balance within floating point precision
        assert np.isclose(j_sum, rot_st.l_orb, rtol=1.0e-5)
        assert rot_st.j_remnant > 0.0
        assert rot_st.j_disk >= 0.0
        assert rot_st.j_ejecta >= 0.0
        assert rot_st.j_gw_loss >= 0.0
        assert rot_st.j_other_loss >= 0.0
