"""
Test suite for reduced-order dynamical ejecta model (Task 008).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import G, c, M_sun
from gw170817.simulation.particles import initialize_taichi, ParticleSystem
from gw170817.physics.inspiral import InspiralModel
from gw170817.physics.tidal import TidalModel
from gw170817.physics.merger import MergerModel
from gw170817.simulation.merger_dynamics import MergerDynamics
from gw170817.physics.ejecta import EjectaModel, EjectaState


def test_ejecta_model():
    print("=== TEST 008 REDUCED-ORDER DYNAMICAL EJECTA MODEL ===")

    # 1. Initialize backend
    backend = initialize_taichi("vulkan")
    print(f"1. Taichi Backend: {backend.upper()}")

    # 2. Setup system
    config = SimConfig(mode="DEV", seed=42)
    psys = ParticleSystem(config)
    inspiral_model = InspiralModel(config)
    tidal_model = TidalModel(config)
    merger_model = MergerModel(config, tidal_model)
    dynamics = MergerDynamics(psys, inspiral_model, tidal_model, merger_model)

    ejecta_model = EjectaModel(config)

    # 3. Initial binary check (a ~ 283.8 km)
    diag_com = psys.compute_diagnostics_numpy()
    ejecta_model.classify(psys, diag_com['com_pos'], diag_com['com_vel'])
    state0 = ejecta_model.compute_state(psys, 0.0)

    print("2. Initial Binary State Check:")
    print(f"   - Ejecta particle count: {state0.ejecta_particle_count} (Expected: 0)")
    print(f"   - Ejecta mass: {state0.ejecta_mass} kg (Expected: 0.0)")

    assert state0.ejecta_particle_count == 0, "Initial binary must have 0 ejecta particles"
    assert state0.ejecta_mass == 0.0, "Initial ejecta mass must be 0.0"

    # 4. Synthetic outward ejecta check
    print("\n3. Synthetic Outward & Inward Particle Tests:")
    com_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    com_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    pos_test = np.zeros((psys.max_particles, 3), dtype=np.float32)
    vel_test = np.zeros((psys.max_particles, 3), dtype=np.float32)

    # P0: Outward fast at x = 100 km, vx = 0.4 c
    pos_test[0] = [100.0e3, 0.0, 0.0]
    vel_test[0] = [0.4 * c, 0.0, 0.0]

    # P1: Inward fast at x = 100 km, vx = -0.4 c
    pos_test[1] = [100.0e3, 0.0, 0.0]
    vel_test[1] = [-0.4 * c, 0.0, 0.0]

    # P2: Slow near origin at x = 50 km, vx = 0.01 c
    pos_test[2] = [50.0e3, 0.0, 0.0]
    vel_test[2] = [0.01 * c, 0.0, 0.0]

    # P3: Outward polar fast at z = 100 km, vz = 0.3 c
    pos_test[3] = [0.0, 0.0, 100.0e3]
    vel_test[3] = [0.0, 0.0, 0.3 * c]

    psys.pos.from_numpy(pos_test)
    psys.vel.from_numpy(vel_test)

    ejecta_model.classify(psys, com_pos, com_vel)

    flags = ejecta_model.ejecta_flag.to_numpy()
    ye = ejecta_model.Ye_field.to_numpy()

    print(f"   - P0 (Outward fast): ejecta={flags[0]} (Expected 1)")
    print(f"   - P1 (Inward fast):  ejecta={flags[1]} (Expected 0)")
    print(f"   - P2 (Slow inner):   ejecta={flags[2]} (Expected 0)")
    print(f"   - P3 (Polar fast):   ejecta={flags[3]} (Expected 1), Ye={ye[3]:.4f}")

    assert flags[0] == 1, "P0 must be classified as ejecta"
    assert flags[1] == 0, "P1 (inward velocity) must NOT be classified as ejecta"
    assert flags[2] == 0, "P2 (sub-escape velocity) must NOT be classified as ejecta"
    assert flags[3] == 1, "P3 (polar fast) must be classified as ejecta"
    assert ye[3] > ye[0], "Polar high-velocity ejecta should have higher Ye"

    # 5. Aggregated State & Conservation Checks
    state_synth = ejecta_model.compute_state(psys, 1.0)
    print("\n4. Aggregated Ejecta Diagnostics Check:")
    print(f"   - Ejecta Particle Count: {state_synth.ejecta_particle_count}")
    print(f"   - Ejecta Mass: {state_synth.ejecta_mass / M_sun:.6e} M_sun")
    print(f"   - Ejecta Fraction: {state_synth.ejecta_fraction:.6e}")
    print(f"   - Mean Velocity: {state_synth.mean_velocity / c:.4f} c")
    print(f"   - Max Velocity: {state_synth.max_velocity / c:.4f} c")
    print(f"   - Kinetic Energy: {state_synth.kinetic_energy:.4e} J")
    print(f"   - Mean Ye: {state_synth.mean_Ye:.4f}")
    print(f"   - Lanthanide Rich Fraction: {state_synth.lanthanide_rich_fraction:.4f}")

    assert state_synth.ejecta_particle_count == 2
    assert state_synth.ejecta_mass <= config.M_total, "Ejecta mass cannot exceed total mass"
    assert 0.0 <= state_synth.ejecta_fraction <= 1.0, "Ejecta fraction must be in [0, 1]"
    assert state_synth.mean_velocity >= 0.0, "Mean velocity must be non-negative"
    assert state_synth.max_velocity >= state_synth.mean_velocity, "Max velocity must be >= mean velocity"
    assert state_synth.kinetic_energy >= 0.0, "Kinetic energy must be non-negative"
    assert 0.05 <= state_synth.mean_Ye <= 0.45, "Mean Ye must be in [0.05, 0.45]"
    assert 0.0 <= state_synth.lanthanide_rich_fraction <= 1.0, "Lanthanide fraction must be in [0, 1]"

    # 6. Determinism Check
    print("\n5. Determinism Check:")
    ejecta_model2 = EjectaModel(config)
    ejecta_model2.classify(psys, com_pos, com_vel)
    state_det = ejecta_model2.compute_state(psys, 1.0)
    assert state_synth.ejecta_mass == state_det.ejecta_mass, "Deterministic ejecta mass mismatch!"

    # 7. Numerical Safety Check
    for field, val in state_synth.__dict__.items():
        if isinstance(val, (float, int)):
            assert np.isfinite(val), f"Field {field} is non-finite: {val}"

    print("\nALL TEST_EJECTA CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_ejecta_model()
