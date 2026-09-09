"""
Test suite for Taichi reduced-order BNS merger particle dynamics.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import M_sun, c
from gw170817.simulation.particles import initialize_taichi, ParticleSystem
from gw170817.physics.inspiral import InspiralModel, InspiralState
from gw170817.physics.tidal import TidalModel
from gw170817.physics.merger import MergerModel
from gw170817.simulation.merger_dynamics import MergerDynamics


def test_merger_dynamics():
    print('=== TEST 007 TAICHI MERGER DYNAMICS VALIDATION ===')

    # 1. Initialize backend
    backend = initialize_taichi('vulkan')
    print(f'1. Taichi Backend: {backend.upper()}')

    # 2. Config and models
    config = SimConfig(mode='DEV', seed=42)
    psys = ParticleSystem(config)
    inspiral_model = InspiralModel(config)
    tidal_model = TidalModel(config)
    merger_model = MergerModel(config, tidal_model)

    dynamics = MergerDynamics(psys, inspiral_model, tidal_model, merger_model)
    diag0 = dynamics.compute_diagnostics()

    print(f'2. Initial Diagnostics:')
    print(f'   - Total Mass: {diag0["total_mass"] / M_sun:.4f} M_sun (Expected: {config.M_total / M_sun:.4f})')
    print(f'   - COM Position [m]: {diag0["com_pos"]}')
    print(f'   - Separation Estimate [km]: {diag0["separation_estimate"] / 1e3:.2f} km')
    print(f'   - Max Particle Speed [m/s]: {diag0["max_particle_speed"]:.4e}')
    print(f'   - Contact Fraction: {diag0["contact_fraction"]:.4f}')

    # Check 1: Total Mass Conservation & COM bounds
    initial_mass = diag0['total_mass']
    assert abs(initial_mass - config.M_total) / config.M_total < 1e-5, 'Initial mass mismatch!'
    assert np.linalg.norm(diag0['com_pos']) / config.initial_separation < 1e-3, 'COM offset too large!'

    # 3. Step physics for 20 steps
    print('\n3. Stepping physics (20 steps)...')
    for step_i in range(20):
        dynamics.step(config.dt_physics)

    diag1 = dynamics.compute_diagnostics()
    print(f'   - Step 20 Separation Estimate [km]: {diag1["separation_estimate"] / 1e3:.2f} km')
    print(f'   - Step 20 Max Particle Speed [m/s]: {diag1["max_particle_speed"]:.4e}')
    print(f'   - Step 20 Contact Fraction: {diag1["contact_fraction"]:.4f}')

    # Check 2: Mass conservation after steps
    assert abs(diag1['total_mass'] - initial_mass) / initial_mass < 1e-6, 'Mass not strictly conserved!'
    assert np.linalg.norm(diag1['com_pos']) / config.initial_separation < 1e-3, 'COM drifted unexpectedly!'
    assert diag1['max_particle_speed'] <= 0.5 * c, 'Speed exceeded clamp limit!'
    assert not diag1['has_nans'], 'NaNs detected!'
    assert not diag1['has_infs'], 'Infs detected!'

    # 4. Synthetic Merger Transition Test
    print('\n4. Running Synthetic Merger Transition Test...')
    # Force inspiral state close to merger (f_gw = 1200 Hz, separation ~ 30 km)
    synthetic_state = InspiralState(
        time=1.0, f_gw=1200.0, orbital_frequency=600.0, omega_orb=1200.0*np.pi,
        separation=30.0e3, orbital_phase=np.pi, df_dt=100.0, chirp_mass=config.chirp_mass
    )
    dynamics.inspiral_state = synthetic_state
    dynamics.step(0.001)

    diag_merger = dynamics.compute_diagnostics()
    print(f'   - Post-Contact Separation Estimate [km]: {diag_merger["separation_estimate"] / 1e3:.2f} km')
    print(f'   - Post-Contact Contact Fraction: {diag_merger["contact_fraction"]:.4f}')
    print(f'   - Post-Contact Max Speed [m/s]: {diag_merger["max_particle_speed"]:.4e}')

    assert diag_merger['contact_fraction'] > 0.5, 'Merger transition contact fraction should be > 0.5'
    assert not diag_merger['has_nans'] and not diag_merger['has_infs'], 'NaN/Inf in merger transition!'

    # 5. Determinism Check
    print('\n5. Determinism Check (re-instantiating with same seed = 42)...')
    dynamics2 = MergerDynamics(ParticleSystem(config), inspiral_model, tidal_model, merger_model)
    diag_det = dynamics2.compute_diagnostics()
    assert abs(diag0['total_mass'] - diag_det['total_mass']) < 1e-9, 'Deterministic mass mismatch!'

    print('\nALL TEST_MERGER_DYNAMICS CHECKS PASSED SUCCESSFULLY!')


if __name__ == '__main__':
    test_merger_dynamics()
