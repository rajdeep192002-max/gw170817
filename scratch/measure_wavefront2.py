import sys, time
# Import real Taichi before modifying sys.path
import taichi as ti
# Now add project root for our modules
PROJECT_ROOT = r'c:/Users/rajde/Documents/Gravitational_wave_events/blackhole'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from gw170817.visualization.wave_propagation import GWWavefrontPropagation

def profile(event_time):
    wp = GWWavefrontPropagation()
    t0 = time.perf_counter()
    state = wp.update(event_time, f_gw=100.0, merger_active=(event_time >= 0), h_plus=1e-21, h_cross=0.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f'event_time={event_time:.3f}s  wall_time={elapsed_ms:.2f}ms  n_vertices={state.n_vertices}')

for et in [-0.5, 0.0, 0.1, 1.0]:
    profile(et)
