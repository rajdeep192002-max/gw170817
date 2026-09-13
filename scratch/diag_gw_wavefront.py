"""
Test expansion and visibility of GW wavefront.
"""
import sys
sys.path.insert(0, '.')
import numpy as np

from gw170817.config import SimConfig
from gw170817.visualization.dashboard import ScientificDashboard

def test_expansion():
    print("=== TESTING GW WAVEFRONT EXPANSION & VISIBILITY ===")
    dash = ScientificDashboard(config=SimConfig(mode="DEV"))

    # Test pre-merger trigger in GW mode
    dash.set_view_mode("GW")
    dash.wave_propagation.trigger(-1.0)
    st = dash.engine.current_state
    wf_st = dash.wave_propagation.update(-1.0, f_gw=100.0, merger_active=False)
    print(f"Pre-merger GW mode active: {wf_st.active}, n_vertices: {wf_st.n_vertices}")

    # Test expansion across event times
    event_times = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    dash.wave_propagation.trigger(0.0)

    for t in event_times:
        wf_st = dash.wave_propagation.update(t, f_gw=100.0, merger_active=True)
        verts = dash.wave_propagation.line_vertices[:wf_st.n_vertices]
        max_r = np.max(np.linalg.norm(verts, axis=1)) / 1e3 if wf_st.n_vertices > 0 else 0.0
        print(f"t = {t:5.2f} s | active={wf_st.active} | base_r={wf_st.radius/1e3:6.2f} km | max_vert_r={max_r:6.2f} km | verts={wf_st.n_vertices}")

if __name__ == "__main__":
    test_expansion()
