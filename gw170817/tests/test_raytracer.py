"""
Unit tests for Vulkan GPU Schwarzschild RK4 Raytracer.
"""
import pytest
import numpy as np
from gw170817.config import SimConfig
from gw170817.simulation.engine import GW170817Simulation
from gw170817.visualization.raytracer import SchwarzschildRaytracer
from gw170817.visualization.background import BackgroundStarfield


def test_raytracer_initialization_and_rendering():
    """Verify SchwarzschildRaytracer initializes, generates sky texture, and renders output image."""
    cfg = SimConfig(mode="DEV")
    sim = GW170817Simulation(config=cfg)
    bg = BackgroundStarfield(width=512, height=288)
    raytracer = SchwarzschildRaytracer(bg=bg, width=256, height=144)

    # Render with lensing OFF
    raytracer.render(
        ns1_pos=(-50.0e3, 0.0, 0.0),
        ns2_pos=(50.0e3, 0.0, 0.0),
        m1=1.36,
        m2=1.36,
        lensing_active=False
    )
    
    img_off = raytracer.output_img.to_numpy()
    assert img_off.shape == (144, 256, 3)
    assert np.all(np.isfinite(img_off))

    # Render with lensing ON (with compact object near ray path)
    raytracer.render(
        ns1_pos=(0.0, 0.0, 1.0e6),
        ns2_pos=(100.0e3, 0.0, 1.0e6),
        m1=1.36,
        m2=1.36,
        lensing_active=True,
        enhanced_scale=100.0
    )
    
    img_on = raytracer.output_img.to_numpy()
    assert img_on.shape == (144, 256, 3)
    assert np.all(np.isfinite(img_on))

    # Curved geodesics (lensing ON) should produce distinct background deflection from straight rays (lensing OFF)
    diff = np.abs(img_on.astype(float) - img_off.astype(float))
    assert np.max(diff) > 0.0
