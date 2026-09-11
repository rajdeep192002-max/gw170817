"""
2D Sky Background Starfield Texture Generator for BNS Mergers.

REDUCED-ORDER APPROXIMATION / SCIENTIFIC TRANSPARENCY:
Renders a high-quality deterministic 2D background starfield sky texture on GPU.
Used as the background texture field sampled by the GPU Schwarzschild RK4 geodesic raytracer.
"""
import taichi as ti
import numpy as np


@ti.data_oriented
class BackgroundStarfield:
    """
    Pre-renders background celestial starfield texture on GPU.
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.w = width
        self.h = height
        self.sky_texture = ti.Vector.field(3, dtype=ti.f32, shape=(self.h, self.w))
        self.generate_sky_texture_kernel(self.w, self.h)

    @ti.kernel
    def generate_sky_texture_kernel(self, w: ti.i32, h: ti.i32):
        """
        Generate deterministic multi-spectral background stars and diffuse galactic plane band.
        """
        for i, j in ti.ndrange(h, w):
            u = float(j) / float(w)
            v = float(i) / float(h)

            # Deep space dark background gradient with faint Milky Way diffuse band
            mw_band = ti.exp(-((v - 0.5) / 0.15) ** 2) * 0.08
            r_bg = 0.01 + mw_band * 0.5
            g_bg = 0.01 + mw_band * 0.6
            b_bg = 0.03 + mw_band * 0.9

            col = ti.Vector([r_bg, g_bg, b_bg])

            # Grid-based pseudo-random star dots
            gx = ti.floor(u * 64.0)
            gy = ti.floor(v * 64.0)
            rnd = ti.sin(gx * 12.9898 + gy * 78.233) * 43758.5453
            rnd -= ti.floor(rnd)

            if rnd > 0.915:
                cx = (gx + 0.5 + 0.3 * ti.sin(rnd * 17.0)) / 64.0
                cy = (gy + 0.5 + 0.3 * ti.cos(rnd * 31.0)) / 64.0

                dist = ti.sqrt((u - cx) ** 2 + (v - cy) ** 2)
                if dist < 0.006:
                    star_intensity = ti.pow(1.0 - (dist / 0.006), 2.0) * (0.6 + 0.8 * rnd)

                    c_idx = int(rnd * 100.0) % 3
                    s_col = ti.Vector([1.0, 1.0, 1.0])
                    if c_idx == 0:
                        s_col = ti.Vector([0.7, 0.85, 1.0])  # Hot blue star
                    elif c_idx == 1:
                        s_col = ti.Vector([1.0, 0.9, 0.6])   # Solar gold star

                    col += s_col * star_intensity

            self.sky_texture[i, j] = col
