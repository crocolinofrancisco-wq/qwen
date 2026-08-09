"""Noise sampled natively on the unit sphere.

All procedural fields are generated in 3D first (this module) and then
projected onto the flat 2D map.  Because longitude 0 and 2*pi are the *same
point* in 3D, the projection is perfectly continuous across the horizontal
edges of the map and contains NO seam-blend artefact (the old
``spherical_fractal`` cosine-blend produced a visible cut at the edges and a
ghost line in the middle of the map — both are gone).

Performance: lattice values come from a fully vectorized integer hash
(no per-cell dict caching), so even high-frequency octaves are fast.
"""
from __future__ import annotations
import numpy as np

from .sphere import unit_sphere_points


def _hash3(i, j, k, seed):
    """Deterministic pseudo-random value in [0, 1) for lattice cell (i, j, k).
    Fully vectorized integer mixing (wraps mod 2^64)."""
    n = (i.astype(np.uint64) * np.uint64(0x8DA6B3431)) ^ \
        (j.astype(np.uint64) * np.uint64(0xD8163841)) ^ \
        (k.astype(np.uint64) * np.uint64(0xCB1AB31F)) ^ \
        np.uint64(seed & 0x7FFFFFFFFFFFFFFF)
    n = (n * np.uint64(0x9E3779B185EBCA6B)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    n ^= n >> np.uint64(29)
    n = (n * np.uint64(0xC2B2AE3D27D4EB4F)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    n ^= n >> np.uint64(32)
    return (n & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


class SphericalNoise:
    """fBm / ridged value noise evaluated directly on sphere points (H, W, 3)."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def noise3(self, x, y, z, octave: int = 0):
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        zi = np.floor(z).astype(np.int64)
        xf = self._fade(x - xi)
        yf = self._fade(y - yi)
        zf = self._fade(z - zi)
        s = self.seed + octave * 1013904223

        c = {}
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    c[(di, dj, dk)] = _hash3(xi + di, yi + dj, zi + dk, s)

        c00 = c[(0, 0, 0)] * (1 - xf) + c[(1, 0, 0)] * xf
        c10 = c[(0, 1, 0)] * (1 - xf) + c[(1, 1, 0)] * xf
        c01 = c[(0, 0, 1)] * (1 - xf) + c[(1, 0, 1)] * xf
        c11 = c[(0, 1, 1)] * (1 - xf) + c[(1, 1, 1)] * xf
        c0 = c00 * (1 - yf) + c10 * yf
        c1 = c01 * (1 - yf) + c11 * yf
        return c0 * (1 - zf) + c1 * zf

    def fbm(self, xyz: np.ndarray, scale: float = 3.0, octaves: int = 4,
            persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        """Fractal Brownian motion on sphere points -> values in [0, 1]."""
        total = np.zeros(xyz.shape[:2], dtype=np.float64)
        amp, freq, norm = 1.0, scale, 0.0
        for o in range(octaves):
            p = xyz * freq + freq * 17.31  # decorrelate octaves
            total += amp * self.noise3(p[..., 0], p[..., 1], p[..., 2], octave=o)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        total /= max(norm, 1e-9)
        mn, mx = total.min(), total.max()
        if mx - mn > 1e-9:
            total = (total - mn) / (mx - mn)
        return total.astype(np.float32)

    def ridged(self, xyz: np.ndarray, scale: float = 4.0, octaves: int = 3,
               shift: float = 0.5) -> np.ndarray:
        """Fractal Ridge Blending (Devote): smooth base in [0,1], shifted by
        `shift`, negative side flipped, whole function flipped and rescaled to
        [0,1] -> sharp peaks."""
        base = self.fbm(xyz, scale=scale, octaves=octaves)
        x = np.abs(base - shift)
        x = 1.0 - x
        mn, mx = x.min(), x.max()
        if mx - mn > 1e-9:
            x = (x - mn) / (mx - mn)
        return x.astype(np.float32)

    def fbm_latlon(self, lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
        """Convenience: sample fBm directly from lat/lon grids."""
        return self.fbm(unit_sphere_points(lat, lon), **kw)

    def ridged_latlon(self, lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
        return self.ridged(unit_sphere_points(lat, lon), **kw)
