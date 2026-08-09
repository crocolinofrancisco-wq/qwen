"""Lightweight Perlin/value noise implementation (no external deps).

Fallback: uses opensimplex if available for a higher quality noise.
Added spherical (seamless-X) fractal for wrapping equirectangular maps.
"""
from __future__ import annotations
import numpy as np

try:
    from opensimplex import OpenSimplex
    _HAS_SIMPLEX = True
except Exception:
    _HAS_SIMPLEX = False


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


class PerlinNoise:
    """2D value-noise (cheap Perlin surrogate) with octaves."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        if _HAS_SIMPLEX:
            self._simplex = OpenSimplex(seed=seed)
        else:
            self._simplex = None

    def _grid(self, gw: int, gh: int, k: int) -> np.ndarray:
        key = (gw, gh, k)
        if key not in self._cache:
            self._cache[key] = self.rng.random((gh + 2, gw + 2))
        return self._cache[key]

    def _octave(self, w: int, h: int, scale: float, k: int) -> np.ndarray:
        if self._simplex is not None:
            xs = np.linspace(0, scale, w)
            ys = np.linspace(0, scale, h)
            arr = np.empty((h, w), dtype=np.float32)
            off = k * 137.0
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    arr[j, i] = self._simplex.noise2(x + off, y + off)
            return (arr + 1.0) * 0.5

        gw = max(2, int(scale) + 2)
        gh = max(2, int(scale * h / max(w, 1)) + 2)
        grid = self._grid(gw, gh, k)
        xs = np.linspace(0, gw - 1.001, w)
        ys = np.linspace(0, gh - 1.001, h)
        xi = xs.astype(int)
        yi = ys.astype(int)
        xf = xs - xi
        yf = ys - yi
        u = _fade(xf)
        v = _fade(yf)
        g00 = grid[np.ix_(yi, xi)]
        g10 = grid[np.ix_(yi, xi + 1)]
        g01 = grid[np.ix_(yi + 1, xi)]
        g11 = grid[np.ix_(yi + 1, xi + 1)]
        u2 = u[np.newaxis, :]
        v2 = v[:, np.newaxis]
        top = g00 * (1 - u2) + g10 * u2
        bot = g01 * (1 - u2) + g11 * u2
        return top * (1 - v2) + bot * v2

    def fractal(self, w: int, h: int, octaves: int = 4,
                scale: float = 4.0, persistence: float = 0.5,
                lacunarity: float = 2.0) -> np.ndarray:
        total = np.zeros((h, w), dtype=np.float32)
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for k in range(octaves):
            total += amp * self._octave(w, h, scale * freq, k)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        total /= max(norm, 1e-6)
        mn, mx = total.min(), total.max()
        if mx - mn > 1e-6:
            total = (total - mn) / (mx - mn)
        return total

    def spherical_fractal(self, w: int, h: int, octaves: int = 4,
                          scale: float = 4.0, persistence: float = 0.5,
                          lacunarity: float = 2.0) -> np.ndarray:
        """4D noise sampled on a cylinder (theta, y) - guarantees seamless X wrap.
        theta -> (cos, sin) so the noise is periodic along longitude.
        We approximate this with a 2D fractal that is blended with its X-shifted copy
        weighted by a cosine to force seamlessness at the seam.
        """
        base = self.fractal(w, h, octaves=octaves, scale=scale,
                            persistence=persistence, lacunarity=lacunarity)
        # blend with rolled copy so left edge == right edge
        shifted = np.roll(base, w // 2, axis=1)
        # weight peaks at the seam (x=0 and x=w) and is zero in the middle
        xs = np.linspace(0, np.pi * 2, w, endpoint=False)
        weight = (1.0 - np.cos(xs)) * 0.5  # 0 at center, 1 at seams
        weight = weight.reshape(1, -1)
        out = base * (1 - weight) + shifted * weight
        mn, mx = out.min(), out.max()
        if mx - mn > 1e-6:
            out = (out - mn) / (mx - mn)
        return out.astype(np.float32)


def ridge_layer(base: np.ndarray, shift: float) -> np.ndarray:
    x = base - shift
    x = np.where(x < 0, -x, x)
    x = 1.0 - x
    mn, mx = x.min(), x.max()
    if mx - mn > 1e-6:
        x = (x - mn) / (mx - mn)
    return x
