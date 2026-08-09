"""Perlin 3D vectorizado (se muestrea sobre la esfera → sin costuras),
fBM de N octavas y Fractal Ridge Blending (Devote) para picos afilados."""
from __future__ import annotations
import numpy as np


class Perlin:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        p = np.arange(256, dtype=np.int64)
        rng.shuffle(p)
        self.p = np.tile(p, 2)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _grad(h, x, y, z):
        h = h & 15
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def noise(self, x, y, z):
        """Perlin 3D clásico. Acepta arrays numpy. Retorna 0..1."""
        p = self.p
        xi = np.floor(x).astype(np.int64) & 255
        yi = np.floor(y).astype(np.int64) & 255
        zi = np.floor(z).astype(np.int64) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        zf = z - np.floor(z)
        u = self._fade(xf); v = self._fade(yf); w = self._fade(zf)
        aaa = p[p[p[xi] + yi] + zi];     aba = p[p[p[xi] + yi + 1] + zi]
        aab = p[p[p[xi] + yi] + zi + 1]; abb = p[p[p[xi] + yi + 1] + zi + 1]
        baa = p[p[p[xi + 1] + yi] + zi];     bba = p[p[p[xi + 1] + yi + 1] + zi]
        bab = p[p[p[xi + 1] + yi] + zi + 1]; bbb = p[p[p[xi + 1] + yi + 1] + zi + 1]
        x1 = self._lerp(self._grad(aaa, xf, yf, zf),     self._grad(baa, xf - 1, yf, zf), u)
        x2 = self._lerp(self._grad(aba, xf, yf - 1, zf), self._grad(bba, xf - 1, yf - 1, zf), u)
        y1 = self._lerp(x1, x2, v)
        x1 = self._lerp(self._grad(aab, xf, yf, zf - 1),     self._grad(bab, xf - 1, yf, zf - 1), u)
        x2 = self._lerp(self._grad(abb, xf, yf - 1, zf - 1), self._grad(bbb, xf - 1, yf - 1, zf - 1), u)
        return (self._lerp(y1, self._lerp(x1, x2, v), w) + 1.0) * 0.5

    @staticmethod
    def _lerp(a, b, t):
        return a + t * (b - a)

    def fbm(self, x, y, z, octaves=4, freq=1.0, lac=2.0, gain=0.5):
        """N capas: cada una más detallada (freq*lac) y con menor influencia (amp*gain)."""
        amp, tot, norm = 1.0, 0.0, 0.0
        fx, fy, fz = x * freq, y * freq, z * freq
        for _ in range(octaves):
            tot = tot + amp * self.noise(fx, fy, fz)
            norm += amp
            amp *= gain
            fx *= lac; fy *= lac; fz *= lac
        return tot / norm  # 0..1

    def ridge(self, x, y, z, freq, shift):
        """Fractal Ridge Blending (Devote), tal cual el DDG:
        1) función suave 0..1  → 2) shift en [0.35..0.65] → 3) espejar valores < 0
        → 4) flip de toda la función → 5) reajustar a 0..1. Resultado: picos afilados."""
        v = self.noise(x * freq, y * freq, z * freq)
        v = v - shift
        v = np.abs(v)
        v = -v
        mn, mx = float(v.min()), float(v.max())
        return (v - mn) / (mx - mn + 1e-12)