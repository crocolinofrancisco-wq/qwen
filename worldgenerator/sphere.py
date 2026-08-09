"""Utilidades de esfera unidad. La Fase 1 se genera sobre la esfera
(los bordes E/O del mapa son el mismo meridiano) y luego se proyecta
a un mapa equirectangular plano de width x height."""
from __future__ import annotations
import numpy as np


def latlon_to_xyz(lat, lon):
    cl = np.cos(lat)
    return np.stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)), axis=-1)


def pixel_grid(w, h):
    """Lat/lon por píxel (equirectangular) + posición xyz en la esfera."""
    lats = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lons = (np.arange(w) + 0.5) / w * 2.0 * np.pi - np.pi
    lon2, lat2 = np.meshgrid(lons, lats)
    return lat2, lon2, latlon_to_xyz(lat2, lon2)


def make_dots(ny, nx, jitter, rng):
    """Grilla de puntos no uniforme (con ruido) sobre la esfera."""
    gy, gx = np.meshgrid((np.arange(ny) + 0.5) / ny,
                         (np.arange(nx) + 0.5) / nx, indexing="ij")
    gy = np.clip(gy + rng.uniform(-jitter, jitter, gy.shape) / ny, 0.02, 0.98)
    gx = (gx + rng.uniform(-jitter, jitter, gx.shape) / nx) % 1.0
    lat = np.pi / 2.0 - gy * np.pi
    lon = gx * 2.0 * np.pi - np.pi
    return latlon_to_xyz(lat.ravel(), lon.ravel())


def slide(xyz, direction, dist):
    """Desplaza puntos sobre la esfera a lo largo de una dirección tangente."""
    v = xyz + direction * dist
    return v / np.linalg.norm(v, axis=-1, keepdims=True)