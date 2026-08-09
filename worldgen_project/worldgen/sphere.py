"""Spherical core: the world is generated ON a sphere (3D) and only later
projected onto a flat 2D equirectangular map, exactly as the design document
requires ("The first phase should be done in an sphere and then projected
into a flat map").

Everything downstream (plates, height, climate, rivers, drift) operates on
these lat/lon grids, so the horizontal edges of the projected map correspond
to *the same points on the sphere* -> perfectly continuous longitude edges,
and no fake "wrap" artefacts anywhere (the old cosine-blend trick produced
both a broken seam at the edges AND a ghost line in the middle of the map).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def latlon_grids(H: int, W: int):
    """Return (lat, lon) arrays shaped (H, W).

    lat in [-pi/2, pi/2]  (row 0 = north pole, row H-1 = south pole)
    lon in [-pi, pi]      (col 0 and col W-1 are adjacent on the sphere)
    """
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
    lon = ((np.arange(W, dtype=np.float64) + 0.5) / W) * 2.0 * np.pi - np.pi
    return lat[:, None].repeat(W, axis=1), lon[None, :].repeat(H, axis=0)


def unit_sphere_points(lat: np.ndarray, lon: np.ndarray):
    """Project lat/lon grids to unit-sphere XYZ points (H, W, 3)."""
    cl = np.cos(lat)
    x = cl * np.cos(lon)
    y = cl * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def lonlat_from_xyz(pts: np.ndarray):
    """(…,3) unit vectors -> (lat, lon) in radians."""
    pts = pts / (np.linalg.norm(pts, axis=-1, keepdims=True) + 1e-12)
    lat = np.arcsin(np.clip(pts[..., 2], -1.0, 1.0))
    lon = np.arctan2(pts[..., 1], pts[..., 0])
    return lat, lon


def rotation_about(axis, angle):
    """3x3 Rodrigues rotation matrix for a rotation of `angle` around `axis`."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    x, y, z = a
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C    ],
    ])


class SphereKD:
    """Fast nearest-neighbour queries on the sphere (chord distance = great-
    circle ordering), used for Voronoi plates and river merging on 3D points."""

    def __init__(self, xyz: np.ndarray):
        self.xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        self.tree = cKDTree(self.xyz)

    def query(self, pts_xyz: np.ndarray, k: int = 1):
        pts = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
        return self.tree.query(pts, k=k)


def lonlat_to_pixel(lat: np.ndarray, lon: np.ndarray, H: int, W: int):
    """Map lat/lon (rad) to pixel indices of the equirectangular projection."""
    j = np.clip(((0.5 - lat / np.pi) * H - 0.5).astype(int), 0, H - 1)
    i = np.clip((((lon + np.pi) / (2 * np.pi)) * W - 0.5).astype(int), 0, W - 1)
    return j, i


def wrap_lon_distance(dlon):
    """Shortest signed longitude difference in radians."""
    return (dlon + np.pi) % (2 * np.pi) - np.pi
