"""Tectonic plates generated directly ON the sphere.

Seed points are distributed on the unit sphere with a jittered
Fibonacci-style grid, major plates grow by Voronoi expansion over sphere
points (K-D tree in 3D), and small plates are carved out of plate borders.
Because everything lives in 3D, a plate that reaches lon=+pi simply continues
at lon=-pi on the projected map — the horizontal edges are truly continuous.
"""
from __future__ import annotations
import numpy as np

from .config import WorldConfig
from .sphere import (latlon_grids, unit_sphere_points, lonlat_from_xyz,
                     rotation_about, SphereKD)


class Plate:
    __slots__ = ("id", "center", "is_oceanic", "move_dir", "base_height",
                 "rot_axis", "rot_rate")

    def __init__(self, pid, center, is_oceanic, move_dir, base_height,
                 rot_axis, rot_rate):
        self.id = pid
        self.center = center             # (lon, lat) of plate centroid
        self.is_oceanic = is_oceanic
        self.move_dir = move_dir         # (dlon, dlat) tangent unit vector
        self.base_height = base_height   # sampled base height offset
        self.rot_axis = rot_axis         # 3D Euler pole (unit vector)
        self.rot_rate = rot_rate         # radians per drift step


def _jittered_sphere_seeds(cfg: WorldConfig, rng: np.random.Generator):
    """Jittered grid of seed points on the sphere surface (lat/lon)."""
    nx, ny = cfg.plate_points_x, cfg.plate_points_y
    pts = []
    for j in range(ny):
        # equal-area-ish rows: latitude of row centers
        lat = (0.5 - (j + 0.5) / ny) * np.pi
        for i in range(nx):
            lon = ((i + 0.5) / nx) * 2 * np.pi - np.pi
            jlon = (rng.random() - 0.5) * (2 * np.pi / nx) * cfg.point_jitter
            jlat = (rng.random() - 0.5) * (np.pi / ny) * cfg.point_jitter
            la = np.clip(lat + jlat, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
            lo = (lon + jlon + np.pi) % (2 * np.pi) - np.pi
            pts.append((la, lo))
    return np.array(pts, dtype=np.float64)  # (N, 2) lat/lon


def _tangent_velocity(axis, rate, pts_xyz):
    """Velocity (3D) of surface points for rotation `rate` about `axis`."""
    v = np.cross(np.broadcast_to(axis, pts_xyz.shape), pts_xyz) * rate
    return v


def _tangent_lonlat(vel3, lat, lon):
    """Project a 3D velocity to (dlon, dlat) unit direction at lat/lon."""
    # east = (-sin lon, cos lon, 0); north = (-sin lat cos lon, -sin lat sin lon, cos lat)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon),
                      -np.sin(lat) * np.sin(lon),
                      np.cos(lat)], axis=-1)
    dlon = np.einsum("...i,...i->...", vel3, east)
    dlat = np.einsum("...i,...i->...", vel3, north)
    n = np.hypot(dlon, dlat) + 1e-9
    return float(dlon / n), float(dlat / n)


def generate_plates(cfg: WorldConfig, rng: np.random.Generator, progress=None):
    """Return (plate_id_map, plates, seeds_xyz, kd) all generated on the sphere."""
    W, H = cfg.sim_width, cfg.sim_height
    lat, lon = latlon_grids(H, W)
    pts_xyz = unit_sphere_points(lat, lon)          # (H, W, 3)

    seeds_ll = _jittered_sphere_seeds(cfg, rng)
    seeds_xyz = unit_sphere_points(seeds_ll[:, 0], seeds_ll[:, 1])  # (N, 3)
    tree = SphereKD(seeds_xyz)

    # fine Voronoi on the sphere (3D nearest neighbour)
    _, idx = tree.query(pts_xyz, k=1)
    fine_map = idx.reshape(H, W)
    if progress: progress("plates:voronoi_fine", 0.15)

    # major plates: pick random seed centers, expand by Voronoi over seeds
    n_seeds = len(seeds_ll)
    n_major = max(2, min(cfg.major_plate_count, n_seeds - cfg.small_plate_count))
    major_ids = rng.choice(n_seeds, size=n_major, replace=False)
    major_tree = SphereKD(seeds_xyz[major_ids])
    _, seed_to_major = major_tree.query(seeds_xyz, k=1)
    if progress: progress("plates:major_clusters", 0.35)

    # border seeds -> small plates
    _, seed_neighbors = tree.query(seeds_xyz, k=6)
    border_seed_ids = []
    for s_idx in range(n_seeds):
        my = seed_to_major[s_idx]
        for nb in np.atleast_1d(seed_neighbors[s_idx])[1:]:
            if seed_to_major[nb] != my:
                border_seed_ids.append(s_idx)
                break
    border_seed_ids = np.array(sorted(set(border_seed_ids)), dtype=int)

    small_count = int(min(cfg.small_plate_count, len(border_seed_ids)))
    if small_count > 0:
        chosen = rng.choice(border_seed_ids, size=small_count, replace=False)
        for sc_idx, s_seed in enumerate(chosen):
            new_pid = n_major + sc_idx
            k = int(rng.integers(2, 6))
            _, nn = tree.query(seeds_xyz[s_seed], k=min(k, n_seeds))
            border_set = set(border_seed_ids.tolist())
            for nb in np.atleast_1d(nn).ravel():
                nb = int(nb)
                if nb in border_set and seed_to_major[nb] < n_major:
                    seed_to_major[nb] = new_pid
    if progress: progress("plates:small_plates", 0.55)

    total_plates = int(seed_to_major.max()) + 1
    plate_map = seed_to_major[fine_map].astype(np.int32)

    plates = []
    for pid in range(total_plates):
        mask = plate_map == pid
        if not mask.any():
            plates.append(Plate(pid, (0.0, 0.0), True, (0.0, 0.0), 0.0,
                                np.array([0.0, 0.0, 1.0]), 0.0))
            continue
        c_xyz = pts_xyz[mask].mean(axis=0)
        c_xyz /= np.linalg.norm(c_xyz) + 1e-12
        c_lat, c_lon = lonlat_from_xyz(c_xyz)

        is_oc = bool(rng.random() < cfg.oceanic_ratio)
        # random Euler pole anywhere on the sphere -> real plate tectonics
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis) + 1e-12
        rate = float(rng.uniform(0.004, 0.02)) * rng.choice([-1.0, 1.0])
        vel3 = _tangent_velocity(axis, rate, c_xyz[None, :])[0]
        mv = _tangent_lonlat(vel3, np.array([c_lat]), np.array([c_lon]))
        bh = float(rng.random() * 2 - 1)
        if is_oc:
            bh -= abs(cfg.oceanic_height_offset)
        plates.append(Plate(pid, (float(c_lon), float(c_lat)), is_oc, mv, bh,
                            axis, rate))
    if progress: progress("plates:done", 0.7)
    return plate_map, plates, seeds_xyz, tree


def compute_boundaries(plate_map: np.ndarray, plates, lat, lon):
    """Boundary map on the projected map (0 none, 1 divergent, 2 convergent,
    3 transform) + stress in [0,1].  Neighbour comparisons wrap horizontally
    (the map is a projection, so column 0 == column W on the sphere)."""
    H, W = plate_map.shape
    boundary_type = np.zeros_like(plate_map, dtype=np.int8)
    stress = np.zeros((H, W), dtype=np.float32)

    pts_xyz = unit_sphere_points(lat, lon)
    up = np.roll(plate_map, -1, axis=0)
    down = np.roll(plate_map, 1, axis=0)
    left = np.roll(plate_map, -1, axis=1)   # horizontal wrap is *correct* here:
    right = np.roll(plate_map, 1, axis=1)   # col W-1 is adjacent to col 0 on the sphere

    diff_mask = ((up != plate_map) | (down != plate_map) |
                 (left != plate_map) | (right != plate_map))
    ys, xs = np.where(diff_mask)

    # Precompute per-plate data once (centroid xyz + Euler pole)
    n_plates = len(plates)
    centroid_xyz = np.zeros((n_plates, 3))
    axes = np.zeros((n_plates, 3))
    rates = np.zeros(n_plates)
    for p in plates:
        ca = unit_sphere_points(np.array([p.center[1]]),
                                np.array([p.center[0]])).reshape(-1, 3)[0]
        centroid_xyz[p.id] = ca
        axes[p.id] = p.rot_axis
        rates[p.id] = p.rot_rate

    for y, x in zip(ys, xs):
        me = plate_map[y, x]
        neighbors = set()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = plate_map[(y + dy) % H, (x + dx) % W]
            if other != me:
                neighbors.add(int(other))
        if not neighbors:
            continue
        here = pts_xyz[y, x]
        va = np.cross(axes[me], here) * rates[me]
        best_type, best_stress = 0, 0.0
        for o in neighbors:
            vb = np.cross(axes[o], here) * rates[o]
            rel = vb - va
            n = centroid_xyz[me] - centroid_xyz[o]
            n = n / (np.linalg.norm(n) + 1e-9)
            nc = float(np.dot(rel, n))
            tangent = np.cross(n, here)
            tc = float(np.dot(rel, tangent))
            mag = float(np.hypot(nc, tc))
            if mag <= best_stress:
                continue
            best_stress = mag
            best_type = 2 if abs(nc) > abs(tc) and nc < 0 else \
                        1 if abs(nc) > abs(tc) else 3
        boundary_type[y, x] = best_type
        stress[y, x] = min(1.0, best_stress * 40.0)
    return boundary_type, stress
