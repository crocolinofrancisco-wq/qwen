"""Tectonic plate generation: Voronoi + K-D tree, majors, minors, oceanic/continental."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

from .config import WorldConfig


class Plate:
    __slots__ = ("id", "center", "is_oceanic", "move_dir", "base_height")

    def __init__(self, pid, center, is_oceanic, move_dir, base_height):
        self.id = pid
        self.center = center
        self.is_oceanic = is_oceanic
        self.move_dir = move_dir           # (dx, dy) unit vector
        self.base_height = base_height     # sampled base height in [-1, 1]


def _jittered_grid(cfg: WorldConfig, rng: np.random.Generator):
    """Generate jittered grid of seed points."""
    W, H = cfg.sim_width, cfg.sim_height
    nx, ny = cfg.plate_points_x, cfg.plate_points_y
    cell_w = W / nx
    cell_h = H / ny
    pts = []
    for j in range(ny):
        for i in range(nx):
            cx = (i + 0.5) * cell_w
            cy = (j + 0.5) * cell_h
            jx = (rng.random() - 0.5) * cell_w * cfg.point_jitter
            jy = (rng.random() - 0.5) * cell_h * cfg.point_jitter
            pts.append((cx + jx, cy + jy))
    return np.array(pts, dtype=np.float32)


def generate_plates(cfg: WorldConfig, rng: np.random.Generator, progress=None):
    """Return (plate_id_map, plates_list, seed_points, kdtree)."""
    W, H = cfg.sim_width, cfg.sim_height
    seeds = _jittered_grid(cfg, rng)
    tree = cKDTree(seeds)

    # Assign each pixel to nearest seed (initial fine Voronoi)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    _, idx = tree.query(pixels, k=1)
    fine_map = idx.reshape(H, W)

    if progress: progress("plates:voronoi_fine", 0.15)

    # Group seed points into major plates (cluster centers)
    n_seeds = len(seeds)
    n_major = max(2, min(cfg.major_plate_count, n_seeds - cfg.small_plate_count))
    # pick major seeds via random selection
    major_ids = rng.choice(n_seeds, size=n_major, replace=False)
    major_centers = seeds[major_ids]
    major_tree = cKDTree(major_centers)
    # each seed is assigned to nearest major center -> forms major plate
    _, seed_to_major = major_tree.query(seeds, k=1)

    # Reserve some border seeds for small plates
    # Detect seeds that lie on major-plate borders
    plate_of_pixel = seed_to_major[fine_map]

    if progress: progress("plates:major_clusters", 0.35)

    # find border seeds: seeds whose neighbors belong to another major
    seed_neighbors = tree.query(seeds, k=6)[1]
    border_seed_ids = []
    for s_idx in range(n_seeds):
        my = seed_to_major[s_idx]
        for nb in seed_neighbors[s_idx][1:]:
            if seed_to_major[nb] != my:
                border_seed_ids.append(s_idx)
                break
    border_seed_ids = np.array(border_seed_ids)

    small_count = min(cfg.small_plate_count, len(border_seed_ids))
    if small_count > 0 and len(border_seed_ids) > 0:
        chosen_small_seeds = rng.choice(border_seed_ids, size=small_count, replace=False)
        # Each small plate absorbs a few nearby border seeds
        for sc_idx, s_seed in enumerate(chosen_small_seeds):
            new_plate_id = n_major + sc_idx
            # find k nearby border seeds
            k = int(rng.integers(2, 6))
            _, nn = tree.query(seeds[s_seed], k=min(k, n_seeds))
            for nb in np.atleast_1d(nn):
                if nb in border_seed_ids and seed_to_major[nb] < n_major:
                    seed_to_major[nb] = new_plate_id

    total_plates = int(seed_to_major.max()) + 1

    if progress: progress("plates:small_plates", 0.55)

    # Rebuild pixel->plate map
    plate_map = seed_to_major[fine_map].astype(np.int32)

    # Build plate objects
    plates = []
    for pid in range(total_plates):
        mask = plate_map == pid
        if not mask.any():
            plates.append(Plate(pid, (W/2, H/2), True, (0.0, 0.0), 0.0))
            continue
        ys_p, xs_p = np.where(mask)
        cx, cy = float(xs_p.mean()), float(ys_p.mean())
        is_oc = rng.random() < cfg.oceanic_ratio
        ang = rng.random() * 2 * np.pi
        mv = (float(np.cos(ang)), float(np.sin(ang)))
        # base height sampled once per plate
        bh = float(rng.random() * 2 - 1)
        if is_oc:
            bh -= abs(cfg.oceanic_height_offset)
        plates.append(Plate(pid, (cx, cy), is_oc, mv, bh))

    if progress: progress("plates:done", 0.7)

    return plate_map, plates, seeds, tree


def compute_boundaries(plate_map: np.ndarray, plates):
    """Return boundary_map with values 0=none, 1=divergent, 2=convergent, 3=transform,
    plus a stress array in [0,1]."""
    H, W = plate_map.shape
    boundary_type = np.zeros_like(plate_map, dtype=np.int8)
    stress = np.zeros((H, W), dtype=np.float32)

    # roll and compare
    up = np.roll(plate_map, -1, axis=0)
    down = np.roll(plate_map, 1, axis=0)
    left = np.roll(plate_map, -1, axis=1)
    right = np.roll(plate_map, 1, axis=1)

    diff_mask = ((up != plate_map) | (down != plate_map) |
                 (left != plate_map) | (right != plate_map))

    ys, xs = np.where(diff_mask)
    for y, x in zip(ys, xs):
        me = plate_map[y, x]
        neighbors = set()
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = (y+dy) % H, (x+dx) % W
            other = plate_map[ny, nx]
            if other != me:
                neighbors.add(int(other))
        if not neighbors:
            continue
        # take strongest interaction
        mv_a = np.array(plates[me].move_dir)
        best_type = 0
        best_stress = 0.0
        for o in neighbors:
            mv_b = np.array(plates[o].move_dir)
            # relative velocity
            rel = mv_b - mv_a
            # boundary normal approx (from other center to me center)
            n = np.array(plates[me].center) - np.array(plates[o].center)
            nlen = np.linalg.norm(n) + 1e-6
            n /= nlen
            # normal component
            nc = float(np.dot(rel, n))
            tc = float(rel[0]*(-n[1]) + rel[1]*n[0])
            mag = float(np.hypot(nc, tc))
            if mag <= best_stress:
                continue
            best_stress = mag
            if abs(nc) > abs(tc):
                best_type = 2 if nc < 0 else 1   # closing=convergent, opening=divergent
            else:
                best_type = 3
        boundary_type[y, x] = best_type
        stress[y, x] = min(1.0, best_stress)
    return boundary_type, stress
