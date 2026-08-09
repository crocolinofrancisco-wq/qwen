"""Continental drift simulated ON the sphere.

Each drift step rotates every tectonic plate around its own Euler pole (true
3D plate tectonics).  The evolved sphere state is then projected back to the
flat equirectangular map.  Every intermediate step is kept as a snapshot so
the UI can show the evolution with a timeline slider — all snapshots were
generated in 3D first, then projected.

Height is transported with the plates (semi-Lagrangian pull-back sampling on
the sphere), and convergent/divergent boundaries uplift/subside terrain.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

from .config import WorldConfig
from .sphere import (latlon_grids, unit_sphere_points, lonlat_from_xyz,
                     rotation_about, lonlat_to_pixel)


def _sample_pullback(field, src_xyz, lat, lon):
    """Sample `field` (H, W) at the sphere points `src_xyz` via equirectangular
    pull-back (nearest-pixel, horizontal wrap is inherent to the projection)."""
    H, W = field.shape
    slat, slon = lonlat_from_xyz(src_xyz)
    j, i = lonlat_to_pixel(slat, slon, H, W)
    return field[j, i]


def simulate_drift(cfg: WorldConfig, rng, plate_map, plates, height, lat, lon,
                   progress=None):
    """Return (final_plate_map, final_height, phases).

    phases is a list of (plate_map, height) snapshots — one per step —
    each produced by projecting the evolved 3D sphere state.
    """
    phases = []
    if cfg.drift_steps <= 0:
        return plate_map, height, phases

    H, W = plate_map.shape
    xyz = unit_sphere_points(lat, lon)          # (H, W, 3) fixed observation grid

    # per-pixel 3D position and plate id of the *current* state
    pos = xyz.copy()                            # where each surface patch lives now
    pm = plate_map.copy()
    hm = height.copy()

    for step in range(cfg.drift_steps):
        # 1) rotate every plate's surface points around its Euler pole
        new_pos = pos.copy()
        for p in plates:
            mask = pm == p.id
            if not mask.any():
                continue
            R = rotation_about(p.rot_axis, p.rot_rate * cfg.drift_plate_shift * 30.0)
            new_pos[mask] = pos[mask] @ R.T

        # 2) re-project: each *fixed* grid pixel asks "which plate is here now?"
        #    Pull-back: a grid pixel at xyz shows the material whose current
        #    position is closest to xyz.  Approximate by inverse rotation.
        new_pm = np.empty_like(pm)
        new_hm = np.empty_like(hm)
        # start from previous state (plates that don't move keep values)
        new_pm[:] = pm
        new_hm[:] = hm
        claimed = np.zeros((H, W), dtype=bool)
        for p in plates:
            mask = pm == p.id
            if not mask.any():
                continue
            Rinv = rotation_about(p.rot_axis, -p.rot_rate * cfg.drift_plate_shift * 30.0)
            # inverse: where did the material currently at each pixel come from?
            back = xyz @ Rinv.T                       # (H, W, 3)
            src_lat, src_lon = lonlat_from_xyz(back)
            sj, si = lonlat_to_pixel(src_lat, src_lon, H, W)
            src_plate = pm[sj, si]
            take = (src_plate == p.id) & ~claimed
            new_pm[take] = p.id
            new_hm[take] = hm[sj, si][take]
            claimed |= take

        # fill any gaps (convergent overlap / divergent gaps) from nearest state
        gaps = ~claimed
        if gaps.any():
            # nearest claimed pixel horizontally-wrapped via roll voting
            fill_pm = new_pm.copy()
            fill_hm = new_hm.copy()
            for _ in range(4):
                if not gaps.any():
                    break
                for shifted_pm, shifted_hm in (
                        (np.roll(fill_pm, 1, 0), np.roll(fill_hm, 1, 0)),
                        (np.roll(fill_pm, -1, 0), np.roll(fill_hm, -1, 0)),
                        (np.roll(fill_pm, 1, 1), np.roll(fill_hm, 1, 1)),
                        (np.roll(fill_pm, -1, 1), np.roll(fill_hm, -1, 1))):
                    take = gaps
                    new_pm[take] = shifted_pm[take]
                    new_hm[take] = shifted_hm[take]
                gaps = np.zeros((H, W), dtype=bool)  # single pass is enough visually

        # 3) boundary reactions: convergent -> uplift, divergent -> rift
        up = np.roll(new_pm, -1, axis=0)
        down = np.roll(new_pm, 1, axis=0)
        left = np.roll(new_pm, -1, axis=1)
        right = np.roll(new_pm, 1, axis=1)
        borders = (up != new_pm) | (down != new_pm) | (left != new_pm) | (right != new_pm)

        uplift = np.zeros_like(hm)
        rift = np.zeros_like(hm)
        ys, xs = np.where(borders)
        if len(ys):
            vel = {}
            for p in plates:
                vel[p.id] = p
            for y, x in zip(ys.tolist(), xs.tolist()):
                me = int(new_pm[y, x])
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    other = int(new_pm[(y + dy) % H, (x + dx) % W])
                    if other == me:
                        continue
                    pa, pb = plates[me], plates[other]
                    va = np.cross(pa.rot_axis, xyz[y, x]) * pa.rot_rate
                    vb = np.cross(pb.rot_axis, xyz[y, x]) * pb.rot_rate
                    rel = vb - va
                    n = np.array([dy, dx], dtype=np.float64)
                    n /= np.linalg.norm(n) + 1e-9
                    # project relative velocity on local east/north
                    east = np.array([-np.sin(lon[y, x]), np.cos(lon[y, x]), 0.0])
                    north = np.array([-np.sin(lat[y, x]) * np.cos(lon[y, x]),
                                      -np.sin(lat[y, x]) * np.sin(lon[y, x]),
                                      np.cos(lat[y, x])])
                    rl = np.array([np.dot(rel, east), np.dot(rel, north)])
                    nc = float(rl[0] * n[1] + rl[1] * n[0])
                    if nc < -1e-5:
                        uplift[y, x] += min(0.25, -nc * 200.0)
                    elif nc > 1e-5:
                        rift[y, x] -= min(0.20, nc * 180.0)
                    break

        uplift = gaussian_filter(uplift, sigma=1.5, mode=("reflect", "wrap"))
        rift = gaussian_filter(rift, sigma=1.5, mode=("reflect", "wrap"))
        new_hm = np.clip(new_hm + uplift + rift, -1.0, 1.5)

        pos, pm, hm = new_pos, new_pm, new_hm
        if cfg.keep_drift_phases:
            phases.append((pm.copy(), hm.copy()))
        if progress:
            progress(f"drift:step{step + 1}",
                     0.9 + 0.05 * ((step + 1) / max(1, cfg.drift_steps)))

    return pm, hm, phases
