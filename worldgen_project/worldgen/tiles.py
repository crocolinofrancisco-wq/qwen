"""Phase 2: Convert dense sim map to tile map + unify similar tiles."""
from __future__ import annotations
import numpy as np
from collections import deque
from .config import WorldConfig


def _material(height, temp, precip):
    """Assign a coarse material to each pixel (not a biome, just ground type)."""
    H, W = height.shape
    mat = np.zeros((H, W), dtype=np.int8)
    # 0=ocean, 1=beach, 2=grass/soil, 3=rock, 4=snow, 5=desert-sand
    mat[height < 0] = 0
    coast = (height >= 0) & (height < 0.03)
    mat[coast] = 1
    land = height >= 0.03
    mat[land] = 2
    rock = height > 0.55
    mat[rock] = 3
    snow = (height > 0.75) | ((temp < 0.15) & land)
    mat[snow] = 4
    desert = (precip < 0.18) & land & (~snow)
    mat[desert] = 5
    return mat


def build_tilemap(cfg: WorldConfig, height, temp, precip, river_map, plate_map, progress=None):
    """Downsample dense sim maps to tile resolution. Return dict of arrays."""
    tw, th = cfg.tile_width, cfg.tile_height
    sw, sh = cfg.sim_width, cfg.sim_height
    sy = sh / th
    sx = sw / tw

    material = _material(height, temp, precip)

    t_height = np.zeros((th, tw), dtype=np.float32)
    t_temp = np.zeros_like(t_height)
    t_precip = np.zeros_like(t_height)
    t_river = np.zeros_like(t_height)
    t_material = np.zeros((th, tw), dtype=np.int8)
    t_plate = np.zeros((th, tw), dtype=np.int32)

    for j in range(th):
        y0 = int(j * sy); y1 = int((j + 1) * sy)
        for i in range(tw):
            x0 = int(i * sx); x1 = int((i + 1) * sx)
            block_h = height[y0:y1, x0:x1]
            block_t = temp[y0:y1, x0:x1]
            block_p = precip[y0:y1, x0:x1]
            block_r = river_map[y0:y1, x0:x1]
            block_m = material[y0:y1, x0:x1]
            block_pl = plate_map[y0:y1, x0:x1]

            t_height[j, i] = float(block_h.mean())
            t_temp[j, i] = float(block_t.mean())
            t_precip[j, i] = float(block_p.mean())
            # river: if any river in block -> full river tile (avoid disconnected)
            if (block_r > 0).any():
                t_river[j, i] = float(block_r[block_r > 0].mean())
                t_material[j, i] = 6  # river material code
            else:
                t_river[j, i] = 0.0
                # most frequent material
                vals, cnts = np.unique(block_m, return_counts=True)
                t_material[j, i] = int(vals[np.argmax(cnts)])
            # plate most frequent
            vals, cnts = np.unique(block_pl, return_counts=True)
            t_plate[j, i] = int(vals[np.argmax(cnts)])
        if progress and j % max(1, th // 10) == 0:
            progress("tilemap:downsample", 0.05 + 0.4 * (j / th))

    return {
        "height": t_height,
        "temperature": t_temp,
        "precipitation": t_precip,
        "river": t_river,
        "material": t_material,
        "plate": t_plate,
    }


def _similarity(a, b, cfg):
    """Return similarity in [0,1] using configured attributes."""
    diffs = []
    for attr in cfg.unify_attributes:
        if attr == "material":
            diffs.append(0.0 if a["material"] == b["material"] else 1.0)
        else:
            key = {"height": "height", "temperature": "temperature",
                   "precipitation": "precipitation"}.get(attr, attr)
            if key in a and key in b:
                diffs.append(abs(a[key] - b[key]))
    if not diffs:
        return 1.0
    return 1.0 - min(1.0, float(np.mean(diffs)))


def unify_tiles(cfg: WorldConfig, tilemap, progress=None):
    """Flood-fill grouping of adjacent tiles with similarity >= threshold.
    Return list of groups: each is dict with 'members' and averaged stats."""
    th, tw = tilemap["height"].shape
    visited = np.zeros((th, tw), dtype=bool)
    groups = []

    def tile_of(j, i):
        return {
            "height": float(tilemap["height"][j, i]),
            "temperature": float(tilemap["temperature"][j, i]),
            "precipitation": float(tilemap["precipitation"][j, i]),
            "material": int(tilemap["material"][j, i]),
            "river": float(tilemap["river"][j, i]),
            "plate": int(tilemap["plate"][j, i]),
        }

    thr = cfg.similarity_threshold
    for j0 in range(th):
        for i0 in range(tw):
            if visited[j0, i0]:
                continue
            seed = tile_of(j0, i0)
            queue = deque([(j0, i0)])
            visited[j0, i0] = True
            members = []
            while queue:
                y, x = queue.popleft()
                members.append((y, x))
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < th and 0 <= nx < tw and not visited[ny, nx]:
                        cand = tile_of(ny, nx)
                        # rivers only merge with rivers
                        if (cand["material"] == 6) != (seed["material"] == 6):
                            continue
                        if _similarity(seed, cand, cfg) >= thr:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            # averaged stats
            m = np.array(members)
            ys, xs = m[:, 0], m[:, 1]
            grp = {
                "id": len(groups),
                "bounds": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "size": len(members),
                "height": float(tilemap["height"][ys, xs].mean()),
                "temperature": float(tilemap["temperature"][ys, xs].mean()),
                "precipitation": float(tilemap["precipitation"][ys, xs].mean()),
                "material": int(seed["material"]),
                "river": float(tilemap["river"][ys, xs].mean()),
                "plate": int(seed["plate"]),
                "member_tiles": [[int(x), int(y)] for (y, x) in members],
            }
            groups.append(grp)
        if progress and j0 % max(1, th // 10) == 0:
            progress("tilemap:unify", 0.5 + 0.5 * (j0 / th))
    return groups
