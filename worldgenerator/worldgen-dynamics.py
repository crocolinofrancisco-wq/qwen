"""Mundo dinámico: deriva continental por eras.
- Cada placa avanza sobre la esfera según su dirección (los límites se
  recalculan y las montañas/terreno se regeneran con la deriva acumulada).
- Puntos interiores al azar se mueven en la dirección de la placa arrastrando
  el terreno en un radio (los puntos cercanos tienen chance de acompañar).
  El punto queda en su posición original: lo que se desplazan son los valores."""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree
from sphere import slide
from plates import classify_boundaries


def advance_era(cfg, plates, rng, xyz, log=print):
    dc = cfg.dynamics
    H, W = plates.id_map.shape
    step_rad = dc.plate_step_px * (2.0 * np.pi) / cfg.world.width
    plates.centers[:] = slide(plates.centers, plates.directions, step_rad)
    plates.offsets += np.cross(plates.centers, plates.directions) * step_rad
    _, ids = cKDTree(plates.centers).query(xyz.reshape(-1, 3))
    plates.id_map = ids.reshape(H, W).astype(np.int32)
    classify_boundaries(plates, cfg.plates)

    blobs = []
    for i in range(plates.n):
        if rng.random() > dc.blob_move_chance:
            continue
        ys, xs = np.nonzero(plates.id_map == i)
        if len(ys) == 0:
            continue
        c = plates.centers[i]
        lon = math.atan2(float(c[1]), float(c[0]))
        lat = math.asin(float(np.clip(c[2], -1, 1)))
        e_lon = np.array([-math.sin(lon), math.cos(lon), 0.0])
        e_lat = np.array([-math.sin(lat) * math.cos(lon),
                          -math.sin(lat) * math.sin(lon), math.cos(lat)])
        d = plates.directions[i]
        dx = int(round(float(np.dot(d, e_lon)) * dc.plate_step_px))
        dy = int(round(-float(np.dot(d, e_lat)) * dc.plate_step_px))
        if dx == 0 and dy == 0:
            dx = 1 if rng.random() < 0.5 else -1
        for _ in range(dc.blobs_per_plate):
            k = int(rng.integers(len(ys)))
            pts = [(int(ys[k]), int(xs[k]))]
            seen = set(pts)
            j = 0
            while j < len(pts) and len(pts) < 40:        # puntos cercanos acompañan
                yy, xx = pts[j]; j += 1
                for dy2, dx2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy2, (xx + dx2) % W
                    if (0 <= ny < H and (ny, nx) not in seen
                            and plates.id_map[ny, nx] == i
                            and rng.random() < dc.blob_follow_chance):
                        seen.add((ny, nx)); pts.append((ny, nx))
            for (yy, xx) in pts:
                blobs.append((yy, xx, dx, dy, dc.blob_radius))
    log(f"Deriva de era: {len(blobs)} desplazamientos locales")
    return blobs


def apply_blobs(height, blobs):
    H, W = height.shape
    for (y0, x0, dx, dy, R) in blobs:
        ys = np.arange(max(0, y0 - R), min(H, y0 + R + 1))
        xs = np.arange(x0 - R, x0 + R + 1) % W
        if len(ys) == 0:
            continue
        yy, xx = np.meshgrid(ys - y0, np.arange(x0 - R, x0 + R + 1) - x0, indexing="ij")
        disk = (yy * yy + xx * xx) <= R * R
        sub = height[np.ix_(ys, xs)]
        rolled = np.roll(np.roll(sub, dy, axis=0), dx, axis=1)
        out = sub.copy()
        out[disk] = rolled[disk]
        height[np.ix_(ys, xs)] = out