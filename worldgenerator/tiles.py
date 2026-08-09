"""Fase 2: conversión a tilemap + optimización por unificación + stats + export.
- Tipo de tile: clase más frecuente del bloque, EXCEPTO ríos: cualquier píxel
  de río convierte al tile en río completo (para no romper la conexión).
- Unificación: tiles vecinos (conexión directa) con stats similares dentro del
  porcentaje configurable se fusionan en una región. Océanos y ríos se unifican
  por componente conexa ("en vez de 50 tiles de océano, uno solo grande")."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label, distance_transform_edt

OCEAN, LAND, RIVER = 0, 1, 2
MAT_NONE, MAT_SOIL, MAT_ROCK, MAT_SAND, MAT_SNOW = range(5)


@dataclass
class TileMap:
    ttype: np.ndarray        # (th,tw) int8
    height: np.ndarray       # medias por tile
    temp: np.ndarray
    precip: np.ndarray
    humidity: np.ndarray
    material: np.ndarray     # (th,tw) int8
    water_dist: np.ndarray   # px a agua (media por tile)
    region: np.ndarray       # (th,tw) int32
    reg_cls: np.ndarray      # (n_regions,) int8
    reg_cells: list          # lista de arrays de índices planos de tiles


def _materials(height, T, P):
    m = np.full(height.shape, MAT_SOIL, np.int8)
    m[height < 0] = MAT_NONE
    m[(height >= 0) & (height > 1500)] = MAT_ROCK
    m[(height >= 0) & (height <= 1500) & (P < 250)] = MAT_SAND
    m[(height >= 0) & (T < -2)] = MAT_SNOW
    return m


def build_tiles(cfg, height, river_ids, T, P, Hm, log=print):
    ts = cfg.tiles.tile_size
    H, W = height.shape
    th, tw = H // ts, W // ts
    if H % ts or W % ts:
        log(f"Aviso: mapa recortado a {th*ts}x{tw*ts} px para tiles de {ts}px")
    Hc, Wc = th * ts, tw * ts

    def blocks(a):
        return a[:Hc, :Wc].reshape(th, ts, tw, ts)

    def bmean(a):
        return blocks(a).mean((1, 3)).astype(np.float32)

    cls = np.where(height < 0, OCEAN, LAND).astype(np.int8)
    cls[river_ids > 0] = RIVER
    river_any = blocks(cls == RIVER).any((1, 3))                 # override de río
    land_cnt = blocks(cls == LAND).sum((1, 3))
    ocean_cnt = blocks(cls == OCEAN).sum((1, 3))
    ttype = np.where(river_any, RIVER,
                     np.where(land_cnt >= ocean_cnt, LAND, OCEAN)).astype(np.int8)

    water = (height < 0) | (river_ids > 0)
    wdist = distance_transform_edt(~water).astype(np.float32)
    mat = _materials(height, T, P)
    mb = blocks(mat).transpose(0, 2, 1, 3).reshape(th * tw, ts * ts)
    material = np.array([np.bincount(r, minlength=5).argmax()
                         for r in mb], np.int8).reshape(th, tw)

    tm = TileMap(ttype, bmean(height), bmean(T), bmean(P), bmean(Hm),
                 material, bmean(wdist), None, None, None)
    _unify(tm, cfg.tiles.similarity_percent)
    n_r = len(tm.reg_cells)
    log(f"Tiles {th}x{tw} -> {n_r} regiones unificadas "
        f"(similitud {cfg.tiles.similarity_percent}%)")
    return tm


def _unify(tm, sim_pct):
    th, tw = tm.ttype.shape
    sim = sim_pct / 100.0
    feats = np.stack([tm.temp / 80.0, tm.precip / 3000.0,
                      tm.humidity / 100.0, tm.height / 4000.0], axis=-1)
    reg = np.full((th, tw), -1, np.int32)
    reg_cls, reg_cells = [], []

    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for cls in (OCEAN, RIVER):                                  # unificación total por componente
        lab, n = label(tm.ttype == cls, structure=struct4)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            rid = len(reg_cells)
            reg[ys, xs] = rid
            reg_cls.append(cls)
            reg_cells.append((ys * tw + xs).astype(np.int64))

    for y0 in range(th):                                        # tierra: similitud + conexión directa
        for x0 in range(tw):
            if tm.ttype[y0, x0] != LAND or reg[y0, x0] >= 0:
                continue
            rid = len(reg_cells)
            reg[y0, x0] = rid
            seed_f = feats[y0, x0]
            seed_m = tm.material[y0, x0]
            cells = []
            dq = deque([(y0, x0)])
            while dq:
                cy, cx = dq.popleft()
                cells.append(cy * tw + cx)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < th and 0 <= nx < tw):
                        continue
                    if reg[ny, nx] >= 0 or tm.ttype[ny, nx] != LAND:
                        continue
                    if (np.all(np.abs(feats[ny, nx] - seed_f) <= sim)
                            and tm.material[ny, nx] == seed_m):
                        reg[ny, nx] = rid
                        dq.append((ny, nx))
            reg_cls.append(LAND)
            reg_cells.append(np.array(cells, np.int64))
    tm.region = reg
    tm.reg_cls = np.array(reg_cls, np.int8)
    tm.reg_cells = reg_cells


def export_phase(year, tm, ts):
    th, tw = tm.ttype.shape
    reg_flat = tm.region.ravel()
    n = len(tm.reg_cells)

    def rmean(a):
        sums = np.bincount(reg_flat, weights=a.ravel(), minlength=n)
        cnt = np.bincount(reg_flat, minlength=n)
        return sums / np.maximum(cnt, 1)

    regions = []
    for i in range(n):
        ys = tm.reg_cells[i] // tw
        xs = tm.reg_cells[i] % tw
        regions.append({
            "id": i, "type": int(tm.reg_cls[i]),
            "tiles": tm.reg_cells[i].tolist(),
            "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
            "stats": {"height": round(float(rmean(tm.height)[i]), 1),
                      "temperature": round(float(rmean(tm.temp)[i]), 2),
                      "precipitation": round(float(rmean(tm.precip)[i]), 1),
                      "humidity": round(float(rmean(tm.humidity)[i]), 1)},
        })
    return {"year": int(year), "tile_size": ts, "tiles_h": th, "tiles_w": tw,
            "tiles": {
                "type": tm.ttype.ravel().tolist(),
                "height": np.round(tm.height, 1).ravel().tolist(),
                "temperature": np.round(tm.temp, 2).ravel().tolist(),
                "precipitation": np.round(tm.precip, 1).ravel().tolist(),
                "humidity": np.round(tm.humidity, 1).ravel().tolist(),
                "material": tm.material.ravel().tolist(),
                "water_dist": np.round(tm.water_dist, 1).ravel().tolist(),
                "region": tm.region.ravel().tolist()},
            "regions": regions}