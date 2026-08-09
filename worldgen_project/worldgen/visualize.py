"""Visualizations (RGB float arrays).

Includes a correct isometric 3D renderer: every map cell becomes a 3D prism
whose height equals the terrain height, projected with a real 2:1 isometric
camera and drawn back-to-front with a proper painter's algorithm, so heights
are rendered correctly (the old renderer used an inverted depth test, which
let background cells overwrite foreground ones and made the view look flat /
scrambled).

Flat layers are returned at native resolution; the UI must display them with
nearest-neighbour interpolation so tilemaps stay crisp.
"""
from __future__ import annotations
import numpy as np


def _lerp(a, b, t):
    return a + (b - a) * t


def color_height(h):
    H, W = h.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    ocean = h < 0
    d = np.clip(-h[ocean], 0, 1)
    rgb[ocean, 0] = _lerp(0.05, 0.02, d)
    rgb[ocean, 1] = _lerp(0.25, 0.05, d)
    rgb[ocean, 2] = _lerp(0.6, 0.25, d)
    land = ~ocean
    lh = np.clip(h[land], 0, 1.5)
    t1 = np.clip(lh / 0.6, 0, 1)
    t2 = np.clip((lh - 0.6) / 0.4, 0, 1)
    rgb[land, 0] = np.where(lh < 0.6, _lerp(0.2, 0.55, t1), _lerp(0.55, 1.0, t2))
    rgb[land, 1] = np.where(lh < 0.6, _lerp(0.55, 0.4, t1), _lerp(0.4, 1.0, t2))
    rgb[land, 2] = np.where(lh < 0.6, _lerp(0.2, 0.25, t1), _lerp(0.25, 1.0, t2))
    return np.clip(rgb, 0, 1)


def color_temperature(t):
    H, W = t.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    tt = np.clip(t, 0, 1)
    rgb[..., 0] = tt
    rgb[..., 2] = 1 - tt
    rgb[..., 1] = 0.4 * (1 - np.abs(tt - 0.5) * 2)
    return rgb


def color_precip(p):
    H, W = p.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    pp = np.clip(p, 0, 1)
    rgb[..., 2] = pp
    rgb[..., 1] = pp * 0.7
    rgb[..., 0] = (1 - pp) * 0.8
    return rgb


def color_plates(plate_map, plates):
    n = max(1, len(plates))
    rng = np.random.default_rng(42)
    palette = rng.random((max(n, int(plate_map.max()) + 1), 3)).astype(np.float32)
    for i, p in enumerate(plates):
        if p.is_oceanic:
            palette[i] *= 0.4
            palette[i, 2] = min(1.0, palette[i, 2] + 0.3)
    return palette[np.clip(plate_map, 0, len(palette) - 1)]


def color_boundaries(boundary_type):
    H, W = boundary_type.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[boundary_type == 1] = [0.2, 0.6, 1.0]   # divergent
    rgb[boundary_type == 2] = [1.0, 0.2, 0.2]   # convergent
    rgb[boundary_type == 3] = [1.0, 0.9, 0.2]   # transform
    return rgb


def color_final(height, temp, precip, river_map):
    rgb = color_height(height)
    dry = (precip < 0.2) & (height >= 0)
    rgb[dry] = rgb[dry] * 0.6 + np.array([0.85, 0.75, 0.5]) * 0.4
    cold = (temp < 0.25) & (height >= 0)
    rgb[cold] = rgb[cold] * 0.5 + 0.5
    r = river_map > 0
    rgb[r] = [0.15, 0.35, 0.85]
    return np.clip(rgb, 0, 1)


def color_tilemap(tilemap):
    return color_final(tilemap["height"], tilemap["temperature"],
                       tilemap["precipitation"], tilemap["river"])


def color_winds(cfg, H=None, W=None):
    from .climate import _wind_field
    H = H or cfg.sim_height
    W = W or cfg.sim_width
    U, V = _wind_field(cfg, H, W)
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    east = U > 0
    west = U < 0
    mag = np.clip(np.abs(U), 0, 1)
    rgb[east] = np.stack([0.2 * np.ones_like(mag[east]),
                          0.4 + 0.5 * mag[east],
                          0.2 + 0.2 * mag[east]], axis=-1)
    rgb[west] = np.stack([0.4 + 0.5 * mag[west],
                          0.35 + 0.2 * mag[west],
                          0.1 * np.ones_like(mag[west])], axis=-1)
    return rgb


def hillshade(height, azimuth_deg=315.0, altitude_deg=55.0, z_exag=1.0):
    """Standard hillshade for a heightmap (wrap-aware horizontally)."""
    h = height.astype(np.float64) * z_exag
    gy = np.gradient(h, axis=0)
    gx = np.zeros_like(h)
    gx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    gx[:, 0] = (h[:, 1] - h[:, -1]) * 0.5
    gx[:, -1] = (h[:, 0] - h[:, -2]) * 0.5
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    shade = (np.sin(alt) * np.cos(slope) +
             np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip((shade + 1) * 0.5, 0, 1).astype(np.float32)


def color_isometric(height, colored=None, height_scale=0.12, sea_level=0.0,
                    max_cells=160):
    """Isometric 3D render of the heightmap with correct occlusion.

    - Each cell is a column whose top sits at its terrain height.
    - True 2:1 isometric projection, cells drawn strictly back-to-front
      (sorted by screen depth) with a per-pixel depth buffer, so taller
      front terrain correctly occludes what is behind it.
    - Ocean is rendered as a flat sea at sea level.
    """
    H, W = height.shape
    # downsample for interactivity while preserving shape
    step = max(1, int(np.ceil(max(H, W) / max_cells)))
    h = height[::step, ::step]
    if colored is not None:
        c = colored[::step, ::step]
    else:
        c = color_height(h)
    hs, ws = h.shape

    shade = hillshade(h, z_exag=1.0)
    c = np.clip(c * (0.55 + 0.6 * shade[..., None]), 0, 1)

    z_scale = max(4.0, height_scale * max(H, W))
    x0, y0 = np.meshgrid(np.arange(ws), np.arange(hs))
    # isometric projection of top corners
    px = (x0 - y0).astype(np.float64)
    py = (x0 + y0).astype(np.float64) * 0.5
    z = np.clip(h, sea_level, None) * z_scale
    top_py = py - z

    px_min, px_max = px.min(), px.max()
    py_min = top_py.min()
    py_max = py.max()
    can_w = int(px_max - px_min) + 3
    can_h = int(py_max - py_min) + 3
    canvas = np.zeros((can_h, can_w, 3), dtype=np.float32)
    canvas[:] = np.array([0.05, 0.06, 0.10], dtype=np.float32)
    depth = np.full((can_h, can_w), np.inf, dtype=np.float64)

    # back-to-front: larger screen-depth first means "farther away".
    # In this projection depth grows with (x + y), so iterate in reverse sum.
    order = np.argsort(-(x0 + y0).ravel())
    xs = x0.ravel()[order]
    ys = y0.ravel()[order]

    ox = -px_min + 1
    oy = -py_min + 1
    for x, y in zip(xs.tolist(), ys.tolist()):
        zc = max(float(h[y, x]), sea_level) * z_scale
        pxx = int(round((x - y) + ox))
        pyy_top = int(round((x + y) * 0.5 - zc + oy))
        pyy_base = int(round((x + y) * 0.5 + oy))
        d = (x + y)  # farther = larger
        c_top = c[y, x]
        c_side = c_top * 0.55
        if not (0 <= pxx < can_w):
            continue
        # top face
        if 0 <= pyy_top < can_h and d <= depth[pyy_top, pxx]:
            canvas[pyy_top, pxx] = c_top
            depth[pyy_top, pxx] = d
            if pxx + 1 < can_w and d <= depth[pyy_top, pxx + 1]:
                canvas[pyy_top, pxx + 1] = c_top
                depth[pyy_top, pxx + 1] = d
        # visible side column (cliff faces)
        for pp in range(max(0, pyy_top + 1), min(can_h, pyy_base + 1)):
            if d <= depth[pp, pxx]:
                canvas[pp, pxx] = c_side
                depth[pp, pxx] = d
                if pxx + 1 < can_w and d <= depth[pp, pxx + 1]:
                    canvas[pp, pxx + 1] = c_side
                    depth[pp, pxx + 1] = d

    return np.clip(canvas, 0, 1)
