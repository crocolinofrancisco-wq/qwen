"""River generation on the projected map, sphere-aware.

Mouths are picked on ocean coasts (with a large-radius ocean detector so no
rivers end in ponds), sources on mountains, and flow descends the real
gradient biased toward the mouth.  All longitudinal distances wrap (the map
is a projection of the sphere), so rivers cross the horizontal edges of the
map seamlessly.  Rivers merge via a KD-tree and can split / form deltas.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

from .config import WorldConfig


def _bresenham(y0, x0, y1, x1):
    pts = []
    dy = abs(y1 - y0); sy = 1 if y0 < y1 else -1
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    err = dx - dy
    y, x = y0, x0
    while True:
        pts.append((y, x))
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
        if len(pts) > 10000:
            break
    return pts


def _wrap_dx(x1, x0, W):
    d = x1 - x0
    if d > W / 2: d -= W
    if d < -W / 2: d += W
    return d


def _pick_ocean_coast_points(height, count, min_ocean_radius, rng):
    H, W = height.shape
    ocean = height < 0
    land = ~ocean
    coast = binary_dilation(ocean, iterations=1) & land
    ys, xs = np.where(coast)
    if len(ys) == 0:
        return []
    candidates = list(zip(ys.tolist(), xs.tolist()))
    rng.shuffle(candidates)
    picked = []
    r = min_ocean_radius
    for (y, x) in candidates:
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        # horizontal wrap: ocean check must consider the sphere
        cols = np.arange(x - r, x + r + 1) % W
        sub = ocean[y0:y1][:, cols]
        if sub.mean() > 0.55:
            picked.append((y, x))
            if len(picked) >= count:
                break
    return picked


def _pick_mountain_sources(height, count, rng):
    mount = height > 0.55
    ys, xs = np.where(mount)
    if len(ys) == 0:
        idx = np.argsort(height.ravel())[-count * 20:]
        ys, xs = np.unravel_index(idx, height.shape)
    picks = list(zip(ys.tolist(), xs.tolist()))
    rng.shuffle(picks)
    return picks[:count]


def _draw_line(river_map, polyline, width):
    H, W = river_map.shape
    for k in range(len(polyline) - 1):
        y0, x0 = polyline[k]
        y1, x1 = polyline[k + 1]
        # split wrapped segments so the line crosses the edge instead of
        # drawing a long streak across the map
        dx = x1 - x0
        if dx > W / 2:
            x0 += W
        elif dx < -W / 2:
            x1 += W
        for (yy, xx) in _bresenham(int(y0), int(x0), int(y1), int(x1)):
            if 0 <= yy < H:
                xx %= W
                if river_map[yy, xx] < width:
                    river_map[yy, xx] = width


def _draw_delta(river_map, mouth, direction, width, rng):
    H, W = river_map.shape
    my, mx = mouth
    base_ang = np.arctan2(direction[0], direction[1])
    for _ in range(int(rng.integers(3, 6))):
        ang = base_ang + rng.uniform(-0.9, 0.9)
        length = int(rng.integers(4, 10))
        ey = my + int(round(np.sin(ang) * length))
        ex = mx + int(round(np.cos(ang) * length))
        for (yy, xx) in _bresenham(my, mx, ey, ex):
            yy = max(0, min(H - 1, yy))
            river_map[yy, xx % W] = max(river_map[yy, xx % W], width * 0.75)


def generate_rivers(cfg: WorldConfig, rng, height, precip, progress=None):
    H, W = height.shape
    river_map = np.zeros((H, W), dtype=np.float32)

    mouths = _pick_ocean_coast_points(height, cfg.river_mouth_count,
                                      cfg.river_mouth_min_ocean_radius, rng)
    sources = _pick_mountain_sources(height, cfg.river_source_count, rng)
    if not mouths or not sources:
        return river_map, []

    mouth_arr = np.array(mouths)
    # wrap-aware nearest mouth: tile mouth x-coordinates by -W, 0, +W
    tiled = np.concatenate([mouth_arr + [0, -W], mouth_arr, mouth_arr + [0, W]])
    mouth_tree = cKDTree(tiled)

    rivers = []
    drawn_points = []
    drawn_tree = None
    if progress: progress("rivers:init", 0.92)

    for sy, sx in sources:
        _, mi = mouth_tree.query([sy, sx], k=1)
        my, mx = tiled[mi]
        mx %= W
        cy, cx = int(sy), int(sx)
        polyline = [(cy, cx)]
        width = cfg.river_base_width

        for _ in range(800):
            candidates = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = cy + dy
                    nx = (cx + dx) % W
                    if not (0 <= ny < H):
                        continue
                    candidates.append((ny, nx, dy, dx))
            if not candidates:
                break

            base_h = height[cy, cx]
            weights = []
            for (ny, nx, dy, dx) in candidates:
                hdiff = base_h - height[ny, nx]
                tdy = my - ny
                tdx = _wrap_dx(mx, nx, W)
                dlen = np.hypot(tdy, tdx) + 1e-6
                slen = np.hypot(dy, dx) + 1e-6
                dirw = (dy * tdy + dx * tdx) / (dlen * slen)
                h_w = max(0.02, 1.0 + hdiff * 6.0)
                d_w = max(0.02, dirw + 1.05)
                similarity = 1.0 - min(1.0, abs(hdiff) * 8.0)
                w = h_w * (1.0 - 0.55 * similarity) + d_w * (0.35 + 0.65 * similarity)
                if hdiff < -0.01 and dirw < 0.3:
                    w *= 0.05
                weights.append(max(0.001, w))

            probs = np.asarray(weights, dtype=np.float32)
            probs /= probs.sum()
            k = rng.choice(len(candidates), p=probs)
            ny, nx, _, _ = candidates[k]
            polyline.append((ny, nx))
            cy, cx = ny, nx
            width = width + 0.04 + precip[cy, cx] * 0.06

            if drawn_tree is not None:
                d, idx = drawn_tree.query([cy, cx], k=1)
                if d <= cfg.river_merge_radius:
                    polyline.append(tuple(drawn_points[idx]))
                    break

            dist_to_mouth = np.hypot(cy - my, _wrap_dx(mx, cx, W))
            if dist_to_mouth < 2 or height[cy, cx] < 0:
                polyline.append((int(my), int(mx)))
                if len(polyline) >= 2:
                    yprev, xprev = polyline[-2]
                    incoming = (my - yprev, _wrap_dx(mx, xprev, W))
                else:
                    incoming = (0, 1)
                _draw_line(river_map, polyline, width)
                if rng.random() < cfg.river_delta_split_chance:
                    _draw_delta(river_map, (int(my), int(mx)), incoming, width, rng)
                break

            if rng.random() < cfg.river_split_chance and len(polyline) > 6:
                fy, fx = cy, cx
                fork = [(fy, fx)]
                for _ in range(int(rng.integers(6, 14))):
                    dy_ = int(np.sign(my - fy) + rng.integers(-1, 2))
                    dx_ = int(np.sign(_wrap_dx(mx, fx, W)) + rng.integers(-1, 2))
                    fy = max(0, min(H - 1, fy + dy_))
                    fx = (fx + dx_) % W
                    fork.append((fy, fx))
                _draw_line(river_map, fork, width * 0.65)
        else:
            _draw_line(river_map, polyline, width)

        rivers.append(polyline)
        drawn_points.extend(polyline)
        if drawn_points:
            drawn_tree = cKDTree(np.array(drawn_points))

    if progress: progress("rivers:done", 0.98)
    return river_map, rivers
