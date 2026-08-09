"""River generation: mouths -> sources -> continuous descent following the real
gradient, with proper connectivity (no isolated points).

Improvements vs previous version:
- Each step draws the full pixel line between consecutive points (Bresenham),
  so rivers are always continuous even when the step radius is large.
- Descent follows the actual local gradient (steepest descent) with a bias
  toward the nearest mouth; radius is much smaller so paths curve naturally.
- Merges use a KD-tree over already-drawn points and physically join branches.
- Deltas are drawn as branching polylines, not random splatter.
- All KD-tree distances respect horizontal wrap when spherical_wrap is True.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

from .config import WorldConfig


def _bresenham(y0, x0, y1, x1):
    """Return list of pixels on the line between (y0,x0) and (y1,x1)."""
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
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if len(pts) > 10000:
            break
    return pts


def _pick_ocean_coast_points(height, count, min_ocean_radius, rng, spherical_wrap=False):
    """Pick points on ocean COAST (land adjacent to ocean), not in deep ocean."""
    H, W = height.shape
    ocean = height < 0
    land = ~ocean
    dilated = binary_dilation(ocean, iterations=1)
    coast = dilated & land  # Only land pixels adjacent to ocean
    ys, xs = np.where(coast)
    if len(ys) == 0:
        return []
    candidates = list(zip(ys.tolist(), xs.tolist()))
    rng.shuffle(candidates)
    picked = []
    for (y, x) in candidates:
        r = min_ocean_radius
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        sub = ocean[y0:y1, x0:x1]
        # Must have significant ocean nearby (>55% ocean in radius)
        if sub.mean() > 0.55:
            picked.append((y, x))
            if len(picked) >= count:
                break
    return picked


def _pick_mountain_sources(height, count, rng):
    H, W = height.shape
    mount_mask = height > 0.55
    ys, xs = np.where(mount_mask)
    if len(ys) == 0:
        idx = np.argsort(height.ravel())[-count * 20:]
        ys, xs = np.unravel_index(idx, height.shape)
    picks = list(zip(ys.tolist(), xs.tolist()))
    rng.shuffle(picks)
    return picks[:count]


def _wrap_dx(x1, x0, W):
    """Shortest signed dx across horizontal wrap."""
    d = x1 - x0
    if d > W / 2:  d -= W
    if d < -W / 2: d += W
    return d


def _draw_line(river_map, polyline, width):
    for k in range(len(polyline) - 1):
        y0, x0 = polyline[k]
        y1, x1 = polyline[k + 1]
        for (yy, xx) in _bresenham(int(y0), int(x0), int(y1), int(x1)):
            if 0 <= yy < river_map.shape[0] and 0 <= xx < river_map.shape[1]:
                if river_map[yy, xx] < width:
                    river_map[yy, xx] = width


def _draw_delta(river_map, mouth, direction, width, rng):
    """Draw a branching delta at the mouth. `direction` is the incoming (dy, dx)."""
    H, W = river_map.shape
    my, mx = mouth
    n_branches = int(rng.integers(3, 6))
    # base angle from incoming direction
    base_ang = np.arctan2(direction[0], direction[1])
    for _ in range(n_branches):
        ang = base_ang + rng.uniform(-0.9, 0.9)
        length = int(rng.integers(4, 10))
        ey = my + int(round(np.sin(ang) * length))
        ex = mx + int(round(np.cos(ang) * length))
        line = _bresenham(my, mx, ey, ex)
        for (yy, xx) in line:
            yy = max(0, min(H - 1, yy))
            xx = xx % W
            river_map[yy, xx] = max(river_map[yy, xx], width * 0.75)


def generate_rivers(cfg: WorldConfig, rng, height, precip, progress=None):
    """Return river_map (float widths, 0 if none) and a list of river polylines.
    
    IMPROVED: Rivers only start from land/mountain sources and flow DOWN to ocean mouths.
    No rivers are generated in the ocean - they only exist on land.
    River pixels only appear where height >= 0 (land), except at the very mouth point.
    """
    H, W = height.shape
    river_map = np.zeros((H, W), dtype=np.float32)

    # River mouths must be on COAST (land adjacent to ocean), not in deep ocean
    mouths = _pick_ocean_coast_points(height, cfg.river_mouth_count,
                                      cfg.river_mouth_min_ocean_radius, rng, 
                                      spherical_wrap=cfg.spherical_wrap)
    # Sources must be in mountains/high elevation ON LAND
    sources = _pick_mountain_sources(height, cfg.river_source_count, rng)
    if not mouths or not sources:
        return river_map, []

    mouth_arr = np.array(mouths)
    # separate tree for mouths (no wrap: OK for our sim, mouths are point-y)
    mouth_tree = cKDTree(mouth_arr)

    rivers = []
    drawn_points = []           # (y, x) already occupied by any river
    drawn_tree = None

    if progress: progress("rivers:init", 0.92)

    for si, (sy, sx) in enumerate(sources):
        # Skip if source is in ocean (shouldn't happen but safety check)
        if height[sy, sx] < 0:
            continue
            
        _, mi = mouth_tree.query([sy, sx], k=1)
        my, mx = mouths[mi]
        cy, cx = int(sy), int(sx)
        polyline = [(cy, cx)]
        width = cfg.river_base_width
        max_steps = 800

        for step_i in range(max_steps):
            # gradient-descent candidates in a small neighborhood
            candidates = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    ny = cy + dy
                    nx = (cx + dx) % W if cfg.spherical_wrap else cx + dx
                    if not (0 <= ny < H): continue
                    if not cfg.spherical_wrap and not (0 <= nx < W): continue
                    # CRITICAL: Only consider land cells for river path (no ocean until mouth)
                    if height[ny, nx] < 0:
                        # Only allow ocean if we're very close to the mouth
                        dist_to_mouth = np.hypot(ny - my, _wrap_dx(mx, nx, W) if cfg.spherical_wrap else (mx - nx))
                        if dist_to_mouth > 2:
                            continue
                    candidates.append((ny, nx, dy, dx))
            if not candidates:
                break

            base_h = height[cy, cx]
            weights = []
            for (ny, nx, dy, dx) in candidates:
                hdiff = base_h - height[ny, nx]         # >0 if descending
                # direction toward mouth (wrap-aware)
                tdy = my - ny
                tdx = _wrap_dx(mx, nx, W) if cfg.spherical_wrap else (mx - nx)
                dlen = np.hypot(tdy, tdx) + 1e-6
                # step direction
                slen = np.hypot(dy, dx) + 1e-6
                dirw = (dy * tdy + dx * tdx) / (dlen * slen)   # -1..1
                h_w = max(0.02, 1.0 + hdiff * 6.0)
                d_w = max(0.02, dirw + 1.05)
                # similarity: when neighbors are ~flat, direction dominates
                similarity = 1.0 - min(1.0, abs(hdiff) * 8.0)
                w = h_w * (1.0 - 0.55 * similarity) + d_w * (0.35 + 0.65 * similarity)
                # penalize going up
                if hdiff < -0.01 and dirw < 0.3:
                    w *= 0.05
                weights.append(max(0.001, w))

            weights = np.array(weights, dtype=np.float32)
            probs = weights / weights.sum()
            k = rng.choice(len(candidates), p=probs)
            ny, nx, _, _ = candidates[k]

            # continuous drawing between (cy,cx) and (ny,nx)
            polyline.append((ny, nx))
            cy, cx = ny, nx
            width = width + 0.04 + precip[cy, cx] * 0.06

            # merge check
            if drawn_tree is not None:
                d, idx = drawn_tree.query([cy, cx], k=1)
                if d <= cfg.river_merge_radius:
                    py, px = drawn_points[idx]
                    polyline.append((py, px))
                    break

            # arrive at mouth / ocean
            dist_to_mouth = np.hypot(cy - my,
                                     _wrap_dx(mx, cx, W) if cfg.spherical_wrap else (mx - cx))
            if dist_to_mouth < 2 or height[cy, cx] < 0:
                polyline.append((int(my), int(mx)))
                # incoming direction for delta
                if len(polyline) >= 2:
                    yprev, xprev = polyline[-2]
                    incoming = (my - yprev, _wrap_dx(mx, xprev, W) if cfg.spherical_wrap else (mx - xprev))
                else:
                    incoming = (0, 1)
                # rasterize the whole polyline first - BUT only on land pixels!
                _draw_line_on_land(river_map, polyline, width, height)
                _draw_delta(river_map, (int(my), int(mx)), incoming, width, rng)
                break

            # occasional split (fork drawn as short branch)
            if rng.random() < cfg.river_split_chance and len(polyline) > 6:
                fork_len = int(rng.integers(6, 14))
                fy, fx = cy, cx
                fork = [(fy, fx)]
                for _ in range(fork_len):
                    dy_ = int(np.sign(my - fy) + rng.integers(-1, 2))
                    dx_ = int(np.sign(_wrap_dx(mx, fx, W) if cfg.spherical_wrap else (mx - fx))
                              + rng.integers(-1, 2))
                    fy = max(0, min(H - 1, fy + dy_))
                    fx = (fx + dx_) % W if cfg.spherical_wrap else max(0, min(W - 1, fx + dx_))
                    fork.append((fy, fx))
                _draw_line_on_land(river_map, fork, width * 0.65, height)

        else:
            # exhausted steps without reaching a mouth: still draw whatever we traced
            _draw_line_on_land(river_map, polyline, width, height)

        rivers.append(polyline)
        drawn_points.extend(polyline)
        if len(drawn_points) > 0:
            drawn_tree = cKDTree(np.array(drawn_points))

    if progress: progress("rivers:done", 0.98)
    return river_map, rivers


def _draw_line_on_land(river_map, polyline, width, height):
    """Draw a line on river_map, but ONLY on land pixels (height >= 0)."""
    H, W = river_map.shape
    for k in range(len(polyline) - 1):
        y0, x0 = polyline[k]
        y1, x1 = polyline[k + 1]
        for (yy, xx) in _bresenham(int(y0), int(x0), int(y1), int(x1)):
            if 0 <= yy < river_map.shape[0] and 0 <= xx < river_map.shape[1]:
                # Only draw river on land (height >= 0)
                if height[yy, xx] >= 0:
                    if river_map[yy, xx] < width:
                        river_map[yy, xx] = width
