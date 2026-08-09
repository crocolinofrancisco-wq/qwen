"""Colored visualizations of the world (matplotlib-compatible RGB arrays)."""
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
    lh = np.clip(h[land], 0, 1)
    rgb[land, 0] = np.where(lh < 0.6, _lerp(0.2, 0.55, lh / 0.6), _lerp(0.55, 1.0, (lh - 0.6) / 0.4))
    rgb[land, 1] = np.where(lh < 0.6, _lerp(0.55, 0.4, lh / 0.6), _lerp(0.4, 1.0, (lh - 0.6) / 0.4))
    rgb[land, 2] = np.where(lh < 0.6, _lerp(0.2, 0.25, lh / 0.6), _lerp(0.25, 1.0, (lh - 0.6) / 0.4))
    # Fix horizontal seam: blend first and last columns for spherical wrap
    if W > 1:
        blend = np.linspace(0, 1, 5)[np.newaxis, :]  # 5-pixel blend zone at edges
        left_blend = int(min(5, W // 10))
        right_blend = int(min(5, W // 10))
        if left_blend > 0 and right_blend > 0:
            for i in range(left_blend):
                weight = i / left_blend
                rgb[:, i] = rgb[:, i] * (1 - weight) + rgb[:, -right_blend + i] * weight
            for i in range(right_blend):
                weight = i / right_blend
                rgb[:, -right_blend + i] = rgb[:, -right_blend + i] * (1 - weight) + rgb[:, i] * weight
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
    palette = rng.random((n, 3)).astype(np.float32)
    for i, p in enumerate(plates):
        if p.is_oceanic:
            palette[i] *= 0.4
            palette[i, 2] = min(1.0, palette[i, 2] + 0.3)
    # protect against out-of-range plate ids (from drift)
    max_id = int(plate_map.max())
    if max_id >= n:
        extra = np.random.default_rng(43).random((max_id - n + 1, 3)).astype(np.float32)
        palette = np.concatenate([palette, extra], axis=0)
    return palette[plate_map]


def color_boundaries(boundary_type):
    H, W = boundary_type.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[boundary_type == 1] = [0.2, 0.6, 1.0]
    rgb[boundary_type == 2] = [1.0, 0.2, 0.2]
    rgb[boundary_type == 3] = [1.0, 0.9, 0.2]
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
    h = tilemap["height"]
    t = tilemap["temperature"]
    p = tilemap["precipitation"]
    r = tilemap["river"]
    return color_final(h, t, p, r)


def color_winds(cfg):
    """Small helper: draw prevailing wind bands to help visualize the model."""
    from .climate import _wind_field
    U, V = _wind_field(cfg)
    H, W = U.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    # east-going (westerlies) green, west-going (trades/polar) orange
    east = U > 0
    west = U < 0
    mag = np.clip(np.abs(U), 0, 1)
    rgb[east] = np.stack([0.2 + 0.0 * mag[east], 0.4 + 0.5 * mag[east], 0.2 + 0.2 * mag[east]], axis=-1)
    rgb[west] = np.stack([0.4 + 0.5 * mag[west], 0.35 + 0.2 * mag[west], 0.1 + 0.0 * mag[west]], axis=-1)
    return rgb


def color_isometric(height, colored=None, tile_h_scale=0.45, sea_level=0.0):
    """Render an isometric 3D-ish view of the heightmap.

    Projection: (x, y, z=height) -> (px, py) with a 2:1 isometric tilt.
        px = (x - y) * 1.0
        py = (x + y) * 0.5 - z * tile_h_scale * H
    Includes directional shading using the height gradient (sun from NW).
    Returns an RGB array (H_iso, W_iso, 3) suitable for st.image.
    
    Improved: Better height rendering with proper vertical columns for terrain relief.
    """
    H, W = height.shape
    if colored is None:
        colored = color_height(height)

    # shading based on gradient (sun from top-left)
    gy, gx = np.gradient(height.astype(np.float32))
    # normal approximation
    nz = 1.0
    nlen = np.sqrt(gx * gx + gy * gy + nz * nz) + 1e-6
    # light vector (from NW, above)
    Lx, Ly, Lz = -0.7, -0.7, 0.9
    Llen = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    Lx, Ly, Lz = Lx / Llen, Ly / Llen, Lz / Llen
    # dot(N, L) where N = (-gx, -gy, 1)/nlen
    dot = (-gx * Lx - gy * Ly + nz * Lz) / nlen
    shade = np.clip(0.45 + 0.85 * dot, 0.15, 1.35).astype(np.float32)
    shaded = np.clip(colored * shade[..., None], 0, 1)

    # canvas dimensions - increased for better quality
    z_scale = tile_h_scale * max(H, W)
    z_max = float(np.clip(height.max(), 0.0, 1.5)) * z_scale
    can_w = int((W + H) * 1.2) + 8
    can_h = int((W + H) * 0.6 + z_max) + 8
    canvas = np.ones((can_h, can_w, 3), dtype=np.float32) * np.array([0.06, 0.07, 0.11], dtype=np.float32)
    depth = np.full((can_h, can_w), -1e9, dtype=np.float32)  # painter's z-buffer proxy

    # draw back-to-front (far y first, then far x). In iso, "back" = small (x+y)
    x_off = int(H * 0.6) + 4  # keep px positive
    y_off = int(W * 0.3) + 4
    
    # Pre-compute all projected points for efficiency
    for y in range(H):
        for x in range(W):
            z = float(height[y, x])
            z_clip = max(z, sea_level)  # ocean stays flat at sea level
            px = int((x - y) + x_off)
            py_top = int((x + y) * 0.5 - z_clip * z_scale) + int(z_max) + y_off
            
            if not (0 <= px < can_w):
                continue
                
            base_py = int((x + y) * 0.5) + int(z_max) + y_off
            base_py = min(base_py, can_h - 1)
            py_top = max(0, min(py_top, can_h - 1))
            
            # column color
            c_top = shaded[y, x]
            c_side = c_top * 0.50  # darker sides for more contrast
            c_cliff = c_top * 0.35  # even darker for steep cliffs
            
            d = -(x + y)
            
            # Calculate cliff intensity based on height difference with neighbors
            cliff_factor = 0.0
            if y > 0:
                cliff_factor = max(cliff_factor, abs(z - height[y-1, x]))
            if y < H-1:
                cliff_factor = max(cliff_factor, abs(z - height[y+1, x]))
            if x > 0:
                cliff_factor = max(cliff_factor, abs(z - height[y, x-1]))
            if x < W-1:
                cliff_factor = max(cliff_factor, abs(z - height[y, x+1]))
            cliff_factor = min(1.0, cliff_factor * 2.0)
            
            # Draw top pixel (larger for better visibility)
            if py_top >= 0 and depth[py_top, px] < d:
                canvas[py_top, px] = c_top
                depth[py_top, px] = d
                # Paint adjacent pixels for chunkier look
                for dx_off in range(-1, 2):
                    for dy_off in range(0, 2):
                        nx, ny_pix = px + dx_off, py_top + dy_off
                        if 0 <= ny_pix < can_h and 0 <= nx < can_w and depth[ny_pix, nx] < d:
                            canvas[ny_pix, nx] = c_top
                            depth[ny_pix, nx] = d
            
            # side (cliff) fill - only if there's significant height
            col_height = base_py - py_top
            if col_height > 1:
                for pp in range(py_top + 1, min(base_py + 1, can_h)):
                    for dx_off in range(-1, 1):
                        nx = px + dx_off
                        if 0 <= nx < can_w and depth[pp, nx] < d:
                            # Gradient from top to bottom of cliff
                            t = (pp - py_top) / max(1, col_height)
                            c_cliff_grad = c_side * (1 - t * 0.4)  # Darker at bottom
                            canvas[pp, nx] = c_cliff_grad
                            depth[pp, nx] = d

    return np.clip(canvas, 0, 1)
