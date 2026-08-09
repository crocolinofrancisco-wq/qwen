"""Continental drift simulation.

Reescrito para que realmente produzca deriva visible:
- Cada paso traslada la placa completa según su vector de movimiento (con wrap X).
- Los bordes convergentes acumulan altura (colisión); los divergentes generan
  fosas (rifts) que hunden terreno; los transformantes no cambian la altura.
- Se puede exportar cada fase como un mapa independiente (world.drift_phases).
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

from .config import WorldConfig


def _wrap_shift(arr, dy, dx, wrap_x):
    """Shift arr by (dy, dx). Wraps in X when wrap_x is True; edge-fills Y."""
    out = np.roll(arr, shift=(dy, dx), axis=(0, 1)) if wrap_x else np.roll(arr, shift=(dy, 0), axis=(0, 1))
    if not wrap_x:
        # zero out wrapped X columns
        if dx > 0:
            out[:, :dx] = arr[:, :1]
        elif dx < 0:
            out[:, dx:] = arr[:, -1:]
    if dy > 0:
        out[:dy, :] = arr[:1, :]
    elif dy < 0:
        out[dy:, :] = arr[-1:, :]
    return out


def simulate_drift(cfg: WorldConfig, rng, plate_map, plates, height, progress=None):
    """Simulate `cfg.drift_steps` phases and return final (plate_map, height) plus
    a list of (plate_map, height) snapshots (drift_phases)."""
    phases = []
    if cfg.drift_steps <= 0:
        return plate_map, height, phases

    H, W = plate_map.shape
    pm = plate_map.copy()
    hm = height.copy()
    wrap_x = cfg.spherical_wrap

    for step in range(cfg.drift_steps):
        new_pm = pm.copy()
        new_hm = hm.copy()

        # For each plate, translate its pixels by its move vector
        for pid, p in enumerate(plates):
            mask = pm == pid
            if not mask.any():
                continue
            dx = int(round(p.move_dir[0] * cfg.drift_radius * cfg.drift_plate_shift))
            dy = int(round(p.move_dir[1] * cfg.drift_radius * cfg.drift_plate_shift))
            if dx == 0 and dy == 0:
                continue
            shifted_mask = _wrap_shift(mask.astype(np.uint8), dy, dx, wrap_x).astype(bool)
            shifted_h    = _wrap_shift(hm * mask, dy, dx, wrap_x)
            # Where the shifted mask lands, adopt this plate's id and height (blended)
            new_pm[shifted_mask] = pid
            new_hm[shifted_mask] = 0.65 * shifted_h[shifted_mask] + 0.35 * new_hm[shifted_mask]

        # Boundary reactions: recompute where plate id changed vs old = collision zone
        up    = np.roll(new_pm, -1, axis=0)
        down  = np.roll(new_pm,  1, axis=0)
        left  = np.roll(new_pm, -1, axis=1)
        right = np.roll(new_pm,  1, axis=1)
        borders = (up != new_pm) | (down != new_pm) | (left != new_pm) | (right != new_pm)

        # Convergent zones -> uplift, divergent zones -> subsidence
        # Heuristic: if two neighboring plates are moving toward each other -> uplift
        uplift = np.zeros_like(hm)
        rift   = np.zeros_like(hm)
        ys, xs = np.where(borders)
        # sample-based (cheap): iterate a fraction
        sample_n = min(len(ys), 4000)
        if sample_n > 0:
            sel = rng.choice(len(ys), size=sample_n, replace=False)
            for k in sel:
                y, x = ys[k], xs[k]
                me = new_pm[y, x]
                # pick a neighbor with different id
                for dy_, dx_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = (y + dy_) % H
                    nx = (x + dx_) % W if wrap_x else max(0, min(W - 1, x + dx_))
                    other = new_pm[ny, nx]
                    if other == me:
                        continue
                    if me >= len(plates) or other >= len(plates):
                        continue
                    mv_a = np.array(plates[me].move_dir)
                    mv_b = np.array(plates[other].move_dir)
                    rel = mv_b - mv_a
                    n = np.array([dy_, dx_], dtype=np.float32)
                    nlen = np.linalg.norm(n) + 1e-6
                    n /= nlen
                    nc = float(np.dot(rel, n))
                    if nc < -0.1:      # converging
                        uplift[y, x] += min(0.25, -nc * 0.4)
                    elif nc > 0.1:     # diverging
                        rift[y, x] -= min(0.20, nc * 0.35)
                    break

        # Diffuse uplift/rift so they form ridges/rifts, not spikes
        uplift = gaussian_filter(uplift, sigma=1.5, mode=("reflect", "wrap") if wrap_x else "reflect")
        rift   = gaussian_filter(rift,   sigma=1.5, mode=("reflect", "wrap") if wrap_x else "reflect")

        new_hm = new_hm + uplift + rift
        new_hm = np.clip(new_hm, -1.0, 1.5)

        pm, hm = new_pm, new_hm
        if cfg.keep_drift_phases:
            phases.append((pm.copy(), hm.copy()))

        if progress:
            progress(f"drift:step{step+1}", 0.9 + 0.05 * ((step + 1) / max(1, cfg.drift_steps)))

    return pm, hm, phases
