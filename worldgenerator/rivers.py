"""Ríos según DDG:
- Bocas en costas oceánicas con detector de radio amplio (evita lagunas).
- Fuentes aleatorias en montañas.
- Flujo: radio con `candidates` puntos en la circunferencia; el punto se elige
  por peso de altura (más bajo = más peso) y dirección a la boca (la dirección
  gana peso cuando las alturas son relativamente similares). Al llegar a la boca
  (distancia < radio) se conectan los puntos.
- Ancho: ancho del punto anterior + promedio de precipitación local.
- Fusión al tocar otro río (también al calcularlos), splits aleatorios,
  splits que se re-unen (islas) y splits extra al final (deltas)."""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import label, binary_dilation

try:
    from numba import njit
except Exception:                                # fallback sin numba
    def njit(*a, **k):
        return lambda f: f


@njit(cache=True)
def _trace(height, river_ids, sy, sx, my, mx, rid, seed,
           cand, r0, dir_w, sim_h, uphill_pen, max_steps):
    np.random.seed(seed)
    H, W = height.shape
    path = np.empty((max_steps, 2), np.int32)
    n = 0
    y, x = sy, sx % W
    path[n, 0] = y; path[n, 1] = x; n += 1
    pdy = 0.0; pdx = 0.0
    merged = -1
    while n < max_steps - 1:
        dym = float(my - y); dxm = float(mx - x)
        if dxm > W / 2: dxm -= W                  # wrap corto (mundo esférico)
        if dxm < -W / 2: dxm += W
        dist = math.hypot(dym, dxm)
        if dist <= r0:
            break                                 # radio > distancia a la boca → conectar
        hmax = -1e18; hmin = 1e18
        hs = np.empty(cand, np.float64)
        for k in range(cand):
            ang = 6.283185307179586 * k / cand
            cy = int(math.floor(y + r0 * math.sin(ang) + 0.5))
            cx = int(math.floor(x + r0 * math.cos(ang) + 0.5)) % W
            cy = min(max(cy, 0), H - 1)
            hh = height[cy, cx]
            hs[k] = hh
            if hh > hmax: hmax = hh
            if hh < hmin: hmin = hh
        ad = dir_w if (hmax - hmin) < sim_h else dir_w * 0.25   # alturas similares → manda dirección
        mdy = dym / dist; mdx = dxm / dist
        ws = np.empty(cand, np.float64)
        tot = 0.0
        for k in range(cand):
            ang = 6.283185307179586 * k / cand
            vy = math.sin(ang); vx = math.cos(ang)
            wh = (hmax - hs[k]) + 1e-3                          # menor altura → más peso
            wd = 1.0 + ad * (vy * mdy + vx * mdx)
            wgt = wh * wd
            if hs[k] > height[y, x] + 0.5:
                wgt *= uphill_pen
            if vy * pdy + vx * pdx < -0.5:
                wgt *= 0.1                                      # no volver sobre sus pasos
            if wgt < 1e-9: wgt = 1e-9
            ws[k] = wgt; tot += wgt
        pick = np.random.random() * tot
        acc = 0.0; chosen = 0
        for k in range(cand):
            acc += ws[k]
            if pick <= acc:
                chosen = k; break
        ang = 6.283185307179586 * chosen / cand
        ny = int(math.floor(y + r0 * math.sin(ang) + 0.5))
        nx = int(math.floor(x + r0 * math.cos(ang) + 0.5)) % W
        ny = min(max(ny, 0), H - 1)
        pdy = (ny - y) / r0; pdx = (nx - x) / r0
        y, x = ny, nx
        path[n, 0] = y; path[n, 1] = x; n += 1
        if height[y, x] < 0:
            break                                               # llegó al mar
        other = river_ids[y, x]
        if other > 0 and other != rid:
            merged = other                                      # fusión con otro río (o isla)
            break
    return path[:n], merged


def _sat(big):
    H, W = big.shape
    s = np.zeros((H + 1, W + 1))
    s[1:, 1:] = big.astype(np.float64).cumsum(0).cumsum(1)
    return s


def _rect(s, y0, y1, x0, x1):
    return s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]


def _stamp(river_ids, width_map, path, precip, rid, w0, k):
    H, W = river_ids.shape
    w = w0
    py, px = int(path[0, 0]), int(path[0, 1])
    for i in range(len(path)):
        y, x = int(path[i, 0]), int(path[i, 1])
        dxr = x - px
        if dxr > W // 2: dxr -= W
        if dxr < -W // 2: dxr += W
        steps = max(abs(y - py), abs(dxr), 1)
        for s in range(steps + 1):
            t = s / steps
            yy = int(round(py + (y - py) * t))
            xx = int(round(px + dxr * t)) % W
            w = w + float(precip[yy, xx]) * k      # ancho = previo + precipitación local
            r = int(min(6, max(1, round(w))))
            y0 = max(0, yy - r); y1 = min(H, yy + r + 1)
            for yyy in range(y0, y1):
                d2 = r * r - (yyy - yy) ** 2
                if d2 < 0:
                    continue
                dx = int(d2 ** 0.5)
                x0 = (xx - dx) % W; x1 = (xx + dx) % W
                if x0 <= x1:
                    river_ids[yyy, x0:x1 + 1] = rid
                    width_map[yyy, x0:x1 + 1] = np.maximum(width_map[yyy, x0:x1 + 1], w)
                else:                              # wrap E/O
                    river_ids[yyy, x0:] = rid; river_ids[yyy, :x1 + 1] = rid
                    width_map[yyy, x0:] = np.maximum(width_map[yyy, x0:], w)
                    width_map[yyy, :x1 + 1] = np.maximum(width_map[yyy, :x1 + 1], w)
        py, px = y, x
    return w


def generate_rivers(cfg, height, precip, rng, log=print):
    rc = cfg.rivers
    H, W = height.shape
    ocean = height < 0
    lab, _ = label(ocean, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    big = np.isin(lab, np.nonzero(sizes >= rc.pond_max_area)[0]) & ocean   # detector de lagunas
    coast = (~ocean) & binary_dilation(big, iterations=1)
    s = _sat(big)
    R = rc.mouth_radius
    ys, xs = np.nonzero(coast)
    y0 = np.clip(ys - R, 0, H); y1 = np.clip(ys + R + 1, 0, H)
    x0 = np.clip(xs - R, 0, W); x1 = np.clip(xs + R + 1, 0, W)
    frac = _rect(s, y0, y1, x0, x1) / ((y1 - y0) * (x1 - x0))
    mys, mxs = ys[frac >= rc.mouth_min_ocean], xs[frac >= rc.mouth_min_ocean]

    mouths = []
    for k in rng.permutation(len(mys)):
        yy, xx = int(mys[k]), int(mxs[k])
        ok = True
        for (a, b) in mouths:
            dxw = min((xx - b) % W, (b - xx) % W)
            if (yy - a) ** 2 + dxw ** 2 < rc.min_mouth_separation ** 2:
                ok = False; break
        if ok:
            mouths.append((yy, xx))
        if len(mouths) >= rc.target:
            break
    log(f"Desembocaduras válidas: {len(mouths)} (radio detector {R}px, anti-lagunas)")
    if not mouths:
        log("Sin desembocaduras: se omite la generación de ríos")
        return np.zeros((H, W), np.int32), np.zeros((H, W), np.float32)

    sys_, sxs_ = np.nonzero(height >= rc.min_source_height)
    river_ids = np.zeros((H, W), np.int32)
    width_map = np.zeros((H, W), np.float32)
    if len(sys_) == 0:
        log("No hay montañas para fuentes de ríos")
        return river_ids, width_map

    ma = np.array(mouths)
    order = rng.permutation(len(sys_))[: rc.target]
    rid = 0
    for si in order:
        sy, sx = int(sys_[si]), int(sxs_[si])
        dxw = np.minimum((ma[:, 1] - sx) % W, (sx - ma[:, 1]) % W)
        d2 = (ma[:, 0] - sy) ** 2 + dxw ** 2
        my, mx = mouths[int(np.argmin(d2))]
        rid += 1
        path, merged = _trace(height, river_ids, sy, sx, my, mx, rid,
                              int(rng.integers(0, 2 ** 31 - 1)),
                              rc.candidates, float(rc.step_radius), rc.dir_weight,
                              rc.similar_height_m, rc.uphill_penalty, 4 * H)
        if len(path) < 3:
            continue
        if merged == -1 and height[path[-1, 0], path[-1, 1]] >= 0:
            path = np.vstack([path, np.array([[my, mx]], np.int32)])   # conectar con la boca
        w_end = _stamp(river_ids, width_map, path, precip, rid, 1.0, rc.width_per_precip)

        # Splits: chance por punto; mayor cerca de la boca (deltas); re-unión → islas
        branches = 0
        for pi in range(3, len(path) - 2):
            if branches >= 3:
                break
            yy, xx = int(path[pi, 0]), int(path[pi, 1])
            near = (yy - my) ** 2 + min((xx - mx) % W, (mx - xx) % W) ** 2 < rc.delta_radius ** 2
            ch = rc.delta_split_chance if near else rc.split_chance
            if rng.random() >= ch:
                continue
            if near:                                            # delta: otra boca cercana
                d2m = (ma[:, 0] - my) ** 2 + np.minimum((ma[:, 1] - mx) % W, (mx - ma[:, 1]) % W) ** 2
                close = np.nonzero(d2m < (3 * rc.delta_radius) ** 2)[0]
                tm_y, tm_x = mouths[int(close[rng.integers(len(close))])] if len(close) else (my, mx)
            else:
                tm_y, tm_x = my, mx
            rid += 1
            bpath, _ = _trace(height, river_ids, yy, xx, tm_y, tm_x, rid,
                              int(rng.integers(0, 2 ** 31 - 1)),
                              rc.candidates, float(rc.step_radius), rc.dir_weight,
                              rc.similar_height_m, rc.uphill_penalty, 2 * H)
            if len(bpath) >= 3:
                _stamp(river_ids, width_map, bpath, precip, rid, w_end * 0.6, rc.width_per_precip)
                branches += 1
    height[river_ids > 0] -= 1.5                                # talla leve del cauce
    log(f"Ríos generados: {rid} cauces (con fusiones, islas y deltas)")
    return river_ids, width_map