"""Placas tectónicas: grilla con ruido → Voronoi esférico (KD-tree) →
micro-placas en bordes → tipo oceánica/continental → dirección de movimiento
→ clasificación de límites (divergente/convergente/transform) con nivel de estrés."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from sphere import make_dots

B_NONE, B_DIVERGENT, B_CONVERGENT, B_TRANSFORM = 0, 1, 2, 3


@dataclass
class Plates:
    id_map: np.ndarray       # (H,W) int32
    oceanic: np.ndarray      # (n,) bool
    directions: np.ndarray   # (n,3) vector tangente * velocidad
    centers: np.ndarray      # (n,3) unitario
    anchors: np.ndarray      # (n,) nivel base del heightmap por placa (punto aleatorio)
    dots: np.ndarray         # grilla inicial de puntos
    btype: np.ndarray        # (H,W) int8  B_*
    bkind: np.ndarray        # (H,W) int8  0 cont-cont, 1 mixto, 2 oce-oce
    stress: np.ndarray       # (H,W) float32
    offsets: np.ndarray      # (n,3) deriva acumulada (rotvec aprox.)

    @property
    def n(self):
        return len(self.oceanic)


def generate_plates(cfg, rng, xyz, log=print):
    pcfg = cfg.plates
    H, W = xyz.shape[:2]
    dots = make_dots(pcfg.dots_y, pcfg.dots_x, pcfg.dot_noise, rng)
    _, dot_of = cKDTree(dots).query(xyz.reshape(-1, 3))          # KD-tree p/ velocidad
    k = min(pcfg.major_plates, len(dots))
    centers0 = dots[rng.choice(len(dots), k, replace=False)]
    _, major_of_dot = cKDTree(centers0).query(dots)              # Voronoi: expansión al más cercano
    id_map = major_of_dot[dot_of].reshape(H, W).astype(np.int32)
    log(f"Placas mayores: {k} (Voronoi sobre {len(dots)} puntos con ruido)")
    id_map = _small_plates(id_map, k, pcfg, rng, log)

    n = int(id_map.max()) + 1
    centers = np.zeros((n, 3))
    for i in range(n):
        m = xyz[id_map == i]
        c = m.mean(0)
        centers[i] = c / (np.linalg.norm(c) + 1e-12)
    oceanic = rng.random(n) < pcfg.oceanic_ratio
    dirs = rng.normal(size=(n, 3))
    dirs -= centers * np.sum(dirs * centers, axis=1, keepdims=True)   # tangente a la esfera
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    dirs *= rng.uniform(0.5, 1.0, (n, 1))
    anchors = rng.random(n).astype(np.float32)                    # "punto aleatorio del heightmap"
    plates = Plates(id_map, oceanic, dirs, centers, anchors, dots,
                    None, None, None, np.zeros((n, 3)))
    classify_boundaries(plates, pcfg)
    log(f"Tipos: {int(oceanic.sum())} oceánicas / {n - int(oceanic.sum())} continentales")
    return plates


def _small_plates(id_map, base_n, pcfg, rng, log):
    """Micro-placas: grupos de formas vecinas ubicados en bordes de placa."""
    H, W = id_map.shape
    border = (id_map != np.roll(id_map, 1, 1)) | (id_map != np.roll(id_map, -1, 1))
    border[1:] |= id_map[1:] != id_map[:-1]
    ys, xs = np.nonzero(border)
    order = rng.permutation(len(ys))
    made, next_id, R = 0, base_n, pcfg.small_plate_radius
    for k in order:
        if made >= pcfg.small_plates:
            break
        y0, x0 = int(ys[k]), int(xs[k])
        if id_map[y0, x0] >= base_n:
            continue
        allowed = {int(id_map[y0, x0]), int(id_map[y0, (x0 + 1) % W]),
                   int(id_map[(y0 + 1) % H, x0])}
        seen = {(y0, x0)}
        dq = deque([(y0, x0, 0)])
        region = []
        while dq and len(region) < R * R * 4:
            y, x, d = dq.popleft()
            if d > R:
                continue
            region.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, (x + dx) % W
                if (0 <= ny < H and (ny, nx) not in seen
                        and int(id_map[ny, nx]) in allowed and id_map[ny, nx] < base_n):
                    seen.add((ny, nx))
                    dq.append((ny, nx, d + 1))
        if len(region) < 12:
            continue
        for y, x in region:
            id_map[y, x] = next_id
        next_id += 1
        made += 1
    log(f"Micro-placas en bordes: {made}")
    return id_map


def classify_boundaries(plates, pcfg):
    """Tipo de límite según movimiento relativo y nivel de estrés por píxel."""
    id_map = plates.id_map
    H, W = id_map.shape
    btype = np.zeros((H, W), np.int8)
    bkind = np.zeros((H, W), np.int8)
    stress = np.zeros((H, W), np.float32)
    ea, eb, ey, ex = [], [], [], []
    for axis, shift in ((1, -1), (0, -1)):
        other = np.roll(id_map, shift, axis)
        edge = other != id_map
        if axis == 0:
            edge[-1, :] = False                      # sin wrap a través de los polos
        ys, xs = np.nonzero(edge)
        ea.append(id_map[ys, xs]); eb.append(other[ys, xs]); ey.append(ys); ex.append(xs)
    ea = np.concatenate(ea); eb = np.concatenate(eb)
    ey = np.concatenate(ey); ex = np.concatenate(ex)
    seen = set()
    for pa, pb in zip(ea.tolist(), eb.tolist()):
        if (pa, pb) in seen or (pb, pa) in seen:
            continue
        seen.add((pa, pb))
        nrm = plates.centers[pb] - plates.centers[pa]
        nrm -= plates.centers[pa] * float(np.dot(nrm, plates.centers[pa]))
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm /= ln
        rel = plates.directions[pa] - plates.directions[pb]
        d = float(np.dot(rel, nrm))
        mag = float(np.linalg.norm(rel)) + 1e-9
        tang = abs(float(np.dot(rel, np.cross(nrm, plates.centers[pa]))))
        if abs(d) < pcfg.transform_thresh * mag or tang > abs(d):
            bt = B_TRANSFORM
        elif d > 0:
            bt = B_CONVERGENT
        else:
            bt = B_DIVERGENT
        m = ((ea == pa) & (eb == pb)) | ((ea == pb) & (eb == pa))
        btype[ey[m], ex[m]] = bt
        oa, ob = bool(plates.oceanic[pa]), bool(plates.oceanic[pb])
        bkind[ey[m], ex[m]] = 2 if (oa and ob) else (1 if (oa != ob) else 0)
        stress[ey[m], ex[m]] = np.maximum(stress[ey[m], ex[m]], mag)
    plates.btype, plates.bkind, plates.stress = btype, bkind, stress