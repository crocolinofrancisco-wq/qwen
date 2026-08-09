"""Terreno y clima.
- Heightmap: base por placa (punto aleatorio de un campo de baja frecuencia,
  estable entre eras) + detalle Perlin de 4 octavas con TOPE (no genera montañas).
  Placas oceánicas: divisor del heightmap. Bordes de continente suavizados.
- Montañas: cerca de límites, forma según tipo de límite; Perlin de 4 octavas
  + 3 capas Fractal Ridge Blending, mezcladas suavemente.
- Clima: latitud (con tilt y ruido) + altura. Lluvias por bandas con la
  transición a árido más prominente que la vuelta a húmedo."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial.transform import Rotation
from plates import B_DIVERGENT, B_CONVERGENT, B_TRANSFORM


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_heightmap(cfg, plates, perlin, xyz, log=print):
    tcfg = cfg.terrain
    H, W = xyz.shape[:2]
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    idm = plates.id_map

    # Detalle por placa: coordenadas rotadas por la deriva acumulada
    # → el terreno "viaja" con su placa entre eras.
    detail = np.zeros((H, W), np.float32)
    for i in range(plates.n):
        m = idm == i
        if not m.any():
            continue
        pts = xyz[m]
        off = plates.offsets[i]
        if np.abs(off).sum() > 1e-9:
            pts = Rotation.from_rotvec(off).inv().apply(pts)
        detail[m] = perlin.fbm(pts[:, 0], pts[:, 1], pts[:, 2],
                               octaves=tcfg.octaves, freq=tcfg.frequency,
                               gain=tcfg.gain, lac=tcfg.lacunarity)
    d = np.clip((detail - 0.5) * 2.0, -1.0, tcfg.perlin_cap)   # tope anti-montañas

    s = plates.anchors                                          # punto aleatorio del heightmap
    cont = tcfg.continent_base_min + (tcfg.continent_base_max - tcfg.continent_base_min) * s
    oce = -(0.55 + 0.45 * s)
    base = np.where(plates.oceanic[idm], oce[idm], cont[idm]).astype(np.float32)
    divider = np.where(plates.oceanic[idm], 1.0 / cfg.plates.oceanic_divider, 1.0)
    h01 = base + d * tcfg.detail_amp * divider

    # Suavizado en bordes de continentes (transiciones más suaves)
    border = plates.btype > 0
    distb = distance_transform_edt(~border)
    w = np.exp(-(distb / max(1, tcfg.edge_smooth_px)) ** 2)
    h01 = h01 * (1 - w) + gaussian_filter(h01, tcfg.edge_smooth_px) * w

    meters = np.where(h01 >= 0, h01 * tcfg.land_scale_m,
                      h01 * tcfg.ocean_depth_m).astype(np.float32)

    # ---- Montañas según tipo de límite ----
    rng_local = np.random.default_rng(cfg.world.seed + 77)      # estable entre eras
    stress = plates.stress / max(float(plates.stress.max()), 1e-6)
    sblur = gaussian_filter(stress, tcfg.boundary_radius / 3.0)
    radius = float(tcfg.boundary_radius)

    pm = perlin.fbm(x, y, z, octaves=tcfg.octaves, freq=6.0,
                    gain=tcfg.gain, lac=tcfg.lacunarity)        # Perlin 4 octavas
    ridge = np.zeros((H, W), np.float32)
    for i in range(tcfg.ridge_layers):                          # 3 capas Fractal Ridge
        shift = rng_local.uniform(tcfg.ridge_shift_min, tcfg.ridge_shift_max)
        ridge += perlin.ridge(x, y, z, freq=5.0 * (2 ** i), shift=shift)
    ridge /= tcfg.ridge_layers
    b = tcfg.ridge_blend
    m = (1.0 - b) * pm + b * ridge                              # mezcla suave

    land = meters >= 0
    for bt, action in ((B_CONVERGENT, "conv"), (B_DIVERGENT, "div"), (B_TRANSFORM, "tra")):
        mask = plates.btype == bt
        if not mask.any():
            continue
        infl = np.exp(-(distance_transform_edt(~mask) / radius) ** 2) * sblur
        if action == "conv":                                    # cordilleras + fosas
            meters += (m * infl * tcfg.max_mountain_m * np.where(land, 1.0, 0.4)).astype(np.float32)
            kind = gaussian_filter((plates.bkind >= 1).astype(np.float32), radius / 3.0)
            meters -= (m * infl * tcfg.trench_depth_m * (~land) * (kind >= 0.4)).astype(np.float32)
        elif action == "div":                                   # rift en tierra / dorsal oceánica
            meters -= (m * infl * tcfg.rift_depth_m * land).astype(np.float32)
            meters += (m * infl * tcfg.ocean_ridge_m * (~land)).astype(np.float32)
        else:                                                   # transform: relieves menores ±
            meters += ((pm - 0.5) * 2.0 * infl * tcfg.transform_relief_m).astype(np.float32)
    log(f"Terreno: min {float(meters.min()):.0f} m / max {float(meters.max()):.0f} m")
    return meters


def climate(cfg, height, perlin, xyz, lat, lon):
    ccfg = cfg.climate
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    lat_eff = lat_deg - ccfg.tilt_deg * np.sin(np.radians(lon_deg))   # tilt: desplaza bandas
    a = np.abs(lat_eff)

    # Temperatura: más cálido hacia el centro; la altura enfría
    # (las cotas cercanas al nivel del mar quedan más cálidas).
    t = ccfg.equator_temp + (ccfg.pole_temp - ccfg.equator_temp) * (a / 90.0) ** ccfg.lat_exp
    t -= np.clip(height, 0, None) * ccfg.lapse_c_per_m
    tn = perlin.fbm(xyz[..., 0], xyz[..., 1], xyz[..., 2], octaves=3, freq=3.0)
    t += (tn - 0.5) * 2.0 * ccfg.noise_amp_c

    # Precipitaciones: alto en ecuador → caída PROMINENTE a árido →
    # subida SUAVE en latitudes medias → desierto polar.
    p = np.full(a.shape, ccfg.equator_rains, np.float32)
    p = p + (ccfg.arid_rains - p) * _smoothstep(*ccfg.arid_fall, a)
    p = p + (ccfg.midlat_rains - p) * _smoothstep(*ccfg.wet_rise, a)
    p = p + (ccfg.polar_rains - p) * _smoothstep(*ccfg.polar_fall, a)
    pn = perlin.fbm(xyz[..., 0], xyz[..., 1], xyz[..., 2], octaves=3, freq=4.0)
    p = np.clip(p * (1.0 + (pn - 0.5) * 2.0 * ccfg.precip_noise), 0.0, None)

    # Humedad: lluvia + cercanía al agua − evaporación por calor + ruido
    land = height >= 0
    distw = distance_transform_edt(land)                        # px al océano
    bonus = np.clip(1.0 - distw / 15.0, 0.0, 1.0) * 30.0
    hn = perlin.fbm(xyz[..., 0], xyz[..., 1], xyz[..., 2], octaves=2, freq=5.0)
    hum = np.clip(100.0 * p / (p + 900.0) + bonus
                  - np.clip(t - 28.0, 0.0, None) * 1.1 + (hn - 0.5) * 16.0, 0.0, 100.0)
    return t.astype(np.float32), p.astype(np.float32), hum.astype(np.float32)