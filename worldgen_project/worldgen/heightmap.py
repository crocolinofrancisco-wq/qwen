"""Heightmap generated natively ON the sphere (3D) and projected to 2D.

All noise is sampled from 3D sphere points, so the projected map is seamless
across the horizontal edges by construction (lon = -pi and lon = +pi are the
same 3D points).  Mountains follow the design: multi-octave Perlin layers +
Fractal Ridge Blending, blended per boundary type.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt

from .config import WorldConfig
from .spherical_noise import SphericalNoise
from .sphere import latlon_grids, unit_sphere_points


def _gauss_wrap(arr, sigma):
    """Gaussian smoothing: horizontal edges wrap (sphere projection),
    poles reflect."""
    return gaussian_filter(arr, sigma=sigma, mode=("reflect", "wrap"))


def _wrap_distance(mask):
    """Distance transform that respects horizontal wrapping."""
    tiled = np.concatenate([mask, mask, mask], axis=1)
    dist = distance_transform_edt(~tiled)
    W = mask.shape[1]
    return dist[:, W:2 * W]


def generate_heightmap(cfg: WorldConfig, rng, plate_map, plates, lat, lon,
                       progress=None):
    W, H = cfg.sim_width, cfg.sim_height
    xyz = unit_sphere_points(lat, lon)
    pn = SphericalNoise(seed=int(rng.integers(0, 1_000_000)))

    # 4 layers of Perlin/fBm, each more detailed but with less influence
    layers, weights = [], []
    scale, w = cfg.perlin_base_scale, 1.0
    for _ in range(cfg.perlin_octaves):
        layers.append(pn.fbm(xyz, scale=scale, octaves=2,
                             persistence=cfg.perlin_persistence,
                             lacunarity=cfg.perlin_lacunarity))
        weights.append(w)
        scale *= 2.0
        w *= 0.5
    weights = np.asarray(weights) / sum(weights)
    height = np.zeros((H, W), dtype=np.float32)
    for layer, wt in zip(layers, weights):
        height += layer * wt

    # Perlin has a limit so it never creates mountains by itself
    height = np.clip(height, 0.0, cfg.perlin_max_height)
    if progress: progress("height:perlin", 0.15)

    # random base height point per tectonic plate
    plate_base = np.zeros(len(plates), dtype=np.float32)
    for i, p in enumerate(plates):
        plate_base[i] = p.base_height * 0.4
    height += plate_base[plate_map]

    oc_mask = np.array([p.is_oceanic for p in plates])[plate_map]
    height[oc_mask] -= 0.15
    if progress: progress("height:plate_offsets", 0.4)

    # smooth continent edges (plate borders) for smoother transitions
    up = np.roll(plate_map, -1, axis=0)
    down = np.roll(plate_map, 1, axis=0)
    left = np.roll(plate_map, -1, axis=1)
    right = np.roll(plate_map, 1, axis=1)
    edges = (up != plate_map) | (down != plate_map) | (left != plate_map) | (right != plate_map)
    edge_zone = binary_dilation(edges, iterations=max(1, cfg.edge_smooth_radius))
    smooth = _gauss_wrap(height, sigma=cfg.edge_smooth_radius)
    height = np.where(edge_zone, smooth, height)

    height = height - np.median(height)
    m = np.max(np.abs(height)) + 1e-6
    height = height / m
    if progress: progress("height:done", 0.6)
    return height


def add_mountains(cfg: WorldConfig, rng, height, boundary_type, stress,
                  plates, plate_map, lat, lon, progress=None):
    """Mountains near boundaries: Perlin layers + Fractal Ridge Blending."""
    xyz = unit_sphere_points(lat, lon)
    pn = SphericalNoise(seed=int(rng.integers(0, 1_000_000)))

    base = pn.fbm(xyz, scale=cfg.perlin_base_scale * 2,
                  octaves=cfg.mountain_octaves, persistence=0.55, lacunarity=2.1)

    # 3 special Fractal Ridge Blending layers (shift between 0.35-0.65)
    ridges = []
    for i in range(cfg.mountain_ridge_layers):
        shift = float(rng.uniform(cfg.mountain_ridge_shift_min,
                                  cfg.mountain_ridge_shift_max))
        ridges.append(pn.ridged(xyz, scale=cfg.perlin_base_scale * (2 + i),
                                octaves=3, shift=shift))
    ridge = np.mean(ridges, axis=0)

    # smooth blend between perlin layers and ridge layers
    mountain = cfg.mountain_blend * ridge + (1 - cfg.mountain_blend) * base
    if progress: progress("mountains:noise", 0.75)

    bmask = boundary_type > 0
    if bmask.any():
        dist = _wrap_distance(bmask)
        influence = np.clip(1.0 - dist / max(1, cfg.mountain_influence_radius), 0, 1)
    else:
        influence = np.zeros_like(height)

    conv = (boundary_type == 2).astype(np.float32)
    dive = (boundary_type == 1).astype(np.float32)
    tran = (boundary_type == 3).astype(np.float32)
    conv_field = _gauss_wrap(conv * stress, sigma=cfg.mountain_influence_radius / 2)
    dive_field = _gauss_wrap(dive * stress, sigma=cfg.mountain_influence_radius / 2)
    tran_field = _gauss_wrap(tran * stress, sigma=cfg.mountain_influence_radius / 2)
    conv_field /= (conv_field.max() + 1e-6)
    dive_field /= (dive_field.max() + 1e-6)
    tran_field /= (tran_field.max() + 1e-6)

    height = height + mountain * conv_field * 0.9
    height = height - mountain * dive_field * 0.3
    height = height + mountain * tran_field * 0.35 * influence

    height = np.clip(height, -1.0, 1.5)
    if progress: progress("mountains:done", 0.9)
    return height
