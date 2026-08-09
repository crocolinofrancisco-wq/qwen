"""Heightmap generation: multi-octave perlin + plate base height + edge smoothing.
Then adds mountains near boundaries using perlin + Fractal Ridge Blending (Devote).
All noise is generated with horizontal (X) wrap when spherical_wrap is True.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt

from .config import WorldConfig
from .noise import PerlinNoise, ridge_layer


def _wrap_mode(cfg):
    return ("reflect", "wrap") if cfg.spherical_wrap else "reflect"


def _wrap_distance(mask, wrap_x):
    """Distance transform that respects horizontal wrapping."""
    if not wrap_x:
        return distance_transform_edt(~mask)
    # tile mask 3x horizontally, compute EDT, take middle
    tiled = np.concatenate([mask, mask, mask], axis=1)
    dist = distance_transform_edt(~tiled)
    W = mask.shape[1]
    return dist[:, W:2*W]


def generate_heightmap(cfg: WorldConfig, rng, plate_map, plates, progress=None):
    W, H = cfg.sim_width, cfg.sim_height
    pn = PerlinNoise(seed=int(rng.integers(0, 1_000_000)))
    fractal = pn.spherical_fractal if cfg.spherical_wrap else pn.fractal

    layers = []
    weights = []
    scale = cfg.perlin_base_scale
    w = 1.0
    for i in range(cfg.perlin_octaves):
        layers.append(fractal(W, H, octaves=2, scale=scale,
                              persistence=cfg.perlin_persistence,
                              lacunarity=cfg.perlin_lacunarity))
        weights.append(w)
        scale *= 2.0
        w *= 0.5
    weights = np.array(weights) / sum(weights)
    height = np.zeros((H, W), dtype=np.float32)
    for l, wt in zip(layers, weights):
        height += l * wt

    height = np.clip(height, 0.0, cfg.perlin_max_height)
    if progress: progress("height:perlin", 0.15)

    plate_base = np.zeros(len(plates), dtype=np.float32)
    for i, p in enumerate(plates):
        plate_base[i] = p.base_height * 0.4
    height += plate_base[plate_map]

    oc_mask = np.array([p.is_oceanic for p in plates])[plate_map]
    height[oc_mask] -= 0.15

    if progress: progress("height:plate_offsets", 0.4)

    # edges of plates (wrap-aware)
    up = np.roll(plate_map, -1, axis=0)
    down = np.roll(plate_map, 1, axis=0)
    left = np.roll(plate_map, -1, axis=1)
    right = np.roll(plate_map, 1, axis=1)
    edges = (up != plate_map) | (down != plate_map) | (left != plate_map) | (right != plate_map)
    edge_zone = binary_dilation(edges, iterations=max(1, cfg.edge_smooth_radius))
    smooth = gaussian_filter(height, sigma=cfg.edge_smooth_radius, mode=_wrap_mode(cfg))
    height = np.where(edge_zone, smooth, height)

    height = height - np.median(height)
    m = np.max(np.abs(height)) + 1e-6
    height = height / m

    if progress: progress("height:done", 0.6)
    return height


def add_mountains(cfg: WorldConfig, rng, height, boundary_type, stress, plates, plate_map, progress=None):
    W, H = cfg.sim_width, cfg.sim_height
    pn = PerlinNoise(seed=int(rng.integers(0, 1_000_000)))
    fractal = pn.spherical_fractal if cfg.spherical_wrap else pn.fractal

    base = fractal(W, H, octaves=cfg.mountain_octaves,
                   scale=cfg.perlin_base_scale * 2, persistence=0.55, lacunarity=2.1)

    ridges = []
    for i in range(cfg.mountain_ridge_layers):
        r_base = fractal(W, H, octaves=3, scale=cfg.perlin_base_scale * (2 + i),
                         persistence=0.5, lacunarity=2.0)
        shift = rng.uniform(cfg.mountain_ridge_shift_min, cfg.mountain_ridge_shift_max)
        ridges.append(ridge_layer(r_base, shift))
    ridge = np.mean(ridges, axis=0)

    mountain = cfg.mountain_blend * ridge + (1 - cfg.mountain_blend) * base
    if progress: progress("mountains:noise", 0.75)

    bmask = boundary_type > 0
    if bmask.any():
        dist = _wrap_distance(bmask, cfg.spherical_wrap)
        influence = np.clip(1.0 - dist / max(1, cfg.mountain_influence_radius), 0, 1)
    else:
        influence = np.zeros_like(height)

    conv = (boundary_type == 2).astype(np.float32)
    dive = (boundary_type == 1).astype(np.float32)
    tran = (boundary_type == 3).astype(np.float32)
    conv_field = gaussian_filter(conv * stress, sigma=cfg.mountain_influence_radius / 2, mode=_wrap_mode(cfg))
    dive_field = gaussian_filter(dive * stress, sigma=cfg.mountain_influence_radius / 2, mode=_wrap_mode(cfg))
    tran_field = gaussian_filter(tran * stress, sigma=cfg.mountain_influence_radius / 2, mode=_wrap_mode(cfg))

    conv_field /= (conv_field.max() + 1e-6)
    dive_field /= (dive_field.max() + 1e-6)
    tran_field /= (tran_field.max() + 1e-6)

    height = height + mountain * conv_field * 0.9
    height = height - mountain * dive_field * 0.3
    height = height + mountain * tran_field * 0.35 * influence

    height = np.clip(height, -1.0, 1.5)
    if progress: progress("mountains:done", 0.9)
    return height
