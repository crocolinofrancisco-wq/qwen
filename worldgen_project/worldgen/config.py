"""Configuration dataclass for world generation.

The world is always generated on a sphere (3D) and then projected onto a
flat equirectangular map, per the design document — there is no optional
"wrap" flag anymore because continuity is inherent to the pipeline.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class WorldConfig:
    # Resolution
    sim_width: int = 384          # internal simulation resolution (longitude)
    sim_height: int = 192         # internal simulation resolution (latitude)
    tile_width: int = 96          # exported tile resolution (same proportion)
    tile_height: int = 48

    # Plates (generated on the sphere)
    plate_points_x: int = 22
    plate_points_y: int = 11
    point_jitter: float = 0.55
    major_plate_count: int = 9
    small_plate_count: int = 14
    oceanic_ratio: float = 0.62
    oceanic_height_offset: float = -0.35

    # Heightmap (multi-octave Perlin sampled on the sphere)
    perlin_octaves: int = 4
    perlin_base_scale: float = 4.0
    perlin_persistence: float = 0.5
    perlin_lacunarity: float = 2.0
    perlin_max_height: float = 0.85
    edge_smooth_radius: int = 3

    # Boundaries / mountains (Fractal Ridge Blending by Devote)
    mountain_influence_radius: int = 8
    mountain_octaves: int = 4
    mountain_ridge_layers: int = 3
    mountain_ridge_shift_min: float = 0.35
    mountain_ridge_shift_max: float = 0.65
    mountain_blend: float = 0.55

    # Climate (axial tilt as fraction of 90 degrees; 21 deg -> 0.2333)
    axial_tilt: float = 0.2333
    climate_noise: float = 0.10
    temp_height_factor: float = 0.9

    # Precipitation
    precip_noise: float = 0.15
    arid_dropoff: float = 1.3
    equator_band_width: float = 0.22
    trade_wind_strength: float = 0.55
    orographic_strength: float = 0.9

    # Continental drift (each step is simulated on the sphere, then
    # projected; every step is kept as a snapshot for the timeline slider)
    drift_steps: int = 3
    drift_chance: float = 0.05
    drift_radius: int = 6
    drift_neighbor_chance: float = 0.55
    drift_plate_shift: float = 0.6
    keep_drift_phases: bool = True

    # Rivers
    river_mouth_count: int = 14
    river_mouth_min_ocean_radius: int = 6
    river_source_count: int = 18
    river_step_radius: int = 3
    river_min_radius: int = 1
    river_merge_radius: int = 2
    river_split_chance: float = 0.02
    river_delta_split_chance: float = 0.35
    river_base_width: float = 1.0

    # Tile unification
    similarity_threshold: float = 0.88
    unify_attributes: Tuple[str, ...] = ("height", "temperature", "precipitation", "material")

    # Isometric viewer
    iso_height_scale: float = 0.12   # vertical exaggeration of the 3D view
    iso_max_cells: int = 160         # max grid cells per side in the 3D view

    # Random
    seed: int = 1337

    def to_dict(self):
        d = asdict(self)
        d["unify_attributes"] = list(d["unify_attributes"])
        return d
