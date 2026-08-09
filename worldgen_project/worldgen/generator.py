"""Top-level orchestrator: full generation pipeline.

Pipeline order follows the design document:
  1. Everything is generated ON THE SPHERE (3D): plates, heightmap,
     mountains, drift steps.
  2. Each drift step is a 3D evolution that is then projected to the flat
     2D equirectangular map (snapshots kept for the timeline slider).
  3. Climate, precipitation and rivers are computed on the projection with
     sphere-aware wrapping.
  4. Phase 2 converts the map to tiles, unifies similar tiles, and exports.
"""
from __future__ import annotations
import numpy as np

from .config import WorldConfig
from .world import World
from .sphere import latlon_grids
from .plates import generate_plates, compute_boundaries
from .heightmap import generate_heightmap, add_mountains
from .climate import generate_temperature, generate_precipitation
from .drift import simulate_drift
from .rivers import generate_rivers
from .tiles import build_tilemap, unify_tiles


class WorldGenerator:
    def __init__(self, cfg: WorldConfig, progress_cb=None):
        self.cfg = cfg
        self.progress_cb = progress_cb or (lambda *a, **k: None)

    def _p(self, name, frac):
        self.progress_cb(name, float(frac))

    def generate(self) -> World:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        world = World(cfg)

        self._p("start", 0.0)
        # --- 3D sphere grids: the world is built here, then projected ---
        lat, lon = latlon_grids(cfg.sim_height, cfg.sim_width)
        world.lat, world.lon = lat, lon

        plate_map, plates, seeds_xyz, tree = generate_plates(cfg, rng, progress=self._p)
        world.plate_map = plate_map
        world.plates = plates

        boundary_type, stress = compute_boundaries(plate_map, plates, lat, lon)
        world.boundary_type = boundary_type
        world.stress = stress
        self._p("boundaries", 0.72)

        height = generate_heightmap(cfg, rng, plate_map, plates, lat, lon,
                                    progress=self._p)
        height = add_mountains(cfg, rng, height, boundary_type, stress,
                               plates, plate_map, lat, lon, progress=self._p)
        world.height = height
        self._p("height_done", 0.9)

        # Dynamic drift: every step is simulated in 3D and projected;
        # snapshots feed the evolution timeline slider in the UI.
        if cfg.drift_steps > 0:
            plate_map2, height, phases = simulate_drift(
                cfg, rng, plate_map, plates, height, lat, lon, progress=self._p)
            world.plate_map = plate_map2
            world.height = height
            world.drift_phases = phases
            # boundaries follow the final plate positions
            world.boundary_type, world.stress = compute_boundaries(
                plate_map2, plates, lat, lon)

        temp = generate_temperature(cfg, rng, world.height, lat, lon,
                                    progress=self._p)
        precip = generate_precipitation(cfg, rng, world.height, temp, lat, lon,
                                        progress=self._p)
        world.temperature = temp
        world.precipitation = precip

        river_map, rivers = generate_rivers(cfg, rng, world.height, precip,
                                            progress=self._p)
        world.river_map = river_map
        world.rivers = rivers

        self._p("phase2:downsample", 0.99)
        tilemap = build_tilemap(cfg, world.height, temp, precip, river_map,
                                world.plate_map, progress=self._p)
        world.tilemap = tilemap
        world.tile_groups = unify_tiles(cfg, tilemap, progress=self._p)

        self._p("done", 1.0)
        return world
