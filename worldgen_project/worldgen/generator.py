"""Top-level orchestrator: runs the full generation pipeline."""
from __future__ import annotations
import numpy as np

from .config import WorldConfig
from .world import World
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
        plate_map, plates, seeds, tree = generate_plates(cfg, rng, progress=self._p)
        world.plate_map = plate_map
        world.plates = plates

        boundary_type, stress = compute_boundaries(plate_map, plates)
        world.boundary_type = boundary_type
        world.stress = stress
        self._p("boundaries", 0.72)

        height = generate_heightmap(cfg, rng, plate_map, plates, progress=self._p)
        height = add_mountains(cfg, rng, height, boundary_type, stress, plates, plate_map, progress=self._p)
        world.height = height
        self._p("height_done", 0.9)

        # Dynamic drift (now ON by default)
        if cfg.drift_steps > 0:
            plate_map2, height, phases = simulate_drift(cfg, rng, plate_map, plates, height, progress=self._p)
            world.plate_map = plate_map2
            world.height = height
            world.drift_phases = phases  # list of (plate_map, height) per phase

        temp = generate_temperature(cfg, rng, height, progress=self._p)
        precip = generate_precipitation(cfg, rng, height, temp, progress=self._p)
        world.temperature = temp
        world.precipitation = precip

        river_map, rivers = generate_rivers(cfg, rng, height, precip, progress=self._p)
        world.river_map = river_map
        world.rivers = rivers

        self._p("phase2:downsample", 0.99)
        tilemap = build_tilemap(cfg, height, temp, precip, river_map, world.plate_map, progress=self._p)
        world.tilemap = tilemap
        groups = unify_tiles(cfg, tilemap, progress=self._p)
        world.tile_groups = groups

        self._p("done", 1.0)
        return world
