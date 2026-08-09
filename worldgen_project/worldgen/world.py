"""World container and export."""
from __future__ import annotations
import json
import numpy as np
from dataclasses import asdict
from .config import WorldConfig


MATERIAL_NAMES = {
    0: "ocean", 1: "beach", 2: "soil", 3: "rock", 4: "snow", 5: "sand", 6: "river"
}


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.plate_map = None
        self.plates = []
        self.boundary_type = None
        self.stress = None
        self.height = None
        self.temperature = None
        self.precipitation = None
        self.river_map = None
        self.rivers = []
        self.tilemap = None
        self.tile_groups = []
        self.drift_phases = []  # list of (plate_map, height) snapshots

    def export_json(self, path: str):
        data = {
            "config": self.cfg.to_dict(),
            "tile_resolution": [self.cfg.tile_width, self.cfg.tile_height],
            "plates": [
                {
                    "id": p.id,
                    "center": list(p.center),
                    "oceanic": bool(p.is_oceanic),
                    "move_dir": list(p.move_dir),
                    "base_height": float(p.base_height),
                }
                for p in self.plates
            ],
            "tile_groups": [
                {
                    **g,
                    "material_name": MATERIAL_NAMES.get(g["material"], "unknown"),
                }
                for g in self.tile_groups
            ],
            "drift_phases_count": len(self.drift_phases),
            "stats": {
                "sim_resolution": [self.cfg.sim_width, self.cfg.sim_height],
                "total_groups": len(self.tile_groups),
                "total_tiles": self.cfg.tile_width * self.cfg.tile_height,
                "compression_ratio": (self.cfg.tile_width * self.cfg.tile_height) /
                                     max(1, len(self.tile_groups)),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path
