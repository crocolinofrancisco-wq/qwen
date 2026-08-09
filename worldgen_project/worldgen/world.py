"""World container and export."""
from __future__ import annotations
import json
from .config import WorldConfig


MATERIAL_NAMES = {
    0: "ocean", 1: "beach", 2: "soil", 3: "rock", 4: "snow", 5: "sand", 6: "river"
}


class World:
    """All arrays are equirectangular PROJECTIONS of the sphere state.

    `drift_phases` holds one snapshot per dynamic step; every snapshot was
    generated on the sphere (3D) first and then projected to the flat map,
    so the evolution timeline is continuous across the horizontal edges.
    """

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.lat = None          # (H, W) latitude grid (rad)
        self.lon = None          # (H, W) longitude grid (rad)
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
        self.drift_phases = []   # list of (plate_map, height) snapshots

    def export_json(self, path: str):
        data = {
            "config": self.cfg.to_dict(),
            "projection": "equirectangular (generated on sphere, then projected)",
            "tile_resolution": [self.cfg.tile_width, self.cfg.tile_height],
            "plates": [
                {
                    "id": p.id,
                    "center_lonlat": [float(p.center[0]), float(p.center[1])],
                    "oceanic": bool(p.is_oceanic),
                    "move_dir": [float(p.move_dir[0]), float(p.move_dir[1])],
                    "euler_pole": [float(a) for a in p.rot_axis],
                    "rotation_rate": float(p.rot_rate),
                    "base_height": float(p.base_height),
                }
                for p in self.plates
            ],
            "tile_groups": [
                {**g, "material_name": MATERIAL_NAMES.get(g["material"], "unknown")}
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
