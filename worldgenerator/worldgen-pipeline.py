"""Orquestador de fases + exportación JSON (todas las fases dinámicas como
mapas separados). Emite eventos para la UI: log / progress / image / done / error."""
from __future__ import annotations
import json
import os
import queue
import threading
import traceback
from datetime import datetime, timezone
import numpy as np

from config import AppConfig
from noise import Perlin
from sphere import pixel_grid
from plates import generate_plates
from terrain import generate_heightmap, climate
from rivers import generate_rivers
from dynamics import advance_era, apply_blobs
from tiles import build_tiles, export_phase
import render

# Tabla de referencia del DDG. Los biomas NO se asignan acá: se definen
# durante la simulación; se exportan sólo como metadato de referencia.
BIOME_REFERENCE = [
    {"name": "River", "tile": "River", "temp": "any", "rains": "—", "humidity": "—", "height": "—", "plant_abundance": "—", "plant_avg_height": "—"},
    {"name": "Ocean", "tile": "Ocean", "temp": "—", "rains": "—", "humidity": "—", "height": "—", "plant_abundance": "—", "plant_avg_height": "—"},
    {"name": "Kelp forest", "tile": "Ocean", "temp": "8-20", "rains": "—", "humidity": "—", "height": "-30 a -5", "plant_abundance": "60-90", "plant_avg_height": "5-30"},
    {"name": "Coral reef", "tile": "Ocean", "temp": "22-29", "rains": "—", "humidity": "—", "height": "-20 a -1", "plant_abundance": "20-40", "plant_avg_height": "0"},
    {"name": "Mangrove", "tile": "Land (4-6 tiles cerca de océano/río)", "temp": "20-32", "rains": "1000-2500", "humidity": "70-100", "height": "-2 a 5", "plant_abundance": "60-80", "plant_avg_height": "5-15"},
    {"name": "Swamp", "tile": "Land", "temp": "15-30", "rains": "1200-2000+", "humidity": "80-100", "height": "-5 a 50", "plant_abundance": "70-90", "plant_avg_height": "2-15"},
    {"name": "Rainforest", "tile": "Land", "temp": "22-32", "rains": "1500-3500+", "humidity": "75-100", "height": "0-800", "plant_abundance": "85-100", "plant_avg_height": "25-45"},
    {"name": "Forest", "tile": "Land", "temp": "5-20", "rains": "750-1500", "humidity": "55-75", "height": "0-1000", "plant_abundance": "65-85", "plant_avg_height": "15-30"},
    {"name": "Savanna", "tile": "Land", "temp": "20-30", "rains": "500-1300", "humidity": "40-60", "height": "0-500", "plant_abundance": "30-55", "plant_avg_height": "0.5-6"},
    {"name": "Grasslands", "tile": "Land", "temp": "10-25", "rains": "500-1000", "humidity": "45-65", "height": "0-600", "plant_abundance": "45-70", "plant_avg_height": "0.3-1.2"},
    {"name": "Plains", "tile": "Land", "temp": "10-20", "rains": "500-900", "humidity": "40-60", "height": "0-500", "plant_abundance": "40-65", "plant_avg_height": "0.3-1"},
    {"name": "Mediterranean shrubland", "tile": "Land", "temp": "15-25", "rains": "300-700", "humidity": "35-55", "height": "0-800", "plant_abundance": "30-50", "plant_avg_height": "1-4"},
    {"name": "Stepe", "tile": "Land", "temp": "0-15", "rains": "200-400", "humidity": "20-40", "height": "200-1200", "plant_abundance": "20-40", "plant_avg_height": "0.2-0.6"},
    {"name": "Sand Desert", "tile": "Land", "temp": "20-40", "rains": "0-250", "humidity": "5-20", "height": "0-800", "plant_abundance": "0-10", "plant_avg_height": "0-0.3"},
    {"name": "Badlands", "tile": "Land", "temp": "15-35", "rains": "0-200", "humidity": "5-15", "height": "200-1500", "plant_abundance": "0-5", "plant_avg_height": "0"},
    {"name": "Taiga", "tile": "Land", "temp": "-10 a 5", "rains": "300-700", "humidity": "50-70", "height": "100-1000", "plant_abundance": "50-75", "plant_avg_height": "10-25"},
    {"name": "Tundra", "tile": "Land", "temp": "-10 a 0", "rains": "150-350", "humidity": "50-70", "height": "0-300", "plant_abundance": "10-25", "plant_avg_height": "0-0.15"},
    {"name": "Snowy plains", "tile": "Land", "temp": "-25 a -5", "rains": "100-400", "humidity": "40-60", "height": "0-500", "plant_abundance": "5-15", "plant_avg_height": "0-0.2"},
    {"name": "Ice sheet / Glacier", "tile": "Land", "temp": "< -20", "rains": "0-200 (nieve)", "humidity": "30-50", "height": "any", "plant_abundance": "0", "plant_avg_height": "0"},
    {"name": "Mountain", "tile": "Land", "temp": "-15 a 10", "rains": "300-1200", "humidity": "30-70", "height": "1500-4000", "plant_abundance": "5-30", "plant_avg_height": "0-8"},
    {"name": "Alpine tundra", "tile": "Land", "temp": "-5 a 5", "rains": "400-900", "humidity": "40-60", "height": "2500-4000", "plant_abundance": "10-25", "plant_avg_height": "0-0.3"},
]


class Pipeline:
    def __init__(self, cfg: AppConfig, events: queue.Queue | None = None,
                 cancel: threading.Event | None = None):
        self.cfg = cfg
        self.ev = events if events is not None else queue.Queue()
        self.cancel = cancel if cancel is not None else threading.Event()

    def _emit(self, kind, *data):
        self.ev.put((kind, *data))

    def log(self, msg):
        self._emit("log", msg)

    def stage(self, name, frac):
        self._emit("progress", name, float(frac))

    def image(self, key, rgb):
        from scipy.ndimage import zoom
        pw, ph = self.cfg.performance.preview_w, self.cfg.performance.preview_h
        z = zoom(rgb.astype(np.float32), (ph / rgb.shape[0], pw / rgb.shape[1], 1), order=1)
        out = np.zeros((ph, pw, 3), np.uint8)
        h, w = min(ph, z.shape[0]), min(pw, z.shape[1])
        out[:h, :w] = np.clip(z[:h, :w], 0, 255).astype(np.uint8)
        self._emit("image", key, out)

    def run(self):
        try:
            self._run()
        except Exception:
            self._emit("error", traceback.format_exc())

    def _run(self):
        cfg = self.cfg
        if cfg.performance.use_numba:
            try:
                from numba import set_num_threads
                set_num_threads(cfg.performance.threads)
            except Exception:
                pass
        rng = np.random.default_rng(cfg.world.seed)
        perlin = Perlin(cfg.world.seed)
        lat, lon, xyz = pixel_grid(cfg.world.width, cfg.world.height)
        self.log(f"Mundo {cfg.world.width}x{cfg.world.height} px — seed {cfg.world.seed}")

        self.stage("Placas", 0.02)
        plates = generate_plates(cfg, rng, xyz, self.log)
        self.image("placas", render.plates_rgb(plates))

        years = [0]
        if cfg.dynamics.enabled:
            years = list(range(0, cfg.dynamics.total_years + 1, cfg.dynamics.step_years))
        self.log(f"Eras dinámicas: {len(years)} mapas (cada {cfg.dynamics.step_years} años)"
                 if cfg.dynamics.enabled else "Mundo estático (dinámica desactivada)")

        phases, last = [], {}
        for ei, year in enumerate(years):
            if self.cancel.is_set():
                self.log("Generación cancelada por el usuario")
                return
            base_frac = ei / max(1, len(years))
            blobs = []
            if ei > 0:
                self.stage(f"Deriva continental (año {year})", base_frac)
                blobs = advance_era(cfg, plates, rng, xyz, self.log)

            self.stage(f"Terreno (año {year})", base_frac + 0.05)
            height = generate_heightmap(cfg, plates, perlin, xyz, self.log)
            if blobs:
                apply_blobs(height, blobs)
            self.image("altura", render.height_rgb(height, plates.btype))

            self.stage(f"Clima (año {year})", base_frac + 0.10)
            T, P, Hm = climate(cfg, height, perlin, xyz, lat, lon)
            self.image("temperatura", render.temp_rgb(T))
            self.image("precipitacion", render.precip_rgb(P))

            self.stage(f"Ríos (año {year})", base_frac + 0.15)
            river_ids, width_map = generate_rivers(cfg, height, P, rng, self.log)
            self.image("rios", render.rivers_rgb(height, river_ids, width_map))

            self.stage(f"Tiles (año {year})", base_frac + 0.20)
            tm = build_tiles(cfg, height, river_ids, T, P, Hm, self.log)
            self.image("tiles", render.tiles_rgb(tm, upscale_to=(tm.ttype.shape[0] * cfg.tiles.tile_size,
                                                                 tm.ttype.shape[1] * cfg.tiles.tile_size)))
            phases.append(export_phase(year, tm, cfg.tiles.tile_size))
            last = dict(height=height, river_ids=river_ids, tm=tm)

        if last:
            self.stage("Visor isométrico 3D", 0.92)
            pw, ph = cfg.performance.preview_w, cfg.performance.preview_h
            self.image("iso_original", render.isometric(
                last["height"], render.height_rgb(last["height"]), pw, ph,
                river_mask=last["river_ids"] > 0))
            self.image("iso_tiles", render.isometric(
                last["tm"].height, render.tiles_rgb(last["tm"]), pw, ph))

        self.stage("Exportando JSON", 0.97)
        path = self._export(phases)
        self._emit("done", path)

    def _export(self, phases):
        cfg = self.cfg
        doc = {"meta": {"generator": "stoneplace-worldgen 0.1",
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "planet": cfg.planet,
                        "config": cfg.model_dump(mode="json")},
               "phases": phases}
        if cfg.export.include_biome_reference:
            doc["biome_reference"] = BIOME_REFERENCE
        d = os.path.dirname(cfg.export.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(cfg.export.path, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
        size_mb = os.path.getsize(cfg.export.path) / 1e6
        self.log(f"Exportado {cfg.export.path} ({size_mb:.1f} MB, {len(phases)} mapas)")
        return cfg.export.path