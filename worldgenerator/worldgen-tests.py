"""Tests mínimos (pytest). El smoke test genera un mundo chico de punta a punta."""
import numpy as np
import pytest

from noise import Perlin
from config import AppConfig


def test_perlin_bounds():
    p = Perlin(1)
    rng = np.random.default_rng(0)
    x = rng.random(5000) * 10
    v = p.fbm(x, x, x, octaves=4)
    assert v.min() >= 0.0 and v.max() <= 1.0
    assert 0.2 < v.mean() < 0.8


def test_ridge_has_sharp_peaks():
    p = Perlin(2)
    x = np.linspace(0, 8, 4000)
    r = p.ridge(x, x * 0.5, x * 0.25, freq=1.0, shift=0.5)
    assert r.min() >= 0.0 and r.max() <= 1.0 + 1e-6
    assert (r > 0.9).mean() < 0.2          # picos afilados: pocos puntos cerca del máximo


def test_tiles_unify_ocean_single_region():
    from tiles import build_tiles, OCEAN
    cfg = AppConfig()
    cfg.tiles.tile_size = 8
    h = np.full((64, 64), -100.0, np.float32)     # todo océano
    h[32:, 32:] = 50.0                            # un cuarto de tierra
    T = np.full_like(h, 15.0); P = np.full_like(h, 800.0); Hm = np.full_like(h, 60.0)
    tm = build_tiles(cfg, h, np.zeros_like(h, np.int32), T, P, Hm)
    ocean_regions = [i for i, c in enumerate(tm.reg_cls) if c == OCEAN]
    assert len(ocean_regions) == 1                # "un solo tile grande de océano"


@pytest.mark.slow
def test_smoke_pipeline(tmp_path):
    from pipeline import Pipeline
    cfg = AppConfig()
    cfg.world.width, cfg.world.height = 128, 64
    cfg.plates.dots_y, cfg.plates.dots_x = 6, 12
    cfg.plates.major_plates, cfg.plates.small_plates = 4, 2
    cfg.rivers.target = 6
    cfg.dynamics.enabled = False
    cfg.tiles.tile_size = 16
    cfg.performance.use_numba = False
    cfg.export.path = str(tmp_path / "world.json")
    import json, os
    Pipeline(cfg).run()
    assert os.path.exists(cfg.export.path)
    doc = json.load(open(cfg.export.path))
    assert len(doc["phases"]) == 1
    assert doc["phases"][0]["tiles_w"] == 8
    assert "biome_reference" in doc