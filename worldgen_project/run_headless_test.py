"""Headless test: generates a small world and dumps JSON + PNG previews (no Tk)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from worldgen import WorldConfig, WorldGenerator
from worldgen import visualize as vz


def main():
    cfg = WorldConfig(
        sim_width=192, sim_height=96,
        tile_width=64, tile_height=32,
        plate_points_x=14, plate_points_y=7,
        major_plate_count=6, small_plate_count=8,
        river_mouth_count=8, river_source_count=10,
        similarity_threshold=0.9,
        drift_steps=2,
        seed=2026,
    )
    def cb(name, frac):
        print(f"  [{frac*100:5.1f}%] {name}")

    gen = WorldGenerator(cfg, progress_cb=cb)
    world = gen.generate()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "world.json")
    world.export_json(json_path)
    print(f"Exported: {json_path}")
    print(f"Tile groups: {len(world.tile_groups)} / {cfg.tile_width*cfg.tile_height}")

    realistic = vz.color_final(world.height, world.temperature, world.precipitation, world.river_map)
    imgs = {
        "plates": vz.color_plates(world.plate_map, world.plates),
        "boundaries": vz.color_boundaries(world.boundary_type),
        "height": vz.color_height(world.height),
        "temperature": vz.color_temperature(world.temperature),
        "precip": vz.color_precip(world.precipitation),
        "winds": vz.color_winds(world.cfg),
        "realistic": realistic,
        "isometric": vz.color_isometric(world.height, realistic),
        "tilemap": vz.color_tilemap(world.tilemap),
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    for ax, (k, im) in zip(axes.ravel(), imgs.items()):
        ax.imshow(im); ax.set_title(k); ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(imgs), 9):
        axes.ravel()[i].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preview.png"), dpi=110)
    print(f"Preview: {os.path.join(out_dir, 'preview.png')}")
    print(f"Drift phases: {len(world.drift_phases)}")


if __name__ == "__main__":
    main()
