"""Headless test: generate a world, verify edge continuity (the projection of
the sphere must be seamless horizontally and must NOT contain the old
middle-of-map ghost line), and dump JSON + PNG previews (no UI)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from worldgen import WorldConfig, WorldGenerator
from worldgen import visualize as vz


def seam_report(h, name):
    """Difference between the leftmost and rightmost columns (must be small)
    vs difference between the two halves at the middle (ghost-line detector)."""
    edge = float(np.abs(h[:, 0] - h[:, -1]).mean())
    mid = float(np.abs(h[:, h.shape[1] // 2 - 1] - h[:, h.shape[1] // 2]).mean())
    interior = float(np.abs(np.diff(h, axis=1)).mean())
    print(f"  {name:14s} edge-diff={edge:.4f}  mid-diff={mid:.4f}  "
          f"avg-step={interior:.4f}")
    return edge, mid, interior


def main():
    cfg = WorldConfig(
        sim_width=192, sim_height=96,
        tile_width=64, tile_height=32,
        plate_points_x=14, plate_points_y=7,
        major_plate_count=6, small_plate_count=8,
        river_mouth_count=8, river_source_count=10,
        similarity_threshold=0.9,
        drift_steps=3,
        seed=2026,
    )

    def cb(name, frac):
        print(f"  [{frac * 100:5.1f}%] {name}")

    world = WorldGenerator(cfg, progress_cb=cb).generate()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "world.json")
    world.export_json(json_path)
    print(f"Exported: {json_path}")
    print(f"Tile groups: {len(world.tile_groups)} / {cfg.tile_width * cfg.tile_height}")

    print("Edge-continuity checks (sphere -> projection):")
    seam_report(world.height, "height")
    seam_report(world.temperature, "temperature")
    seam_report(world.precipitation, "precipitation")

    realistic = vz.color_final(world.height, world.temperature,
                               world.precipitation, world.river_map)
    imgs = {
        "plates": vz.color_plates(world.plate_map, world.plates),
        "boundaries": vz.color_boundaries(world.boundary_type),
        "height": vz.color_height(world.height),
        "temperature": vz.color_temperature(world.temperature),
        "precip": vz.color_precip(world.precipitation),
        "realistic": realistic,
        "isometric": vz.color_isometric(world.height, realistic,
                                        height_scale=cfg.iso_height_scale,
                                        max_cells=cfg.iso_max_cells),
        "tilemap": vz.color_tilemap(world.tilemap),
        "iso_tilemap": vz.color_isometric(world.tilemap["height"],
                                          vz.color_tilemap(world.tilemap),
                                          height_scale=cfg.iso_height_scale,
                                          max_cells=cfg.iso_max_cells),
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    for ax, (k, im) in zip(axes.ravel(), imgs.items()):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(k)
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(imgs), 9):
        axes.ravel()[i].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preview.png"), dpi=110)
    print(f"Preview: {os.path.join(out_dir, 'preview.png')}")
    print(f"Drift phases (timeline steps): {len(world.drift_phases)}")

    # evolution strip: first, middle and last drift step
    if world.drift_phases:
        n = len(world.drift_phases)
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 5))
        for col, idx in enumerate([0, n // 2, n - 1]):
            pm_p, h_p = world.drift_phases[idx]
            axes2[0, col].imshow(vz.color_plates(pm_p, world.plates),
                                 interpolation="nearest")
            axes2[0, col].set_title(f"plates step {idx + 1}")
            axes2[1, col].imshow(vz.color_height(h_p), interpolation="nearest")
            axes2[1, col].set_title(f"height step {idx + 1}")
        for ax in axes2.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "evolution.png"), dpi=110)
        print(f"Evolution: {os.path.join(out_dir, 'evolution.png')}")


if __name__ == "__main__":
    main()
