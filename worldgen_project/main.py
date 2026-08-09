"""WorldGen — Procedural World Generator (single entry point).

Run with:  streamlit run main.py

Everything lives here: configuration UI, live generation progress, layer
visualizations (with a correct isometric 3D view), a timeline slider to
scrub through the plate-tectonics evolution steps (each simulated on the
sphere in 3D and then projected to the flat map), and JSON export.

The world is generated on a sphere first and projected to an equirectangular
map, so the horizontal edges are continuous by construction.
"""
import os
import sys
import json

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldgen import WorldConfig, WorldGenerator
from worldgen import visualize as vz
from worldgen.world import MATERIAL_NAMES


st.set_page_config(page_title="WorldGen", page_icon="🌍", layout="wide")


# --------------------------------------------------------------------------
# Sidebar: configuration
# --------------------------------------------------------------------------
def build_config() -> WorldConfig:
    cfg0 = st.session_state.get("cfg", WorldConfig())
    st.sidebar.header("🔧 Configuration")

    with st.sidebar.expander("📊 Resolution", expanded=True):
        sim_width = st.number_input("Simulation width (longitude)", 32, 2048, cfg0.sim_width, 16)
        sim_height = st.number_input("Simulation height (latitude)", 16, 1024, cfg0.sim_height, 16)
        tile_width = st.number_input("Tile width", 8, 1024, cfg0.tile_width, 8)
        tile_height = st.number_input("Tile height", 8, 1024, cfg0.tile_height, 8)

    with st.sidebar.expander("🌐 Sphere & Projection"):
        st.caption("The world is generated on a sphere (3D) and projected to a "
                   "flat equirectangular map — horizontal edges are continuous.")
        axial_tilt = st.slider("Axial tilt (fraction of 90°)", 0.0, 1.0, cfg0.axial_tilt, 0.01)

    with st.sidebar.expander("🍽️ Plates"):
        plate_points_x = st.number_input("Plate points X", 2, 100, cfg0.plate_points_x)
        plate_points_y = st.number_input("Plate points Y", 2, 100, cfg0.plate_points_y)
        point_jitter = st.slider("Point jitter", 0.0, 1.0, cfg0.point_jitter, 0.05)
        major_plate_count = st.number_input("Major plate count", 1, 50, cfg0.major_plate_count)
        small_plate_count = st.number_input("Small plate count", 0, 50, cfg0.small_plate_count)
        oceanic_ratio = st.slider("Oceanic ratio", 0.0, 1.0, cfg0.oceanic_ratio, 0.05)
        oceanic_height_offset = st.slider("Oceanic height offset", -1.0, 1.0,
                                          cfg0.oceanic_height_offset, 0.05)

    with st.sidebar.expander("⛰️ Heightmap"):
        perlin_octaves = st.number_input("Perlin octaves", 1, 10, cfg0.perlin_octaves)
        perlin_base_scale = st.number_input("Perlin base scale", 0.1, 50.0,
                                            cfg0.perlin_base_scale, 0.5)
        perlin_persistence = st.slider("Perlin persistence", 0.0, 1.0,
                                       cfg0.perlin_persistence, 0.05)
        perlin_lacunarity = st.number_input("Perlin lacunarity", 0.1, 10.0,
                                            cfg0.perlin_lacunarity, 0.1)
        perlin_max_height = st.slider("Perlin max height (mountain cap)", 0.0, 2.0,
                                      cfg0.perlin_max_height, 0.05)
        edge_smooth_radius = st.number_input("Continent edge smooth radius", 0, 50,
                                             cfg0.edge_smooth_radius)

    with st.sidebar.expander("🏔️ Mountains"):
        mountain_influence_radius = st.number_input("Influence radius", 1, 50,
                                                    cfg0.mountain_influence_radius)
        mountain_octaves = st.number_input("Mountain octaves", 1, 10, cfg0.mountain_octaves)
        mountain_ridge_layers = st.number_input("Ridge layers", 1, 10,
                                                cfg0.mountain_ridge_layers)
        mountain_ridge_shift_min = st.slider("Ridge shift min", 0.0, 1.0,
                                             cfg0.mountain_ridge_shift_min, 0.05)
        mountain_ridge_shift_max = st.slider("Ridge shift max", 0.0, 1.0,
                                             cfg0.mountain_ridge_shift_max, 0.05)
        mountain_blend = st.slider("Perlin/ridge blend", 0.0, 1.0, cfg0.mountain_blend, 0.05)

    with st.sidebar.expander("🌡️ Climate"):
        climate_noise = st.slider("Climate noise", 0.0, 1.0, cfg0.climate_noise, 0.01)
        temp_height_factor = st.slider("Temperature height factor", 0.0, 2.0,
                                       cfg0.temp_height_factor, 0.05)
        precip_noise = st.slider("Precipitation noise", 0.0, 1.0, cfg0.precip_noise, 0.01)
        arid_dropoff = st.slider("Arid dropoff", 0.1, 5.0, cfg0.arid_dropoff, 0.1)
        equator_band_width = st.slider("Equator band width", 0.05, 0.5,
                                       cfg0.equator_band_width, 0.01)
        trade_wind_strength = st.slider("Trade wind strength", 0.0, 2.0,
                                        cfg0.trade_wind_strength, 0.05)
        orographic_strength = st.slider("Orographic strength", 0.0, 2.0,
                                        cfg0.orographic_strength, 0.05)

    with st.sidebar.expander("🪨 Continental drift (dynamic world)"):
        drift_steps = st.number_input("Drift steps (timeline phases)", 0, 50, cfg0.drift_steps)
        drift_plate_shift = st.slider("Plate shift per step", 0.0, 2.0,
                                      cfg0.drift_plate_shift, 0.05)
        keep_drift_phases = st.checkbox("Keep every phase (timeline slider)",
                                        value=cfg0.keep_drift_phases)

    with st.sidebar.expander("🌊 Rivers"):
        river_mouth_count = st.number_input("River mouths", 0, 100, cfg0.river_mouth_count)
        river_mouth_min_ocean_radius = st.number_input("Min ocean radius", 1, 50,
                                                       cfg0.river_mouth_min_ocean_radius)
        river_source_count = st.number_input("River sources", 0, 100, cfg0.river_source_count)
        river_merge_radius = st.number_input("Merge radius", 1, 10, cfg0.river_merge_radius)
        river_split_chance = st.slider("Split chance", 0.0, 1.0, cfg0.river_split_chance, 0.01)
        river_delta_split_chance = st.slider("Delta split chance", 0.0, 1.0,
                                             cfg0.river_delta_split_chance, 0.05)
        river_base_width = st.slider("Base width", 0.1, 5.0, cfg0.river_base_width, 0.1)

    with st.sidebar.expander("🔗 Tile unification"):
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0,
                                         cfg0.similarity_threshold, 0.01)

    with st.sidebar.expander("🧊 Isometric 3D view"):
        iso_height_scale = st.slider("Height exaggeration", 0.02, 0.5,
                                     cfg0.iso_height_scale, 0.01)
        iso_max_cells = st.slider("3D grid resolution", 48, 320, cfg0.iso_max_cells, 16)

    with st.sidebar.expander("🎲 Random", expanded=True):
        seed = st.number_input("Seed", 1, 9999999, cfg0.seed)
        if st.button("🔄 Randomize seed", use_container_width=True):
            seed = int(np.random.randint(1, 1000000))

    cfg = WorldConfig(
        sim_width=sim_width, sim_height=sim_height,
        tile_width=tile_width, tile_height=tile_height,
        plate_points_x=plate_points_x, plate_points_y=plate_points_y,
        point_jitter=point_jitter,
        major_plate_count=major_plate_count, small_plate_count=small_plate_count,
        oceanic_ratio=oceanic_ratio, oceanic_height_offset=oceanic_height_offset,
        perlin_octaves=perlin_octaves, perlin_base_scale=perlin_base_scale,
        perlin_persistence=perlin_persistence, perlin_lacunarity=perlin_lacunarity,
        perlin_max_height=perlin_max_height, edge_smooth_radius=edge_smooth_radius,
        mountain_influence_radius=mountain_influence_radius,
        mountain_octaves=mountain_octaves, mountain_ridge_layers=mountain_ridge_layers,
        mountain_ridge_shift_min=mountain_ridge_shift_min,
        mountain_ridge_shift_max=mountain_ridge_shift_max,
        mountain_blend=mountain_blend,
        axial_tilt=axial_tilt, climate_noise=climate_noise,
        temp_height_factor=temp_height_factor, precip_noise=precip_noise,
        arid_dropoff=arid_dropoff, equator_band_width=equator_band_width,
        trade_wind_strength=trade_wind_strength, orographic_strength=orographic_strength,
        drift_steps=drift_steps, drift_plate_shift=drift_plate_shift,
        keep_drift_phases=keep_drift_phases,
        river_mouth_count=river_mouth_count,
        river_mouth_min_ocean_radius=river_mouth_min_ocean_radius,
        river_source_count=river_source_count, river_merge_radius=river_merge_radius,
        river_split_chance=river_split_chance,
        river_delta_split_chance=river_delta_split_chance,
        river_base_width=river_base_width,
        similarity_threshold=similarity_threshold,
        iso_height_scale=iso_height_scale, iso_max_cells=iso_max_cells,
        seed=seed,
    )
    st.session_state.cfg = cfg
    return cfg


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def crisp(img, caption=None):
    """Display an image at full quality with nearest-neighbour upscaling so
    tilemaps stay crisp instead of blurry."""
    st.image(img, caption=caption, use_container_width=True, clamp=True)


def rivers_overlay(world):
    base = vz.color_height(world.height)
    base[world.river_map > 0] = [0.1, 0.35, 0.9]
    return base


def groups_view(world):
    th, tw = world.cfg.tile_height, world.cfg.tile_width
    img = np.zeros((th, tw, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    for g in world.tile_groups:
        c = rng.random(3)
        for (x, y) in g["member_tiles"]:
            img[y, x] = c
    return img


def iso(world, height, colored=None):
    cfg = world.cfg
    return vz.color_isometric(height, colored,
                              height_scale=cfg.iso_height_scale,
                              max_cells=cfg.iso_max_cells)


def render_layers(world):
    realistic = vz.color_final(world.height, world.temperature,
                               world.precipitation, world.river_map)
    show = {
        "Plates": vz.color_plates(world.plate_map, world.plates),
        "Boundaries": vz.color_boundaries(world.boundary_type),
        "Height": vz.color_height(world.height),
        "Temperature": vz.color_temperature(world.temperature),
        "Precipitation": vz.color_precip(world.precipitation),
        "Winds": vz.color_winds(world.cfg),
        "Rivers": rivers_overlay(world),
        "Realistic": realistic,
        "Isometric 3D (world)": iso(world, world.height, realistic),
        "Tilemap": vz.color_tilemap(world.tilemap),
        "Isometric 3D (tilemap)": iso(world, world.tilemap["height"],
                                      vz.color_tilemap(world.tilemap)),
        "Unified groups": groups_view(world),
    }
    st.subheader("🗺️ Layers")
    tabs = st.tabs(list(show.keys()))
    for tab, (key, img) in zip(tabs, show.items()):
        with tab:
            st.caption(f"{key} — {img.shape[1]}×{img.shape[0]}")
            crisp(img)


def render_evolution(world):
    """Timeline slider over the dynamic drift phases.  Phase 0 is the initial
    projection; every later phase was simulated on the sphere in 3D and then
    projected to the flat map."""
    phases = [("Initial", world.plate_map if not world.drift_phases else None, None)]
    # build the real list: initial state (from first snapshot is not stored, so
    # we show final state as "current") + each stored phase
    snaps = world.drift_phases
    if not snaps:
        st.info("Set 'Drift steps' > 0 to generate a dynamic timeline.")
        return

    st.subheader("⏳ World evolution — drift timeline")
    step = st.slider("Evolution step", 0, len(snaps) - 1, len(snaps) - 1,
                     help="Each step was simulated on the sphere (3D) and then "
                          "projected to the flat map.")
    pm_p, h_p = snaps[step]
    c1, c2, c3 = st.columns(3)
    with c1:
        crisp(vz.color_plates(pm_p, world.plates), f"Plates — step {step + 1}")
    with c2:
        crisp(vz.color_height(h_p), f"Height — step {step + 1}")
    with c3:
        crisp(iso(world, h_p), f"Isometric 3D — step {step + 1}")


def render_export(world):
    st.subheader("💾 Export")
    data = {
        "config": world.cfg.to_dict(),
        "projection": "equirectangular (generated on sphere, then projected)",
        "tile_resolution": [world.cfg.tile_width, world.cfg.tile_height],
        "plates": [
            {"id": p.id,
             "center_lonlat": [float(p.center[0]), float(p.center[1])],
             "oceanic": bool(p.is_oceanic),
             "move_dir": [float(p.move_dir[0]), float(p.move_dir[1])],
             "euler_pole": [float(a) for a in p.rot_axis],
             "rotation_rate": float(p.rot_rate),
             "base_height": float(p.base_height)}
            for p in world.plates
        ],
        "tile_groups": [{**g, "material_name": MATERIAL_NAMES.get(g["material"], "unknown")}
                        for g in world.tile_groups],
        "drift_phases_count": len(world.drift_phases),
        "stats": {
            "sim_resolution": [world.cfg.sim_width, world.cfg.sim_height],
            "total_groups": len(world.tile_groups),
            "total_tiles": world.cfg.tile_width * world.cfg.tile_height,
            "compression_ratio": (world.cfg.tile_width * world.cfg.tile_height) /
                                 max(1, len(world.tile_groups)),
        },
    }
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download world JSON", data=json.dumps(data, indent=2),
                           file_name=f"world_seed_{world.cfg.seed}.json",
                           mime="application/json", use_container_width=True)
    with c2:
        st.download_button("📥 Download config JSON",
                           data=json.dumps(world.cfg.to_dict(), indent=2),
                           file_name=f"world_config_seed_{world.cfg.seed}.json",
                           mime="application/json", use_container_width=True)


def render_stats(world):
    st.subheader("📊 Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tile resolution", f"{world.cfg.tile_width}×{world.cfg.tile_height}")
    c2.metric("Sim resolution", f"{world.cfg.sim_width}×{world.cfg.sim_height}")
    c3.metric("Tile groups", f"{len(world.tile_groups)}")
    c4.metric("Compression",
              f"{(world.cfg.tile_width * world.cfg.tile_height) / max(1, len(world.tile_groups)):.2f}x")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.title("🌍 WorldGen — Procedural World Generator")
    st.markdown("Worlds are generated **on a sphere (3D)** and then **projected "
                "to a flat equirectangular map** — the horizontal edges are "
                "continuous, and every drift step is available on the timeline slider.")

    cfg = build_config()

    if st.sidebar.button("🚀 Generate world", use_container_width=True, type="primary"):
        progress = st.progress(0.0)
        status = st.empty()

        def cb(name, frac):
            progress.progress(min(1.0, frac))
            status.text(f"{name}  ({frac * 100:.0f}%)")

        with st.spinner("Generating on the sphere and projecting…"):
            try:
                world = WorldGenerator(cfg, progress_cb=cb).generate()
                st.session_state.world = world
                progress.progress(1.0)
                status.text("Done.")
                st.toast("World generated", icon="🌍")
            except Exception as e:
                import traceback
                st.error(f"Generation error: {e}")
                st.code(traceback.format_exc())

    world = st.session_state.get("world")
    if world is None:
        st.info("Adjust parameters in the sidebar and click **Generate world**.")
        return

    render_layers(world)
    render_evolution(world)
    render_export(world)
    render_stats(world)


if __name__ == "__main__":
    main()
else:
    main()
