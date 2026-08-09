import streamlit as st
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worldgen import WorldConfig, WorldGenerator
from worldgen import visualize as vz
from worldgen.world import MATERIAL_NAMES

st.set_page_config(
    page_title="WorldGen - Procedural World Generator",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 WorldGen: Procedural World Generator")
st.markdown(
    "Configure parameters, click **Generate**, and explore layers - now with "
    "spherical wrap, tilted ITCZ, wind-driven precipitation, plate drift, "
    "connected rivers and an isometric 3D view."
)

if "world" not in st.session_state:
    st.session_state.world = None
if "cfg" not in st.session_state:
    st.session_state.cfg = WorldConfig()

st.sidebar.header("🔧 Configuration Parameters")

with st.sidebar.expander("📊 Resolution", expanded=True):
    sim_width = st.number_input("Simulation Width", 16, 2048, int(st.session_state.cfg.sim_width), 16)
    sim_height = st.number_input("Simulation Height", 16, 2048, int(st.session_state.cfg.sim_height), 16)
    tile_width = st.number_input("Tile Width", 8, 1024, int(st.session_state.cfg.tile_width), 8)
    tile_height = st.number_input("Tile Height", 8, 1024, int(st.session_state.cfg.tile_height), 8)

with st.sidebar.expander("🍽️ Plates"):
    plate_points_x = st.number_input("Plate Points X", 2, 100, int(st.session_state.cfg.plate_points_x))
    plate_points_y = st.number_input("Plate Points Y", 2, 100, int(st.session_state.cfg.plate_points_y))
    point_jitter = st.slider("Point Jitter", 0.0, 1.0, float(st.session_state.cfg.point_jitter), 0.05)
    major_plate_count = st.number_input("Major Plate Count", 1, 50, int(st.session_state.cfg.major_plate_count))
    small_plate_count = st.number_input("Small Plate Count", 0, 50, int(st.session_state.cfg.small_plate_count))
    oceanic_ratio = st.slider("Oceanic Ratio", 0.0, 1.0, float(st.session_state.cfg.oceanic_ratio), 0.05)
    oceanic_height_offset = st.slider("Oceanic Height Offset", -1.0, 1.0, float(st.session_state.cfg.oceanic_height_offset), 0.05)

with st.sidebar.expander("⛰️ Heightmap"):
    perlin_octaves = st.number_input("Perlin Octaves", 1, 10, int(st.session_state.cfg.perlin_octaves))
    perlin_base_scale = st.number_input("Perlin Base Scale", 0.1, 50.0, float(st.session_state.cfg.perlin_base_scale), 0.5)
    perlin_persistence = st.slider("Perlin Persistence", 0.0, 1.0, float(st.session_state.cfg.perlin_persistence), 0.05)
    perlin_lacunarity = st.number_input("Perlin Lacunarity", 0.1, 10.0, float(st.session_state.cfg.perlin_lacunarity), 0.1)
    perlin_max_height = st.slider("Perlin Max Height", 0.0, 2.0, float(st.session_state.cfg.perlin_max_height), 0.05)
    edge_smooth_radius = st.number_input("Edge Smooth Radius", 0, 50, int(st.session_state.cfg.edge_smooth_radius))

with st.sidebar.expander("🏔️ Mountains"):
    mountain_influence_radius = st.number_input("Mountain Influence Radius", 1, 50, int(st.session_state.cfg.mountain_influence_radius))
    mountain_octaves = st.number_input("Mountain Octaves", 1, 10, int(st.session_state.cfg.mountain_octaves))
    mountain_ridge_layers = st.number_input("Mountain Ridge Layers", 1, 10, int(st.session_state.cfg.mountain_ridge_layers))
    mountain_ridge_shift_min = st.slider("Ridge Shift Min", 0.0, 1.0, float(st.session_state.cfg.mountain_ridge_shift_min), 0.05)
    mountain_ridge_shift_max = st.slider("Ridge Shift Max", 0.0, 1.0, float(st.session_state.cfg.mountain_ridge_shift_max), 0.05)
    mountain_blend = st.slider("Mountain Blend", 0.0, 1.0, float(st.session_state.cfg.mountain_blend), 0.05)

with st.sidebar.expander("🌡️ Climate"):
    axial_tilt = st.slider("Axial Tilt (0=none, 1=90°)", 0.0, 1.0, float(st.session_state.cfg.axial_tilt), 0.01)
    climate_noise = st.slider("Climate Noise", 0.0, 1.0, float(st.session_state.cfg.climate_noise), 0.01)
    temp_height_factor = st.slider("Temp Height Factor", 0.0, 2.0, float(st.session_state.cfg.temp_height_factor), 0.05)
    precip_noise = st.slider("Precipitation Noise", 0.0, 1.0, float(st.session_state.cfg.precip_noise), 0.01)
    arid_dropoff = st.slider("Arid Dropoff", 0.1, 5.0, float(st.session_state.cfg.arid_dropoff), 0.1)
    equator_band_width = st.slider("Equator Band Width", 0.05, 0.5, float(st.session_state.cfg.equator_band_width), 0.01)
    trade_wind_strength = st.slider("Trade Wind Strength", 0.0, 2.0, float(st.session_state.cfg.trade_wind_strength), 0.05)
    orographic_strength = st.slider("Orographic Strength", 0.0, 2.0, float(st.session_state.cfg.orographic_strength), 0.05)

with st.sidebar.expander("🪵 Continental Drift"):
    drift_steps = st.number_input("Drift Steps (dynamic phases)", 0, 50, int(st.session_state.cfg.drift_steps))
    drift_chance = st.slider("Drift Chance", 0.0, 1.0, float(st.session_state.cfg.drift_chance), 0.05)
    drift_radius = st.number_input("Drift Radius", 1, 50, int(st.session_state.cfg.drift_radius))
    drift_neighbor_chance = st.slider("Drift Neighbor Chance", 0.0, 1.0, float(st.session_state.cfg.drift_neighbor_chance), 0.05)
    drift_plate_shift = st.slider("Plate Shift/step", 0.0, 2.0, float(st.session_state.cfg.drift_plate_shift), 0.05)

with st.sidebar.expander("🌊 Rivers"):
    river_mouth_count = st.number_input("River Mouth Count", 0, 100, int(st.session_state.cfg.river_mouth_count))
    river_mouth_min_ocean_radius = st.number_input("Min Ocean Radius", 1, 50, int(st.session_state.cfg.river_mouth_min_ocean_radius))
    river_source_count = st.number_input("River Source Count", 0, 100, int(st.session_state.cfg.river_source_count))
    river_step_radius = st.number_input("River Step Radius", 1, 50, int(st.session_state.cfg.river_step_radius))
    river_min_radius = st.number_input("River Min Radius", 1, 50, int(st.session_state.cfg.river_min_radius))
    river_split_chance = st.slider("River Split Chance", 0.0, 1.0, float(st.session_state.cfg.river_split_chance), 0.01)
    river_delta_split_chance = st.slider("River Delta Split Chance", 0.0, 1.0, float(st.session_state.cfg.river_delta_split_chance), 0.05)
    river_base_width = st.slider("River Base Width", 0.1, 5.0, float(st.session_state.cfg.river_base_width), 0.1)

with st.sidebar.expander("🔗 Unification"):
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, float(st.session_state.cfg.similarity_threshold), 0.01)

with st.sidebar.expander("🌐 World Shape"):
    spherical_wrap = st.checkbox("Spherical wrap (horizontal edges continuous)",
                                 value=bool(st.session_state.cfg.spherical_wrap))

with st.sidebar.expander("🎲 Random", expanded=True):
    seed = st.number_input("Seed", 1, 9999999, int(st.session_state.cfg.seed))
    if st.button("🔄 Randomize Seed"):
        seed = int(np.random.randint(1, 1000000))
        st.session_state.cfg.seed = seed
        st.rerun()

new_cfg = WorldConfig(
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
    drift_steps=drift_steps, drift_chance=drift_chance,
    drift_radius=drift_radius, drift_neighbor_chance=drift_neighbor_chance,
    drift_plate_shift=drift_plate_shift,
    river_mouth_count=river_mouth_count,
    river_mouth_min_ocean_radius=river_mouth_min_ocean_radius,
    river_source_count=river_source_count,
    river_step_radius=river_step_radius,
    river_min_radius=river_min_radius,
    river_split_chance=river_split_chance,
    river_delta_split_chance=river_delta_split_chance,
    river_base_width=river_base_width,
    similarity_threshold=similarity_threshold,
    spherical_wrap=spherical_wrap,
    seed=seed,
)
st.session_state.cfg = new_cfg

if st.sidebar.button("🚀 Generate World", use_container_width=True):
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def cb(name, frac):
        progress_bar.progress(min(1.0, frac))
        status_text.text(f"Generating: {name} ({frac*100:.0f}%)")

    with st.spinner("Simulating plates, drift, climate, and rivers..."):
        try:
            gen = WorldGenerator(new_cfg, progress_cb=cb)
            world = gen.generate()
            st.session_state.world = world
            progress_bar.progress(1.0)
            status_text.text("World generation completed successfully!")
            st.toast("Success! World generated.", icon="🌍")
        except Exception as e:
            st.error(f"Error during generation: {e}")
            import traceback
            st.code(traceback.format_exc())


if st.session_state.world is not None:
    w = st.session_state.world

    def _rivers_overlay(world):
        base = vz.color_height(world.height)
        r = world.river_map > 0
        base[r] = [0.1, 0.35, 0.9]
        return base

    def _groups_view(world):
        th, tw = world.cfg.tile_height, world.cfg.tile_width
        img = np.zeros((th, tw, 3), dtype=np.float32)
        rng = np.random.default_rng(0)
        for g in world.tile_groups:
            c = rng.random(3)
            for (x, y) in g["member_tiles"]:
                img[y, x] = c
        return img

    realistic = vz.color_final(w.height, w.temperature, w.precipitation, w.river_map)
    show = {
        "Plates": vz.color_plates(w.plate_map, w.plates),
        "Boundaries": vz.color_boundaries(w.boundary_type),
        "Height": vz.color_height(w.height),
        "Temperature": vz.color_temperature(w.temperature),
        "Precipitation": vz.color_precip(w.precipitation),
        "Winds": vz.color_winds(w.cfg),
        "Rivers": _rivers_overlay(w),
        "Realistic": realistic,
        "Isometric 3D (world)": vz.color_isometric(w.height, realistic),
        "Tilemap": vz.color_tilemap(w.tilemap),
        "Isometric 3D (tilemap)": vz.color_isometric(w.tilemap["height"], vz.color_tilemap(w.tilemap)),
        "Unified Groups": _groups_view(w),
    }

    st.subheader("🗺️ Layer Visualizations")
    tabs = st.tabs(list(show.keys()))
    for tab, (key, img) in zip(tabs, show.items()):
        with tab:
            st.markdown(f"**Layer:** {key} | **Resolution:** {img.shape[1]}x{img.shape[0]}")
            st.image(img, use_container_width=True)

    # Drift phases
    if w.drift_phases:
        st.subheader("🪵 Continental Drift - Dynamic Phases")
        phase_tabs = st.tabs([f"Phase {i+1}" for i in range(len(w.drift_phases))])
        for i, (pt, (pm_p, h_p)) in enumerate(zip(phase_tabs, w.drift_phases)):
            with pt:
                c1, c2 = st.columns(2)
                c1.image(vz.color_plates(pm_p, w.plates), caption=f"Plates - Phase {i+1}", use_container_width=True)
                c2.image(vz.color_height(h_p), caption=f"Height - Phase {i+1}", use_container_width=True)

    st.subheader("💾 Export World Data")
    col1, col2 = st.columns(2)
    with col1:
        data = {
            "config": w.cfg.to_dict(),
            "tile_resolution": [w.cfg.tile_width, w.cfg.tile_height],
            "plates": [
                {"id": p.id, "center": list(p.center), "oceanic": bool(p.is_oceanic),
                 "move_dir": list(p.move_dir), "base_height": float(p.base_height)}
                for p in w.plates
            ],
            "tile_groups": [{**g, "material_name": MATERIAL_NAMES.get(g["material"], "unknown")}
                            for g in w.tile_groups],
            "drift_phases_count": len(w.drift_phases),
            "stats": {
                "sim_resolution": [w.cfg.sim_width, w.cfg.sim_height],
                "total_groups": len(w.tile_groups),
                "total_tiles": w.cfg.tile_width * w.cfg.tile_height,
                "compression_ratio": (w.cfg.tile_width * w.cfg.tile_height) /
                                     max(1, len(w.tile_groups)),
            },
        }
        st.download_button("📥 Download World JSON",
                           data=json.dumps(data, indent=2),
                           file_name=f"world_seed_{w.cfg.seed}.json",
                           mime="application/json", use_container_width=True)
    with col2:
        st.download_button("📥 Download Config JSON",
                           data=json.dumps(w.cfg.to_dict(), indent=2),
                           file_name=f"world_config_seed_{w.cfg.seed}.json",
                           mime="application/json", use_container_width=True)

    st.subheader("📊 Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tile Resolution", f"{w.cfg.tile_width}x{w.cfg.tile_height}")
    c2.metric("Sim Resolution", f"{w.cfg.sim_width}x{w.cfg.sim_height}")
    c3.metric("Total Tile Groups", f"{len(w.tile_groups)}")
    c4.metric("Compression Ratio", f"{(w.cfg.tile_width*w.cfg.tile_height)/max(1,len(w.tile_groups)):.2f}x")
else:
    st.info("💡 Adjust parameters in the sidebar and click **Generate World** to begin!")
