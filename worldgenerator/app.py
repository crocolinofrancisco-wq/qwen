import queue
import threading
import time
import numpy as np
import streamlit as st

from config import AppConfig
from pipeline import Pipeline

st.set_page_config(page_title="Stoneplace WorldGen", layout="wide", page_icon="🌍")

st.title("🌍 Stoneplace WorldGen - Generador de Mundos")
st.markdown("Generador procedural de mundos esféricos con placas tectónicas, clima, ríos y tilemap.")

# Sidebar de configuración
st.sidebar.header("⚙️ Configuración")

seed = st.sidebar.number_input("Semilla (Seed)", value=2026, step=1)
width = st.sidebar.number_input("Ancho (px)", value=512, step=64)
height = st.sidebar.number_input("Alto (px)", value=256, step=32)

with st.sidebar.expander("Placas Tectónicas"):
    major_plates = st.number_input("Placas Mayores", value=12, min_value=2, max_value=50)
    small_plates = st.number_input("Micro-placas", value=8, min_value=0, max_value=30)
    oceanic_ratio = st.slider("Proporción Oceánica", 0.0, 1.0, 0.5)

with st.sidebar.expander("Dinámica Continental"):
    dynamics_enabled = st.checkbox("Habilitar deriva continental", value=True)
    step_years = st.number_input("Años por era", value=500000, step=100000)
    total_years = st.number_input("Años totales", value=2000000, step=500000)

with st.sidebar.expander("Tiles"):
    tile_size = st.number_input("Tamaño de tile (px)", value=16, min_value=4, max_value=64)
    similarity = st.slider("Similitud de unificación (%)", 0.0, 50.0, 10.0)

run_button = st.sidebar.button("▶ Generar mundo", type="primary", use_container_width=True)

# Layout principal
col_status, col_progress = st.columns([1, 3])
with col_status:
    stage_text = st.empty()
    stage_text.markdown("**Estado:** Esperando ejecución...")

with col_progress:
    progress_bar = st.progress(0.0)

log_expander = st.expander("📋 Logs de generación", expanded=False)
log_area = log_expander.empty()

# Pestañas para previews
tab_names = ["Placas", "Altura", "Temperatura", "Precipitación", "Ríos", "Tiles", "3D Isométrico"]
tabs = st.tabs(tab_names)
tab_dict = {name: tab.empty() for name, tab in zip(["placas", "altura", "temperatura", "precipitacion", "rios", "tiles", "iso_original"], tabs)}

if run_button:
    cfg = AppConfig()
    cfg.world.seed = int(seed)
    cfg.world.width = int(width)
    cfg.world.height = int(height)
    cfg.plates.major_plates = int(major_plates)
    cfg.plates.small_plates = int(small_plates)
    cfg.plates.oceanic_ratio = float(oceanic_ratio)
    cfg.dynamics.enabled = bool(dynamics_enabled)
    cfg.dynamics.step_years = int(step_years)
    cfg.dynamics.total_years = int(total_years)
    cfg.tiles.tile_size = int(tile_size)
    cfg.tiles.similarity_percent = float(similarity)
    cfg.performance.use_numba = False

    ev = queue.Queue()
    cancel_ev = threading.Event()
    pipe = Pipeline(cfg, ev, cancel_ev)

    thread = threading.Thread(target=pipe.run, daemon=True)
    thread.start()

    logs = []
    
    while thread.is_alive() or not ev.empty():
        try:
            kind, *data = ev.get(timeout=0.1)
            if kind == "log":
                logs.append(str(data[0]))
                log_area.code("\n".join(logs[-20:]), language="text")
            elif kind == "progress":
                stage_text.markdown(f"**Etapa:** {data[0]}")
                progress_bar.progress(min(1.0, max(0.0, float(data[1]))))
            elif kind == "image":
                img_key, rgb_data = data[0], data[1]
                if img_key in tab_dict:
                    tab_dict[img_key].image(rgb_data, use_container_width=True)
            elif kind == "done":
                stage_text.markdown("✅ **¡Generación completada!**")
                progress_bar.progress(1.0)
                json_path = data[0]
                with open(json_path, "rb") as f:
                    st.download_button("💾 Descargar JSON del Mundo", f, file_name="world.json", mime="application/json")
                break
            elif kind == "error":
                stage_text.markdown("❌ **Error en la generación**")
                st.error(data[0])
                break
        except queue.Empty:
            time.sleep(0.05)
