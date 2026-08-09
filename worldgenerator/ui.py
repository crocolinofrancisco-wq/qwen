"""UI (Dear PyGui): personalización total de la generación + vista en vivo
del proceso (logs, progreso, previews por etapa) + visor isométrico 3D
del mapa original y del tilemap."""
from __future__ import annotations
import queue
import threading
import typing
import numpy as np
import dearpygui.dearpygui as dpg

from config import AppConfig
from pipeline import Pipeline

SECTIONS = [("Mundo", "world"), ("Placas", "plates"), ("Terreno / montañas", "terrain"),
            ("Clima", "climate"), ("Ríos", "rivers"), ("Dinámica (deriva)", "dynamics"),
            ("Tiles / unificación", "tiles"), ("Rendimiento", "performance"),
            ("Exportación", "export")]
TABS = [("placas", "Placas"), ("altura", "Altura"), ("temperatura", "Temperatura"),
        ("precipitacion", "Precipitación"), ("rios", "Ríos"), ("tiles", "Tiles"),
        ("iso_original", "3D · mapa original"), ("iso_tiles", "3D · tilemap")]


class WorldGenUI:
    def __init__(self):
        self.cfg = AppConfig()
        self.events = queue.Queue()
        self.cancel_ev = threading.Event()
        self.running = False
        self.widgets = {}          # (sección, campo) -> tag
        self.textures = {}         # key -> (tex_tag, w, h)
        self.log_lines = []

    # ---------------- construcción ----------------
    def build(self):
        with dpg.window(tag="main", label="Stoneplace WorldGen", no_close=True):
            with dpg.menu_bar():
                dpg.add_button(label="▶ Generar mundo", tag="runbtn", callback=self.on_run)
                dpg.add_button(label="■ Cancelar", callback=lambda: self.cancel_ev.set())
                dpg.add_text(" | config YAML: ")
                dpg.add_input_text(tag="yaml_path", default_value="worldgen.yaml", width=180)
                dpg.add_button(label="Cargar", callback=self.on_load)
                dpg.add_button(label="Guardar", callback=self.on_save)
            with dpg.group(horizontal=True):
                with dpg.child_window(width=400, tag="cfgpanel"):
                    self._build_config_form()
                with dpg.group():
                    with dpg.tab_bar(tag="tabs"):
                        for key, label in TABS:
                            with dpg.tab(label=label, tag=f"tab::{key}"):
                                pass
                    with dpg.group(horizontal=True):
                        dpg.add_text("Etapa: ", color=(140, 140, 140))
                        dpg.add_text("—", tag="pstage")
                    dpg.add_progress_bar(tag="pbar", default_value=0.0, width=-1)
                    dpg.add_text("Log", color=(140, 140, 140))
                    with dpg.child_window(tag="logwin", height=210):
                        pass

    def _build_config_form(self):
        for label, sec in SECTIONS:
            model = getattr(self.cfg, sec)
            with dpg.collapsing_header(label=label, default_open=(sec == "world")):
                for name, finfo in type(model).model_fields.items():
                    value = getattr(model, name)
                    if isinstance(value, dict):
                        continue
                    ann = finfo.annotation
                    tag = f"cfg::{sec}.{name}"
                    kw = dict(label=name, tag=tag, callback=self._on_change,
                              user_data=(sec, name))
                    if ann is bool:
                        dpg.add_checkbox(default_value=bool(value), **kw)
                    elif ann is int:
                        dpg.add_input_int(default_value=int(value), step=0, **kw)
                    elif ann is float:
                        dpg.add_input_float(default_value=float(value), format="%.6g", **kw)
                    elif typing.get_origin(ann) is tuple:
                        dpg.add_input_floatx(default_value=list(value), size=len(value), **kw)
                    else:
                        dpg.add_input_text(default_value=str(value), **kw)
                    self.widgets[(sec, name)] = tag

    # ---------------- callbacks ----------------
    def _on_change(self, sender, app_data, user_data):
        sec, name = user_data
        model = getattr(self.cfg, sec)
        cur = getattr(model, name)
        if isinstance(cur, bool):
            val = bool(app_data)
        elif isinstance(cur, int):
            val = int(app_data)
        elif isinstance(cur, float):
            val = float(app_data)
        elif isinstance(cur, tuple):
            val = tuple(float(v) for v in app_data)
        else:
            val = str(app_data)
        setattr(model, name, val)

    def on_run(self):
        if self.running:
            return
        self.running = True
        self.cancel_ev.clear()
        dpg.configure_item("runbtn", enabled=False, label="Generando…")
        self._log("─" * 60)
        pipe = Pipeline(self.cfg, self.events, self.cancel_ev)
        threading.Thread(target=pipe.run, daemon=True).start()

    def on_load(self):
        path = dpg.get_value("yaml_path")
        try:
            self.cfg = AppConfig.load(path)
            for (sec, name), tag in self.widgets.items():
                v = getattr(getattr(self.cfg, sec), name)
                dpg.set_value(tag, list(v) if isinstance(v, tuple) else v)
            self._log(f"Config cargada de {path}")
        except Exception as e:
            self._log(f"Error cargando {path}: {e}")

    def on_save(self):
        path = dpg.get_value("yaml_path")
        self.cfg.save(path)
        self._log(f"Config guardada en {path}")

    # ---------------- eventos del pipeline ----------------
    def poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "log":
                    self._log(ev[1])
                elif kind == "progress":
                    dpg.set_value("pstage", ev[1])
                    dpg.set_value("pbar", min(1.0, ev[2]))
                elif kind == "image":
                    self._set_image(ev[1], ev[2])
                elif kind == "error":
                    self._log("ERROR:\n" + ev[1])
                    self._finish()
                elif kind == "done":
                    self._log(f"✔ Listo: {ev[1]}")
                    dpg.set_value("pbar", 1.0)
                    dpg.set_value("pstage", "finalizado")
                    self._finish()
        except queue.Empty:
            pass

    def _finish(self):
        self.running = False
        dpg.configure_item("runbtn", enabled=True, label="▶ Generar mundo")

    def _log(self, msg):
        for line in str(msg).splitlines() or [""]:
            tag = f"log{len(self.log_lines)}"
            dpg.add_text(line, parent="logwin", tag=tag, wrap=0)
            self.log_lines.append(tag)
        if len(self.log_lines) > 600:
            for tag in self.log_lines[:-600]:
                dpg.delete_item(tag)
            self.log_lines = self.log_lines[-600:]
        dpg.set_y_scroll("logwin", 10 ** 9)

    def _set_image(self, key, rgb):
        h, w = rgb.shape[:2]
        rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=-1)
        vals = (rgba.astype(np.float32) / 255.0).ravel().tolist()
        if key not in self.textures or self.textures[key][1:] != (w, h):
            if key in self.textures:
                dpg.delete_item(self.textures[key][0])
                dpg.delete_item(f"img::{key}")
            if not dpg.does_item_exist("texreg"):
                with dpg.texture_registry(tag="texreg"):
                    pass
            tex = dpg.add_raw_texture(w, h, vals, format=dpg.mvFormat_Float_rgba,
                                      parent="texreg", tag=f"tex::{key}")
            dpg.add_image(tex, parent=f"tab::{key}", tag=f"img::{key}")
            self.textures[key] = (tex, w, h)
        else:
            dpg.set_value(self.textures[key][0], vals)

    # ---------------- loop ----------------
    def start(self):
        dpg.create_context()
        self.build()
        dpg.create_viewport(title="Stoneplace WorldGen", width=1560, height=940)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        while dpg.is_dearpygui_running():
            self.poll()
            dpg.render_dearpygui_frame()
        self.cancel_ev.set()
        dpg.destroy_context()


def main():
    WorldGenUI().start()