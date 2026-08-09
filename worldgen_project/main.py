"""Interactive UI for World Generation.

Left panel: configuration parameters.
Right panel: live process visualization with layer tabs.
"""
import os
import sys
import threading
import queue
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worldgen import WorldConfig, WorldGenerator
from worldgen import visualize as vz


PARAM_GROUPS = [
    ("Resolution", [
        ("sim_width", int), ("sim_height", int),
        ("tile_width", int), ("tile_height", int),
    ]),
    ("Plates", [
        ("plate_points_x", int), ("plate_points_y", int),
        ("point_jitter", float), ("major_plate_count", int),
        ("small_plate_count", int), ("oceanic_ratio", float),
        ("oceanic_height_offset", float),
    ]),
    ("Heightmap", [
        ("perlin_octaves", int), ("perlin_base_scale", float),
        ("perlin_persistence", float), ("perlin_lacunarity", float),
        ("perlin_max_height", float), ("edge_smooth_radius", int),
    ]),
    ("Mountains", [
        ("mountain_influence_radius", int), ("mountain_octaves", int),
        ("mountain_ridge_layers", int),
        ("mountain_ridge_shift_min", float), ("mountain_ridge_shift_max", float),
        ("mountain_blend", float),
    ]),
    ("Climate", [
        ("axial_tilt", float), ("climate_noise", float),
        ("temp_height_factor", float),
        ("precip_noise", float), ("arid_dropoff", float),
    ]),
    ("Drift", [
        ("drift_steps", int), ("drift_chance", float),
        ("drift_radius", int), ("drift_neighbor_chance", float),
    ]),
    ("Rivers", [
        ("river_mouth_count", int), ("river_mouth_min_ocean_radius", int),
        ("river_source_count", int), ("river_step_radius", int),
        ("river_min_radius", int), ("river_split_chance", float),
        ("river_delta_split_chance", float), ("river_base_width", float),
    ]),
    ("Unification", [
        ("similarity_threshold", float),
    ]),
    ("Random", [("seed", int)]),
]


LAYERS = [
    ("Plates", "plates"),
    ("Boundaries", "boundaries"),
    ("Height", "height"),
    ("Temperature", "temperature"),
    ("Precipitation", "precipitation"),
    ("Rivers", "rivers"),
    ("Realistic", "realistic"),
    ("Tilemap", "tilemap"),
    ("Unified Groups", "groups"),
]


class WorldGenApp:
    def __init__(self, root):
        self.root = root
        root.title("WorldGen - Procedural World Generator")
        root.geometry("1400x820")

        self.cfg = WorldConfig()
        self.world = None
        self.msg_queue = queue.Queue()
        self.worker = None
        self._build_ui()
        self._draw_placeholder()
        self.root.after(100, self._poll)

    # ---------------- UI ----------------
    def _build_ui(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: params
        left = ttk.Frame(main, width=420)
        main.add(left, weight=0)

        canvas = tk.Canvas(left, width=400)
        vsb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self.vars = {}
        for group_name, params in PARAM_GROUPS:
            lf = ttk.LabelFrame(inner, text=group_name)
            lf.pack(fill="x", padx=6, pady=4)
            for pname, ptype in params:
                row = ttk.Frame(lf); row.pack(fill="x", padx=4, pady=1)
                ttk.Label(row, text=pname, width=28).pack(side="left")
                v = tk.StringVar(value=str(getattr(self.cfg, pname)))
                self.vars[pname] = (v, ptype)
                ttk.Entry(row, textvariable=v, width=12).pack(side="left")

        # Buttons
        btns = ttk.Frame(inner); btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Generate", command=self.on_generate).pack(side="left", padx=4)
        ttk.Button(btns, text="Randomize Seed", command=self.on_randomize).pack(side="left", padx=4)
        ttk.Button(btns, text="Export JSON", command=self.on_export).pack(side="left", padx=4)
        ttk.Button(btns, text="Save Config", command=self.on_save_cfg).pack(side="left", padx=4)
        ttk.Button(btns, text="Load Config", command=self.on_load_cfg).pack(side="left", padx=4)

        # Right: visualization
        right = ttk.Frame(main)
        main.add(right, weight=1)

        self.status_var = tk.StringVar(value="Idle")
        status = ttk.Frame(right); status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(status, mode="determinate", length=400)
        self.progress.pack(side="right", padx=6, pady=4)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        self.axes = {}
        self.canvases = {}
        for label, key in LAYERS:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=label)
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_xticks([]); ax.set_yticks([])
            can = FigureCanvasTkAgg(fig, master=frame)
            can.get_tk_widget().pack(fill="both", expand=True)
            self.axes[key] = (fig, ax)
            self.canvases[key] = can

    def _draw_placeholder(self):
        for k, (fig, ax) in self.axes.items():
            ax.clear()
            ax.text(0.5, 0.5, f"[{k}] - press Generate", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self.canvases[k].draw()

    # ---------------- callbacks ----------------
    def _collect_cfg(self):
        cfg = WorldConfig()
        for pname, (var, ptype) in self.vars.items():
            try:
                val = ptype(var.get())
                setattr(cfg, pname, val)
            except Exception as e:
                messagebox.showerror("Invalid value", f"{pname}: {e}")
                return None
        return cfg

    def on_randomize(self):
        v, _ = self.vars["seed"]
        v.set(str(np.random.randint(1, 1_000_000)))

    def on_generate(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Generation already in progress.")
            return
        cfg = self._collect_cfg()
        if cfg is None:
            return
        self.cfg = cfg
        self.progress["value"] = 0
        self.status_var.set("Starting...")
        self.worker = threading.Thread(target=self._run_generation, args=(cfg,), daemon=True)
        self.worker.start()

    def _run_generation(self, cfg):
        try:
            def cb(name, frac):
                self.msg_queue.put(("progress", name, frac))
            gen = WorldGenerator(cfg, progress_cb=cb)
            world = gen.generate()
            self.world = world
            self.msg_queue.put(("done", world))
        except Exception as e:
            import traceback
            self.msg_queue.put(("error", str(e), traceback.format_exc()))

    def _poll(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg[0] == "progress":
                    _, name, frac = msg
                    self.status_var.set(f"{name}  ({frac*100:.0f}%)")
                    self.progress["value"] = frac * 100
                elif msg[0] == "done":
                    self.status_var.set("Done.")
                    self.progress["value"] = 100
                    self._render_all()
                elif msg[0] == "error":
                    self.status_var.set("Error.")
                    messagebox.showerror("Generation error", msg[1] + "\n\n" + msg[2])
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _render_all(self):
        w = self.world
        if not w:
            return
        show = {
            "plates":       vz.color_plates(w.plate_map, w.plates),
            "boundaries":   vz.color_boundaries(w.boundary_type),
            "height":       vz.color_height(w.height),
            "temperature":  vz.color_temperature(w.temperature),
            "precipitation":vz.color_precip(w.precipitation),
            "rivers":       self._rivers_overlay(w),
            "realistic":    vz.color_final(w.height, w.temperature, w.precipitation, w.river_map),
            "tilemap":      vz.color_tilemap(w.tilemap),
            "groups":       self._groups_view(w),
        }
        for key, img in show.items():
            fig, ax = self.axes[key]
            ax.clear()
            ax.imshow(img, origin="upper", interpolation="nearest")
            ax.set_title(f"{key}  ({img.shape[1]}x{img.shape[0]})")
            ax.set_xticks([]); ax.set_yticks([])
            self.canvases[key].draw()

    def _rivers_overlay(self, w):
        base = vz.color_height(w.height)
        r = w.river_map > 0
        base[r] = [0.1, 0.35, 0.9]
        return base

    def _groups_view(self, w):
        th, tw = self.cfg.tile_height, self.cfg.tile_width
        img = np.zeros((th, tw, 3), dtype=np.float32)
        rng = np.random.default_rng(0)
        for g in w.tile_groups:
            c = rng.random(3)
            for (x, y) in g["member_tiles"]:
                img[y, x] = c
        return img

    def on_export(self):
        if not self.world:
            messagebox.showinfo("No world", "Generate a world first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.world.export_json(path)
        messagebox.showinfo("Exported", f"World exported to:\n{path}\n\n"
                                       f"Tile groups: {len(self.world.tile_groups)}\n"
                                       f"Compression: "
                                       f"{(self.cfg.tile_width*self.cfg.tile_height)/max(1,len(self.world.tile_groups)):.2f}x")

    def on_save_cfg(self):
        cfg = self._collect_cfg()
        if not cfg: return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path: return
        with open(path, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    def on_load_cfg(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path: return
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            if k in self.vars:
                self.vars[k][0].set(str(v))


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = WorldGenApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
