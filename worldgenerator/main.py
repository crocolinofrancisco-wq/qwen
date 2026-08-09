"""Punto de entrada.
  python main.py                     → UI (por defecto)
  python main.py --headless          → genera sin UI (logs por consola)
  python main.py --config mio.yaml --headless --out mundo.json"""
from __future__ import annotations
import argparse
import queue
import threading

from config import AppConfig


def headless(cfg: AppConfig):
    from pipeline import Pipeline
    ev = queue.Queue()
    pipe = Pipeline(cfg, ev)
    t = threading.Thread(target=pipe.run, daemon=True)
    t.start()
    while True:
        kind, *data = ev.get()
        if kind == "log":
            print(data[0])
        elif kind == "progress":
            print(f"  [{data[0]}] {data[1] * 100:5.1f}%")
        elif kind == "error":
            print(data[0])
            break
        elif kind == "done":
            print(f"Exportado: {data[0]}")
            break
    t.join()


def main():
    ap = argparse.ArgumentParser(description="Stoneplace WorldGen")
    ap.add_argument("--config", default=None, help="YAML de configuración")
    ap.add_argument("--headless", action="store_true", help="sin UI")
    ap.add_argument("--out", default=None, help="ruta del JSON de salida")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config) if args.config else AppConfig()
    if args.out:
        cfg.export.path = args.out
    if args.headless:
        headless(cfg)
    else:
        try:
            import ui
            ui.main()
        except ImportError as e:
            print(f"No se pudo iniciar la UI ({e}). Usá --headless o instalá dearpygui.")


if __name__ == "__main__":
    main()