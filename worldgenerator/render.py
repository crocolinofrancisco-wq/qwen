"""Render 2D (previews de la UI) y visor isométrico 3D (matplotlib off-screen)
para el mapa original y el tilemap."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import zoom
from plates import B_DIVERGENT, B_CONVERGENT, B_TRANSFORM
from tiles import OCEAN, RIVER


def _cmap(stops):
    pos = [s[0] for s in stops]
    r = [s[1][0] for s in stops]; g = [s[1][1] for s in stops]; b = [s[1][2] for s in stops]

    def f(v):
        v = np.clip(v, 0, 1)
        return np.stack([np.interp(v, pos, r), np.interp(v, pos, g),
                         np.interp(v, pos, b)], axis=-1).astype(np.uint8)
    return f


_LAND = _cmap([(0.0, (96, 148, 78)), (0.25, (178, 190, 120)), (0.45, (206, 190, 140)),
               (0.65, (140, 105, 70)), (0.85, (170, 160, 150)), (1.0, (248, 248, 248))])
_TEMP = _cmap([(0.0, (10, 25, 90)), (0.35, (90, 165, 220)), (0.5, (235, 235, 190)),
               (0.7, (240, 160, 60)), (1.0, (185, 30, 10))])
_PREC = _cmap([(0.0, (240, 230, 200)), (0.15, (140, 190, 120)), (0.4, (40, 120, 60)),
               (0.75, (30, 60, 140)), (1.0, (10, 20, 60))])


def height_rgb(height, btype=None):
    v = np.clip(height / 4000.0, -1, 1)
    rgb = np.zeros(height.shape + (3,), np.uint8)
    ocean = v < 0
    t = np.clip(-v, 0, 1)
    deep = np.array([8, 24, 64]); shallow = np.array([46, 116, 181])
    rgb[ocean] = (shallow + (deep - shallow) * t[ocean][:, None]).astype(np.uint8)
    rgb[~ocean] = _LAND(np.clip(v[~ocean] * 1.6, 0, 1))
    if btype is not None:                                       # límites de placa
        for bt, col in ((B_CONVERGENT, (220, 60, 40)), (B_DIVERGENT, (60, 110, 230)),
                        (B_TRANSFORM, (230, 200, 60))):
            m = btype == bt
            rgb[m] = (0.5 * rgb[m] + 0.5 * np.array(col)).astype(np.uint8)
    return rgb


def temp_rgb(T):
    return _TEMP((T + 40.0) / 85.0)


def precip_rgb(P):
    return _PREC(P / 3000.0)


def plates_rgb(plates):
    rng = np.random.default_rng(7)
    cols = rng.integers(40, 230, (plates.n, 3), dtype=np.uint8)
    rgb = cols[plates.id_map]
    b = plates.btype > 0
    rgb[b] = (rgb[b] * 0.4).astype(np.uint8)
    return rgb


def rivers_rgb(height, river_ids, width_map):
    rgb = height_rgb(height)
    m = river_ids > 0
    a = np.clip(width_map[m] / 5.0, 0.45, 1.0)[:, None]
    rgb[m] = ((1 - a) * rgb[m] + a * np.array([36, 84, 200])).astype(np.uint8)
    return rgb


def tiles_rgb(tm, upscale_to=None):
    rng = np.random.default_rng(11)
    cols = rng.integers(50, 235, (len(tm.reg_cells), 3), dtype=np.uint8)
    rgb = cols[tm.region]
    rgb[tm.ttype == OCEAN] = (rgb[tm.ttype == OCEAN] * np.array([0.55, 0.7, 1.0])).astype(np.uint8)
    rgb[tm.ttype == RIVER] = (rgb[tm.ttype == RIVER] * np.array([0.5, 0.7, 1.3])
                              .clip(0, 255)).astype(np.uint8)
    if upscale_to is not None:                                  # grilla de tiles
        ts = upscale_to[0] // tm.ttype.shape[0]
        rgb = np.repeat(np.repeat(rgb, ts, 0), ts, 1)
        rgb[::ts, :] = (rgb[::ts, :] * 0.55).astype(np.uint8)
        rgb[:, ::ts] = (rgb[:, ::ts] * 0.55).astype(np.uint8)
    return rgb


def isometric(height, face_rgb, out_w, out_h, azim=-55, elev=32, river_mask=None,
              rows=110, cols=200):
    """Visor isométrico 3D (matplotlib Agg → imagen). Sirve para el mapa
    original (pasar height + height_rgb) y para el tilemap (pasar tm.height +
    tiles_rgb(tm) a resolución de tiles)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra projection='3d')

    fy = rows / height.shape[0]
    fx = cols / height.shape[1]
    hz = zoom(height, (fy, fx), order=1).astype(np.float32)
    cz = zoom(face_rgb.astype(np.float32), (fy, fx, 1), order=1) / 255.0
    if river_mask is not None:
        rm = zoom(river_mask.astype(np.float32), (fy, fx), order=1) > 0.3
        cz[rm] = np.array([0.15, 0.35, 0.85])
    zscale = (hz.shape[1] / 7.0) / max(float(np.abs(hz).max()), 1.0)
    X, Y = np.meshgrid(np.arange(hz.shape[1]), np.arange(hz.shape[0]))
    fc = np.zeros((hz.shape[0] - 1, hz.shape[1] - 1, 4))
    fc[..., :3] = cz[:-1, :-1, :3]
    fc[..., 3] = 1.0

    fig = Figure(figsize=(out_w / 100.0, out_h / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, hz * zscale, rstride=1, cstride=1,
                    facecolors=fc, shade=False, antialiased=False, linewidth=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_zlim(0, hz.shape[1] / 7.0)
    fig.subplots_adjust(0, 0, 1, 1)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()