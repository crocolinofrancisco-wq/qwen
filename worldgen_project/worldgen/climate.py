"""Climate: temperature and precipitation.

Improvements vs previous version:
- Precipitation follows the tilted ITCZ (equatorial band bends with axial tilt across
  longitude), not a flat horizontal band.
- Equatorial rain band is wider and less abrupt.
- Trade winds / westerlies advect moisture zonally, breaking horizontal striping.
- Orographic effect: windward slopes are wetter, leeward drier (rain shadows).
- Everything is computed with horizontal wrap (spherical projection).
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import WorldConfig
from .noise import PerlinNoise


def _wrap_gaussian(arr, sigma):
    """Gaussian filter that wraps horizontally (X) for spherical maps."""
    return gaussian_filter(arr, sigma=sigma, mode=("reflect", "wrap"))


def _tilted_latitude(cfg: WorldConfig):
    """Return a 2D array of 'effective latitude' [-1, 1] where the ITCZ (0) is
    displaced by the axial tilt as a function of longitude.
    """
    W, H = cfg.sim_width, cfg.sim_height
    ys = np.linspace(-1, 1, H).reshape(-1, 1)
    lon = np.linspace(0, 2 * np.pi, W, endpoint=False).reshape(1, -1)
    # ITCZ offset oscillates sinusoidally along longitude with amplitude = tilt
    itcz = cfg.axial_tilt * 0.5 * np.sin(lon)   # -tilt/2 .. tilt/2
    eff_lat = ys - itcz                          # broadcast to (H, W)
    return np.broadcast_to(eff_lat, (H, W)).copy()


def generate_temperature(cfg: WorldConfig, rng, height, progress=None):
    W, H = cfg.sim_width, cfg.sim_height
    eff_lat = _tilted_latitude(cfg)
    lat_temp = np.cos(eff_lat * np.pi / 2)   # 1 at (tilted) equator, 0 at poles
    lat_temp = np.clip(lat_temp, 0, 1)

    # Height effect: high altitudes are colder
    height_effect = np.clip(height, 0, None) * cfg.temp_height_factor
    temp = lat_temp - height_effect * 0.9

    # Ocean moderating
    ocean = height < 0
    temp[ocean] = temp[ocean] * 0.85 + 0.15 * lat_temp[ocean]

    # 2D noise (seamless in X)
    pn = PerlinNoise(seed=int(rng.integers(0, 1_000_000)))
    if cfg.spherical_wrap:
        n = pn.spherical_fractal(W, H, octaves=4, scale=5)
    else:
        n = pn.fractal(W, H, octaves=4, scale=5)
    temp += (n - 0.5) * cfg.climate_noise * 2

    temp = np.clip(temp, -0.2, 1.2)
    if progress: progress("climate:temp", 0.55)
    return temp


def _wind_field(cfg: WorldConfig):
    """Return zonal (u) and meridional (v) wind components per pixel.
    Model of prevailing winds:
      - 0..15 deg lat  : trade winds -> easterly (u < 0)
      - 15..35 deg lat : subtropical high (weak, poleward)
      - 35..60 deg lat : westerlies (u > 0)
      - 60..90 deg lat : polar easterlies (u < 0)
    """
    W, H = cfg.sim_width, cfg.sim_height
    ys = np.linspace(-1, 1, H)          # -1 south pole, 1 north pole
    lat_abs = np.abs(ys)
    u = np.zeros(H, dtype=np.float32)
    # trade
    mask = lat_abs < 0.33
    u[mask] = -1.0 * (1 - lat_abs[mask] / 0.33)
    # westerlies
    mask = (lat_abs >= 0.33) & (lat_abs < 0.75)
    u[mask] = 1.0 * np.sin((lat_abs[mask] - 0.33) / 0.42 * np.pi)
    # polar easterlies
    mask = lat_abs >= 0.75
    u[mask] = -0.6 * ((lat_abs[mask] - 0.75) / 0.25)
    u = u * cfg.trade_wind_strength
    v = -0.15 * np.sign(ys) * (lat_abs)   # weak equator-ward flow near the tropics
    U = np.broadcast_to(u.reshape(-1, 1), (H, W)).copy()
    V = np.broadcast_to(v.reshape(-1, 1), (H, W)).copy()
    return U, V


def _advect_moisture(base, U, V, steps=6):
    """Row-wise semi-lagrangian advection: each latitude row is rolled by its own
    wind velocity (so trades and westerlies really move air in opposite directions),
    which is what breaks the horizontal striping.  Wraps in X."""
    H, W = base.shape
    out = base.copy()
    # per-row integer shift, scaled per iteration
    row_u = U[:, 0]  # zonal wind is constant along a row
    row_v = V[:, 0]
    for it in range(steps):
        strength = 3 + it  # each iteration moves farther
        new = np.empty_like(out)
        for j in range(H):
            dx = int(round(row_u[j] * strength))
            new[j] = np.roll(out[j], shift=dx)
        # meridional (weak) shift
        dy = int(round(np.mean(row_v) * 2))
        if dy != 0:
            new = np.roll(new, shift=dy, axis=0)
        out = 0.6 * out + 0.4 * new
    return out


def _orographic(height, U, cfg):
    """Windward slopes gain moisture, leeward slopes lose it.
    Slope along wind = d(height)/dx * sign(U) using centered differences.
    """
    dh_dx = np.zeros_like(height)
    dh_dx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) * 0.5
    dh_dx[:, 0]    = (height[:, 1] - height[:, -1]) * 0.5    # wrap
    dh_dx[:, -1]   = (height[:, 0] - height[:, -2]) * 0.5    # wrap

    # component in the direction of the wind
    along = dh_dx * np.sign(U)
    # windward (positive) -> more rain; leeward (negative) -> less
    return along * cfg.orographic_strength


def generate_precipitation(cfg: WorldConfig, rng, height, temp, progress=None):
    W, H = cfg.sim_width, cfg.sim_height

    # 1) Tilted latitude for the ITCZ
    eff_lat = _tilted_latitude(cfg)
    lat = np.abs(eff_lat)  # in [0, ~1]

    # 2) Base zonal precipitation curve, but per-pixel because eff_lat is 2D
    band = cfg.equator_band_width
    # equatorial peak (wider gaussian than before)
    eq   = np.exp(-(eff_lat ** 2) / (band ** 2 * 2.0))
    # subtropical arid belts ~ lat 0.33
    arid = 1.0 - 0.6 * np.exp(-((lat - 0.36) ** 2) / 0.020) * cfg.arid_dropoff
    arid = np.clip(arid, 0.15, 1.0)
    # mid-latitude wet band ~0.6
    mid  = np.exp(-((lat - 0.60) ** 2) / 0.030) * 0.75
    # polar dryness
    polar = np.exp(-((lat - 1.00) ** 2) / 0.030) * -0.45

    curve = np.clip(eq + mid + polar, 0, None) * arid
    curve = np.clip(curve, 0.0, 1.0)
    precip = curve.astype(np.float32).copy()

    # 3) Domain-warped noise (seamless X) - warps the latitudinal curve so it
    #    stops looking like straight stripes.
    pn  = PerlinNoise(seed=int(rng.integers(0, 1_000_000)))
    pnw = PerlinNoise(seed=int(rng.integers(0, 1_000_000)))
    fractal = pn.spherical_fractal if cfg.spherical_wrap else pn.fractal
    warp_fn = pnw.spherical_fractal if cfg.spherical_wrap else pnw.fractal

    n_low  = fractal(W, H, octaves=5, scale=4.5)
    n_high = fractal(W, H, octaves=6, scale=11)
    warp_x = (warp_fn(W, H, octaves=3, scale=3.0) - 0.5) * 0.25 * H   # up to 12% H displacement
    warp_y = (warp_fn(W, H, octaves=3, scale=3.5) - 0.5) * 0.25 * H

    # Warp the curve itself: sample precip at (j + warp_y, i + warp_x)
    js, is_ = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    sj = np.clip((js + warp_y).astype(int), 0, H - 1)
    si = ((is_ + warp_x).astype(int)) % W if cfg.spherical_wrap else np.clip((is_ + warp_x).astype(int), 0, W - 1)
    precip = precip[sj, si]

    noise = 0.55 * n_low + 0.45 * n_high
    precip = precip * (0.55 + 0.9 * noise)
    precip += (noise - 0.5) * cfg.precip_noise * 1.5

    # 4) Wind advection: humidity spreads horizontally along prevailing winds
    U, V = _wind_field(cfg)
    # baseline humidity source over oceans
    ocean = height < 0
    humidity_src = np.where(ocean, 0.9, precip)
    advected = _advect_moisture(humidity_src, U, V, steps=8)
    precip = 0.55 * precip + 0.45 * advected

    # 5) Orographic effect: windward wet, leeward dry
    oro = _orographic(np.clip(height, 0, None), U, cfg)
    precip = precip + oro * 0.6
    # rain shadow: strong leeward drying
    precip = np.where(oro < 0, precip * (1.0 + oro * 0.5), precip)

    # 6) Ocean baseline
    precip[ocean] = np.maximum(precip[ocean], 0.4)

    # 7) Smooth with horizontal wrap so the seam is continuous
    precip = _wrap_gaussian(precip, sigma=1.3)
    precip = np.clip(precip, 0.0, 1.0)

    if progress: progress("climate:precip", 0.7)
    return precip
