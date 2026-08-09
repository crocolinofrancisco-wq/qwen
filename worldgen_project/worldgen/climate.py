"""Climate computed on the sphere (3D) and projected to the flat map.

Temperature follows true spherical latitude (with axial tilt applied as a
rotation of the sphere itself).  Precipitation starts high at the equator,
drops sharply into arid belts, and rises again at mid latitudes — with
trade winds / westerlies advecting moisture along the sphere and orographic
rain shadows.  Noise is 3D sphere noise, so the projected map is seamless.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import WorldConfig
from .spherical_noise import SphericalNoise
from .sphere import latlon_grids, unit_sphere_points, rotation_about, lonlat_from_xyz


def _gauss_wrap(arr, sigma):
    return gaussian_filter(arr, sigma=sigma, mode=("reflect", "wrap"))


def _tilted_lat(cfg: WorldConfig, lat, lon):
    """Apply the axial tilt as a rotation of the planet: returns the
    'effective' latitude (in [-1, 1], relative to the tilted equator)."""
    xyz = unit_sphere_points(lat, lon)
    tilt = cfg.axial_tilt * (np.pi / 2)  # cfg fraction of 90 degrees
    R = rotation_about([1.0, 0.0, 0.0], -tilt)  # tilt the axis about X
    rot = xyz @ R.T
    tlat, _ = lonlat_from_xyz(rot)
    return tlat / (np.pi / 2)  # normalized to [-1, 1]


def generate_temperature(cfg: WorldConfig, rng, height, lat, lon, progress=None):
    eff_lat = _tilted_lat(cfg, lat, lon)
    lat_temp = np.cos(np.clip(eff_lat, -1, 1) * np.pi / 2)  # 1 at equator, 0 poles

    # altitude cools; heights near sea level stay warm
    height_effect = np.clip(height, 0, None) * cfg.temp_height_factor
    temp = lat_temp - height_effect * 0.9

    # ocean moderating
    ocean = height < 0
    temp[ocean] = temp[ocean] * 0.85 + 0.15 * lat_temp[ocean]

    pn = SphericalNoise(seed=int(rng.integers(0, 1_000_000)))
    n = pn.fbm_latlon(lat, lon, scale=5.0, octaves=4)
    temp += (n - 0.5) * cfg.climate_noise * 2

    temp = np.clip(temp, -0.2, 1.2)
    if progress: progress("climate:temp", 0.55)
    return temp


def _wind_field(cfg: WorldConfig, H: int, W: int):
    """Prevailing winds per row (zonal u, meridional v), sphere-consistent."""
    ys = np.linspace(-1, 1, H)  # -1 south pole, 1 north pole
    lat_abs = np.abs(ys)
    u = np.zeros(H, dtype=np.float32)
    mask = lat_abs < 0.33                        # trade winds
    u[mask] = -1.0 * (1 - lat_abs[mask] / 0.33)
    mask = (lat_abs >= 0.33) & (lat_abs < 0.75)  # westerlies
    u[mask] = np.sin((lat_abs[mask] - 0.33) / 0.42 * np.pi)
    mask = lat_abs >= 0.75                       # polar easterlies
    u[mask] = -0.6 * ((lat_abs[mask] - 0.75) / 0.25)
    u = u * cfg.trade_wind_strength
    v = -0.15 * np.sign(ys) * lat_abs
    U = np.broadcast_to(u.reshape(-1, 1), (H, W)).copy()
    V = np.broadcast_to(v.reshape(-1, 1), (H, W)).copy()
    return U, V


def _advect_moisture(base, U, V, steps=8):
    """Row-wise semi-lagrangian advection with horizontal wrap."""
    H, W = base.shape
    out = base.copy()
    row_u = U[:, 0]
    row_v = V[:, 0]
    for it in range(steps):
        strength = 3 + it
        new = np.empty_like(out)
        for j in range(H):
            dx = int(round(row_u[j] * strength))
            new[j] = np.roll(out[j], shift=dx)  # wraps: correct on a sphere
        dy = int(round(np.mean(row_v) * 2))
        if dy != 0:
            new = np.roll(new, shift=dy, axis=0)
        out = 0.6 * out + 0.4 * new
    return out


def _orographic(height, U, cfg):
    """Windward slopes gain moisture, leeward slopes lose it (wrap in X)."""
    dh_dx = np.zeros_like(height)
    dh_dx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) * 0.5
    dh_dx[:, 0] = (height[:, 1] - height[:, -1]) * 0.5
    dh_dx[:, -1] = (height[:, 0] - height[:, -2]) * 0.5
    return dh_dx * np.sign(U) * cfg.orographic_strength


def generate_precipitation(cfg: WorldConfig, rng, height, temp, lat, lon,
                           progress=None):
    W, H = cfg.sim_width, cfg.sim_height
    eff_lat = _tilted_lat(cfg, lat, lon)
    alat = np.abs(eff_lat)

    band = cfg.equator_band_width
    eq = np.exp(-(eff_lat ** 2) / (band ** 2 * 2.0))          # equatorial rain
    arid = 1.0 - 0.6 * np.exp(-((alat - 0.36) ** 2) / 0.020) * cfg.arid_dropoff
    arid = np.clip(arid, 0.15, 1.0)                           # prominent arid drop
    mid = np.exp(-((alat - 0.60) ** 2) / 0.030) * 0.75        # mid-latitude wet
    polar = np.exp(-((alat - 1.00) ** 2) / 0.030) * -0.45     # polar dryness
    precip = np.clip(np.clip(eq + mid + polar, 0, None) * arid, 0.0, 1.0)
    precip = precip.astype(np.float32)

    pn = SphericalNoise(seed=int(rng.integers(0, 1_000_000)))
    n_low = pn.fbm_latlon(lat, lon, scale=4.5, octaves=5)
    n_high = pn.fbm_latlon(lat, lon, scale=11.0, octaves=6)

    noise = 0.55 * n_low + 0.45 * n_high
    precip = precip * (0.55 + 0.9 * noise)
    precip += (noise - 0.5) * cfg.precip_noise * 1.5

    U, V = _wind_field(cfg, H, W)
    ocean = height < 0
    humidity_src = np.where(ocean, 0.9, precip)
    advected = _advect_moisture(humidity_src, U, V, steps=8)
    precip = 0.55 * precip + 0.45 * advected

    oro = _orographic(np.clip(height, 0, None), U, cfg)
    precip = precip + oro * 0.6
    precip = np.where(oro < 0, precip * (1.0 + oro * 0.5), precip)

    precip[ocean] = np.maximum(precip[ocean], 0.4)
    precip = _gauss_wrap(precip, sigma=1.3)
    precip = np.clip(precip, 0.0, 1.0)
    if progress: progress("climate:precip", 0.7)
    return precip
