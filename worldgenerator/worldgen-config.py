"""Configuración del generador de mundos — Stoneplace (WOI-02026b).
Todos los parámetros del DDG son editables desde la UI o por YAML."""
from __future__ import annotations
import yaml
from pydantic import BaseModel, Field


class WorldCfg(BaseModel):
    width: int = 1024            # px, mapa equirectangular (proyección de la esfera)
    height: int = 512
    seed: int = 2026


class PlatesCfg(BaseModel):
    dots_y: int = 18             # puntos de la grilla inicial (eje y)
    dots_x: int = 36             # puntos de la grilla inicial (eje x)
    dot_noise: float = 0.35      # jitter de la grilla (0..1, fracción de celda)
    major_plates: int = 12
    small_plates: int = 8        # micro-placas nacidas en bordes
    small_plate_radius: int = 26 # px, crecimiento máx. de micro-placas
    oceanic_ratio: float = 0.5
    oceanic_divider: float = 2.2 # divisor del heightmap para placas oceánicas
    transform_thresh: float = 0.35  # umbral p/ clasificar transform vs conv/div


class TerrainCfg(BaseModel):
    octaves: int = 4             # capas de Perlin (más detalle, menor influencia)
    frequency: float = 2.0
    gain: float = 0.5            # influencia decreciente por octava
    lacunarity: float = 2.0
    detail_amp: float = 0.35
    perlin_cap: float = 0.55     # tope: el Perlin SOLO no puede generar montañas
    edge_smooth_px: int = 6      # suavizado en bordes de continentes
    ridge_layers: int = 3        # capas Fractal Ridge Blending (Devote)
    ridge_shift_min: float = 0.35
    ridge_shift_max: float = 0.65
    ridge_blend: float = 0.55    # mezcla suave perlin/ridge en montañas
    boundary_radius: int = 42    # px de influencia de los límites de placa
    land_scale_m: float = 2200.0
    max_mountain_m: float = 4200.0
    rift_depth_m: float = 900.0
    ocean_ridge_m: float = 300.0
    transform_relief_m: float = 500.0
    ocean_depth_m: float = 3800.0
    trench_depth_m: float = 7000.0
    continent_base_min: float = 0.05
    continent_base_max: float = 0.30


class ClimateCfg(BaseModel):
    tilt_deg: float = 21.0       # Stoneplace: inclinación axial 21°
    equator_temp: float = 30.0
    pole_temp: float = -28.0
    lat_exp: float = 1.35
    lapse_c_per_m: float = 0.0055  # alturas cercanas al nivel del mar: más cálidas
    noise_amp_c: float = 2.5
    equator_rains: float = 2400.0  # mm/año
    arid_rains: float = 120.0
    midlat_rains: float = 1100.0
    polar_rains: float = 180.0
    arid_fall: tuple[float, float] = (12.0, 26.0)  # transición a árido: prominente (abrupta)
    wet_rise: tuple[float, float] = (34.0, 58.0)   # vuelta a húmedo: suave
    polar_fall: tuple[float, float] = (62.0, 78.0)
    precip_noise: float = 0.18


class RiversCfg(BaseModel):
    target: int = 90
    mouth_radius: int = 22          # detector de radio amplio
    mouth_min_ocean: float = 0.55   # fracción mínima de océano en el radio
    pond_max_area: int = 600        # px: masas menores = "lagunas" (no valen)
    min_source_height: float = 1400.0  # nacen en montañas
    candidates: int = 10            # puntos posibles en la circunferencia
    step_radius: int = 10
    dir_weight: float = 1.6         # peso de dirección cuando alturas son similares
    similar_height_m: float = 60.0
    uphill_penalty: float = 0.15
    split_chance: float = 0.015     # bifurcación por paso
    delta_split_chance: float = 0.35  # mayor chance al final (deltas)
    delta_radius: int = 18
    rejoin_allow: bool = True       # ramas que se reúnen → islas
    width_per_precip: float = 0.0008  # ancho += precip_local * factor (por punto)
    min_mouth_separation: int = 26


class DynamicsCfg(BaseModel):
    enabled: bool = True
    total_years: int = 2_000_000
    step_years: int = 500_000       # se exporta un mapa por cada step_years
    plate_step_px: float = 6.0      # deriva por era
    blob_move_chance: float = 0.6   # chance de punto interior en movimiento
    blobs_per_plate: int = 2
    blob_radius: int = 14
    blob_follow_chance: float = 0.45  # puntos cercanos acompañan el movimiento


class TilesCfg(BaseModel):
    tile_size: int = 16
    similarity_percent: float = 10.0  # criterio de unificación (configurable)


class PerfCfg(BaseModel):
    threads: int = 16   # default pensado para i7-13700 + 16GB RAM
    use_numba: bool = True
    preview_w: int = 512
    preview_h: int = 256


class ExportCfg(BaseModel):
    path: str = "world.json"
    include_biome_reference: bool = True  # tabla de referencia (los biomas se definen en la simulación)


PLANET = {
    "name": "Stoneplace (WOI-02026b)",
    "star": "WOI-2026a",
    "moon": "WOI-2026b I",
    "year_days": 400,
    "day_hours": 25,
    "gravity_earth": 0.8,
    "axial_tilt_deg": 21.0,
    "magnetosphere": True,
    "atmosphere_percent": {"N2": 71.0, "O2": 28.0, "Ar": 0.8, "CO2": 0.1, "CH4+H2": 0.1},
}


class AppConfig(BaseModel):
    world: WorldCfg = Field(default_factory=WorldCfg)
    plates: PlatesCfg = Field(default_factory=PlatesCfg)
    terrain: TerrainCfg = Field(default_factory=TerrainCfg)
    climate: ClimateCfg = Field(default_factory=ClimateCfg)
    rivers: RiversCfg = Field(default_factory=RiversCfg)
    dynamics: DynamicsCfg = Field(default_factory=DynamicsCfg)
    tiles: TilesCfg = Field(default_factory=TilesCfg)
    performance: PerfCfg = Field(default_factory=PerfCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    planet: dict = Field(default_factory=lambda: dict(PLANET))

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False, allow_unicode=True)