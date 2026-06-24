"""Feature extraction helpers for sustainable agriculture modeling.

Future/t+1 values are written only to gee_targets.csv. They are never written to
gee_features.csv, so model input files cannot silently include future data.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

S2_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
S1_DATASET = "COPERNICUS/S1_GRD"
CHIRPS_DATASET = "UCSB-CHG/CHIRPS/DAILY"
ERA5_DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
SRTM_DATASET = "USGS/SRTMGL1_003"
DYNAMICWORLD_DATASET = "GOOGLE/DYNAMICWORLD/V1"
JRC_WATER_DATASET = "JRC/GSW1_4/GlobalSurfaceWater"
DEFAULT_GEE_CREDENTIALS_PATH = Path("config") / "gee_credentials.json"

ID_COLUMNS = ["sample_id", "label", "latitude", "longitude", "frame_index", "acquisition_date"]
OBSERVATION_COLUMNS = ID_COLUMNS + [
    "ndvi_mean", "evi_mean", "ndwi_mean", "ndmi_mean", "s2_image_count",
    "rainfall_30d_mm", "rainfall_90d_mm", "dry_days_30d", "heavy_rain_days_30d", "heavy_rain_days_90d",
    "temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_30d_m",
    "elevation_m", "slope_deg", "lowland_flag",
    "vv_mean_db", "vh_mean_db", "vv_minus_vh_db", "s1_image_count",
    "built_probability_1km", "built_probability_5km", "flooded_vegetation_probability", "dynamicworld_image_count",
    "water_occurrence_mean", "max_water_extent_fraction",
]
OPTICAL_COLUMNS = ["ndvi_mean", "evi_mean", "ndwi_mean", "ndmi_mean"]
SAR_COLUMNS = ["vv_mean_db", "vh_mean_db", "vv_minus_vh_db"]
CLIMATE_COLUMNS = [
    "rainfall_30d_mm", "rainfall_90d_mm", "dry_days_30d", "heavy_rain_days_30d", "heavy_rain_days_90d",
    "temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_30d_m",
]
CONTEXT_COLUMNS = [
    "built_probability_1km", "built_probability_5km", "flooded_vegetation_probability",
    "water_occurrence_mean", "max_water_extent_fraction", "elevation_m", "slope_deg", "lowland_flag",
]
IMPUTE_COLUMNS = OPTICAL_COLUMNS + SAR_COLUMNS + CLIMATE_COLUMNS + CONTEXT_COLUMNS
FEATURE_COLUMNS = ID_COLUMNS + [
    "month_sin", "month_cos",
    "optical_imputed_flag", "sar_imputed_flag", "climate_imputed_flag", "any_imputed_flag", "leading_backfill_flag",
    "ndvi_lag_1", "ndvi_lag_2", "ndvi_rolling_mean_3", "ndvi_rolling_std_3", "ndvi_rolling_mean_5", "ndvi_rolling_std_5", "ndvi_trend",
    "evi_lag_1", "evi_lag_2", "evi_rolling_mean_3", "evi_rolling_std_3", "evi_rolling_mean_5", "evi_rolling_std_5", "evi_trend",
    "built_growth_rate", "built_growth_trend", "urban_encroachment_index",
    "rainfall_rolling_mean_3", "heavy_rain_rolling_sum_3", "rain_to_green_ratio",
    "sar_moisture_trend", "vv_vh_ratio_linear", "flood_risk_proxy_score",
]
TARGET_COLUMNS = [
    "sample_id", "label", "frame_index", "acquisition_date", "target_date",
    "target_ndvi_delta_1", "target_evi_delta_1", "target_built_delta_1",
    "target_sustainability_proxy_score", "target_available_flag", "target_uses_imputed_observation",
]

@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Settings for slow Google Earth Engine observation fetching."""

    gee_project_id: str | None = None
    gee_credentials_path: str | None = None
    data_dir: str = "data"
    sample_index_csv: str = "data/processed/sample_index.csv"
    buffer_radius_m: int = 1000
    context_buffer_radius_m: int = 5000
    s2_lookback_days: int = 45
    s1_tolerance_days: int = 15
    dynamicworld_lookback_days: int = 180
    dry_day_threshold_mm: float = 1.0
    heavy_rain_threshold_mm: float = 20.0
    lowland_elevation_threshold_m: float = 50.0
    force: bool = False
    verbose: bool = True
    log_feature_groups: bool = True

    @property
    def project_id(self) -> str | None:
        if self.gee_project_id:
            return self.gee_project_id
        project_root = project_root_from_sample_index(self.sample_index_csv)
        return load_gee_project_id(
            self.gee_credentials_path or DEFAULT_GEE_CREDENTIALS_PATH,
            required=False,
            project_root=project_root,
        )

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Settings for local feature and target engineering."""

    data_dir: str = "data"
    sample_index_csv: str = "data/processed/sample_index.csv"
    force: bool = True
    verbose: bool = True
    ffill_enabled: bool = True
    leading_bfill_limit: int = 2
    epsilon: float = 1e-6


def initialize_earth_engine(project_id: str | None = None, credentials_path: str | Path | None = None):
    """Authenticate and initialize the Earth Engine Python API."""

    import ee

    resolved_project = project_id or load_gee_project_id(credentials_path, required=False)
    try:
        if resolved_project:
            ee.Initialize(project=resolved_project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if resolved_project:
            ee.Initialize(project=resolved_project)
        else:
            ee.Initialize()
    return ee


def project_root_from_sample_index(sample_index_csv: str | Path) -> Path:
    """Infer the project root from the sample index location."""

    sample_index_path = Path(sample_index_csv).resolve()
    if (
        sample_index_path.name == "sample_index.csv"
        and sample_index_path.parent.name == "processed"
        and sample_index_path.parent.parent.name == "data"
    ):
        return sample_index_path.parent.parent.parent
    return Path.cwd().resolve()


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve project-relative paths without changing stored CSV values."""

    value = str(path).strip()
    candidate = Path(value)
    if candidate.is_absolute() and not value.startswith(("\\", "/")):
        return candidate
    if value.startswith(("\\", "/")):
        value = value.lstrip("\\/")
        candidate = Path(value)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    return root / candidate


def project_relative_path(path: str | Path, project_root: str | Path) -> str:
    """Format a path relative to the project root when possible."""

    resolved_path = Path(path).resolve()
    resolved_root = Path(project_root).resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(path)


def display_project_path(path: str | Path, project_root: str | Path | None = None) -> str:
    """Return a privacy-safe path for logs and error messages."""

    if project_root is None:
        project_root = Path.cwd()
    return project_relative_path(resolve_project_path(path, project_root), project_root)


def load_gee_project_id(
    credentials_path: str | Path | None = DEFAULT_GEE_CREDENTIALS_PATH,
    *,
    env_var: str = "GEE_PROJECT_ID",
    required: bool = True,
    project_root: str | Path | None = None,
) -> str | None:
    """Load the Earth Engine project id from env or an ignored JSON file."""

    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return env_value

    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    credentials_file = resolve_project_path(credentials_path or DEFAULT_GEE_CREDENTIALS_PATH, root)
    if not credentials_file.exists():
        if required:
            safe_path = display_project_path(credentials_file, root)
            raise FileNotFoundError(
                f"Missing GEE credentials file: {safe_path}. "
                "Create it from config/gee_credentials.example.json or set GEE_PROJECT_ID."
            )
        return None

    try:
        credentials = json.loads(credentials_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        safe_path = display_project_path(credentials_file, root)
        raise ValueError(f"Invalid GEE credentials JSON: {safe_path}") from exc

    project_id = str(credentials.get("project_id") or credentials.get("gee_project_id") or "").strip()
    if project_id:
        return project_id
    if required:
        safe_path = display_project_path(credentials_file, root)
        raise ValueError(f"GEE credentials file must contain project_id: {safe_path}")
    return None


def load_sample_index(path: str | Path) -> pd.DataFrame:
    """Load sample_index.csv and attach the inferred project root."""

    path = Path(path)
    project_root = project_root_from_sample_index(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing sample index: {display_project_path(path, project_root)}")
    sample_index = pd.read_csv(path)
    sample_index.attrs["project_root"] = project_root
    return sample_index


def load_frame_metadata(path: str | Path, project_root: str | Path | None = None) -> pd.DataFrame:
    """Load one sample frame_metadata.csv from a relative or absolute path."""

    resolved_path = resolve_project_path(path, project_root)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing frame metadata: {display_project_path(resolved_path, project_root)}")
    return pd.read_csv(resolved_path)


def _log(config: FeatureExtractionConfig | FeatureEngineeringConfig, message: str) -> None:
    """Print a message when verbose mode is enabled."""

    if config.verbose:
        print(message, flush=True)


def _progress(iterable: Any, *, total: int | None, desc: str, config: FeatureExtractionConfig | FeatureEngineeringConfig, leave: bool = True) -> Any:
    """Wrap an iterable with tqdm when available and verbose mode is on."""

    if not config.verbose:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)


def _timed_group(config: FeatureExtractionConfig, sample_id: str, acquisition_date: str, group_name: str, extractor: Any) -> dict[str, Any]:
    """Run one GEE feature group with optional timing logs."""

    if config.log_feature_groups:
        _log(config, f"      {sample_id} {acquisition_date}: {group_name} start")
    started_at = time.perf_counter()
    result = extractor()
    if config.log_feature_groups:
        elapsed = time.perf_counter() - started_at
        _log(config, f"      {sample_id} {acquisition_date}: {group_name} done ({elapsed:.1f}s)")
    return result


def extract_raw_observations_from_gee(config: FeatureExtractionConfig, sample_limit: int | None = None, ee_module: Any | None = None) -> list[Path]:
    """Fetch raw GEE observations only and write them under data/raw."""
    ee = ee_module or initialize_earth_engine(config.project_id)
    sample_index = load_sample_index(config.sample_index_csv)
    project_root = sample_index.attrs.get("project_root", project_root_from_sample_index(config.sample_index_csv))
    if sample_limit is not None:
        sample_index = sample_index.head(sample_limit)
    _log(config, f"Starting raw GEE observation fetch: {len(sample_index)} sample(s), project={config.project_id}")
    written_paths: list[Path] = []
    sample_iter = _progress(sample_index.iterrows(), total=len(sample_index), desc="GEE samples", config=config)
    for sample_number, (_, sample) in enumerate(sample_iter, start=1):
        sample_id = str(sample["sample_id"])
        output_path = resolve_project_path(sample["gee_observations_csv"], project_root)
        metadata_path = resolve_project_path(sample["gee_feature_metadata_json"], project_root)
        _log(config, f"[{sample_number}/{len(sample_index)}] {sample_id}: raw observation fetch start")
        if output_path.exists() and not config.force:
            _log(config, f"    {sample_id}: using cached raw observations from {display_project_path(output_path, project_root)}")
            written_paths.append(output_path)
            continue
        observations = extract_sample_observations(sample, config, ee)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        observations.to_csv(output_path, index=False)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(build_feature_metadata(sample, observations, config), indent=2), encoding="utf-8")
        written_paths.append(output_path)
        _log(config, f"    {sample_id}: wrote {len(observations)} raw rows")
    return written_paths


def extract_sample_observations(sample: pd.Series, config: FeatureExtractionConfig, ee: Any) -> pd.DataFrame:
    """Fetch raw GEE observations for every frame in one sample."""

    sample_id = str(sample["sample_id"])
    frame_metadata = load_frame_metadata(
        sample["frame_metadata_csv"],
        project_root_from_sample_index(config.sample_index_csv),
    )
    rows = []
    frame_iter = _progress(frame_metadata.iterrows(), total=len(frame_metadata), desc=f"{sample_id} frames", config=config, leave=False)
    for frame_number, (_, frame) in enumerate(frame_iter, start=1):
        acquisition_date = str(frame["acquisition_date"])
        _log(config, f"    {sample_id}: frame {frame_number}/{len(frame_metadata)} {acquisition_date} start")
        rows.append(extract_observation(sample, frame, acquisition_date, config, ee))
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)

def extract_observation(sample: pd.Series, frame: pd.Series, acquisition_date: str, config: FeatureExtractionConfig, ee: Any) -> dict[str, Any]:
    """Fetch all raw GEE feature groups for one coordinate-date row."""

    latitude = float(sample["latitude"])
    longitude = float(sample["longitude"])
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(config.buffer_radius_m)
    context_region = point.buffer(config.context_buffer_radius_m)
    current_date = date.fromisoformat(acquisition_date)

    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "latitude": latitude,
        "longitude": longitude,
        "frame_index": int(frame["frame_index"]),
        "acquisition_date": acquisition_date,
    }
    sample_id = str(sample["sample_id"])
    row.update(_timed_group(config, sample_id, acquisition_date, "Sentinel-2 vegetation", lambda: _extract_s2(ee, region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "CHIRPS rainfall", lambda: _extract_chirps(ee, region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "ERA5-Land climate", lambda: _extract_era5(ee, region, current_date)))
    row.update(_timed_group(config, sample_id, acquisition_date, "SRTM terrain", lambda: _extract_terrain(ee, region, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "Sentinel-1 SAR", lambda: _extract_s1(ee, region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "Dynamic World urban/water", lambda: _extract_dynamic_world(ee, region, context_region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "JRC water", lambda: _extract_water(ee, region)))
    return {column: row.get(column, np.nan) for column in OBSERVATION_COLUMNS}


def engineer_all_samples(config: FeatureEngineeringConfig, sample_limit: int | None = None) -> pd.DataFrame:
    """Create per-sample gee_features.csv and gee_targets.csv files."""

    sample_index = load_sample_index(config.sample_index_csv)
    project_root = sample_index.attrs.get("project_root", project_root_from_sample_index(config.sample_index_csv))
    if sample_limit is not None:
        sample_index = sample_index.head(sample_limit)
    summaries: list[dict[str, Any]] = []
    sample_iter = _progress(sample_index.iterrows(), total=len(sample_index), desc="Feature engineering", config=config)
    for _, sample in sample_iter:
        sample_id = str(sample["sample_id"])
        observations_path = resolve_project_path(sample["gee_observations_csv"], project_root)
        if not observations_path.exists():
            raise FileNotFoundError(f"Missing raw GEE observations for {sample_id}: {display_project_path(observations_path, project_root)}")
        features_path = resolve_project_path(sample["gee_features_csv"], project_root)
        targets_path = resolve_project_path(sample["gee_targets_csv"], project_root)
        if features_path.exists() and targets_path.exists() and not config.force:
            _log(config, f"    {sample_id}: using cached engineered outputs")
        else:
            observations = pd.read_csv(observations_path)
            features, targets = engineer_features_and_targets(observations, config)
            write_engineered_outputs(sample, features, targets, project_root)
            _log(config, f"    {sample_id}: wrote {len(features)} feature rows and {len(targets)} target rows")
        summaries.append({
            "sample_id": sample_id,
            "label": sample["label"],
            "gee_observations_csv": project_relative_path(observations_path, project_root),
            "gee_features_csv": project_relative_path(features_path, project_root),
            "gee_targets_csv": project_relative_path(targets_path, project_root),
        })
    return pd.DataFrame(summaries)


def engineer_features_and_targets(observations: pd.DataFrame, config: FeatureEngineeringConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw observations into leakage-separated features and targets."""

    config = config or FeatureEngineeringConfig()
    if observations.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.DataFrame(columns=TARGET_COLUMNS)
    df = observations.reindex(columns=OBSERVATION_COLUMNS).copy()
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce")
    df = df.sort_values(["sample_id", "acquisition_date", "frame_index"]).reset_index(drop=True)
    df = apply_leakage_safe_imputation(df, config)
    group = df.groupby("sample_id", group_keys=False)
    features = df[ID_COLUMNS].copy()
    features["acquisition_date"] = features["acquisition_date"].dt.date.astype(str)
    month = df["acquisition_date"].dt.month.astype(float)
    features["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    for flag in ["optical_imputed_flag", "sar_imputed_flag", "climate_imputed_flag", "any_imputed_flag", "leading_backfill_flag"]:
        features[flag] = df[flag].astype(int)
    for prefix in ["ndvi", "evi"]:
        source = f"{prefix}_mean"
        lag_1 = group[source].shift(1)
        lag_2 = group[source].shift(2)
        features[f"{prefix}_lag_1"] = lag_1.fillna(df[source])
        features[f"{prefix}_lag_2"] = lag_2.fillna(features[f"{prefix}_lag_1"])
        features[f"{prefix}_rolling_mean_3"] = group[source].transform(lambda s: s.rolling(3, min_periods=1).mean())
        features[f"{prefix}_rolling_std_3"] = group[source].transform(lambda s: s.rolling(3, min_periods=1).std()).fillna(0.0)
        features[f"{prefix}_rolling_mean_5"] = group[source].transform(lambda s: s.rolling(5, min_periods=1).mean())
        features[f"{prefix}_rolling_std_5"] = group[source].transform(lambda s: s.rolling(5, min_periods=1).std()).fillna(0.0)
        features[f"{prefix}_trend"] = group[source].transform(lambda s: s.diff(2) / 2.0).fillna(0.0)
    built = df["built_probability_5km"]
    built_group = group["built_probability_5km"]
    features["built_growth_rate"] = built_group.diff().fillna(0.0)
    features["built_growth_trend"] = built_group.transform(lambda s: s.diff().rolling(3, min_periods=1).mean()).fillna(0.0)
    features["urban_encroachment_index"] = built - df["built_probability_1km"]
    features["rainfall_rolling_mean_3"] = group["rainfall_30d_mm"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    features["heavy_rain_rolling_sum_3"] = group["heavy_rain_days_30d"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    rainfall_lag = group["rainfall_30d_mm"].shift(1).fillna(df["rainfall_30d_mm"])
    features["rain_to_green_ratio"] = df["ndvi_mean"] / (rainfall_lag.abs() + config.epsilon)
    features["sar_moisture_trend"] = group["vh_mean_db"].transform(lambda s: s.diff(2) / 2.0).fillna(0.0)
    features["vv_vh_ratio_linear"] = np.power(10.0, df["vv_minus_vh_db"] / 10.0)
    features["flood_risk_proxy_score"] = _flood_risk_proxy(df)
    features = features.reindex(columns=FEATURE_COLUMNS)
    assert_no_target_leakage(features)
    targets = df[["sample_id", "label", "frame_index"]].copy()
    targets["acquisition_date"] = df["acquisition_date"].dt.date.astype(str)
    target_date = group["acquisition_date"].shift(-1)
    targets["target_date"] = target_date.dt.date.astype(str).where(target_date.notna(), pd.NA)
    targets["target_ndvi_delta_1"] = group["ndvi_mean"].shift(-1) - df["ndvi_mean"]
    targets["target_evi_delta_1"] = group["evi_mean"].shift(-1) - df["evi_mean"]
    targets["target_built_delta_1"] = group["built_probability_5km"].shift(-1) - df["built_probability_5km"]
    targets["target_sustainability_proxy_score"] = targets["target_ndvi_delta_1"] + targets["target_evi_delta_1"] - targets["target_built_delta_1"] - features["flood_risk_proxy_score"].fillna(0) * 0.1
    targets["target_available_flag"] = targets["target_date"].notna().astype(int)
    next_imputed = group["any_imputed_flag"].shift(-1).fillna(0).astype(int)
    targets["target_uses_imputed_observation"] = ((df["any_imputed_flag"].astype(int) == 1) | (next_imputed == 1)).astype(int)
    return features, targets.reindex(columns=TARGET_COLUMNS)


def apply_leakage_safe_imputation(df: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """Fill missing observations within each sample without creating target columns."""

    working = df.copy()
    for column in IMPUTE_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["optical_imputed_flag"] = working[OPTICAL_COLUMNS].isna().any(axis=1).astype(int)
    working["sar_imputed_flag"] = working[SAR_COLUMNS].isna().any(axis=1).astype(int)
    working["climate_imputed_flag"] = working[CLIMATE_COLUMNS].isna().any(axis=1).astype(int)
    working["any_imputed_flag"] = working[["optical_imputed_flag", "sar_imputed_flag", "climate_imputed_flag"]].any(axis=1).astype(int)
    after_ffill = working.groupby("sample_id", group_keys=False)[IMPUTE_COLUMNS].ffill() if config.ffill_enabled else working[IMPUTE_COLUMNS].copy()
    after_fill = after_ffill.groupby(working["sample_id"], group_keys=False).bfill(limit=config.leading_bfill_limit)
    working[IMPUTE_COLUMNS] = after_fill
    working["leading_backfill_flag"] = (after_ffill.isna() & after_fill.notna()).any(axis=1).astype(int)
    static_columns = ["elevation_m", "slope_deg", "lowland_flag", "water_occurrence_mean", "max_water_extent_fraction"]
    working[static_columns] = working.groupby("sample_id", group_keys=False)[static_columns].ffill()
    working[static_columns] = working.groupby("sample_id", group_keys=False)[static_columns].bfill()
    return working


def write_engineered_outputs(sample: pd.Series, features: pd.DataFrame, targets: pd.DataFrame, project_root: str | Path | None = None) -> None:
    """Write one sample's engineered feature and target tables."""

    features_path = resolve_project_path(sample["gee_features_csv"], project_root)
    targets_path = resolve_project_path(sample["gee_targets_csv"], project_root)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)


def load_per_sample_features(sample_index_csv: str | Path = "data/processed/sample_index.csv") -> pd.DataFrame:
    """Load and stack all per-sample gee_features.csv files."""

    return _load_per_sample_table(sample_index_csv, "gee_features_csv", FEATURE_COLUMNS)


def load_per_sample_targets(sample_index_csv: str | Path = "data/processed/sample_index.csv") -> pd.DataFrame:
    """Load and stack all per-sample gee_targets.csv files."""

    return _load_per_sample_table(sample_index_csv, "gee_targets_csv", TARGET_COLUMNS)


def _load_per_sample_table(sample_index_csv: str | Path, path_column: str, columns: list[str]) -> pd.DataFrame:
    """Load one kind of per-sample CSV listed in sample_index.csv."""

    sample_index = load_sample_index(sample_index_csv)
    project_root = sample_index.attrs.get("project_root", project_root_from_sample_index(sample_index_csv))
    frames = []
    for _, sample in sample_index.iterrows():
        path = resolve_project_path(sample[path_column], project_root)
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

def build_feature_metadata(sample: pd.Series, observations: pd.DataFrame, config: FeatureExtractionConfig) -> dict[str, Any]:
    """Build reproducibility metadata for a raw GEE observation file."""

    dates = pd.to_datetime(observations["acquisition_date"], errors="coerce").dropna()
    end_date = dates.max().date().isoformat() if len(dates) else ""
    s2_start = (dates.min().date() - timedelta(days=config.s2_lookback_days)).isoformat() if len(dates) else ""
    chirps_start = (dates.min().date() - timedelta(days=90)).isoformat() if len(dates) else ""
    era5_start = (dates.min().date() - timedelta(days=30)).isoformat() if len(dates) else ""
    return {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "gee_project_id": config.project_id,
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "s2_date_range": {"start": s2_start, "end": end_date},
        "s1_tolerance_days": config.s1_tolerance_days,
        "chirps_date_range": {"start": chirps_start, "end": end_date},
        "era5_date_range": {"start": era5_start, "end": end_date},
        "srtm_version": SRTM_DATASET,
        "dynamicworld_version": DYNAMICWORLD_DATASET,
        "buffer_radius_m": config.buffer_radius_m,
        "context_buffer_radius_m": config.context_buffer_radius_m,
    }


def assert_no_target_leakage(features: pd.DataFrame) -> None:
    """Fail if future or target columns appear in model-input features."""

    leaks = [c for c in features.columns if c.startswith("target_") or c.startswith("future_") or "delta_1" in c]
    if leaks:
        raise ValueError(f"Future target columns found in gee_features.csv: {leaks}")



def build_features_and_targets(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility alias for the local feature-engineering step."""
    return engineer_features_and_targets(observations, FeatureEngineeringConfig())


def extract_all_samples(
    config: FeatureExtractionConfig,
    sample_limit: int | None = None,
    ee_module: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deprecated compatibility wrapper. Does not write combined CSV files."""
    extract_raw_observations_from_gee(config, sample_limit=sample_limit, ee_module=ee_module)
    engineering_config = FeatureEngineeringConfig(
        data_dir=config.data_dir,
        sample_index_csv=config.sample_index_csv,
        force=True,
        verbose=config.verbose,
    )
    engineer_all_samples(engineering_config, sample_limit=sample_limit)
    return load_per_sample_features(config.sample_index_csv), load_per_sample_targets(config.sample_index_csv)

def _extract_s2(ee: Any, region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(S2_DATASET)
        .filterBounds(region)
        .filterDate(*_window(current_date, config.s2_lookback_days, 1))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(_mask_s2_clouds)
        .map(lambda image: _add_s2_indices(ee, image))
    )
    count = _safe_size(collection)
    if count == 0:
        return _nan_values(["ndvi_mean", "evi_mean", "ndwi_mean", "ndmi_mean"], {"s2_image_count": 0})
    stats = _reduce_mean(ee, collection.mean(), region, 20, ["ndvi", "evi", "ndwi", "ndmi"])
    return {"ndvi_mean": stats.get("ndvi"), "evi_mean": stats.get("evi"), "ndwi_mean": stats.get("ndwi"), "ndmi_mean": stats.get("ndmi"), "s2_image_count": count}


def _mask_s2_clouds(image: Any) -> Any:
    scl = image.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)


def _add_s2_indices(ee: Any, image: Any) -> Any:
    scaled = image.select(["B2", "B3", "B4", "B8", "B11"]).multiply(0.0001)
    blue = scaled.select("B2")
    green = scaled.select("B3")
    red = scaled.select("B4")
    nir = scaled.select("B8")
    swir = scaled.select("B11")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {"NIR": nir, "RED": red, "BLUE": blue}).rename("evi")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("ndwi")
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename("ndmi")
    return ee.Image.cat([ndvi, evi, ndwi, ndmi])


def _extract_chirps(ee: Any, region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = ee.ImageCollection(CHIRPS_DATASET).filterBounds(region)
    precip_30 = collection.filterDate(*_window(current_date, 30, 1)).select("precipitation")
    precip_90 = collection.filterDate(*_window(current_date, 90, 1)).select("precipitation")
    if _safe_size(precip_30) == 0:
        return _nan_values(["rainfall_30d_mm", "rainfall_90d_mm", "dry_days_30d", "heavy_rain_days_30d", "heavy_rain_days_90d"])
    dry_days = precip_30.map(lambda image: image.lt(config.dry_day_threshold_mm).rename("dry_days_30d")).sum()
    heavy_30 = precip_30.map(lambda image: image.gt(config.heavy_rain_threshold_mm).rename("heavy_rain_days_30d")).sum()
    heavy_90 = precip_90.map(lambda image: image.gt(config.heavy_rain_threshold_mm).rename("heavy_rain_days_90d")).sum()
    image = ee.Image.cat([
        precip_30.sum().rename("rainfall_30d_mm"),
        precip_90.sum().rename("rainfall_90d_mm"),
        dry_days,
        heavy_30,
        heavy_90,
    ])
    return _reduce_mean(ee, image, region, 5500, ["rainfall_30d_mm", "rainfall_90d_mm", "dry_days_30d", "heavy_rain_days_30d", "heavy_rain_days_90d"])


def _extract_era5(ee: Any, region: Any, current_date: date) -> dict[str, Any]:
    keys = ["temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_30d_m"]
    collection = ee.ImageCollection(ERA5_DATASET).filterBounds(region).filterDate(*_window(current_date, 30, 1))
    if _safe_size(collection) == 0:
        return _nan_values(keys)
    mean = collection.mean()
    temp_c = mean.select("temperature_2m").subtract(273.15).rename("temperature_2m_mean_c")
    dew_c = mean.select("dewpoint_temperature_2m").subtract(273.15)
    rh = ee.Image().expression("100 * (exp((17.625 * D) / (243.04 + D)) / exp((17.625 * T) / (243.04 + T)))", {"D": dew_c, "T": temp_c}).rename("relative_humidity_mean_pct")
    soil = mean.select("volumetric_soil_water_layer_1").rename("soil_water_layer1_mean")
    runoff = collection.select("surface_runoff_sum").sum().rename("surface_runoff_30d_m")
    return _reduce_mean(ee, ee.Image.cat([temp_c, rh, soil, runoff]), region, 9000, keys)


def _extract_terrain(ee: Any, region: Any, config: FeatureExtractionConfig) -> dict[str, Any]:
    elevation = ee.Image(SRTM_DATASET).select("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope_deg")
    stats = _reduce_mean(ee, ee.Image.cat([elevation.rename("elevation_m"), slope]), region, 30, ["elevation_m", "slope_deg"])
    elevation_value = stats.get("elevation_m")
    stats["lowland_flag"] = int(elevation_value is not None and elevation_value < config.lowland_elevation_threshold_m)
    return stats

def _extract_s1(ee: Any, region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(S1_DATASET)
        .filterBounds(region)
        .filterDate(*_window(current_date, config.s1_tolerance_days, config.s1_tolerance_days + 1))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    count = _safe_size(collection)
    if count == 0:
        return _nan_values(["vv_mean_db", "vh_mean_db", "vv_minus_vh_db"], {"s1_image_count": 0})
    mean = collection.mean()
    image = ee.Image.cat([
        mean.select("VV").rename("vv_mean_db"),
        mean.select("VH").rename("vh_mean_db"),
        mean.select("VV").subtract(mean.select("VH")).rename("vv_minus_vh_db"),
    ])
    stats = _reduce_mean(ee, image, region, 30, ["vv_mean_db", "vh_mean_db", "vv_minus_vh_db"])
    stats["s1_image_count"] = count
    return stats


def _extract_dynamic_world(ee: Any, region: Any, context_region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(DYNAMICWORLD_DATASET)
        .filterBounds(context_region)
        .filterDate(*_window(current_date, config.dynamicworld_lookback_days, 1))
        .select(["built", "flooded_vegetation"])
    )
    count = _safe_size(collection)
    if count == 0:
        return _nan_values(["built_probability_1km", "built_probability_5km", "flooded_vegetation_probability"], {"dynamicworld_image_count": 0})
    image = collection.mean()
    built_1km = _reduce_mean(ee, image.select("built"), region, 10, ["built"]).get("built")
    built_5km = _reduce_mean(ee, image.select("built"), context_region, 10, ["built"]).get("built")
    flooded = _reduce_mean(ee, image.select("flooded_vegetation"), region, 10, ["flooded_vegetation"]).get("flooded_vegetation")
    return {
        "built_probability_1km": built_1km,
        "built_probability_5km": built_5km,
        "flooded_vegetation_probability": flooded,
        "dynamicworld_image_count": count,
    }


def _extract_water(ee: Any, region: Any) -> dict[str, Any]:
    stats = _reduce_mean(ee, ee.Image(JRC_WATER_DATASET).select(["occurrence", "max_extent"]), region, 30, ["occurrence", "max_extent"])
    return {"water_occurrence_mean": stats.get("occurrence"), "max_water_extent_fraction": stats.get("max_extent")}


def _flood_risk_proxy(df: pd.DataFrame) -> pd.Series:
    """Combine terrain, water, rainfall, and SAR signals into a flood proxy."""

    heavy = _minmax(df.get("heavy_rain_days_90d"), df.index)
    water = _minmax(df.get("water_occurrence_mean"), df.index)
    flooded = _minmax(df.get("flooded_vegetation_probability"), df.index)
    lowland = df.get("lowland_flag", pd.Series(0, index=df.index)).fillna(0)
    return (0.35 * heavy + 0.25 * water + 0.25 * flooded + 0.15 * lowland).clip(0, 1)


def _minmax(values: pd.Series | None, index: pd.Index) -> pd.Series:
    """Scale a series to 0-1 while tolerating missing or constant values."""

    if values is None:
        return pd.Series(0.0, index=index)
    series = values.astype(float)
    min_value = series.min(skipna=True)
    max_value = series.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def _window(current_date: date, days_before: int, days_after: int) -> tuple[str, str]:
    """Return an ISO date window around an acquisition date."""

    return (current_date - timedelta(days=days_before)).isoformat(), (current_date + timedelta(days=days_after)).isoformat()


def _safe_size(collection: Any) -> int:
    """Return an Earth Engine collection size, falling back to zero."""

    try:
        return int(collection.size().getInfo())
    except Exception:
        return 0


def _reduce_mean(ee: Any, image: Any, region: Any, scale: int, keys: list[str]) -> dict[str, float | None]:
    """Reduce selected Earth Engine bands to regional mean values."""

    try:
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
        ).getInfo()
    except Exception:
        values = {}
    result = {}
    for key in keys:
        value = values.get(key)
        try:
            result[key] = None if value is None else float(value)
        except (TypeError, ValueError):
            result[key] = None
    return result


def _nan_values(keys: list[str], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a dictionary of NaN placeholders plus optional overrides."""

    values = {key: np.nan for key in keys}
    if overrides:
        values.update(overrides)
    return values
