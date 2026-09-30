"""Hazard layer: cyclone track loading and distance/intensity exposure scoring."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

GEOGRAPHIC_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32645"  # UTM 45N, suitable for the Odisha coast


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_track_points(config=None):
    """Track points with wind >= hazard.min_wind_kt (the same subset used to
    build the hazard line), each carrying a wind_kt (WMO_WIND, falling back to
    USA_WIND where missing), its ISO_TIME, and geometry in METRIC_CRS."""
    config = config or load_config()
    min_wind_kt = config.get("hazard", {}).get("min_wind_kt", 64)

    df = pd.read_csv(PROCESSED / "fani_track.csv").sort_values("ISO_TIME")
    df = df[df["USA_WIND"] >= min_wind_kt].copy()
    if len(df) < 2:
        raise ValueError(f"Not enough track points with wind >= {min_wind_kt} kt to build a line")

    df["wind_kt"] = df["WMO_WIND"].fillna(df["USA_WIND"])

    points = gpd.GeoDataFrame(
        df[["wind_kt", "ISO_TIME"]].reset_index(drop=True),
        geometry=gpd.points_from_xy(df["LON"], df["LAT"]),
        crs=GEOGRAPHIC_CRS,
    )
    return points.to_crs(METRIC_CRS)


def load_track(config=None):
    """Return a single-row GeoDataFrame with the hazard LineString in METRIC_CRS."""
    config = config or load_config()
    min_wind_kt = config.get("hazard", {}).get("min_wind_kt", 64)

    points = load_track_points(config)
    line = LineString([(p.x, p.y) for p in points.geometry])
    track = gpd.GeoDataFrame(
        {"name": [config["event"]["name"]], "min_wind_kt": [min_wind_kt]},
        geometry=[line],
        crs=points.crs,
    )
    return track


def exposure_band_for_distance(distance_km, bands_km):
    """The categorical band (high/medium/low) for a given distance (km) from
    the track. Pulled out of assign_exposure so src/countdown.py can re-derive
    bands from a hypothetical (forecast-error-adjusted) distance without
    duplicating - and risking drifting from - this threshold logic."""
    if distance_km <= bands_km["high"]:
        return "high"
    if distance_km <= bands_km["medium"]:
        return "medium"
    return "low"


def exposure_score_for_distance(distance_km, nearest_wind_kt, peak_wind_kt, bands_km):
    """The continuous, intensity-weighted exposure score for a given distance
    (km) from the track. See assign_exposure's docstring for the formula's
    rationale. Also reused by src/countdown.py for the same reason as above."""
    distance_factor = np.clip(1 - distance_km / bands_km["medium"], 0, None)
    return distance_factor * (nearest_wind_kt / peak_wind_kt)


def exposure_columns_for_track(points_metric, track_points):
    """Core geometry: distance_km, nearest_wind_kt, and track_peak_wind_kt of
    points_metric (already in METRIC_CRS) against an arbitrary track_points
    GeoDataFrame (METRIC_CRS, with a wind_kt column).

    Shared by assign_exposure() (against the true, realised track) and
    src/countdown.py (against a hypothetical shifted/forecast track) so the
    distance-and-nearest-wind math has exactly one implementation regardless
    of which track it's measured against.
    """
    track_line = LineString([(p.x, p.y) for p in track_points.geometry])
    track_peak_wind_kt = track_points["wind_kt"].max()

    distance_km = points_metric.geometry.distance(track_line) / 1000.0

    track_coords = np.array([(p.x, p.y) for p in track_points.geometry])
    point_coords = np.array([(p.x, p.y) for p in points_metric.geometry])
    pairwise_dist = np.hypot(
        point_coords[:, None, 0] - track_coords[None, :, 0],
        point_coords[:, None, 1] - track_coords[None, :, 1],
    )
    nearest_wind_kt = track_points["wind_kt"].to_numpy()[pairwise_dist.argmin(axis=1)]

    return distance_km, nearest_wind_kt, track_peak_wind_kt


def assign_exposure(points_gdf, config=None):
    """Return a copy of points_gdf with distance_km, exposure_band,
    nearest_wind_kt, track_peak_wind_kt, exposure_score.

    exposure_band (high/medium/low) is categorical and purely distance-based;
    it's used for the priority gate and for labels. exposure_score is
    intensity-weighted: the distance falloff (max(0, 1 - distance_km /
    bands_km.medium)) times how strong the storm was at the nearest track
    point, relative to the track's own peak wind (track_peak_wind_kt).

    This is a fixed reference: it comes only from the track data, not from
    whichever points happen to be passed in. Normalising against the max seen
    in the *input* would make a point's score depend on what else was in the
    same batch - the same village could score differently depending on
    whether it was scored alongside the whole portfolio or on its own. A
    fixed reference keeps exposure_score for a given location stable no
    matter what subset of points assign_exposure is called on.

    Accepts points in any CRS; the CRS conversion needed to measure distance
    against the (metric) track happens internally.
    """
    config = config or load_config()
    bands_km = config["hazard"]["bands_km"]

    track_points = load_track_points(config)
    points_metric = points_gdf.to_crs(METRIC_CRS)
    distance_km, nearest_wind_kt, track_peak_wind_kt = exposure_columns_for_track(
        points_metric, track_points
    )

    result = points_gdf.copy()
    result["distance_km"] = distance_km.values
    result["exposure_band"] = result["distance_km"].apply(
        lambda d: exposure_band_for_distance(d, bands_km)
    )
    result["nearest_wind_kt"] = nearest_wind_kt
    result["track_peak_wind_kt"] = track_peak_wind_kt
    result["exposure_score"] = exposure_score_for_distance(
        result["distance_km"], result["nearest_wind_kt"], track_peak_wind_kt, bands_km
    )
    return result


NM_TO_KM = 1.852  # IBTrACS wind radii are reported in nautical miles

# Compass-bearing ranges (degrees, clockwise from north) matching IBTrACS's
# quadrant convention: NE = the arc from due north to due east, etc.
RADII_QUADRANTS = ("NE", "SE", "SW", "NW")
_QUADRANT_BEARING_RANGES = {"NE": (0, 90), "SE": (90, 180), "SW": (180, 270), "NW": (270, 360)}


def load_radii_track_points(config=None):
    """Every track point that reports at least one quadrant's R34 (>=34kt
    gale) radius, in METRIC_CRS - deliberately NOT filtered to
    hazard.min_wind_kt like load_track_points(): R34 radii exist well below
    the 64kt threshold used to build the hazard line, and dropping those
    weaker-stage points would understate how far gale-force winds actually
    reached. Used only by assign_exposure_wind_radii(); the hazard line and
    Countdown's forecast-track shifting keep using load_track_points().
    """
    config = config or load_config()
    r34_cols = [f"USA_R34_{q}" for q in RADII_QUADRANTS]
    radii_cols = r34_cols + [f"USA_R50_{q}" for q in RADII_QUADRANTS] + [f"USA_R64_{q}" for q in RADII_QUADRANTS]

    df = pd.read_csv(PROCESSED / "fani_track.csv").sort_values("ISO_TIME")
    df = df[df[r34_cols].notna().any(axis=1)].copy()
    df["wind_kt"] = df["WMO_WIND"].fillna(df["USA_WIND"])

    points = gpd.GeoDataFrame(
        df[["wind_kt", "ISO_TIME", *radii_cols]].reset_index(drop=True),
        geometry=gpd.points_from_xy(df["LON"], df["LAT"]),
        crs=GEOGRAPHIC_CRS,
    )
    return points.to_crs(METRIC_CRS)


def _wind_radii_swath(points_metric, radii_track_points):
    """For each point in points_metric, across every radii-track-point and
    compass quadrant with a reported radius: ever_high = ever inside that
    quadrant's R64 (>=64kt, hurricane-force) radius; ever_medium = ever
    inside R34 (>=34kt, gale-force) but never R64; best_ratio = the minimum
    distance_km / R34_km seen (the closest normalised approach to any gale
    radius edge - 0 means dead-center, 1 means exactly at the edge). A point
    never within any reported R34 radius keeps best_ratio == inf (it was
    never under gale-force winds at any point in the storm's life) - the
    wind-radii equivalent of "far away" in the circular model, except
    grounded in the storm's actual measured wind field instead of a fixed
    120km cutoff.
    """
    track_coords = np.array([(p.x, p.y) for p in radii_track_points.geometry])
    point_coords = np.array([(p.x, p.y) for p in points_metric.geometry])

    dx = point_coords[:, None, 0] - track_coords[None, :, 0]  # (n_points, n_track)
    dy = point_coords[:, None, 1] - track_coords[None, :, 1]
    dist_km = np.hypot(dx, dy) / 1000.0
    bearing_deg = np.degrees(np.arctan2(dx, dy)) % 360  # 0=north, 90=east, clockwise

    n_points = point_coords.shape[0]
    ever_high = np.zeros(n_points, dtype=bool)
    ever_medium = np.zeros(n_points, dtype=bool)
    best_ratio = np.full(n_points, np.inf)

    for j, (_, row) in enumerate(radii_track_points.iterrows()):
        for quadrant, (lo, hi) in _QUADRANT_BEARING_RANGES.items():
            in_quad = (bearing_deg[:, j] >= lo) & (bearing_deg[:, j] < hi)
            if not in_quad.any():
                continue

            r64 = row.get(f"USA_R64_{quadrant}")
            if pd.notna(r64) and r64 > 0:
                r64_km = r64 * NM_TO_KM
                ever_high[in_quad & (dist_km[:, j] <= r64_km)] = True

            r34 = row.get(f"USA_R34_{quadrant}")
            if pd.notna(r34) and r34 > 0:
                r34_km = r34 * NM_TO_KM
                ratio = dist_km[:, j] / r34_km
                ever_medium[in_quad & (ratio <= 1.0)] = True
                improve = in_quad & (ratio < best_ratio)
                best_ratio[improve] = ratio[improve]

    return ever_high, ever_medium, best_ratio


def assign_exposure_wind_radii(points_gdf, config=None):
    """Alternative to assign_exposure(): quadrant-aware exposure using
    Fani's real per-quadrant wind radii (R34/R64, IBTrACS) instead of a flat
    circular distance band. exposure_band becomes physically grounded:
    `high` = ever inside the R64 (hurricane-force) radius in the
    village-to-storm-center quadrant at any point in the storm's life,
    `medium` = ever inside R34 (gale-force) but never R64, `low` = never
    under gale-force winds at all.

    distance_km and nearest_wind_kt are left exactly as assign_exposure()
    computes them (true closest-approach distance to the track line, wind at
    that nearest point) - they're display/reasons-text values, not part of
    the tier gate, and changing their meaning here would make the two
    models' "Why flagged?" text inconsistent for no reason. Only
    exposure_band and exposure_score change: exposure_score's distance
    falloff uses the matched R34 radius as its scale (max(0, 1 -
    best_ratio)) in place of the fixed bands_km.medium constant.
    """
    config = config or load_config()

    track_points = load_track_points(config)
    points_metric = points_gdf.to_crs(METRIC_CRS)
    distance_km, nearest_wind_kt, track_peak_wind_kt = exposure_columns_for_track(
        points_metric, track_points
    )

    radii_track_points = load_radii_track_points(config)
    ever_high, ever_medium, best_ratio = _wind_radii_swath(points_metric, radii_track_points)
    exposure_band = np.where(ever_high, "high", np.where(ever_medium, "medium", "low"))
    distance_factor = np.clip(1 - best_ratio, 0, None)  # best_ratio == inf -> factor 0

    result = points_gdf.copy()
    result["distance_km"] = distance_km.values if hasattr(distance_km, "values") else np.asarray(distance_km)
    result["exposure_band"] = exposure_band
    result["nearest_wind_kt"] = nearest_wind_kt
    result["track_peak_wind_kt"] = track_peak_wind_kt
    result["exposure_score"] = distance_factor * (nearest_wind_kt / track_peak_wind_kt)
    return result


def save_track_line():
    track = load_track().to_crs(GEOGRAPHIC_CRS)
    out_path = PROCESSED / "fani_track_line.geojson"
    track.to_file(out_path, driver="GeoJSON")
    print(f"[write] {out_path}")
    return out_path


def save_exposure_zones(config=None):
    """Buffer the hazard line at bands_km.medium and bands_km.high (meters, in
    METRIC_CRS) and save both rings to data/processed/exposure_zones.geojson.

    Row order is medium-then-high so a naive single-layer render already
    draws the smaller high-band disc on top of the larger medium-band one;
    the map still renders them as two separate layers for reliable z-order.
    """
    config = config or load_config()
    bands_km = config["hazard"]["bands_km"]
    track_line = load_track(config).geometry.iloc[0]

    zones = gpd.GeoDataFrame(
        {"band": ["medium", "high"]},
        geometry=[
            track_line.buffer(bands_km["medium"] * 1000),
            track_line.buffer(bands_km["high"] * 1000),
        ],
        crs=METRIC_CRS,
    ).to_crs(GEOGRAPHIC_CRS)

    out_path = PROCESSED / "exposure_zones.geojson"
    zones.to_file(out_path, driver="GeoJSON")
    print(f"[write] {out_path}")
    return out_path


def _tehsil_centroids():
    tehsils = gpd.read_file(PROCESSED / "odisha_tehsils.geojson")
    metric = tehsils.to_crs(METRIC_CRS)
    tehsils["geometry"] = metric.geometry.centroid
    return tehsils.set_geometry("geometry").set_crs(METRIC_CRS, allow_override=True).to_crs(GEOGRAPHIC_CRS)


def _plot_hazard_check(centroids):
    import matplotlib.pyplot as plt

    track = load_track().to_crs(GEOGRAPHIC_CRS)
    colors = {"high": "#d62828", "medium": "#f4a261", "low": "#8ecae6"}

    fig, ax = plt.subplots(figsize=(8, 8))
    track.plot(ax=ax, color="#1f4e79", linewidth=2, label="track (>= min wind)")
    for band, color in colors.items():
        subset = centroids[centroids["exposure_band"] == band]
        if not subset.empty:
            subset.plot(ax=ax, color=color, markersize=40, label=band)

    ax.set_title("Hazard check: tehsil centroids vs. cyclone track")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    out_path = PROCESSED / "hazard_check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[write] {out_path}")


def main():
    save_track_line()
    save_exposure_zones()

    centroids = assign_exposure(_tehsil_centroids())

    table = centroids[["NAME_2", "NAME_3", "distance_km", "exposure_band"]].rename(
        columns={"NAME_2": "district", "NAME_3": "subdivision"}
    )
    print(table.sort_values("distance_km").to_string(index=False))

    _plot_hazard_check(centroids)


if __name__ == "__main__":
    main()
