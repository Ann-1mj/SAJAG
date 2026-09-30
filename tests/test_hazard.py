import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.hazard import PROCESSED, assign_exposure, assign_exposure_wind_radii

# 2019-05-03 00:00 UTC — a landfall-time vertex of the filtered track.
# USA_WIND=135 kt there, but WMO_WIND=105 kt is preferred when present.
ON_TRACK = (85.5, 19.1)
# Same latitude, ~300 km further inland/west of the track.
FAR_FROM_TRACK = (82.3, 19.1)
# This specific point (near Koraput) is confirmed never inside any reported
# R34 quadrant radius at any point in Fani's life - a genuine "never under
# gale-force winds" negative case for the wind-radii model. NOT a claim that
# all of Koraput is unaffected: other villages in the district do fall
# within Fani's real gale-force swath under this model (see config.yaml's
# comment on region.reference_districts).
DEEP_INLAND = (82.7, 18.8)


def _points_gdf():
    return gpd.GeoDataFrame(
        {"label": ["on_track", "far"]},
        geometry=[Point(*ON_TRACK), Point(*FAR_FROM_TRACK)],
        crs="EPSG:4326",
    )


def test_point_on_track_is_high_exposure():
    result = assign_exposure(_points_gdf())
    row = result[result["label"] == "on_track"].iloc[0]
    assert row["exposure_band"] == "high"


def test_point_300km_away_is_low_exposure():
    result = assign_exposure(_points_gdf())
    row = result[result["label"] == "far"].iloc[0]
    assert row["distance_km"] > 120
    assert row["exposure_band"] == "low"


def test_nearest_wind_kt_prefers_wmo_wind_over_usa_wind():
    result = assign_exposure(_points_gdf())
    row = result[result["label"] == "on_track"].iloc[0]
    assert row["nearest_wind_kt"] == 105.0


def test_exposure_score_is_zero_when_distance_factor_is_zero():
    result = assign_exposure(_points_gdf())
    row = result[result["label"] == "far"].iloc[0]
    # nearest_wind_kt is still positive here, but the distance factor clips to
    # 0 beyond bands_km.medium, so intensity can't pull the score back up.
    assert row["nearest_wind_kt"] > 0
    assert row["exposure_score"] == 0


def test_exposure_score_is_independent_of_subset():
    """assign_exposure on a subset of villages must give the same
    exposure_score for those villages as assign_exposure on the full set -
    a location's score shouldn't depend on what else was batched in with it.
    (Intensity normalises against the track's fixed peak, not the max seen
    in whatever points happen to be passed in.)"""
    villages = gpd.read_file(PROCESSED / "villages.geojson")
    full_result = assign_exposure(villages)

    subset = villages.iloc[[0, 5, 10]]
    subset_result = assign_exposure(subset)

    expected = full_result.loc[subset.index, "exposure_score"].to_numpy()
    actual = subset_result["exposure_score"].to_numpy()
    assert actual == pytest.approx(expected)


def _radii_points_gdf():
    return gpd.GeoDataFrame(
        {"label": ["on_track", "deep_inland"]},
        geometry=[Point(*ON_TRACK), Point(*DEEP_INLAND)],
        crs="EPSG:4326",
    )


def test_wind_radii_point_on_track_is_high_exposure():
    result = assign_exposure_wind_radii(_radii_points_gdf())
    row = result[result["label"] == "on_track"].iloc[0]
    assert row["exposure_band"] == "high"
    assert row["exposure_score"] > 0


def test_wind_radii_point_never_in_any_quadrant_radius_is_low():
    result = assign_exposure_wind_radii(_radii_points_gdf())
    row = result[result["label"] == "deep_inland"].iloc[0]
    assert row["exposure_band"] == "low"
    assert row["exposure_score"] == 0


def test_wind_radii_distance_and_wind_match_circle_model():
    """distance_km/nearest_wind_kt are deliberately left as the true
    closest-approach values (same as assign_exposure) - only exposure_band
    and exposure_score should differ between the two models."""
    circle = assign_exposure(_radii_points_gdf()).set_index("label")
    radii = assign_exposure_wind_radii(_radii_points_gdf()).set_index("label")
    assert circle["distance_km"].to_numpy() == pytest.approx(radii["distance_km"].to_numpy())
    assert circle["nearest_wind_kt"].to_numpy() == pytest.approx(radii["nearest_wind_kt"].to_numpy())
