import pandas as pd

from src.scoring import load_config, load_inputs, score_dataframe

# Fani's historical peak (WMO_WIND, within the filtered track) — mirrors the
# peak_wind_kt hazard.assign_exposure() derives from the track data itself.
PEAK_WIND_KT = 115.0


def _synthetic_row(distance_km, exposure_band, config, occupation="farmer", missed_emis=0, nearest_wind_kt=PEAK_WIND_KT):
    bands_km = config["hazard"]["bands_km"]
    distance_factor = max(0.0, 1 - distance_km / bands_km["medium"])
    exposure_score = distance_factor * (nearest_wind_kt / PEAK_WIND_KT)
    return {
        "occupation": occupation,
        "missed_emis": missed_emis,
        "distance_km": distance_km,
        "exposure_band": exposure_band,
        "nearest_wind_kt": nearest_wind_kt,
        "exposure_score": exposure_score,
    }


def test_reference_districts_are_never_high_or_medium():
    """Far-inland reference districts (Koraput, Kalahandi) should never
    reach high/medium priority - but note this does NOT mean 'unaffected':
    under hazard.exposure_model: wind_radii some of their villages do fall
    within Fani's real gale-force wind swath and score 'low' (see
    tests/test_hazard.py's DEEP_INLAND fixture and config.yaml's comment on
    reference_districts)."""
    config = load_config()
    df = score_dataframe(load_inputs(config), config)

    reference_districts = config["region"]["reference_districts"]
    reference_tiers = df.loc[df["district"].isin(reference_districts), "tier"]
    assert not reference_tiers.isin(["high", "medium"]).any()


def test_high_exposure_farmer_with_two_missed_emis_is_high():
    config = load_config()
    row = _synthetic_row(0.0, "high", config, occupation="farmer", missed_emis=2)
    df = score_dataframe(pd.DataFrame([row]), config)

    assert df.loc[0, "tier"] == "high"


def test_far_away_borrower_with_three_missed_emis_is_not_exposed():
    config = load_config()
    row = _synthetic_row(300.0, "low", config, occupation="farmer", missed_emis=3)
    df = score_dataframe(pd.DataFrame([row]), config)

    assert df.loc[0, "tier"] == "not_exposed"
