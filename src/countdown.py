"""Countdown replay: rescore the whole portfolio at each 6h timestep before
landfall, under the forecast-track scenarios in config hazard.scenarios.

Each scenario is an offset_fraction: the forecast track is the true track
displaced sideways (perpendicular to its direction of travel) by
offset_fraction * forecast_error(lead); the residual (the full
forecast_error, not the reduced remainder) is subtracted from distance as
radial uncertainty. Whenever offset_fraction <= 1.0 that residual buffer is
large enough to always recapture a truly-close village even when the offset
points the wrong way, so recall == 100% at every lead time by construction
(inland, seaward). forecast_miss uses offset_fraction > 1.0 - big enough that
the buffer can no longer fully compensate - to model the storm's true
position falling outside the forecast cone entirely; since which direction
is worse can vary by lead time, both signs are scored and the one with lower
recall (the worse one) is kept.

Weights, tiers, and the exposure formula itself are frozen (reused unchanged
from src.scoring / src.hazard) - only the track a village's distance is
measured against, and the residual error subtracted from that distance, vary
by lead time and scenario.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hazard import (  # noqa: E402
    METRIC_CRS,
    exposure_band_for_distance,
    exposure_columns_for_track,
    exposure_score_for_distance,
    load_track_points,
)
from src.scoring import score_dataframe  # noqa: E402

PROCESSED = ROOT / "data" / "processed"

# Every 6h from landfall-72h to landfall itself, descending (72 = earliest, 0 = landfall).
LEAD_TIMES_H = list(range(72, -1, -6))


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def forecast_error_km(lead_time_h, config=None):
    """Linear interpolation of hazard.forecast_error_km between its anchor
    points (hours before landfall -> km of assumed position uncertainty)."""
    config = config or load_config()
    table = config["hazard"]["forecast_error_km"]
    xs = sorted(table.keys())
    ys = [table[x] for x in xs]
    return float(np.interp(lead_time_h, xs, ys))


def _local_perpendiculars(coords):
    """Unit vectors perpendicular to the local direction of travel at each
    track point (central differences; forward/backward at the endpoints)."""
    tangents = np.zeros_like(coords, dtype=float)
    tangents[1:-1] = coords[2:] - coords[:-2]
    tangents[0] = coords[1] - coords[0]
    tangents[-1] = coords[-1] - coords[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms

    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def _inland_sign(perpendiculars):
    """Which sign of the perpendicular direction points inland (west, toward
    Odisha) vs seaward (east, toward the Bay of Bengal). Determined from the
    track geometry itself (mean perpendicular x-component), rather than
    assumed, so it's robust to whichever way Fani's track happens to curve."""
    return 1.0 if perpendiculars[:, 0].mean() < 0 else -1.0


def build_scenario_tracks(config=None):
    """Precompute, once, everything needed to shift the true track by any
    offset: its points' coordinates, wind_kt, and a fixed inland/seaward sign
    (independent of lead time - only the offset magnitude varies by lead)."""
    config = config or load_config()
    track_points = load_track_points(config)
    coords = np.array([(p.x, p.y) for p in track_points.geometry])
    perpendiculars = _local_perpendiculars(coords)
    inland_sign = _inland_sign(perpendiculars)
    return {
        "track_points": track_points,
        "coords": coords,
        "perpendiculars": perpendiculars,
        "inland_sign": inland_sign,
    }


def shifted_track_points(scenario_tracks, sign, offset_km):
    """The true track's points, shifted offset_km perpendicular to the local
    direction of travel (sign > 0 or < 0), keeping their original wind_kt.

    Public: also used by app.py to redraw the exact forecast track a frame
    was scored against (paired with resolve_offset's sign/offset_km).
    """
    shifted_coords = scenario_tracks["coords"] + scenario_tracks["perpendiculars"] * (offset_km * 1000) * sign

    shifted = scenario_tracks["track_points"].copy()
    shifted["geometry"] = [Point(x, y) for x, y in shifted_coords]
    return shifted


def load_base_villages(config=None):
    """Villages in METRIC_CRS, ready to measure distance against any track."""
    config = config or load_config()
    villages = gpd.read_file(PROCESSED / "villages.geojson")
    return villages.to_crs(METRIC_CRS)


def load_final_high_ids():
    """Ground-truth (landfall) high-priority borrower ids, straight from the
    already-scored portfolio - independent of any countdown run."""
    final = pd.read_csv(PROCESSED / "scored_portfolio.csv")
    return set(final.loc[final["tier"] == "high", "borrower_id"])


def _recall_pct(scored_df, final_high_ids):
    """% of final_high_ids already flagged high-or-medium in scored_df."""
    if not final_high_ids:
        return float("nan")
    flagged = set(scored_df.loc[scored_df["tier"].isin(["high", "medium"]), "borrower_id"])
    return 100 * len(final_high_ids & flagged) / len(final_high_ids)


def _score_with_sign(lead_time_h, offset_km, sign, portfolio, villages_metric, scenario_tracks, config, error_km):
    """Rescore the portfolio against the true track shifted by offset_km
    (along the given sign) and shrunk by the full residual error_km."""
    bands_km = config["hazard"]["bands_km"]
    forecast_track_points = shifted_track_points(scenario_tracks, sign, offset_km)
    distance_to_forecast, nearest_wind_kt, track_peak_wind_kt = exposure_columns_for_track(
        villages_metric, forecast_track_points
    )

    villages = villages_metric.copy()
    # Carried through as "distance_km" too: score_dataframe's reasons-text
    # builder reads that column name; here it holds the effective (shrunk,
    # forecast-track-relative) distance rather than the true one.
    villages["distance_km"] = (distance_to_forecast - error_km).clip(lower=0)
    villages["exposure_band"] = villages["distance_km"].apply(
        lambda d: exposure_band_for_distance(d, bands_km)
    )
    villages["nearest_wind_kt"] = nearest_wind_kt
    villages["exposure_score"] = exposure_score_for_distance(
        villages["distance_km"], villages["nearest_wind_kt"], track_peak_wind_kt, bands_km
    )

    village_cols = villages[["village_id", "distance_km", "exposure_band", "nearest_wind_kt", "exposure_score"]]
    df = portfolio.merge(village_cols, on="village_id", how="left")
    return score_dataframe(df, config)


def resolve_offset(lead_time_h, scenario, portfolio, villages_metric, scenario_tracks, config, final_high_ids):
    """The (offset_km, sign, error_km, scored_df) actually used to score one
    (lead_time, scenario) combination.

    inland/seaward have one fixed sign each, so scored_df is None (the
    caller can score cheaply with a single _score_with_sign call).
    forecast_miss tries both signs and keeps whichever produces lower recall
    (the worse case); scored_df is that winning frame, returned so the
    caller (rescore_at) doesn't have to redo the rescore.

    Public: also lets app.py redraw the exact forecast track/cone a given
    frame was scored against, without duplicating the "try both signs, keep
    the worse" selection logic.
    """
    offset_fraction = config["hazard"]["scenarios"][scenario]
    error_km = forecast_error_km(lead_time_h, config)
    offset_km = offset_fraction * error_km
    inland_sign = scenario_tracks["inland_sign"]

    if scenario == "inland":
        return offset_km, inland_sign, error_km, None
    if scenario == "seaward":
        return offset_km, -inland_sign, error_km, None

    candidates = [
        (sign, _score_with_sign(lead_time_h, offset_km, sign, portfolio, villages_metric, scenario_tracks, config, error_km))
        for sign in (inland_sign, -inland_sign)
    ]
    sign, scored_df = min(candidates, key=lambda pair: _recall_pct(pair[1], final_high_ids))
    return offset_km, sign, error_km, scored_df


def rescore_at(lead_time_h, scenario, portfolio, villages_metric, scenario_tracks, config, final_high_ids):
    """Rescore every borrower under one (lead_time, scenario) combination."""
    offset_km, sign, error_km, cached_scored = resolve_offset(
        lead_time_h, scenario, portfolio, villages_metric, scenario_tracks, config, final_high_ids
    )
    scored_df = cached_scored if cached_scored is not None else _score_with_sign(
        lead_time_h, offset_km, sign, portfolio, villages_metric, scenario_tracks, config, error_km
    )
    scored = scored_df.copy()
    scored["timestep"] = lead_time_h
    scored["scenario"] = scenario
    return scored


def run_countdown(config=None):
    """Rescore the full portfolio at every (timestep, scenario) combination;
    return one long DataFrame with 'timestep' and 'scenario' columns."""
    config = config or load_config()
    portfolio = pd.read_csv(PROCESSED / "portfolio.csv")
    villages_metric = load_base_villages(config)
    scenario_tracks = build_scenario_tracks(config)
    final_high_ids = load_final_high_ids()
    scenarios = list(config["hazard"]["scenarios"].keys())

    frames = [
        rescore_at(lead, scenario, portfolio, villages_metric, scenario_tracks, config, final_high_ids)
        for lead in LEAD_TIMES_H
        for scenario in scenarios
    ]
    return pd.concat(frames, ignore_index=True)


def summarize(countdown_df):
    """Per (lead_time, scenario): high/medium counts, plus three precisely-
    defined CONSISTENCY-WITH-LANDFALL metrics against the T-0h/inland High
    set (High_T0). Not forecast "accuracy" in any absolute sense - the T-0h
    read is itself only ever an estimate, not ground truth.

    Let High_t / Medium_t be this lead time's tiers, High_T0 the final set:
      - recall_pct (LENIENT recall)   = |High_T0 ∩ (High_t ∪ Medium_t)| / |High_T0|
        Of borrowers who end up High, what % were already flagged High-or-
        Medium at this lead time. This is the only variant the app showed
        before this fix (used unlabeled as "recall").
      - strict_recall_pct (STRICT recall) = |High_T0 ∩ High_t| / |High_T0|
        Of borrowers who end up High, what % were already High (not just
        Medium) at this lead time. Always <= lenient recall.
      - strict_precision_pct (STRICT precision) = |High_t ∩ High_T0| / |High_t|
        Of borrowers flagged High at this lead time, what % end up High.
        The app's previous "precision" instead used High_T0 ∪ Medium_T0 as
        the target set (i.e. it credited a t-High borrower who only ends up
        Medium, not High, as a match) - that inflated precision versus this
        strict definition and has been replaced.
    """
    final = countdown_df[(countdown_df["timestep"] == 0) & (countdown_df["scenario"] == "inland")]
    final_tier = final.set_index("borrower_id")["tier"]
    final_high_ids = set(final_tier[final_tier == "high"].index)

    scenarios = list(dict.fromkeys(countdown_df["scenario"]))  # preserve first-seen order

    rows = []
    for lead in LEAD_TIMES_H:
        for scenario in scenarios:
            step = countdown_df[(countdown_df["timestep"] == lead) & (countdown_df["scenario"] == scenario)]

            high_ids = set(step.loc[step["tier"] == "high", "borrower_id"])
            medium_ids = set(step.loc[step["tier"] == "medium", "borrower_id"])
            high_or_medium_ids = high_ids | medium_ids

            recall_pct = (
                100 * len(final_high_ids & high_or_medium_ids) / len(final_high_ids) if final_high_ids else float("nan")
            )
            strict_recall_pct = (
                100 * len(final_high_ids & high_ids) / len(final_high_ids) if final_high_ids else float("nan")
            )
            strict_precision_pct = (
                100 * len(high_ids & final_high_ids) / len(high_ids) if high_ids else float("nan")
            )

            rows.append(
                {
                    "lead_time_h": lead,
                    "scenario": scenario,
                    "high": len(high_ids),
                    "medium": len(medium_ids),
                    "recall_pct": round(recall_pct, 1),
                    "strict_recall_pct": round(strict_recall_pct, 1),
                    "strict_precision_pct": round(strict_precision_pct, 1),
                }
            )

    return pd.DataFrame(rows)


# Full column set persisted per (timestep, scenario, borrower_id) row so the
# dashboard can read one frame per lead/scenario with everything it needs
# (demographics, exposure, score components, tier, action, reasons) without
# re-joining against the frozen T-0h scored_portfolio.csv - each row here
# already carries its own timestep/scenario-specific scoring.
COUNTDOWN_COLUMNS = [
    "timestep", "scenario", "borrower_id", "village_id", "branch_id",
    "district", "subdivision", "lat", "lon", "occupation",
    "loan_amount", "emi", "months_paid", "outstanding", "missed_emis",
    "distance_km", "exposure_band", "nearest_wind_kt", "exposure_score",
    "vulnerability_score", "stress_score",
    "exposure_component", "vulnerability_component", "stress_component",
    "priority_score", "tier", "suggested_action", "reasons",
]


def main():
    config = load_config()
    countdown_df = run_countdown(config)

    out_path = PROCESSED / "countdown.parquet"
    countdown_df[COUNTDOWN_COLUMNS].to_parquet(out_path, index=False)
    print(f"[write] {len(countdown_df)} rows -> {out_path}")

    table = summarize(countdown_df)
    print("\nCountdown summary (lead time in hours before landfall):")
    print(table.to_string(index=False))

    worst_48 = table[table["lead_time_h"] == 48]["recall_pct"].min()
    worst_24 = table[table["lead_time_h"] == 24]["recall_pct"].min()
    print(f"\nWorst-case recall across scenarios at T-48h: {worst_48:.1f}%")
    print(f"Worst-case recall across scenarios at T-24h: {worst_24:.1f}%")


if __name__ == "__main__":
    main()
