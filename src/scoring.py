"""Priority scoring: combine hazard exposure, livelihood vulnerability, and
repayment stress into a per-borrower priority score, tier, and explanation."""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hazard import assign_exposure, assign_exposure_wind_radii  # noqa: E402
from src.labels import OCCUPATION_LABELS  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_inputs(config=None):
    """Portfolio joined to village-level distance_km / exposure_band /
    nearest_wind_kt / exposure_score.

    Exposure is computed once per village (not once per borrower) since every
    borrower at a village shares the same location.
    """
    config = config or load_config()
    portfolio = pd.read_csv(PROCESSED / "portfolio.csv")

    exposure_fn = (
        assign_exposure_wind_radii if config["hazard"].get("exposure_model") == "wind_radii" else assign_exposure
    )
    villages = gpd.read_file(PROCESSED / "villages.geojson")
    villages = exposure_fn(villages, config)
    exposure_cols = villages[
        ["village_id", "distance_km", "exposure_band", "nearest_wind_kt", "exposure_score"]
    ]

    return portfolio.merge(exposure_cols, on="village_id", how="left")


def _build_reasons(row, event_name, hazard_type, gate, reason_cfg):
    reasons = []

    if row["exposure_band"] in gate:
        reasons.append(
            f"{row['exposure_band'].capitalize()} exposure: {row['distance_km']:.0f} km "
            f"from {event_name}'s track, storm at ~{row['nearest_wind_kt']:.0f} kt nearby"
        )

    if row["vulnerability_score"] >= reason_cfg["vulnerability_high"]:
        label = OCCUPATION_LABELS.get(row["occupation"], str(row["occupation"]))
        reasons.append(f"Livelihood ({label}) highly sensitive to {hazard_type}s")

    if row["missed_emis"] >= reason_cfg["min_missed_emis"]:
        n = int(row["missed_emis"])
        reasons.append(f"Already missed {n} {'EMI' if n == 1 else 'EMIs'} before the event")

    return reasons


def score_dataframe(df, config=None):
    """Add vulnerability/stress/priority scoring, tier, action, and reasons.

    Expects df to already carry: occupation, missed_emis, distance_km,
    exposure_band, nearest_wind_kt, exposure_score (e.g. from load_inputs()).
    """
    config = config or load_config()
    df = df.copy()

    hazard_type = config["event"]["hazard_type"]
    weights = config["priority"]["weights"]
    gate = set(config["priority"]["exposure_gate"])
    tiers_cfg = config["priority"]["tiers"]
    actions_cfg = config["priority"]["actions"]
    reason_cfg = config["priority"]["reason_thresholds"]
    vulnerability_cfg = config["vulnerability"][hazard_type]
    stress_cfg = {int(k): v for k, v in config["repayment_stress"]["scores"].items()}

    df["vulnerability_score"] = df["occupation"].map(vulnerability_cfg)
    df["stress_score"] = df["missed_emis"].clip(upper=3).map(stress_cfg)

    df["exposure_component"] = weights["exposure"] * df["exposure_score"]
    df["vulnerability_component"] = weights["vulnerability"] * df["vulnerability_score"]
    df["stress_component"] = weights["stress"] * df["stress_score"]
    df["priority_score"] = (
        df["exposure_component"] + df["vulnerability_component"] + df["stress_component"]
    )

    def tier_for(row):
        if row["exposure_band"] not in gate:
            return "not_exposed"
        if row["priority_score"] >= tiers_cfg["high"]:
            return "high"
        if row["priority_score"] >= tiers_cfg["medium"]:
            return "medium"
        return "low"

    df["tier"] = df.apply(tier_for, axis=1)
    df["suggested_action"] = df["tier"].map(actions_cfg)

    event_name = config["event"]["name"]
    df["reasons"] = df.apply(
        lambda row: "; ".join(_build_reasons(row, event_name, hazard_type, gate, reason_cfg)),
        axis=1,
    )
    return df


def _print_summary(df, config):
    total = len(df)
    exposed = df["tier"] != "not_exposed"

    print(f"\nTotal borrowers: {total}")
    print(f"Exposed (band in {config['priority']['exposure_gate']}): {exposed.sum()}")

    print("\nCount per tier:")
    tier_counts = df["tier"].value_counts().reindex(["high", "medium", "low", "not_exposed"], fill_value=0)
    tier_pct = (100 * tier_counts / total).round(1)
    print(pd.DataFrame({"count": tier_counts, "pct": tier_pct}).to_string())

    high_scores = df.loc[df["tier"] == "high", "priority_score"]
    if not high_scores.empty:
        p50, p90, p99 = high_scores.quantile([0.5, 0.9, 0.99])
        print("\npriority_score percentiles within 'high' tier:")
        print(f"  p50: {p50:.3f}  p90: {p90:.3f}  p99: {p99:.3f}  max: {high_scores.max():.3f}")

    exposed_cr = df.loc[exposed, "outstanding"].sum() / 1e7
    high_cr = df.loc[df["tier"] == "high", "outstanding"].sum() / 1e7
    print(f"\nOutstanding - exposed borrowers: Rs {exposed_cr:.2f} Cr")
    print(f"Outstanding - high-priority borrowers: Rs {high_cr:.2f} Cr")

    print("\nTier counts by district:")
    print(pd.crosstab(df["district"], df["tier"]).to_string())

    print("\nOccupation mix - high tier vs. all exposed:")
    high_mix = df.loc[df["tier"] == "high", "occupation"].value_counts(normalize=True)
    exposed_mix = df.loc[exposed, "occupation"].value_counts(normalize=True)
    mix = pd.DataFrame({"high_tier": high_mix, "all_exposed": exposed_mix}).fillna(0).round(3)
    print(mix.to_string())

    print("\nTop 10 by priority score:")
    top10 = df.head(10)[
        ["borrower_id", "district", "subdivision", "occupation", "priority_score", "tier", "reasons"]
    ]
    print(top10.to_string(index=False))


def main():
    config = load_config()
    df = load_inputs(config)
    scored = score_dataframe(df, config)
    scored = scored.sort_values("priority_score", ascending=False).reset_index(drop=True)

    out_path = PROCESSED / "scored_portfolio.csv"
    scored.to_csv(out_path, index=False)
    print(f"[write] {len(scored)} scored borrowers -> {out_path}")

    _print_summary(scored, config)


if __name__ == "__main__":
    main()
