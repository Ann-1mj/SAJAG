"""Synthetic portfolio: villages, branches, and borrower-level loan records."""
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

from src.regions import clean_subdivision_names, normalise_names  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
GEOGRAPHIC_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32645"  # UTM 45N, suitable for the Odisha coast


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_tehsils(config=None):
    config = config or load_config()
    gdf = gpd.read_file(PROCESSED / "odisha_tehsils.geojson")
    aliases = config["region"].get("name_aliases", {})
    gdf["NAME_2"] = normalise_names(gdf["NAME_2"], aliases)
    gdf = clean_subdivision_names(gdf)
    return gdf


def _sample_points_in_polygon(polygon, n, rng, max_batches=200):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    for _ in range(max_batches):
        if len(points) >= n:
            break
        xs = rng.uniform(minx, maxx, n)
        ys = rng.uniform(miny, maxy, n)
        for x, y in zip(xs, ys):
            if len(points) >= n:
                break
            p = Point(x, y)
            if polygon.contains(p):
                points.append(p)
    if len(points) < n:
        raise RuntimeError(f"could not sample {n} points inside polygon after {max_batches} batches")
    return points


def build_villages(tehsils, config, rng):
    """One row per sampled village: village_id, district, subdivision, geometry."""
    per_tehsil = config["portfolio"]["villages_per_tehsil"]
    rows = []
    for _, row in tehsils.iterrows():
        for point in _sample_points_in_polygon(row.geometry, per_tehsil, rng):
            rows.append({"district": row["NAME_2"], "subdivision": row["NAME_3"], "geometry": point})

    villages = gpd.GeoDataFrame(rows, crs=tehsils.crs)
    villages.insert(0, "village_id", [f"V{i + 1:04d}" for i in range(len(villages))])
    return villages


def build_branches(tehsils):
    """One branch per subdivision, at its (metric-computed) centroid."""
    centroids = tehsils.to_crs(METRIC_CRS).geometry.centroid
    centroids = gpd.GeoSeries(centroids, crs=METRIC_CRS).to_crs(GEOGRAPHIC_CRS)

    branches = tehsils[["NAME_2", "NAME_3"]].rename(columns={"NAME_2": "district", "NAME_3": "subdivision"})
    branches = branches.reset_index(drop=True)
    branches["lat"] = centroids.y.values
    branches["lon"] = centroids.x.values
    branches.insert(0, "branch_id", [f"BR{i + 1:03d}" for i in range(len(branches))])
    return branches


def _occupation_mixes(config):
    """Coastal keeps the configured mix; inland drops fisherfolk, redistributed
    proportionally to farmer + dairy_livestock (the two other land-based occupations)."""
    coastal = dict(config["portfolio"]["occupation_mix"])

    inland = dict(coastal)
    fisher_share = inland.pop("fisherfolk")
    land_base = inland["farmer"] + inland["dairy_livestock"]
    inland["farmer"] += fisher_share * inland["farmer"] / land_base
    inland["dairy_livestock"] += fisher_share * inland["dairy_livestock"] / land_base
    inland["fisherfolk"] = 0.0
    return coastal, inland


def build_borrowers(villages, branches, config, rng):
    portfolio_cfg = config["portfolio"]
    n = portfolio_cfg["n_borrowers"]
    tenure_months = portfolio_cfg["tenure_months"]
    coastal_districts = set(portfolio_cfg["coastal_districts"])

    assigned = villages.iloc[rng.integers(0, len(villages), size=n)].reset_index(drop=True)
    is_coastal = assigned["district"].isin(coastal_districts).to_numpy()

    coastal_mix, inland_mix = _occupation_mixes(config)
    occ_names = list(coastal_mix.keys())
    occupations = np.empty(n, dtype=object)
    if is_coastal.any():
        occupations[is_coastal] = rng.choice(
            occ_names, size=is_coastal.sum(), p=[coastal_mix[o] for o in occ_names]
        )
    if (~is_coastal).any():
        occupations[~is_coastal] = rng.choice(
            occ_names, size=(~is_coastal).sum(), p=[inland_mix[o] for o in occ_names]
        )

    loan_min = portfolio_cfg["loan_amount_inr"]["min"]
    loan_max = portfolio_cfg["loan_amount_inr"]["max"]
    skew = rng.beta(2, 5, size=n)  # skewed toward the lower end of the range
    loan_amount = np.round((loan_min + skew * (loan_max - loan_min)) / 500) * 500

    # No interest rate is modelled (none is in config.yaml yet); flat EMI over tenure.
    emi = np.round(loan_amount / tenure_months)
    months_paid = rng.integers(1, tenure_months, size=n)

    missed_dist = portfolio_cfg["missed_emi_distribution"]
    missed_emis = rng.choice([int(k) for k in missed_dist], size=n, p=list(missed_dist.values()))
    missed_emis = np.minimum(missed_emis, months_paid)  # can't miss more EMIs than months elapsed

    # Only (months_paid - missed_emis) EMIs actually reduced principal; missed ones are still owed.
    outstanding = np.maximum(loan_amount - emi * (months_paid - missed_emis), 0)

    branch_lookup = {
        (row.district, row.subdivision): row.branch_id for row in branches.itertuples(index=False)
    }
    branch_ids = [
        branch_lookup[(d, s)] for d, s in zip(assigned["district"], assigned["subdivision"])
    ]

    return pd.DataFrame({
        "borrower_id": [f"B{i + 1:04d}" for i in range(n)],
        "village_id": assigned["village_id"].values,
        "branch_id": branch_ids,
        "district": assigned["district"].values,
        "subdivision": assigned["subdivision"].values,
        "lat": assigned.geometry.y.values,
        "lon": assigned.geometry.x.values,
        "occupation": occupations,
        "loan_amount": loan_amount,
        "emi": emi,
        "months_paid": months_paid,
        "outstanding": outstanding,
        "missed_emis": missed_emis,
    })


def generate_portfolio(config=None, tehsils=None):
    """Run the full, seeded pipeline: villages -> branches -> borrowers."""
    config = config or load_config()
    tehsils = tehsils if tehsils is not None else load_tehsils(config)
    rng = np.random.default_rng(config["portfolio"]["seed"])

    villages = build_villages(tehsils, config, rng)
    branches = build_branches(tehsils)
    borrowers = build_borrowers(villages, branches, config, rng)
    return villages, branches, borrowers


def _print_validation(borrowers, config):
    coastal_districts = set(config["portfolio"]["coastal_districts"])
    is_coastal = borrowers["district"].isin(coastal_districts)

    print("\nBorrowers per district:")
    print(borrowers["district"].value_counts().to_string())

    print("\nOccupation mix - coastal:")
    print(borrowers.loc[is_coastal, "occupation"].value_counts(normalize=True).round(3).to_string())
    print("\nOccupation mix - inland:")
    print(borrowers.loc[~is_coastal, "occupation"].value_counts(normalize=True).round(3).to_string())

    print("\nMissed EMI shares (actual vs config):")
    actual = borrowers["missed_emis"].value_counts(normalize=True).sort_index()
    expected = pd.Series({int(k): v for k, v in config["portfolio"]["missed_emi_distribution"].items()})
    print(pd.DataFrame({"actual": actual, "expected": expected}).round(3).to_string())

    total_outstanding_cr = borrowers["outstanding"].sum() / 1e7
    print(f"\nTotal outstanding: Rs {total_outstanding_cr:.2f} Cr")


def _plot_portfolio_check(tehsils, villages, branches):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    tehsils.boundary.plot(ax=ax, color="#999999", linewidth=0.6)
    villages.plot(ax=ax, color="#8ecae6", markersize=8, label="village")
    ax.scatter(branches["lon"], branches["lat"], color="#d62828", marker="^", s=60, label="branch", zorder=3)

    ax.set_title("Portfolio check: subdivisions, villages, branches")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    out_path = PROCESSED / "portfolio_check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[write] {out_path}")


def main():
    config = load_config()
    tehsils = load_tehsils(config)
    villages, branches, borrowers = generate_portfolio(config, tehsils)

    villages.to_file(PROCESSED / "villages.geojson", driver="GeoJSON")
    print(f"[write] {len(villages)} villages -> {PROCESSED / 'villages.geojson'}")

    branches.to_csv(PROCESSED / "branches.csv", index=False)
    print(f"[write] {len(branches)} branches -> {PROCESSED / 'branches.csv'}")

    borrowers.to_csv(PROCESSED / "portfolio.csv", index=False)
    print(f"[write] {len(borrowers)} borrowers -> {PROCESSED / 'portfolio.csv'}")

    _print_validation(borrowers, config)
    _plot_portfolio_check(tehsils, villages, branches)


if __name__ == "__main__":
    main()
