"""Phase 0 sanity check: plot the cyclone track over the selected tehsils."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def main():
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    tehsils = gpd.read_file(PROCESSED / "odisha_tehsils.geojson")
    track = pd.read_csv(PROCESSED / "fani_track.csv")

    affected = set(config["region"]["affected_districts"])
    is_affected = tehsils["NAME_2"].isin(affected)

    fig, ax = plt.subplots(figsize=(8, 8))
    tehsils[~is_affected].plot(ax=ax, color="#dddddd", edgecolor="white", label="other")
    tehsils[is_affected].plot(ax=ax, color="#f4a261", edgecolor="white", label="affected")
    ax.plot(track["LON"], track["LAT"], "-o", color="#1f4e79", markersize=3, linewidth=1.5, label=config["event"]["name"])

    ax.set_title(f"{config['event']['name']} ({config['event']['season']}) track over {config['region']['state']} tehsils")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    out_path = PROCESSED / "check_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
