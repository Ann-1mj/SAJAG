"""Phase 0 data prep: cyclone track (IBTrACS) and district/tehsil boundaries (GADM).

Reads region and event settings from config.yaml, downloads the raw sources into
data/raw/, and writes the filtered, analysis-ready files into data/processed/.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regions import clean_subdivision_names, normalise_names  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_3.json.zip"

TRACK_COLUMNS = [
    "SID", "NAME", "SEASON", "ISO_TIME", "NATURE",
    "LAT", "LON", "WMO_WIND", "USA_WIND", "USA_PRES", "STORM_SPEED", "STORM_DIR",
    # Per-quadrant wind radii (nautical miles) + radius of max wind - real
    # IBTrACS fields, previously left out of the processed extract. Powers
    # the wind-radii exposure model (src/hazard.assign_exposure_wind_radii),
    # a physically-grounded alternative to the flat circular distance bands.
    "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW",
    "USA_R50_NE", "USA_R50_SE", "USA_R50_SW", "USA_R50_NW",
    "USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW",
    "USA_RMW",
]


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def download(url, dest):
    if dest.exists():
        print(f"[skip] already have {dest}")
        return
    print(f"[download] {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def build_track_csv(config):
    raw_path = RAW / "ibtracs.NI.list.v04r01.csv"
    download(IBTRACS_URL, raw_path)

    df = pd.read_csv(raw_path, skiprows=[1], low_memory=False, na_values=[" ", ""])

    event = config["event"]
    track = df[
        (df["NAME"].str.upper() == event["ibtracs_name"].upper())
        & (df["SEASON"] == event["season"])
    ].copy()

    if track.empty:
        raise SystemExit(
            f"No IBTrACS rows found for NAME={event['ibtracs_name']!r} SEASON={event['season']!r}"
        )

    columns = [c for c in TRACK_COLUMNS if c in track.columns]
    track = track[columns]
    track["LAT"] = pd.to_numeric(track["LAT"], errors="coerce")
    track["LON"] = pd.to_numeric(track["LON"], errors="coerce")
    track = track.dropna(subset=["LAT", "LON"]).sort_values("ISO_TIME")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "fani_track.csv"
    track.to_csv(out_path, index=False)
    print(f"[write] {len(track)} track points -> {out_path}")


def build_tehsils_geojson(config):
    zip_path = RAW / "gadm41_IND_3.json.zip"
    download(GADM_URL, zip_path)

    json_path = RAW / "gadm41_IND_3.json"
    if not json_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW)

    gdf = gpd.read_file(json_path)

    region = config["region"]
    aliases = region.get("name_aliases", {})
    target_districts = set(region["affected_districts"]) | set(region["reference_districts"])

    unused_aliases = set(aliases.values()) - target_districts
    assert not unused_aliases, (
        f"name_aliases target(s) not in affected/reference districts: {sorted(unused_aliases)}"
    )

    gdf["NAME_2"] = normalise_names(gdf["NAME_2"], aliases)

    subset = gdf[
        (gdf["NAME_1"] == region["state"]) & (gdf["NAME_2"].isin(target_districts))
    ].copy()

    subdivision_counts = subset["NAME_2"].value_counts()
    missing = target_districts - set(subdivision_counts.index)
    assert not missing, f"districts missing from GADM output: {sorted(missing)}"
    assert (subdivision_counts >= 1).all(), "every matched district must have >= 1 subdivision"

    subset = clean_subdivision_names(subset)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "odisha_tehsils.geojson"
    subset.to_file(out_path, driver="GeoJSON")
    print(f"[write] {len(subset)} tehsils across {len(subdivision_counts)} districts -> {out_path}")


def main():
    config = load_config()
    build_track_csv(config)
    build_tehsils_geojson(config)


if __name__ == "__main__":
    main()
