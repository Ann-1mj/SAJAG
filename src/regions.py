"""Shared helpers for normalising district/subdivision names against config.yaml."""
import re

_PLACEHOLDER_NAME_RE = re.compile(r"^n\.a\.", re.IGNORECASE)
_ALT_NAME_SENTINELS = {"", "NA", "N.A", "N.A.", "NAN"}


def normalise_names(names, aliases):
    """Map raw shapefile district names to the canonical spelling used in config.yaml.

    `names` is a pandas/geopandas Series; `aliases` is config['region']['name_aliases']
    (raw spelling -> canonical spelling).
    """
    return names.replace(aliases)


def _first_real_altname(value):
    """Return the first usable name out of a (possibly '|'-separated) VARNAME_3
    value, or None if it's empty/a GADM 'no alt name' placeholder like 'NA'."""
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in _ALT_NAME_SENTINELS:
        return None
    return text.split("|")[0].strip() or None


def clean_subdivision_names(gdf, district_col="NAME_2", name_col="NAME_3", varname_col="VARNAME_3"):
    """Replace GADM placeholder subdivision names (e.g. 'n.a.(1507)') with a
    readable label: the first real VARNAME_3 alternate name if present, else
    '<District> - Subdivision N' (numbered per district, in row order).
    """
    gdf = gdf.copy()
    counters = {}
    cleaned = []
    for _, row in gdf.iterrows():
        name = str(row[name_col]).strip()
        if not _PLACEHOLDER_NAME_RE.match(name):
            cleaned.append(name)
            continue

        alt = _first_real_altname(row[varname_col]) if varname_col in gdf.columns else None
        if alt:
            cleaned.append(alt)
            continue

        district = row[district_col]
        counters[district] = counters.get(district, 0) + 1
        # Plain ASCII hyphen: pyogrio/GDAL on this Windows setup corrupts
        # non-ASCII characters (e.g. an en dash) written into GeoJSON properties.
        cleaned.append(f"{district} - Subdivision {counters[district]}")

    gdf[name_col] = cleaned
    return gdf
