"""Shared display labels for occupation and tier categories.

Used by src/scoring.py (reasons text) and the Streamlit dashboard (table,
filters, map legend) so labels and colors stay consistent everywhere they're
shown.
"""

OCCUPATION_LABELS = {
    "farmer": "Farmer",
    "dairy_livestock": "Dairy",
    "fisherfolk": "Fisherfolk",
    "kirana_shop": "Kirana shop",
    "tailoring_handicraft": "Tailoring/handicraft",
    "other_services": "Other services",
}

TIER_LABELS = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "not_exposed": "Not exposed",
}

# Status colors: reserved for the tier dimension only, consistent everywhere
# tier is shown (table, charts, map legend). "Calm ops room" palette.
TIER_COLORS = {
    "high": "#E5484D",
    "medium": "#F5A524",
    "low": "#5B6B7F",
    "not_exposed": "#3A4653",
}

EXPOSURE_BAND_LABELS = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}
