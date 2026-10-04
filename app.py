"""SANKALP / SAJAG dashboard - landing page + 4-step app (Sense, Prioritize,
Act, Measure). "Calm ops room" visual design: dark blue-slate palette, IBM
Plex fonts, Plotly charts, one shared free folium dark basemap for every map.

This module only changes how things are drawn - every number still comes
from get_current_frame() / summarize_countdown() / compare_strategies() in
src/countdown.py and src/comparison.py, unchanged.
"""
import html
import json
import re
import sys
import time
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import LineString, Point
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hazard import METRIC_CRS, load_track_points  # noqa: E402
from src.labels import EXPOSURE_BAND_LABELS, OCCUPATION_LABELS, TIER_LABELS  # noqa: E402
from src.countdown import (  # noqa: E402
    build_scenario_tracks,
    load_base_villages,
    load_final_high_ids,
)
from src.countdown import load_config as load_countdown_config  # noqa: E402
from src.countdown import resolve_offset, shifted_track_points, summarize as summarize_countdown  # noqa: E402
from src.formatting import format_inr, format_inr_lakh_cr  # noqa: E402
from src.comparison import (  # noqa: E402
    DEFAULT_ASSUMPTIONS,
    breakeven_effectiveness_pct,
    compare_strategies,
    sensitivity_table,
)
from src.briefings import (  # noqa: E402
    OUTREACH_PATH,
    build_branch_facts,
    build_borrower_facts,
    template_branch_briefing,
    template_outreach_message,
)

PROCESSED = ROOT / "data" / "processed"

TIER_ORDER = ["high", "medium", "low", "not_exposed"]

# Midpoint between Puri (19.8135N, 85.8312E) and Bhubaneswar (20.2961N, 85.8245E).
MAP_INITIAL_VIEW = {"latitude": 20.0548, "longitude": 85.8279, "zoom": 7}
DARK_TILE_URL = "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
DARK_TILE_ATTR = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"

PURI_LATLON = (19.8135, 85.8312)  # Puri town - landfall time is derived from this

# "Calm ops room" design tokens.
BG = "#0E141B"
SURFACE = "#151D26"
TEXT = "#E6EAF0"
MUTED = "#8A96A6"
ACCENT = "#4CC3D9"
DIVIDER = "#232E3A"
GRID = "#232E3A"
RISK_PALETTE = {
    "high": "#C85A61",
    "medium": "#C49A45",
    "low": "#617A92",
    "not_exposed": "#4A5663",
}

# Landing-page refinement (landing only; the dashboard keeps the tokens above).
LANDING_BG = "#0B1219"
LANDING_SURFACE = "#111B24"
LANDING_BORDER = "#263541"
LANDING_TEXT = "#E8EEF3"
LANDING_MUTED = "#8295A8"
LANDING_CYAN = "#35C5E5"
LANDING_RED = "#E84B52"

COUNTDOWN_SCENARIO_LABELS = {"Inland": "inland", "Seaward": "seaward", "Forecast missed": "forecast_miss"}
COUNTDOWN_LINE_STYLE = {
    "inland": (ACCENT, "Inland"),
    "seaward": ("#3E9C7D", "Seaward"),
    "forecast_miss": (RISK_PALETTE["high"], "Forecast missed"),
}

TIMELINE_STOPS = (72, 48, 24, 12, 0)

# Individual-borrower jitter: fixed so a given borrower always lands in the same
# spot around its village regardless of which filters are currently applied.
JITTER_SEED = 42
JITTER_STD_DEG = 0.004  # ~400m, enough to fan borrowers out within a village dot

OUTREACH_LANGUAGE_LABELS = {"en": "English", "hi": "Hindi", "or": "Odia"}


def inject_custom_css():
    """Fonts, chrome removal, and every custom CSS class the app's HTML
    fragments below reference. Purely visual - no behavior here affects any
    computed value."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: "IBM Plex Sans", sans-serif;
        }}
        .stApp [data-testid="stAppViewContainer"] *,
        .stApp [data-testid="stSidebar"] * {{
            caret-color: transparent;
        }}
        .stApp [data-testid="stAppViewContainer"] input,
        .stApp [data-testid="stAppViewContainer"] textarea,
        .stApp [data-testid="stAppViewContainer"] select,
        .stApp [data-testid="stAppViewContainer"] [contenteditable="true"],
        .stApp [data-testid="stAppViewContainer"] [role="textbox"],
        .stApp [data-testid="stSidebar"] input,
        .stApp [data-testid="stSidebar"] textarea,
        .stApp [data-testid="stSidebar"] select,
        .stApp [data-testid="stSidebar"] [contenteditable="true"],
        .stApp [data-testid="stSidebar"] [role="textbox"] {{
            caret-color: auto;
        }}

        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stSidebarCollapsedControl"] {{
            visibility: hidden;
            height: 0;
        }}
        div.block-container {{
            padding-top: 0.6rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}
        /* Landing: container grows with the viewport (same 1400px at a ~1920px
           window) so zooming out widens the layout instead of leaving a fixed
           1400px column in the middle of a larger screen. */
        div.block-container:has([class*="st-key-sajag_landing_cta"]) {{
            max-width: min(100%, 1680px);
        }}
        /* App pages (not landing): one shared content container for header, nav,
           controls, KPIs, map panel and tables. */
        div.block-container:has([class*="st-key-sajag_topbar"]) {{
            max-width: 1680px;
            padding-left: clamp(1rem, 3vw, 3rem); padding-right: clamp(1rem, 3vw, 3rem);
        }}

        [data-testid="stMetricValue"], [data-testid="stDataFrame"], .sajag-mono {{
            font-family: "IBM Plex Mono", monospace;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED};
        }}
        [data-testid="stMetric"] {{ background: transparent; border: none; padding: 0; }}

        [data-testid="stExpander"] {{ border: 1px solid {DIVIDER}; border-radius: 8px; background: transparent; }}

        /* Top bar */
        [class*="st-key-sajag_topbar"] {{
            border-bottom: 1px solid {DIVIDER};
            padding: 6px 0 4px 0; margin-bottom: 4px; gap: 22px;
            background-image: repeating-radial-gradient(circle at 92% -30%, rgba(76,195,217,0.05) 0, rgba(76,195,217,0.05) 1px, transparent 1px, transparent 42px);
        }}
        /* Header: 1fr | auto | 1fr grid. The right track stays empty so the
           centre zone sits on the true horizontal centre. Tops are aligned so
           T-Xh reads on the same line as the SAJAG wordmark. */
        [class*="st-key-sajag_header"] {{ padding: 14px 0 0 0; }}
        [class*="st-key-sajag_header"] [data-testid="stHorizontalBlock"] {{
            display: grid !important;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            align-items: center; column-gap: 56px;
        }}
        [class*="st-key-sajag_header"] [data-testid="stColumn"],
        [class*="st-key-sajag_row2"] [data-testid="stColumn"] {{
            width: auto !important; min-width: 0 !important; flex: none !important;
        }}

        /* Row 2: compact vertical control on the left, timeline takes the rest */
        [class*="st-key-sajag_row2"] {{ padding: 6px 0 14px 0; }}
        [class*="st-key-sajag_row2"] [data-testid="stHorizontalBlock"]:has([class*="st-key-sajag_timeline"]) {{
            display: grid !important;
            grid-template-columns: auto minmax(0, 1fr);
            align-items: center; column-gap: 72px;
        }}

        .sajag-brand {{ display: flex; flex-direction: column; gap: 8px; }}
        [class*="st-key-sajag_header"] [data-testid="stVerticalBlock"] {{ gap: 8px; }}

        /* Clickable wordmark (st.button styled to match .sajag-wordmark) */
        .st-key-brand_home button {{
            background: transparent !important; border: none !important; box-shadow: none !important;
            padding: 0 !important; min-height: 0 !important; justify-content: flex-start;
            cursor: pointer; transition: opacity 150ms ease;
        }}
        .st-key-brand_home button p {{
            font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 1.9rem;
            line-height: 1; letter-spacing: 0.22em; color: {TEXT};
        }}
        .st-key-brand_home button:hover {{ opacity: 0.75; }}
        .st-key-brand_home button:focus-visible {{ outline: 1px solid rgba(76,195,217,0.5); outline-offset: 4px; }}
        .sajag-wordmark {{
            font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 1.9rem;
            line-height: 1; letter-spacing: 0.22em; color: {TEXT};
        }}
        .sajag-event {{ color: {MUTED}; font-size: 0.85rem; letter-spacing: 0; white-space: nowrap; }}
        .sajag-countdown {{
            font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 1.5rem;
            color: {TEXT}; font-variant-numeric: tabular-nums;
        }}
        /* Current-status block: compact, flat surface, no glow */
        .sajag-status {{
            display: flex; flex-direction: column; align-items: center;
            padding: 12px 32px 11px 32px;
            background: rgba(21, 29, 38, 0.6);
            border: 1px solid rgba(138, 150, 166, 0.13); border-radius: 8px;
        }}
        [class*="st-key-sajag_header"] .sajag-countdown {{
            font-size: 2.1rem; font-weight: 600; line-height: 1; letter-spacing: -0.01em; text-align: center;
        }}
        .sajag-status-sub {{
            margin-top: 7px; font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase;
            color: {MUTED}; white-space: nowrap;
        }}
        .sajag-demo-label {{
            color: {MUTED}; font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase;
            text-align: right; opacity: 0.55; padding: 2px 0 2px 0;
        }}

        /* Scenario segmented control */
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] {{
            display: inline-flex; flex-direction: column; align-items: stretch; width: auto; gap: 1px; padding: 3px;
            background: rgba(21, 29, 38, 0.5); border: 1px solid rgba(138, 150, 166, 0.12); border-radius: 7px;
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label {{
            display: flex; align-items: center; min-height: 26px;
            background: transparent; border: none; border-radius: 4px;
            padding: 0 14px !important; margin: 0 !important;
            transition: background-color 150ms ease;
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label p {{
            color: {MUTED} !important; font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase;
            white-space: nowrap; transition: color 150ms ease;
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.025); }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label:hover p {{ color: {TEXT} !important; }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label:has(input:checked) {{
            background: rgba(76,195,217,0.08);
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label:has(input:checked) p {{
            color: {ACCENT} !important; font-weight: 500;
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label div:first-child {{ display: none; }}

        /* Timeline: five connected stops spanning the full content width.
           Tracks 1fr 2fr 2fr 2fr 1fr put the stops at exactly 0/25/50/75/100%:
           the first stop is left-aligned in its track, the last right-aligned,
           and the middle three centred. */
        [class*="st-key-sajag_timeline"] {{ margin-top: 4px; }}
        [class*="st-key-sajag_timeline"] [data-testid="stHorizontalBlock"] {{
            display: grid !important; position: relative;
            grid-template-columns: 1fr 2fr 2fr 2fr 1fr; gap: 0 !important;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stColumn"] {{
            width: auto !important; min-width: 0 !important; flex: none !important;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stHorizontalBlock"]::before,
        [class*="st-key-sajag_timeline"] [data-testid="stHorizontalBlock"]::after {{
            content: ""; position: absolute; top: 8px; left: 4px; height: 1px; pointer-events: none;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stHorizontalBlock"]::before {{
            right: 4px; background: #1C252F;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stHorizontalBlock"]::after {{
            width: 0; background: rgba(76,195,217,0.35); transition: width 300ms ease;
        }}
        [class*="st-key-sajag_timeline"]:has(.st-key-tl_48 button[kind="primary"]) [data-testid="stHorizontalBlock"]::after {{ width: 25%; }}
        [class*="st-key-sajag_timeline"]:has(.st-key-tl_24 button[kind="primary"]) [data-testid="stHorizontalBlock"]::after {{ width: 50%; }}
        [class*="st-key-sajag_timeline"]:has(.st-key-tl_12 button[kind="primary"]) [data-testid="stHorizontalBlock"]::after {{ width: 75%; }}
        [class*="st-key-sajag_timeline"]:has(.st-key-tl_0 button[kind="primary"]) [data-testid="stHorizontalBlock"]::after {{ width: calc(100% - 8px); }}

        [class*="st-key-sajag_timeline"] [data-testid="stButton"] button {{
            position: relative; z-index: 1;
            background: transparent !important; border: none !important; box-shadow: none !important;
            border-radius: 0 !important; padding: 22px 4px 6px 4px !important; min-height: 0;
            transition: color 150ms ease;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stButton"] button p {{
            font-family: "IBM Plex Mono", monospace; font-size: 0.74rem; letter-spacing: 0.02em;
            color: #6F7B8B; transition: color 150ms ease;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stButton"] button::before {{
            content: ""; position: absolute; top: 4px; left: 50%; transform: translateX(-50%);
            width: 9px; height: 9px; border-radius: 50%; box-sizing: border-box;
            background: {BG}; border: 1px solid #2E3A47;
            transition: border-color 150ms ease, box-shadow 150ms ease, background-color 150ms ease;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stButton"] button:hover p {{ color: {TEXT}; }}
        [class*="st-key-sajag_timeline"] [data-testid="stButton"] button:hover::before {{ border-color: {MUTED}; }}

        [class*="st-key-sajag_timeline"] [data-testid="stButton"] button[kind="primary"] p {{
            color: {ACCENT}; font-weight: 600;
        }}
        [class*="st-key-sajag_timeline"] [data-testid="stButton"] button[kind="primary"]::before {{
            background: {ACCENT}; border-color: {ACCENT};
            box-shadow: 0 0 0 3px rgba(76,195,217,0.14);
        }}

        /* LANDFALL endpoint: diamond marker, visible but secondary */
        [class*="st-key-sajag_timeline"] .st-key-tl_0 [data-testid="stButton"] button p {{
            letter-spacing: 0.08em; font-weight: 500; color: {MUTED};
        }}
        [class*="st-key-sajag_timeline"] .st-key-tl_0 [data-testid="stButton"] button::before {{
            width: 10px; height: 10px; border-radius: 1px; top: 3px;
            transform: translateX(-50%) rotate(45deg); border-color: #5C6878;
        }}
        [class*="st-key-sajag_timeline"] .st-key-tl_0 [data-testid="stButton"] button[kind="primary"] p {{ color: {ACCENT}; }}

        /* End stops sit flush with the content edges */
        [class*="st-key-sajag_timeline"] .st-key-tl_72 [data-testid="stButton"] button {{
            justify-content: flex-start; padding-left: 0 !important;
        }}
        [class*="st-key-sajag_timeline"] .st-key-tl_72 [data-testid="stButton"] button::before {{
            left: 0; transform: none;
        }}
        [class*="st-key-sajag_timeline"] .st-key-tl_0 [data-testid="stButton"] button {{
            justify-content: flex-end; padding-right: 0 !important;
        }}
        [class*="st-key-sajag_timeline"] .st-key-tl_0 [data-testid="stButton"] button::before {{
            left: auto; right: 1px; transform: rotate(45deg);
        }}

        /* Step-nav pills, rendered as minimal st.buttons */
        [class*="st-key-sajag_step_nav"] [data-testid="stButton"] button {{
            background: transparent !important; border: none !important; color: {MUTED} !important;
            font-family: "IBM Plex Mono", monospace; font-size: 0.78rem;
            padding: 4px 4px !important; border-bottom: 2px solid transparent !important;
            border-radius: 0 !important; box-shadow: none !important;
        }}
        [class*="st-key-sajag_step_nav"] [data-testid="stButton"] button {{
            font-family: "IBM Plex Sans", sans-serif; font-size: 0.82rem; font-weight: 600;
            letter-spacing: 0.03em; justify-content: flex-start;
        }}
        [class*="st-key-sajag_step_nav"] [data-testid="stButton"] button[kind="primary"] {{
            color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important;
        }}
        [class*="st-key-sajag_step_nav"] [data-testid="stButton"] button:hover {{
            color: {ACCENT} !important;
        }}
        [class*="st-key-sajag_step_nav"] {{ border-bottom: 1px solid {DIVIDER}; margin-bottom: 12px; padding-bottom: 6px; }}

        /* Section labels & dividers */
        .sajag-section-label {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED};
            margin: 18px 0 8px 0;
        }}
        .sajag-divider {{ border-top: 1px solid {DIVIDER}; margin: 20px 0; }}

        /* Filters + Play: one compact row (filters fill, Play hugs the right) */
        [class*="st-key-sajag_controls"] {{ margin-bottom: 8px; }}
        [class*="st-key-sajag_controls"] [data-testid="stHorizontalBlock"]:has([data-testid="stExpander"]) {{
            display: grid !important; grid-template-columns: minmax(0, 1fr) auto;
            align-items: start; column-gap: 12px;
        }}
        [class*="st-key-sajag_controls"] [data-testid="stColumn"]:not([data-testid="stExpander"] *) {{
            width: auto !important; min-width: 0 !important; flex: none !important;
        }}
        [class*="st-key-sajag_controls"] [data-testid="stExpander"] summary {{
            padding-top: 8px; padding-bottom: 8px; font-size: 0.85rem; color: {MUTED};
        }}
        [class*="st-key-sajag_controls"] [data-testid="stExpander"] summary:hover {{ color: {TEXT}; }}
        [class*="st-key-sajag_controls"] [data-testid="stExpander"] {{
            border: 0;
            border-radius: 0;
            background: transparent;
        }}
        [class*="st-key-sajag_controls"] [data-testid="stExpander"] details {{
            border: 0 !important;
            background: transparent;
        }}
        [class*="st-key-sajag_controls"] [data-testid="stExpander"] summary {{
            padding: 4px 0;
        }}
        [class*="st-key-sajag_controls"] [data-testid="stButton"] button {{
            min-height: 0; padding: 8px 18px; border-radius: 8px;
            background: transparent; border: 1px solid {DIVIDER}; color: {TEXT}; font-size: 0.85rem;
            transition: border-color 150ms ease, color 150ms ease;
        }}
        [class*="st-key-sajag_controls"] [data-testid="stButton"] button:hover {{
            border-color: rgba(76,195,217,0.45); color: {ACCENT};
        }}

        /* Map + ranking: one information panel */
        [class*="st-key-sajag_risk_panel"] {{
            border: 1px solid rgba(138,150,166,0.12); border-radius: 8px;
            padding: 18px 20px 12px 20px; margin-bottom: 24px;
        }}
        .sajag-panel-head {{
            display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
            min-height: 20px; margin-bottom: 4px;
        }}
        .sajag-panel-head .sajag-section-label {{ margin: 0; }}
        .sajag-panel-context {{ color: {MUTED}; font-size: 0.75rem; white-space: nowrap; }}
        .sajag-panel-context .sajag-mono {{ color: {TEXT}; font-weight: 600; }}

        /* Section heading with hairline divider (Next Actions) */
        .sajag-section-head {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED};
            padding-bottom: 8px; border-bottom: 1px solid {DIVIDER}; margin: 0 0 4px 0;
        }}

        /* KPI cards */
        .sajag-kpi-row {{
            display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px; margin: 4px 0 24px 0;
        }}
        @media (max-width: 720px) {{
            .sajag-kpi-row {{ grid-template-columns: minmax(0, 1fr); }}
        }}
        .sajag-kpi {{
            position: relative; overflow: hidden;
            background: {SURFACE};
            border: 1px solid rgba(138, 150, 166, 0.14);
            border-radius: 8px;
            padding: 18px 22px 16px 22px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35), 0 6px 16px rgba(0, 0, 0, 0.18);
            animation: sajag-kpi-in 360ms ease-out both;
        }}
        /* Hairline accent along the top edge */
        .sajag-kpi::before {{
            content: ""; position: absolute; top: 0; left: 22px; width: 28px; height: 2px;
            background: rgba(138, 150, 166, 0.35); border-radius: 0 0 2px 2px;
        }}
        .sajag-kpi:nth-child(2) {{ animation-delay: 60ms; }}
        .sajag-kpi:nth-child(3) {{ animation-delay: 120ms; }}
        .sajag-kpi-value {{
            font-family: "IBM Plex Mono", monospace;
            font-size: 2.4rem; font-weight: 600; line-height: 1.05; color: {TEXT};
            font-variant-numeric: tabular-nums; letter-spacing: -0.01em; white-space: nowrap;
        }}
        .sajag-kpi-label {{
            font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
            color: {MUTED}; margin-top: 8px;
        }}
        .sajag-kpi--primary {{
            background: #17222C;
            border-color: rgba(76, 195, 217, 0.30);
        }}
        .sajag-kpi--primary::before {{ background: {ACCENT}; width: 40px; }}
        .sajag-kpi--primary .sajag-kpi-label {{ color: rgba(76, 195, 217, 0.85); }}
        @keyframes sajag-kpi-in {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: none; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .sajag-kpi {{ animation: none; }}
        }}

        /* Why-this-borrower "case file" panel */
        .sajag-case-header {{ display:flex; align-items:center; gap:14px; margin-bottom:10px; }}
        .sajag-case-id {{ font-family:"IBM Plex Mono", monospace; font-size:1.7rem; font-weight:600; color:{TEXT}; }}
        .sajag-tier-pill {{
            font-family:"IBM Plex Sans", sans-serif; font-size:0.7rem; font-weight:700;
            text-transform:uppercase; letter-spacing:0.05em; padding:4px 11px; border-radius:999px;
            color:#0E141B;
        }}
        .sajag-action-strip {{
            border-left: 3px solid {ACCENT}; background: rgba(76,195,217,0.08);
            padding: 10px 14px; border-radius: 4px; margin-top: 12px; color: {TEXT};
        }}

        /* Landing */
        /* Landing page only */
        /* Deep navy page background on the landing page; the hero sits on it directly.
           ":not(:has(topbar))": landing elements linger until the dashboard's first run
           finishes, so key off the dashboard top bar too to keep the dashboard untouched. */
        .stApp:has([class*="st-key-sajag_landing_cta"]):not(:has([class*="st-key-sajag_topbar"])),
        .stApp:has([class*="st-key-sajag_landing_cta"]):not(:has([class*="st-key-sajag_topbar"])) [data-testid="stAppViewContainer"] {{ background: {LANDING_BG}; }}
        .stApp:has([class*="st-key-sajag_landing_cta"]) .sajag-hero-wrap {{ background: transparent; }}
        .sajag-hero-summary {{ font-size: clamp(0.74rem, 0.7rem + 0.15vw, 0.8rem); color: {MUTED}; margin: 0; opacity: 0.85; }}
        .sajag-landing-head {{
            font-size: clamp(0.66rem, 0.63rem + 0.12vw, 0.72rem); font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.1em; color: {LANDING_TEXT}; margin: 56px 0 14px 0;
        }}
        /* Funnel -> How it works: hairline divider; same total spacing (16 + 40 = 56px) */
        .sajag-landing-head--divided {{ margin-top: 16px; padding-top: 40px; border-top: 1px solid {LANDING_BORDER}; }}
        /* "How the list narrows" funnel (landing). ~78% of the content width, centred;
           label | bar track | mirror spacer keeps the bars on the true centre line.
           Stage height follows width (~48px at the normal 1240px content width). */
        .sajag-funnel {{
            --fh: clamp(38px, 2.5vw, 58px); --side: clamp(96px, 11%, 136px);
            width: 78%; margin: 4px auto 0 auto;
            cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 18 18'%3E%3Ccircle cx='9' cy='9' r='5.5' fill='%230B1219' fill-opacity='.18' stroke='%23C5D0DA' stroke-opacity='.7' stroke-width='1'/%3E%3C/svg%3E") 9 9, auto;
        }}
        .sajag-fstage, .sajag-fconn {{ display: grid; grid-template-columns: var(--side) 1fr var(--side); align-items: center; }}
        .sajag-flabel {{ display: flex; align-items: baseline; gap: 10px; font-family: "IBM Plex Mono", monospace; white-space: nowrap; }}
        .sajag-fidx {{ font-size: 0.62rem; color: {LANDING_MUTED}; opacity: 0.7; letter-spacing: 0.04em; }}
        .sajag-ftime {{ font-size: clamp(0.72rem, 0.68rem + 0.15vw, 0.8rem); color: {LANDING_TEXT}; letter-spacing: 0.02em; }}
        .sajag-ftrack {{ position: relative; height: var(--fh); display: flex; justify-content: center; }}
        /* faint guide tying the label to its stage */
        .sajag-ftrack::before {{
            content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: rgba(38, 53, 65, 0.45);
        }}
        .sajag-fbar {{
            position: relative; width: var(--w); height: 100%; box-sizing: border-box; border-radius: 2px;
            background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0) 55%), var(--c);
            border: 1px solid rgba(255,255,255,0.05);
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            transform-origin: 50% 50%; transition: filter 180ms ease;
        }}
        .sajag-fnum {{
            font-family: "IBM Plex Mono", monospace; font-weight: 600; color: #FFFFFF; line-height: 1.05;
            font-size: clamp(0.92rem, 0.8rem + 0.45vw, 1.2rem); font-variant-numeric: tabular-nums;
        }}
        .sajag-fprec {{
            margin-top: 3px; font-size: clamp(0.54rem, 0.52rem + 0.1vw, 0.6rem); letter-spacing: 0.08em;
            text-transform: uppercase; color: rgba(255,255,255,0.78); transition: color 180ms ease;
        }}
        /* connector: a thin trapezoid narrowing from one stage's width to the next */
        .sajag-fconn > div:nth-child(2) {{ height: 10px; background: var(--c); opacity: 0.16; clip-path: var(--clip); }}
        /* hover: hovered stage a touch brighter, the rest slightly dimmer; no size change */
        .sajag-fstage, .sajag-fconn {{ transition: opacity 180ms ease; }}
        .sajag-funnel:has(.sajag-fstage:hover) .sajag-fstage:not(:hover),
        .sajag-funnel:has(.sajag-fstage:hover) .sajag-fconn {{ opacity: 0.8; }}
        .sajag-fstage:hover .sajag-fbar {{ filter: brightness(1.08); }}
        .sajag-fstage:hover .sajag-fprec {{ color: #FFFFFF; }}
        /* entrance (JS adds .is-armed, then .is-in when the funnel scrolls into view) */
        .sajag-funnel.is-armed .sajag-fstage, .sajag-funnel.is-armed .sajag-fconn {{ opacity: 0; }}
        .sajag-funnel.is-in .sajag-fstage {{ animation: sajag-fstage-in 0.5s ease-out calc(var(--i) * 110ms) backwards; }}
        .sajag-funnel.is-in .sajag-fbar {{ animation: sajag-fbar-in 0.55s cubic-bezier(.2,.7,.3,1) calc(var(--i) * 110ms) backwards; }}
        .sajag-funnel.is-in .sajag-fconn {{ animation: sajag-fstage-in 0.4s ease-out calc(var(--i) * 110ms + 160ms) backwards; }}
        @keyframes sajag-fstage-in {{ from {{ opacity: 0; }} }}
        @keyframes sajag-fbar-in {{ from {{ transform: scaleX(0.9); }} }}
        @media (max-width: 640px) {{ .sajag-funnel {{ width: 100%; }} }}
        .sajag-howstep {{
            border-top: 1px solid {LANDING_BORDER}; padding: 14px 16px 12px 16px;
            background: rgba(17, 27, 36, 0.55);  /* {LANDING_SURFACE}, faint */
        }}
        /* Equal-height blocks: stretch the four columns and let each block fill its column */
        [data-testid="stHorizontalBlock"]:has(.sajag-howstep) {{ align-items: stretch !important; }}
        [data-testid="stColumn"]:has(.sajag-howstep) > [data-testid="stVerticalBlock"] {{ height: 100%; }}
        [data-testid="stColumn"]:has(.sajag-howstep) [data-testid="stElementContainer"]:has(.sajag-howstep) {{ flex: 1 1 auto; display: flex; flex-direction: column; }}
        [data-testid="stColumn"]:has(.sajag-howstep) [data-testid="stMarkdown"]:has(.sajag-howstep),
        [data-testid="stColumn"]:has(.sajag-howstep) [data-testid="stMarkdown"]:has(.sajag-howstep) > div,
        [data-testid="stColumn"]:has(.sajag-howstep) [data-testid="stMarkdownContainer"]:has(.sajag-howstep) {{
            flex: 1 1 auto; display: flex; flex-direction: column; align-items: stretch;
        }}
        .sajag-howstep {{ flex: 1 1 auto; box-sizing: border-box; }}
        .sajag-howstep-num, .sajag-howstep-name {{ font-size: clamp(0.8rem, 0.76rem + 0.15vw, 0.88rem); }}
        .sajag-howstep-num {{ font-family:"IBM Plex Mono", monospace; color:{LANDING_CYAN}; font-weight:600; }}
        .sajag-howstep-name {{ font-weight:600; letter-spacing:0.03em; margin-left:6px; color:{LANDING_TEXT}; }}
        .sajag-howstep p {{ color: {LANDING_MUTED}; font-size: clamp(0.72rem, 0.69rem + 0.12vw, 0.78rem); margin-top: 6px; margin-bottom: 0; line-height:1.5; }}
        [class*="st-key-sajag_landing_cta"] {{
            margin-top: 48px; padding-top: 32px; border-top: 1px solid {DIVIDER}; gap: 0;
        }}
        .stApp:has([class*="st-key-sajag_landing_cta"]) .st-key-landing_cta button {{
            background: #E8E4DC !important;
            color: #0B1117 !important;
            border: 1px solid rgba(11, 17, 23, 0.12) !important;
            box-shadow: none !important;
            padding: 9px 26px;
            font-weight: 600;
            transition: background-color 150ms ease;
        }}
        .stApp:has([class*="st-key-sajag_landing_cta"]) .st-key-landing_cta button:hover {{
            background: #F4F1EA !important;
            color: #0B1117 !important;
            border-color: rgba(11, 17, 23, 0.12) !important;
            box-shadow: none !important;
        }}
        .stApp:has([class*="st-key-sajag_landing_cta"]) .st-key-landing_cta button p {{
            color: #0B1117 !important;
        }}
        .st-key-landing_cta button p {{ font-size: clamp(0.84rem, 0.8rem + 0.15vw, 0.92rem); }}
        .sajag-footer-note {{ color: {MUTED}; font-size: clamp(0.66rem, 0.64rem + 0.1vw, 0.7rem); margin-top: 12px; opacity: 0.8; }}

        .stApp:has([class*="st-key-sajag_topbar"]) div.block-container {{
            padding-top: 0.3rem;
            padding-bottom: 1.5rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_topbar"] {{
            background-image: none;
            padding-top: 2px;
            gap: 10px;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_header"] {{
            padding-top: 6px;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .st-key-brand_home button p {{
            font-size: 1.65rem;
            letter-spacing: 0.18em;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-event {{
            font-size: 0.78rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-status {{
            padding: 5px 18px;
            background: transparent;
            border: 0;
            border-radius: 0;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_header"] .sajag-countdown {{
            font-size: 1.55rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-status-sub {{
            margin-top: 4px;
            font-size: 0.6rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_row2"] {{
            padding: 2px 0 8px;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_row2"] [data-testid="stHorizontalBlock"]:has([class*="st-key-sajag_timeline"]) {{
            grid-template-columns: minmax(320px, 0.9fr) minmax(0, 2.4fr);
            column-gap: clamp(14px, 2vw, 30px);
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_topbar"] div[role="radiogroup"] {{
            flex-direction: row;
            flex-wrap: nowrap;
            width: max-content;
            max-width: 100%;
            gap: 2px;
            padding: 2px;
            background: transparent;
            border: 0;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_topbar"] div[role="radiogroup"] label {{
            min-height: 28px;
            padding: 0 7px !important;
            white-space: nowrap;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_topbar"] div[role="radiogroup"] label p {{
            font-size: 0.62rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_timeline"] {{
            margin-top: 0;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_timeline"] [data-testid="stButton"] button {{
            padding-top: 18px !important;
            padding-bottom: 3px !important;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_timeline"] .st-key-tl_0 [data-testid="stButton"] button p {{
            font-size: 0.62rem;
            letter-spacing: 0.03em;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_step_nav"] {{
            padding-bottom: 4px;
            margin-bottom: 7px;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_step_nav"] [data-testid="stHorizontalBlock"] {{
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            align-items: stretch;
            gap: clamp(8px, 1vw, 16px);
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_step_nav"] [data-testid="stColumn"] {{
            width: 100% !important;
            min-width: 0 !important;
            flex: none !important;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_step_nav"] [data-testid="stButton"] button {{
            min-height: 0;
            padding: 3px 2px 6px !important;
            font-size: 0.76rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_controls"] {{
            margin-bottom: 4px;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_controls"] [data-testid="stExpander"] {{
            border: 0;
            background: transparent;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_controls"] [data-testid="stExpander"] summary {{
            padding: 5px 0;
            font-size: 0.78rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_controls"] [data-testid="stButton"] button {{
            padding: 5px 14px;
            border-radius: 4px;
            font-size: 0.78rem;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi-row {{
            gap: 0;
            margin: 4px 0 16px;
            border-top: 1px solid rgba(138, 150, 166, 0.14);
            border-bottom: 1px solid rgba(138, 150, 166, 0.14);
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi {{
            min-width: 0;
            padding: 12px 18px 11px;
            background: transparent;
            border: 0;
            border-radius: 0;
            box-shadow: none;
            animation: none;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi + .sajag-kpi {{
            border-left: 1px solid rgba(138, 150, 166, 0.14);
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi::before {{
            display: none;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi-value {{
            font-size: clamp(1.65rem, 2vw, 2rem);
            line-height: 1.1;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi-label {{
            margin-top: 4px;
            font-size: 0.64rem;
            letter-spacing: 0.08em;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi--primary .sajag-kpi-value {{
            color: {ACCENT};
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_risk_panel"] {{
            padding: 4px 0 0;
            margin-bottom: 16px;
            border: 0;
            border-radius: 0;
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-panel-head {{
            min-height: 24px;
            margin-bottom: 2px;
            padding-bottom: 7px;
            border-bottom: 1px solid rgba(138, 150, 166, 0.16);
        }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-section-label {{
            margin-top: 8px;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
        }}
        .sajag-actions-scroll {{
            width: 100%;
            overflow-x: auto;
        }}
        .sajag-actions-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.86rem;
            background: #10171E;
            border: 1px solid #26323D;
        }}
        .sajag-actions-table th {{
            padding: 8px 12px;
            color: {MUTED};
            font-size: 0.62rem;
            font-weight: 500;
            letter-spacing: 0.09em;
            text-align: left;
            text-transform: uppercase;
            border-bottom: 1px solid #26323D;
        }}
        .sajag-actions-table td {{
            padding: 11px 12px;
            color: {TEXT};
            line-height: 1.45;
            vertical-align: middle;
            border-bottom: 1px solid #26323D;
        }}
        .sajag-actions-table tbody tr:last-child td {{ border-bottom: 0; }}
        .sajag-actions-table th:first-child, .sajag-actions-table td:first-child {{ width: 14%; }}
        .sajag-actions-table th:nth-child(2), .sajag-actions-table td:nth-child(2) {{ width: 13%; }}
        .sajag-actions-table th:nth-child(2) {{ white-space: nowrap; }}
        .sajag-actions-table th:nth-child(3), .sajag-actions-table td:nth-child(3) {{ width: 51%; }}
        .sajag-actions-table th:last-child, .sajag-actions-table td:last-child {{
            width: 22%;
            text-align: right;
            white-space: nowrap;
        }}
        .sajag-action-tier {{
            display: inline-block;
            padding-left: 9px;
            border-left: 2px solid {MUTED};
            font-weight: 600;
        }}
        .sajag-actions-table tbody tr:has(.sajag-action-tier--high) {{ background: #4A252B; }}
        .sajag-actions-table tbody tr:has(.sajag-action-tier--medium) {{ background: #403522; }}
        .sajag-actions-table tbody tr:has(.sajag-action-tier--low) {{ background: #202B38; }}
        .sajag-actions-table tbody tr:has(.sajag-action-tier--not-exposed) {{ background: #171F28; }}
        .sajag-action-tier--high {{ color: #F2B8B8; border-color: #C85A61; }}
        .sajag-action-tier--medium {{ color: #E5C98A; border-color: #C49A45; }}
        .sajag-action-tier--low {{ color: #9FB1C4; border-color: #617A92; }}
        .sajag-action-tier--not-exposed {{ color: #7F8B98; border-color: #4A5663; }}
        .sajag-actions-count, .sajag-actions-amount {{
            font-family: "IBM Plex Mono", monospace;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }}
        .sajag-actions-amount {{ color: #B6C0CB; }}
        .stApp:has([class*="st-key-sajag_topbar"]) .sajag-section-head {{
            margin-bottom: 3px;
            padding-bottom: 7px;
        }}
        @media (max-width: 900px) {{
            [class*="st-key-sajag_timeline"] [data-testid="stHorizontalBlock"] {{
                grid-template-columns: 0.8fr 1.65fr 1.65fr 1.65fr 1.2fr;
            }}
            [class*="st-key-sajag_timeline"] .st-key-tl_0 [data-testid="stButton"] button p {{
                font-size: 0.58rem;
                letter-spacing: 0.02em;
            }}
            .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_row2"] [data-testid="stHorizontalBlock"]:has([class*="st-key-sajag_timeline"]) {{
                grid-template-columns: minmax(300px, 0.9fr) minmax(0, 2fr);
                column-gap: 8px;
            }}
            .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_step_nav"] [data-testid="stHorizontalBlock"] {{
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 5px;
            }}
        }}
        @media (max-width: 700px) {{
            .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_row2"] [data-testid="stHorizontalBlock"]:has([class*="st-key-sajag_timeline"]) {{
                grid-template-columns: minmax(0, 1fr);
                row-gap: 4px;
            }}
            .stApp:has([class*="st-key-sajag_topbar"]) [class*="st-key-sajag_topbar"] div[role="radiogroup"] {{
                justify-content: flex-start;
                width: 100%;
            }}
            .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi-row {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}
            .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi {{
                padding: 10px 8px;
            }}
            .stApp:has([class*="st-key-sajag_topbar"]) .sajag-kpi-value {{
                font-size: clamp(1.15rem, 5vw, 1.55rem);
            }}
            .sajag-actions-table {{ min-width: 720px; }}
        }}
        [class*="st-key-sajag_act_page"] {{
            --act-line: #26323D;
            --act-muted: #8A96A6;
            --act-text: #E6EAF0;
            --act-surface: #10171E;
        }}
        [class*="st-key-sajag_act_page"] h1,
        [class*="st-key-sajag_act_page"] h2,
        [class*="st-key-sajag_act_page"] h3,
        [class*="st-key-sajag_act_page"] h4 {{
            letter-spacing: -0.02em;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-intro {{
            color: #9AA7B5;
            margin: -8px 0 14px;
            font-size: 0.92rem;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-eyebrow {{
            color: #8A96A6;
            font-size: 0.65rem;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin: 0 0 8px;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-summary {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr)) auto;
            align-items: center;
            gap: 0;
            margin: 4px 0 24px;
            padding: 14px 0;
            border-top: 1px solid var(--act-line);
            border-bottom: 1px solid var(--act-line);
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-stat {{
            min-width: 0;
            padding: 0 18px;
            border-right: 1px solid var(--act-line);
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-stat:first-child {{ padding-left: 0; }}
        [class*="st-key-sajag_act_page"] .sajag-act-stat-value {{
            color: var(--act-text);
            font: 600 clamp(1.45rem, 2vw, 1.9rem)/1.15 "IBM Plex Mono", monospace;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-stat--high .sajag-act-stat-value {{ color: #E2A0A2; }}
        [class*="st-key-sajag_act_page"] .sajag-act-stat-label {{
            color: var(--act-muted);
            font-size: 0.64rem;
            letter-spacing: 0.09em;
            margin-top: 4px;
            text-transform: uppercase;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-jump {{
            display: inline-flex;
            justify-content: center;
            align-items: center;
            margin-left: 20px;
            padding: 9px 13px;
            color: #0B1117 !important;
            background: #E8E4DC;
            text-decoration: none !important;
            font-size: 0.82rem;
            font-weight: 600;
            white-space: nowrap;
            transition: background-color 140ms ease;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-jump:hover {{
            color: #0B1117 !important;
            background: #F4F1EA;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-section {{
            margin: 22px 0 9px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--act-line);
            color: var(--act-text);
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-scroll {{
            max-height: 360px;
            overflow: auto;
            border-top: 1px solid var(--act-line);
            border-bottom: 1px solid var(--act-line);
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table {{
            width: 100%;
            min-width: 610px;
            border-collapse: collapse;
            table-layout: fixed;
            background: var(--act-surface);
            font-size: 0.84rem;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table th {{
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 9px 12px;
            color: #98A5B3;
            background: #111922;
            border-bottom: 1px solid var(--act-line);
            font-size: 0.66rem;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-align: left;
            text-transform: uppercase;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table td {{
            padding: 8px 12px;
            color: #D8DEE5;
            border-bottom: 1px solid rgba(138, 150, 166, 0.1);
            line-height: 1.35;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table tbody tr:last-child td {{ border-bottom: 0; }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table tbody tr:hover {{ background: #18222C; }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-id {{
            width: 12%;
            color: #D9E0E7 !important;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table th:nth-child(2),
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table td:nth-child(2) {{ width: 34%; }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table th:nth-child(3),
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table td:nth-child(3) {{ width: 12%; }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table th:nth-child(4),
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table td:nth-child(4) {{ width: 15%; }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table th:nth-child(5),
        [class*="st-key-sajag_act_page"] .sajag-act-branch-table td:nth-child(5) {{ width: 20%; }}
        [class*="st-key-sajag_act_page"] .sajag-act-number {{
            text-align: right !important;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.8rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-high-count {{ color: #E2A0A2 !important; }}
        [class*="st-key-sajag_act_page"] .sajag-act-currency {{ color: #C5CED7; }}
        [class*="st-key-sajag_act_page"] .sajag-act-branch-meta {{
            color: var(--act-muted);
            font-size: 0.78rem;
            margin: -3px 0 15px;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-list {{
            margin: 0;
            padding: 0;
            list-style: none;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-list li {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 14px;
            padding: 8px 0;
            color: #D5DCE3;
            border-bottom: 1px solid rgba(138, 150, 166, 0.1);
            font-size: 0.88rem;
            line-height: 1.4;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-list li:last-child {{ border-bottom: 0; }}
        [class*="st-key-sajag_act_page"] .sajag-act-list-count {{
            color: #AEB9C4;
            font: 500 0.78rem "IBM Plex Mono", monospace;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-recommendation {{
            margin-top: 6px;
            padding: 2px 0 2px 14px;
            border-left: 2px solid #C85A61;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-recommendation li {{
            padding: 5px 0;
            color: #E2E5E9;
            font-size: 0.88rem;
        }}
        [class*="st-key-sajag_act_page"] .sajag-act-briefing {{
            margin: 3px 0 10px;
            color: #CBD3DC;
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        [class*="st-key-sajag_act_page"] [data-testid="stDataFrame"] {{
            border: 1px solid var(--act-line);
            border-radius: 2px;
            overflow: hidden;
        }}
        [class*="st-key-sajag_act_page"] [data-testid="stDataFrame"] [role="columnheader"] {{
            color: #98A5B3;
            font-size: 0.7rem;
            letter-spacing: 0.04em;
        }}
        [class*="st-key-sajag_act_page"] [data-testid="stDataFrame"] [role="gridcell"] {{
            border-bottom-color: var(--act-line);
        }}
        [class*="st-key-sajag_act_page"] [data-testid="stSelectbox"] label,
        [class*="st-key-sajag_act_page"] [data-testid="stTextInput"] label {{
            color: var(--act-muted);
            font-size: 0.72rem;
        }}
        [class*="st-key-sajag_act_page"] [data-testid="stExpander"] {{
            border: 1px solid var(--act-line);
            border-radius: 2px;
            background: transparent;
        }}
        .sajag-page-intro {{
            max-width: 72ch;
            margin: -8px 0 18px;
            color: #9AA7B5;
            font-size: 0.95rem;
            line-height: 1.55;
        }}
        .sajag-kpi-row {{
            gap: clamp(12px, 2vw, 28px);
            margin: 10px 0 24px;
        }}
        .sajag-kpi {{
            padding: 15px 18px 13px;
            border: 0;
            border-right: 1px solid #26323D;
            border-radius: 0;
            box-shadow: none;
            animation: none;
        }}
        .sajag-kpi-row {{
            gap: 0;
            border-top: 1px solid #26323D;
            border-bottom: 1px solid #26323D;
            background: #10171E;
        }}
        .sajag-kpi:last-child {{ border-right: 0; }}
        .sajag-kpi::before {{ left: 18px; top: 0; width: 20px; border-radius: 0; }}
        .sajag-kpi-value {{ font-size: clamp(1.75rem, 2.4vw, 2.35rem); }}
        .sajag-kpi--primary {{
            background: rgba(200, 90, 97, 0.045);
        }}
        .sajag-kpi--primary::before {{ background: {RISK_PALETTE["high"]}; width: 34px; }}
        .sajag-kpi--primary .sajag-kpi-label {{ color: #D79A9D; }}
        [class*="st-key-sajag_risk_panel"] {{
            border: 0;
            border-radius: 0;
            padding: 4px 0 12px;
            margin-bottom: 18px;
        }}
        .sajag-actions-table {{
            border-color: #26323D;
            border-radius: 2px;
            background: #10171E;
        }}
        .sajag-actions-table tbody tr:has(.sajag-action-tier) {{ background: transparent; }}
        .sajag-action-tier {{
            padding: 3px 8px;
            border: 1px solid transparent;
            border-left-width: 2px;
            background: #171F28;
            font-size: 0.74rem;
            line-height: 1.5;
        }}
        .sajag-action-tier--high {{ background: #281D22; border-color: #5C353A; border-left-color: #C85A61; color: #F2B8B8; }}
        .sajag-action-tier--medium {{ background: #272319; border-color: #51452B; border-left-color: #C49A45; color: #E5C98A; }}
        .sajag-action-tier--low {{ background: #1A232D; border-color: #303E4C; border-left-color: #617A92; color: #9FB1C4; }}
        .sajag-action-tier--not-exposed {{ background: #171F28; border-color: #2E3945; border-left-color: #4A5663; color: #7F8B98; }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] {{
            flex-direction: row;
            flex-wrap: wrap;
            align-items: center;
            gap: 2px;
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label {{ padding: 0 9px !important; }}
        .sajag-hero-stage {{
            min-height: clamp(300px, 35vw, 520px);
            max-height: 680px;
        }}
        .sajag-fbar {{
            background: var(--c);
            border: 1px solid rgba(235, 192, 193, 0.2);
        }}
        .sajag-fnum {{ color: #F3E9E7; }}
        .sajag-fprec {{ color: rgba(243, 233, 231, 0.78); }}
        .sajag-fconn > div:nth-child(2) {{ opacity: 0.28; }}
        .sajag-measure-intro {{
            color: #9AA7B5;
            font-size: 0.95rem;
            margin: -7px 0 16px;
        }}
        .sajag-measure-break-even {{
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 6px 10px;
            margin: 12px 0 18px;
            padding: 12px 0;
            border-top: 1px solid #26323D;
            border-bottom: 1px solid #26323D;
            color: #A9B4C0;
            font-size: 0.86rem;
        }}
        .sajag-measure-break-even strong {{
            color: #9AD9E3;
            font: 600 1.15rem "IBM Plex Mono", monospace;
            font-variant-numeric: tabular-nums;
        }}
        .sajag-measure-comparison {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            border-top: 1px solid #26323D;
            border-bottom: 1px solid #26323D;
            background: #10171E;
        }}
        .sajag-measure-strategy {{ min-width: 0; padding: 18px clamp(14px, 2vw, 26px); }}
        .sajag-measure-strategy + .sajag-measure-strategy {{ border-left: 1px solid #26323D; }}
        .sajag-measure-strategy--targeted {{
            background: rgba(76, 195, 217, 0.035);
            box-shadow: inset 2px 0 0 rgba(76, 195, 217, 0.62);
        }}
        .sajag-measure-strategy-title {{
            color: #E6EAF0;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .sajag-measure-strategy-note {{
            min-height: 2.8em;
            margin: 5px 0 14px;
            color: #8A96A6;
            font-size: 0.8rem;
            line-height: 1.45;
        }}
        .sajag-measure-metrics {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px 20px;
        }}
        .sajag-measure-metric-label {{
            color: #8A96A6;
            font-size: 0.65rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }}
        .sajag-measure-metric-value {{
            margin-top: 3px;
            color: #E6EAF0;
            font: 500 1rem "IBM Plex Mono", monospace;
            font-variant-numeric: tabular-nums;
        }}
        .sajag-measure-metric--net .sajag-measure-metric-value {{ color: #9AD9E3; }}
        .sajag-measure-detail {{ margin-top: 18px; border-top: 1px solid #26323D; }}
        .sajag-priority-overview {{
            margin: 10px 0 22px;
            padding: 14px 0 15px;
            border-top: 1px solid #26323D;
            border-bottom: 1px solid #26323D;
        }}
        .sajag-priority-counts {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
        }}
        .sajag-priority-count {{ min-width: 0; }}
        .sajag-priority-value {{
            font: 600 clamp(1.35rem, 2.2vw, 1.9rem)/1.15 "IBM Plex Mono", monospace;
            font-variant-numeric: tabular-nums;
            color: #E6EAF0;
        }}
        .sajag-priority-label {{
            margin-top: 3px;
            color: #8A96A6;
            font-size: 0.66rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .sajag-priority-count--high .sajag-priority-value {{ color: #E2A0A2; }}
        .sajag-priority-count--medium .sajag-priority-value {{ color: #D9BF85; }}
        .sajag-priority-count--low .sajag-priority-value {{ color: #9FB1C4; }}
        .sajag-priority-meter {{
            display: flex;
            height: 5px;
            margin: 14px 0 17px;
            overflow: hidden;
            background: #202B35;
        }}
        .sajag-priority-meter span {{ display: block; min-width: 1px; }}
        .sajag-priority-driver-head {{
            margin-bottom: 7px;
            color: #8A96A6;
            font-size: 0.64rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}
        .sajag-priority-drivers {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
        }}
        .sajag-priority-driver {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding-top: 7px;
            border-top: 1px solid #26323D;
            color: #CBD3DC;
            font-size: 0.82rem;
        }}
        .sajag-priority-driver strong {{
            color: #AEB9C4;
            font: 500 0.78rem "IBM Plex Mono", monospace;
            white-space: nowrap;
        }}
        [data-testid="stDataFrame"] {{ border: 1px solid #26323D; border-radius: 2px; }}
        [data-testid="stExpander"] {{ border-radius: 2px; }}
        @media (max-width: 720px) {{
            .sajag-panel-head {{ flex-wrap: wrap; align-items: flex-start; }}
            .sajag-panel-context {{ max-width: 100%; white-space: normal; }}
            [class*="st-key-sajag_risk_panel"] [data-testid="stHorizontalBlock"] {{
                grid-template-columns: minmax(0, 1fr);
                row-gap: 12px;
            }}
            .sajag-measure-comparison {{ grid-template-columns: minmax(0, 1fr); }}
            .sajag-measure-strategy + .sajag-measure-strategy {{
                border-left: 0;
                border-top: 1px solid #26323D;
            }}
            .sajag-measure-strategy--targeted {{ box-shadow: inset 0 2px 0 rgba(76, 195, 217, 0.62); }}
            .sajag-priority-counts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 12px; }}
            .sajag-priority-drivers {{ grid-template-columns: minmax(0, 1fr); gap: 8px; }}
            .sajag-funnel {{ width: 100%; }}
            .sajag-flabel {{ gap: 5px; }}
            .sajag-fidx {{ display: none; }}
            [data-testid="stHorizontalBlock"]:has(.sajag-howstep) {{
                display: grid !important;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }}
        }}
        @media (max-width: 720px) {{
            [class*="st-key-sajag_act_page"] .sajag-act-summary {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
                row-gap: 14px;
            }}
            [class*="st-key-sajag_act_page"] .sajag-act-stat {{
                padding: 0 9px;
            }}
            [class*="st-key-sajag_act_page"] .sajag-act-stat:first-child {{ padding-left: 0; }}
            [class*="st-key-sajag_act_page"] .sajag-act-stat:nth-child(3) {{ border-right: 0; }}
            [class*="st-key-sajag_act_page"] .sajag-act-jump {{
                grid-column: 1 / -1;
                justify-self: start;
                margin: 0;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_ist(dt_utc):
    """IBTrACS timestamps are UTC; the top bar labels its time IST, so
    actually shift it (+5:30) rather than just relabeling the UTC value."""
    ist = dt_utc + pd.Timedelta(hours=5, minutes=30)
    return f"{ist.day} {ist.strftime('%b %Y %H:%M')}"


@st.cache_data
def load_outreach_cache():
    """Cached per-borrower outreach messages written by `python -m src.briefings`."""
    if not OUTREACH_PATH.exists():
        return {}
    return json.loads(OUTREACH_PATH.read_text(encoding="utf-8"))


def get_borrower_outreach(borrower_row):
    """(messages dict, source) for one HIGH-tier borrower: cached AI messages
    if present, else a freshly-computed plain-template fallback."""
    entry = load_outreach_cache().get(borrower_row["borrower_id"])
    if entry is not None:
        return entry["messages"], entry["source"]
    facts = build_borrower_facts(borrower_row)
    return template_outreach_message(facts), "template"


@st.cache_data
def load_countdown():
    return pd.read_parquet(PROCESSED / "countdown.parquet")


@st.cache_data
def get_current_frame(lead, scenario):
    """The single scored frame every screen reads for a given (lead,
    scenario) - countdown.parquet already carries the full per-borrower
    scoring for every timestep x scenario, so no re-join is needed here."""
    countdown_df = load_countdown()
    frame = countdown_df[(countdown_df["timestep"] == lead) & (countdown_df["scenario"] == scenario)]
    return frame.reset_index(drop=True)


def _init_countdown_state(lead_options, scenario_labels):
    """Idempotent: seeds the shared (lead, scenario) session state once, on
    first run. Default is lead_options[0] (T-72h, since options are sorted
    descending) and scenario_labels[0] (Inland, dict-first)."""
    if "countdown_lead" not in st.session_state:
        st.session_state["countdown_lead"] = lead_options[0]
    if "countdown_scenario_label" not in st.session_state:
        st.session_state["countdown_scenario_label"] = scenario_labels[0]


def current_lead_and_scenario():
    """Reads the shared (lead, scenario) selection - assumes
    _init_countdown_state() already ran earlier in this script run."""
    lead = st.session_state["countdown_lead"]
    scenario = COUNTDOWN_SCENARIO_LABELS[st.session_state["countdown_scenario_label"]]
    return lead, scenario


@st.cache_data
def _countdown_static_portfolio():
    return pd.read_csv(PROCESSED / "portfolio.csv")


@st.cache_data
def _countdown_scenario_tracks():
    return build_scenario_tracks(load_countdown_config())


@st.cache_data
def _countdown_villages_metric():
    return load_base_villages(load_countdown_config())


@st.cache_data
def _countdown_final_high_ids():
    return load_final_high_ids()


@st.cache_data
def _landfall_time():
    """Landfall time derived from the track itself: the ISO_TIME of whichever
    track point is closest to Puri town, not a hardcoded date."""
    points_metric = load_track_points()
    puri_metric = gpd.GeoSeries([Point(PURI_LATLON[1], PURI_LATLON[0])], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]
    nearest_idx = points_metric.geometry.distance(puri_metric).idxmin()
    landfall = pd.to_datetime(points_metric.loc[nearest_idx, "ISO_TIME"])
    print(f"[countdown] Landfall time derived from nearest track point to Puri: {landfall}")
    return landfall


def _single_select(container, label, values, format_func=str):
    """One compact 'All'-default selectbox instead of a multiselect. Returns
    None for 'All' (no filtering on this column)."""
    options = ["All"] + values
    choice = container.selectbox(label, options, format_func=lambda v: "All" if v == "All" else format_func(v))
    return None if choice == "All" else choice


def apply_filters(df):
    with st.expander("Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        district = _single_select(col1, "District", sorted(df["district"].unique()))
        branch = _single_select(col2, "Branch", sorted(df["branch_id"].unique()))
        tier = _single_select(
            col3, "Tier", [t for t in TIER_ORDER if t in df["tier"].unique()],
            format_func=lambda t: TIER_LABELS.get(t, t),
        )
        occupation = _single_select(
            col4, "Occupation", sorted(df["occupation"].unique()),
            format_func=lambda o: OCCUPATION_LABELS.get(o, o),
        )

    mask = pd.Series(True, index=df.index)
    if district is not None:
        mask &= df["district"] == district
    if branch is not None:
        mask &= df["branch_id"] == branch
    if tier is not None:
        mask &= df["tier"] == tier
    if occupation is not None:
        mask &= df["occupation"] == occupation
    return df[mask]


def _style_tier_column(display_df):
    """Keep risk tiers legible and muted rather than using saturated fills."""
    label_to_tier = {v: k for k, v in TIER_LABELS.items()}
    style_by_tier = {
        "high": ("#281D22", "#F2B8B8", "#C85A61"),
        "medium": ("#272319", "#E5C98A", "#C49A45"),
        "low": ("#1A232D", "#9FB1C4", "#617A92"),
        "not_exposed": ("#171F28", "#7F8B98", "#4A5663"),
    }

    def _cell_style(value):
        tier = label_to_tier.get(value)
        if tier not in style_by_tier:
            return ""
        background, text, accent = style_by_tier[tier]
        return (
            f"background-color: {background}; color: {text}; "
            f"border-left: 2px solid {accent}; font-weight: 600"
        )

    return display_df.style.map(_cell_style, subset=["Tier"])


def _style_act_tier_column(display_df):
    tier_styles = {
        "High": ("#4A252B", "#F2B8B8", "#C85A61"),
        "Medium": ("#403522", "#E5C98A", "#C49A45"),
        "Low": ("#202B38", "#9FB1C4", "#617A92"),
        "Not exposed": ("#171F28", "#7F8B98", "#4A5663"),
    }

    def _tier_cell_style(value):
        background, text, accent = tier_styles.get(value, ("#171F28", "#7F8B98", "#4A5663"))
        return (
            f"background-color: {background}; color: {text}; "
            f"border-left: 2px solid {accent}; font-weight: 600"
        )

    return display_df.style.map(_tier_cell_style, subset=["Tier"])


def _approved_outreach_ids():
    """Borrower ids marked 'Approved' - shared across every location the
    outreach widget is rendered."""
    return st.session_state.setdefault("approved_outreach_ids", set())


def _whatsapp_bubble(text):
    escaped = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="background:#e5ddd5;padding:16px;border-radius:8px;">
          <div style="background:#dcf8c6;border-radius:8px;padding:10px 14px;
                      max-width:480px;margin-left:auto;box-shadow:0 1px 1px rgba(0,0,0,0.15);
                      font-size:0.95rem;line-height:1.45;color:#111;">
            {escaped}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_outreach_widget(borrower_id, messages, source, location):
    """3-language WhatsApp-style preview of one borrower's outreach message,
    with Approve/Edit controls. `location` keeps widget keys unique when the
    same borrower is shown in two places at once."""
    prefix = f"outreach_{location}_{borrower_id}"
    editing = st.session_state.get(f"{prefix}_editing", False)

    tabs = st.tabs([OUTREACH_LANGUAGE_LABELS[lang] for lang in ("en", "hi", "or")])
    for lang, tab in zip(("en", "hi", "or"), tabs):
        with tab:
            text_key = f"{prefix}_{lang}_text"
            current_text = st.session_state.get(text_key, messages[lang])
            if editing:
                st.session_state[text_key] = st.text_area(
                    "Edit message", value=current_text, key=f"{text_key}_input", height=130,
                    label_visibility="collapsed",
                )
            else:
                _whatsapp_bubble(current_text)

    approve_col, edit_col = st.columns([1, 1])
    with approve_col:
        if borrower_id in _approved_outreach_ids():
            st.success("Approved")
        elif st.button("Approve", key=f"{prefix}_approve_btn"):
            _approved_outreach_ids().add(borrower_id)
            st.rerun()
    with edit_col:
        if st.button("Done editing" if editing else "Edit", key=f"{prefix}_edit_btn"):
            st.session_state[f"{prefix}_editing"] = not editing
            st.rerun()


def _plotly_layout(**overrides):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXT, size=12),
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        legend=dict(font=dict(color=MUTED, size=11), bgcolor="rgba(0,0,0,0)"),
    )
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return layout


def _plot(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_detail_panel(borrower):
    """"Why this borrower?" case-file panel: mono ID + tier pill, raw inputs,
    reasons as short lines, tier-colored contribution bars, and the
    recommended action highlighted in a callout strip."""
    st.markdown("<div class='sajag-divider'></div>", unsafe_allow_html=True)
    tier = borrower["tier"]
    tier_color = RISK_PALETTE.get(tier, MUTED)
    st.markdown(
        f"""
        <div class="sajag-case-header">
          <span class="sajag-case-id">{borrower['borrower_id']}</span>
          <span class="sajag-tier-pill" style="background:{tier_color};">{TIER_LABELS.get(tier, tier)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**District:** {borrower['district']}")
        st.markdown(f"**Subdivision:** {borrower['subdivision']}")
        st.markdown(f"**Branch:** {borrower['branch_id']}")
    with col2:
        occupation_label = OCCUPATION_LABELS.get(borrower["occupation"], borrower["occupation"])
        st.markdown(f"**Occupation:** {occupation_label}")
        st.markdown(f"**Loan amount:** {format_inr(borrower['loan_amount'])}")
        st.markdown(f"**EMI:** {format_inr(borrower['emi'])}")
    with col3:
        st.markdown(f"**Outstanding:** {format_inr(borrower['outstanding'])}")
        st.markdown(f"**Missed EMIs:** {borrower['missed_emis']}")
        band_label = EXPOSURE_BAND_LABELS.get(borrower["exposure_band"], borrower["exposure_band"])
        st.markdown(f"**Exposure band:** {band_label} ({borrower['distance_km']:.0f} km from track)")

    st.markdown("<div class='sajag-section-label'>WHY FLAGGED</div>", unsafe_allow_html=True)
    reasons_raw = borrower["reasons"]
    reasons = [r.strip() for r in str(reasons_raw).split(";") if r.strip()] if pd.notna(reasons_raw) else []
    if reasons:
        st.markdown("\n".join(f"- {reason}" for reason in reasons[:3]))
    else:
        st.markdown("- No scoring component crossed its reporting threshold.")

    components_fig = go.Figure(
        go.Bar(
            x=[borrower["exposure_component"], borrower["vulnerability_component"], borrower["stress_component"]],
            y=["Exposure", "Vulnerability", "Stress"],
            orientation="h",
            marker_color=tier_color,
        )
    )
    components_fig.update_layout(
        _plotly_layout(height=170, xaxis=dict(range=[0, 1]), margin=dict(l=90, r=20, t=8, b=8))
    )
    _plot(components_fig)

    st.markdown(
        f"<div class='sajag-action-strip'><strong>Recommended action:</strong> {borrower['suggested_action']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Outreach message**")
    if borrower["tier"] == "high":
        messages, source = get_borrower_outreach(borrower)
        render_outreach_widget(borrower["borrower_id"], messages, source, location="queue")
    else:
        st.caption("Outreach messages are drafted for High-priority borrowers only.")


def _comparison_assumptions_ui(config):
    """Assumptions expander: every number Measure uses, editable, seeded from
    config.yaml's comparison: block."""
    defaults = config.get("comparison", DEFAULT_ASSUMPTIONS)
    with st.expander("Assumptions", expanded=False):
        st.caption("Prototype assumptions - not fitted to real lender data. Edit freely; all values shown below.")
        outreach_cost = st.number_input(
            "Outreach cost per contacted borrower (₹)", min_value=0, value=int(defaults["outreach_cost_inr"]), step=10
        )
        col_h, col_m, col_l = st.columns(3)
        pd_uplift = {
            "high": col_h.number_input("PD uplift - High (pp)", min_value=0, max_value=100, value=int(defaults["pd_uplift_pp"]["high"])),
            "medium": col_m.number_input("PD uplift - Medium (pp)", min_value=0, max_value=100, value=int(defaults["pd_uplift_pp"]["medium"])),
            "low": col_l.number_input("PD uplift - Low (pp)", min_value=0, max_value=100, value=int(defaults["pd_uplift_pp"]["low"])),
        }
        lgd = st.slider("Loss given default (%)", min_value=0, max_value=100, value=int(defaults["lgd_pct"]))
        col_eff, col_eff_lite = st.columns(2)
        effectiveness = col_eff.slider(
            "Effectiveness - deferral + review (%)", min_value=0, max_value=100,
            value=int(defaults["effectiveness_pct"]),
        )
        outreach_only_effectiveness = col_eff_lite.slider(
            "Effectiveness - outreach only, no deferral (%)", min_value=0, max_value=100,
            value=int(defaults.get("outreach_only_effectiveness_pct", DEFAULT_ASSUMPTIONS["outreach_only_effectiveness_pct"])),
        )
        col_cof, col_mm = st.columns(2)
        cost_of_funds = col_cof.slider(
            "Annual cost of funds (%)", min_value=0, max_value=50,
            value=int(defaults.get("cost_of_funds_pct", DEFAULT_ASSUMPTIONS["cost_of_funds_pct"])),
        )
        moratorium_months = col_mm.number_input(
            "Moratorium length (months)", min_value=1, max_value=24,
            value=int(defaults.get("moratorium_months", DEFAULT_ASSUMPTIONS["moratorium_months"])),
        )
        st.caption(
            "Every strategy's cost is outreach/admin PLUS a liquidity cost on whatever it actually defers: "
            "(EMIs deferred) x cost of funds x (moratorium months / 12). Deferred cash itself is not a cost - "
            "it's money whose repayment is paused, not lost."
        )
    return {
        "outreach_cost_inr": outreach_cost,
        "pd_uplift_pp": pd_uplift,
        "lgd_pct": lgd,
        "effectiveness_pct": effectiveness,
        "outreach_only_effectiveness_pct": outreach_only_effectiveness,
        "cost_of_funds_pct": cost_of_funds,
        "moratorium_months": moratorium_months,
    }


def _forecast_quality_table(countdown_df, lead_times=(72, 48, 24, 12, 0)):
    """High/High+Medium counts and three consistency-with-landfall-assessment
    metrics vs. the T-0h/inland High set - labeled "consistency with landfall
    assessment" throughout, not accuracy."""
    summary = summarize_countdown(countdown_df)
    summary = summary[summary["lead_time_h"].isin(lead_times)].copy()
    summary["high_medium"] = summary["high"] + summary["medium"]
    summary = summary.sort_values(["lead_time_h", "scenario"], ascending=[False, True])

    return pd.DataFrame(
        {
            "Lead time": summary["lead_time_h"].map(lambda h: f"T-{h}h"),
            "Scenario": summary["scenario"].map(lambda s: COUNTDOWN_LINE_STYLE.get(s, (None, s))[1]),
            "High": summary["high"],
            "High + Medium": summary["high_medium"],
            "Strict precision (%)": summary["strict_precision_pct"],
            "Strict recall (%)": summary["strict_recall_pct"],
            "Lenient recall (%)": summary["recall_pct"],
        }
    )


_STRATEGY_LABELS = [("Do nothing", "do_nothing"), ("Targeted", "targeted"), ("Blanket", "blanket"), ("Staged", "staged")]
_STRATEGY_METRIC_ROWS = [
    ("Borrowers contacted", "contacted", lambda v: f"{v:,}"),
    ("Deferred cash (EMIs)", "deferred_inr", format_inr_lakh_cr),
    ("Outreach/admin cost", "outreach_cost_inr", format_inr_lakh_cr),
    ("Liquidity cost of deferral", "liquidity_cost_inr", format_inr_lakh_cr),
    ("Total cost", "total_cost_inr", format_inr_lakh_cr),
    ("Expected loss avoided", "expected_loss_avoided_inr", format_inr_lakh_cr),
    ("Net vs. do nothing", "net_vs_do_nothing_inr", format_inr_lakh_cr),
]


def _strategy_comparison_table(result):
    columns = {}
    for label, key in _STRATEGY_LABELS:
        if key not in result:
            continue
        r = result[key]
        columns[label] = [fmt(r[metric_key]) for _, metric_key, fmt in _STRATEGY_METRIC_ROWS]
    return pd.DataFrame(columns, index=[row[0] for row in _STRATEGY_METRIC_ROWS])


def _district_high_bar_chart(df):
    """Districts ranked by High-tier count. None (caller shows a caption
    instead) if there are zero High borrowers to rank."""
    counts = df[df["tier"] == "high"].groupby("district").size().sort_values(ascending=True)
    if counts.empty:
        return None
    colors = [
        "#C85A61" if value == counts.max() else "rgba(153, 82, 88, 0.72)"
        for value in counts.values
    ]
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h", marker_color=colors,
        text=counts.values, textposition="outside", cliponaxis=False,
        textfont=dict(family="IBM Plex Mono, monospace", color=MUTED, size=11),
    ))
    fig.update_layout(_plotly_layout(
        height=max(210, 31 * len(counts) + 20), bargap=0.72,
        margin=dict(l=8, r=40, t=4, b=4),
        xaxis=dict(title=None, showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(color=TEXT, ticksuffix="  ", automargin=True),
    ))
    return fig


def _next_actions_table(df, config):
    actions_cfg = config["priority"]["actions"]
    agg = (
        df.groupby("tier")
        .agg(borrowers=("borrower_id", "count"), outstanding=("outstanding", "sum"))
        .reindex(TIER_ORDER, fill_value=0)
        .reset_index()
    )
    return pd.DataFrame(
        {
            "Tier": agg["tier"].map(lambda t: TIER_LABELS.get(t, t)),
            "Borrowers": agg["borrowers"],
            "Recommended action": agg["tier"].map(lambda t: actions_cfg.get(t, "")),
            "₹ Outstanding": agg["outstanding"].map(format_inr_lakh_cr),
        }
    )


def _render_next_actions(df, config):
    rows = _next_actions_table(df, config).to_dict(orient="records")
    tier_styles = {
        "high": "high",
        "medium": "medium",
        "low": "low",
        "not exposed": "not-exposed",
    }
    table_rows = []
    for row in rows:
        tier = str(row["Tier"])
        tier_style = tier_styles.get(tier.casefold(), "not-exposed")
        borrower_count = f"{row['Borrowers']:,}"
        table_rows.append(
            "<tr>"
            f"<td><span class='sajag-action-tier sajag-action-tier--{tier_style}'>{html.escape(tier)}</span></td>"
            f"<td class='sajag-actions-count'>{html.escape(borrower_count)}</td>"
            f"<td>{html.escape(str(row['Recommended action']))}</td>"
            f"<td class='sajag-actions-amount'>{html.escape(str(row['₹ Outstanding']))}</td>"
            "</tr>"
        )
    st.markdown(
        "<div class='sajag-actions-scroll'>"
        "<table class='sajag-actions-table'>"
        "<thead><tr><th>Risk tier</th><th>Borrowers</th><th>Recommended action</th>"
        "<th>Outstanding</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )


def _lead_time_line_chart(summary_table, y_col, y_title, height=260):
    """One trace per scenario, colored via COUNTDOWN_LINE_STYLE - used for
    both High count and strict precision vs. lead time on Measure."""
    fig = go.Figure()
    for scenario, (color, label) in COUNTDOWN_LINE_STYLE.items():
        sub = summary_table[summary_table["scenario"] == scenario].sort_values("lead_time_h", ascending=False)
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["lead_time_h"], y=sub[y_col], mode="lines+markers", name=label,
                line=dict(color=color, width=2), marker=dict(size=5),
            )
        )
    fig.update_layout(
        _plotly_layout(
            xaxis=dict(title="Lead time (hours before landfall)", autorange="reversed"),
            yaxis=dict(title=y_title),
            height=height,
            legend=dict(orientation="h", y=-0.25),
        )
    )
    return fig


def _precision_funnel_rows(summary_table, scenario="inland", stops=TIMELINE_STOPS):
    """[(label, high_count, strict_precision_pct)] at each canonical stop -
    the data behind both funnel renderings (Measure's Plotly chart and the
    landing page's HTML funnel)."""
    rows = []
    for h in stops:
        row = summary_table[(summary_table["lead_time_h"] == h) & (summary_table["scenario"] == scenario)]
        if row.empty:
            continue
        r = row.iloc[0]
        label = "Landfall" if h == 0 else f"T-{h}h"
        rows.append((label, int(r["high"]), float(r["strict_precision_pct"])))
    return rows


def _precision_funnel_chart(summary_table, scenario="inland", stops=TIMELINE_STOPS):
    """The "list narrowing" visual: a native Plotly funnel (decreasing-width
    by construction, no custom animation needed) of High-tier count at each
    canonical stop, strict precision labeled beside each one."""
    rows = _precision_funnel_rows(summary_table, scenario, stops)
    if not rows:
        return None
    fig = go.Figure(
        go.Funnel(
            y=[r[0] for r in rows],
            x=[r[1] for r in rows],
            text=[f"{v:,} · {p:.0f}% precision" for _, v, p in rows],
            textinfo="text",
            textposition="inside",
            textfont=dict(color="#F3E9E7", size=13),
            marker=dict(color=RISK_PALETTE["high"]),
            connector=dict(line=dict(color=GRID, width=1)),
        )
    )
    fig.update_layout(_plotly_layout(height=320))
    return fig


def render_measure(df, config):
    st.subheader("Measure")
    st.markdown(
        "<p class='sajag-measure-intro'>Does targeted outreach reduce expected losses enough to justify its cost?</p>",
        unsafe_allow_html=True,
    )
    if df.empty:
        st.info("No borrowers match the current filters.")
        return

    assumptions = _comparison_assumptions_ui(config)
    _, scenario = current_lead_and_scenario()
    df_t72 = get_current_frame(72, scenario)
    df_t24 = get_current_frame(24, scenario)
    result = compare_strategies(df, assumptions, df_t72=df_t72, df_t24=df_t24)

    breakeven = breakeven_effectiveness_pct(df, assumptions)
    if breakeven is None:
        break_even_value = "n/a"
        if result["targeted"]["contacted"] == 0:
            break_even_note = "No High- or Medium-priority borrowers are present in the current filtered view."
        else:
            break_even_note = "No modeled avoided-loss base is available for the targeted cohort under these assumptions."
    else:
        pct_display = f"{min(breakeven, 999):.1f}%" if breakeven <= 100 else f"{breakeven:.0f}%"
        break_even_value = pct_display
        if breakeven > 100:
            break_even_note = (
                "Exceeds 100%; under these assumptions, no achievable effectiveness makes outreach's benefit exceed its cost."
            )
        else:
            break_even_note = (
                "Below this share, targeted outreach's cost exceeds its expected benefit; above it, benefit exceeds cost."
            )
    st.markdown(
        f"<div class='sajag-measure-break-even'><span>Break-even effectiveness</span>"
        f"<strong>{html.escape(break_even_value)}</strong>"
        "<span>of excess post-disaster defaults prevented</span>"
        f"<span>· {html.escape(break_even_note)}</span></div>",
        unsafe_allow_html=True,
    )

    def strategy_metrics(summary, is_targeted=False):
        net_value = summary["net_vs_do_nothing_inr"]
        metric_items = (
            ("Borrowers contacted", f"{summary['contacted']:,}", ""),
            ("Expected loss avoided", format_inr_lakh_cr(summary["expected_loss_avoided_inr"]), ""),
            ("Total strategy cost", format_inr_lakh_cr(summary["total_cost_inr"]), ""),
            ("Net vs. do nothing", format_inr_lakh_cr(net_value), " sajag-measure-metric--net"),
        )
        metrics_html = "".join(
            f"<div class='{metric_class.strip()}'>"
            f"<div class='sajag-measure-metric-label'>{html.escape(label)}</div>"
            f"<div class='sajag-measure-metric-value'>{html.escape(value)}</div></div>"
            for label, value, metric_class in metric_items
        )
        strategy_class = " sajag-measure-strategy--targeted" if is_targeted else ""
        note = (
            "Reactive baseline · no proactive outreach"
            if not is_targeted
            else "Targeted outreach to High + Medium borrowers under the current policy"
        )
        return (
            f"<section class='sajag-measure-strategy{strategy_class}'>"
            f"<div class='sajag-measure-strategy-title'>{'Targeted outreach' if is_targeted else 'Do nothing'}</div>"
            f"<div class='sajag-measure-strategy-note'>{note}</div>"
            f"<div class='sajag-measure-metrics'>{metrics_html}</div></section>"
        )

    comparison_html = (
        "<div class='sajag-measure-comparison'>"
        f"{strategy_metrics(result['do_nothing'])}"
        f"{strategy_metrics(result['targeted'], is_targeted=True)}</div>"
    )
    st.markdown(comparison_html, unsafe_allow_html=True)

    with st.expander("Strategy comparison and methodology", expanded=False):
        st.caption(
            "Do nothing is the reactive baseline, with zero cost and zero avoided loss by definition. "
            "Targeted contacts High + Medium borrowers. Blanket offers a moratorium to every exposed borrower. "
            "Staged uses outreach-only at T-72h, then deferral and review for borrowers still High at T-24h "
            f"under the {COUNTDOWN_LINE_STYLE.get(scenario, (None, scenario))[1]} scenario. Deferred cash is "
            "repayment paused, not a cost; strategy costs include outreach/admin and liquidity cost. Net vs. "
            "do-nothing shows each strategy's outcome relative to the baseline, not a ranking."
        )
        st.dataframe(_strategy_comparison_table(result), width="stretch")

    with st.expander("Sensitivity analysis", expanded=False):
        st.caption("Net benefit if the post-disaster PD uplift is 0.5x / 1x / 1.5x the assumption above.")
        sens_rows = sensitivity_table(df, assumptions)
        sens_display = pd.DataFrame(
            {
                "PD uplift": [f"{r['multiplier']:.1f}x" for r in sens_rows],
                "Targeted net benefit": [format_inr_lakh_cr(r["targeted_net_inr"]) for r in sens_rows],
                "Blanket net benefit": [format_inr_lakh_cr(r["blanket_net_inr"]) for r in sens_rows],
            }
        )
        st.dataframe(sens_display, hide_index=True, width="stretch")

    with st.expander("Forecast evidence and list narrowing", expanded=False):
        st.markdown("<div class='sajag-section-label'>LIST NARROWING</div>", unsafe_allow_html=True)
        st.caption("How the High-tier list narrows as landfall approaches, inland scenario.")
        summary_table = summarize_countdown(load_countdown())
        funnel_fig = _precision_funnel_chart(summary_table, scenario="inland")
        if funnel_fig is not None:
            _plot(funnel_fig)

        st.markdown("<div class='sajag-section-label'>HIGH COUNT & STRICT PRECISION VS. LEAD TIME</div>", unsafe_allow_html=True)
        st.caption(
            "As landfall approaches, the High-tier list narrows and gets more precise - consistency with the "
            "landfall assessment, not accuracy in an absolute sense, since the T-0h read is itself only ever an estimate."
        )
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            _plot(_lead_time_line_chart(summary_table, "high", "High-tier count"))
        with chart_col2:
            _plot(_lead_time_line_chart(summary_table, "strict_precision_pct", "Strict precision (%)"))

        st.markdown(
            "<div class='sajag-section-label'>FORECAST QUALITY: CONSISTENCY WITH LANDFALL ASSESSMENT</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Strict precision = of borrowers flagged High at this lead time, % still High at T-0h. "
            "Strict recall = of borrowers who end up High at T-0h, % already High at this lead time. "
            "Lenient recall = of borrowers who end up High at T-0h, % already High-or-Medium at this lead time."
        )
        st.dataframe(_forecast_quality_table(load_countdown()), hide_index=True, width="stretch")


def _subdivision_outline_geojson(districts):
    tehsils = gpd.read_file(PROCESSED / "odisha_tehsils.geojson")
    tehsils = tehsils[tehsils["NAME_2"].isin(districts)]
    return json.loads(tehsils.to_json())


def _exposure_zone_geojson(band):
    zones = gpd.read_file(PROCESSED / "exposure_zones.geojson")
    return json.loads(zones[zones["band"] == band].to_json())


def _village_agg(df):
    df = df.copy()
    df["is_high"] = df["tier"] == "high"
    df["is_medium"] = df["tier"] == "medium"

    agg = (
        df.groupby("village_id")
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            district=("district", "first"),
            subdivision=("subdivision", "first"),
            borrowers=("borrower_id", "count"),
            high_count=("is_high", "sum"),
            medium_count=("is_medium", "sum"),
            outstanding=("outstanding", "sum"),
        )
        .reset_index()
    )
    agg["tooltip"] = agg.apply(
        lambda r: (
            f"<b>Village {r['village_id']}</b><br/>{r['subdivision']}, {r['district']}<br/>"
            f"Borrowers: {r['borrowers']}<br/>High: {r['high_count']} &middot; Medium: {r['medium_count']}<br/>"
            f"Outstanding: {format_inr_lakh_cr(r['outstanding'])}"
        ),
        axis=1,
    )
    return agg


def _branch_agg(df):
    """Per-branch borrowers/high/medium counts and exposed outstanding, joined
    to each branch's actual (subdivision-centroid) location."""
    branches = pd.read_csv(PROCESSED / "branches.csv")

    df = df.copy()
    df["is_high"] = df["tier"] == "high"
    df["is_medium"] = df["tier"] == "medium"
    df["is_exposed"] = df["tier"] != "not_exposed"
    df["exposed_outstanding"] = df["outstanding"].where(df["is_exposed"], 0.0)

    agg = (
        df.groupby("branch_id")
        .agg(
            borrowers=("borrower_id", "count"),
            high_count=("is_high", "sum"),
            medium_count=("is_medium", "sum"),
            exposed_outstanding=("exposed_outstanding", "sum"),
        )
        .reset_index()
    )

    merged = branches.merge(agg, on="branch_id", how="inner")
    merged["position"] = merged[["lon", "lat"]].values.tolist()
    merged["marker"] = "^"
    merged["tooltip"] = merged.apply(
        lambda r: (
            f"<b>Branch {r['branch_id']}</b><br/>{r['subdivision']}, {r['district']}<br/>"
            f"Borrowers: {r['borrowers']}<br/>High priority: {r['high_count']}<br/>"
            f"Exposed outstanding: {format_inr_lakh_cr(r['exposed_outstanding'])}"
        ),
        axis=1,
    )
    return merged


@st.cache_data
def _borrower_jitter_offsets(borrower_ids):
    """One fixed jitter offset per borrower_id, computed once over the full
    portfolio so a borrower's dot doesn't move when unrelated filters change."""
    rng = np.random.default_rng(JITTER_SEED)
    offsets = rng.normal(0, JITTER_STD_DEG, size=(len(borrower_ids), 2))
    return pd.DataFrame({"borrower_id": borrower_ids, "dlon": offsets[:, 0], "dlat": offsets[:, 1]})


def _borrower_jittered_points(df, jitter):
    merged = df.merge(jitter, on="borrower_id", how="left")
    merged["lat"] = merged["lat"] + merged["dlat"]
    merged["lon"] = merged["lon"] + merged["dlon"]
    return merged


def _branch_alert_table(df):
    agg = _branch_agg(df)
    table = pd.DataFrame(
        {
            "Branch": agg["branch_id"],
            "District": agg["district"],
            "High": agg["high_count"],
            "Medium": agg["medium_count"],
            "₹ Exposed": agg["exposed_outstanding"].map(format_inr_lakh_cr),
            "_sort": agg["high_count"],
        }
    ).sort_values("_sort", ascending=False).drop(columns="_sort")
    return table


def _render_act_branch_table(branch_table):
    rows = "".join(
        "<tr>"
        f"<td class='sajag-act-branch-id'>{html.escape(str(row['Branch']))}</td>"
        f"<td>{html.escape(str(row['District']))}</td>"
        f"<td class='sajag-act-number sajag-act-high-count'>{int(row['High']):,}</td>"
        f"<td class='sajag-act-number'>{int(row['Medium']):,}</td>"
        f"<td class='sajag-act-number sajag-act-currency'>{html.escape(str(row['₹ Exposed']))}</td>"
        "</tr>"
        for _, row in branch_table.iterrows()
    )
    st.markdown(
        "<div class='sajag-act-branch-scroll'>"
        "<table class='sajag-act-branch-table'>"
        "<thead><tr><th>Branch</th><th>District</th><th class='sajag-act-number'>High</th>"
        "<th class='sajag-act-number'>Medium</th><th class='sajag-act-number'>₹ Exposed</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_legend():
    st.markdown("**Legend**")
    for tier in TIER_ORDER:
        color = RISK_PALETTE[tier]
        label = TIER_LABELS[tier]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="width:14px;height:14px;border-radius:50%;background:{color};'
            f'display:inline-block;"></span><span>{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption("Circle size = outstanding ₹ at that village. High markers carry a soft halo.")
    st.caption("Zones: light red = within 50 km of track, light amber = 50-120 km.")


MAP_INDIVIDUAL_BORROWER_CAP = 800

_BORROWER_ID_RE = re.compile(r"\bB\d{4}\b")
_VILLAGE_ID_RE = re.compile(r"\bV\d{4}\b")


def _geojson_style_fn(color, weight=1, fill=False, fill_opacity=0.0, dash=None):
    style = {"color": color, "weight": weight, "fillOpacity": fill_opacity if fill else 0}
    if fill:
        style["fillColor"] = color
    if dash:
        style["dashArray"] = dash
    return lambda feature: style


def _borrower_popup_html(row):
    reasons = [r.strip() for r in str(row.get("reasons", "")).split(";") if r.strip()][:2]
    reasons_html = "".join(f"<div>&bull; {r}</div>" for r in reasons) if reasons else ""
    return (
        f"<b>{row['borrower_id']}</b><br/>"
        f"{TIER_LABELS.get(row['tier'], row['tier'])} &middot; "
        f"{OCCUPATION_LABELS.get(row['occupation'], row['occupation'])}<br/>"
        f"Outstanding: {format_inr(row['outstanding'])}<br/>"
        f"Action: {row['suggested_action']}<br/>{reasons_html}"
    )


def _folium_base_map():
    """The one dark basemap every folium map in the app uses - free, no API
    key (Esri's World Dark Gray Canvas; Carto's raster tiles now require a
    paid key)."""
    return folium.Map(
        location=[MAP_INITIAL_VIEW["latitude"], MAP_INITIAL_VIEW["longitude"]],
        zoom_start=MAP_INITIAL_VIEW["zoom"],
        tiles=DARK_TILE_URL,
        attr=DARK_TILE_ATTR,
    )


def _add_tier_marker(m, lat, lon, tier, radius, tooltip, popup_html=None):
    """A village/borrower marker in its tier color. High-tier markers get a
    soft halo (a larger, low-opacity circle underneath) - the one place
    glow/decoration is allowed, so the highest-risk markers visually pop."""
    color = RISK_PALETTE.get(tier, RISK_PALETTE["not_exposed"])
    if tier == "high":
        folium.CircleMarker(
            location=[lat, lon], radius=radius * 1.8, color=color, weight=0,
            fill=True, fill_color=color, fill_opacity=0.06,
        ).add_to(m)
    popup = folium.Popup(popup_html, max_width=260) if popup_html else None
    fill_opacity = 0.82 if tier == "high" else (0.76 if tier == "medium" else 0.5)
    folium.CircleMarker(
        location=[lat, lon], radius=radius, color=color, fill=True, fill_color=color,
        fill_opacity=fill_opacity, weight=0.8, tooltip=tooltip, popup=popup,
    ).add_to(m)


def _add_countdown_track_layers(m, lead, scenario):
    """Track/cone for the currently selected (lead, scenario). Forecast
    (cone + dashed segment) in accent cyan, observed track solid off-white -
    keeps "the forecast" visually distinct from the tier-red risk signal."""
    config = load_countdown_config()
    scenario_tracks = _countdown_scenario_tracks()
    villages_metric = _countdown_villages_metric()
    cd_portfolio = _countdown_static_portfolio()
    final_high_ids = _countdown_final_high_ids()
    landfall_time = _landfall_time()

    offset_km, sign, error_km, _ = resolve_offset(
        lead, scenario, cd_portfolio, villages_metric, scenario_tracks, config, final_high_ids
    )
    observed_geojson, forecast_geojson, cone_geojson = _countdown_track_geojson(
        lead, sign, offset_km, error_km, scenario_tracks, landfall_time
    )

    if cone_geojson["features"]:
        folium.GeoJson(cone_geojson, style_function=_geojson_style_fn(ACCENT, fill=True, fill_opacity=0.16)).add_to(m)
    folium.GeoJson(forecast_geojson, style_function=_geojson_style_fn(ACCENT, weight=2, dash="6,6")).add_to(m)
    if observed_geojson["features"]:
        folium.GeoJson(observed_geojson, style_function=_geojson_style_fn(TEXT, weight=4)).add_to(m)


def _resolve_map_click(map_state, df):
    """Match the clicked marker's tooltip text back to a borrower_id (direct)
    or village_id (proxy: that village's highest-priority borrower)."""
    tooltip_text = (map_state or {}).get("last_object_clicked_tooltip")
    if not tooltip_text:
        return None

    borrower_match = _BORROWER_ID_RE.search(tooltip_text)
    if borrower_match and (df["borrower_id"] == borrower_match.group()).any():
        return df[df["borrower_id"] == borrower_match.group()].iloc[0], None

    village_match = _VILLAGE_ID_RE.search(tooltip_text)
    if village_match:
        village_rows = df[df["village_id"] == village_match.group()]
        if not village_rows.empty:
            top = village_rows.sort_values("priority_score", ascending=False).iloc[0]
            return top, village_match.group()

    return None


def _render_priority_overview(df, weights):
    counts = df["tier"].value_counts()
    tier_rows = []
    meter = []
    for tier in TIER_ORDER:
        count = int(counts.get(tier, 0))
        label = TIER_LABELS.get(tier, tier)
        tier_rows.append(
            f"<div class='sajag-priority-count sajag-priority-count--{tier}'>"
            f"<div class='sajag-priority-value'>{count:,}</div>"
            f"<div class='sajag-priority-label'>{html.escape(label)}</div></div>"
        )
        percentage = count / len(df) * 100 if len(df) else 0
        meter.append(
            f"<span style='width:{percentage:.2f}%;background:{RISK_PALETTE[tier]}' "
            f"title='{html.escape(label)}: {count:,}'></span>"
        )

    driver_labels = (
        ("Exposure", "exposure"),
        ("Livelihood vulnerability", "vulnerability"),
        ("Repayment stress", "stress"),
    )
    drivers = "".join(
        f"<div class='sajag-priority-driver'><span>{label}</span>"
        f"<strong>{weights[key] * 100:.0f}% weight</strong></div>"
        for label, key in driver_labels
    )
    st.markdown(
        f"<div class='sajag-priority-overview'>"
        f"<div class='sajag-priority-counts'>{''.join(tier_rows)}</div>"
        f"<div class='sajag-priority-meter' role='img' aria-label='Risk distribution'>{''.join(meter)}</div>"
        "<div class='sajag-priority-driver-head'>WHAT SHAPES THE PRIORITY SCORE</div>"
        f"<div class='sajag-priority-drivers'>{drivers}</div></div>",
        unsafe_allow_html=True,
    )


def render_prioritize(df, config):
    st.subheader("Prioritize")
    st.markdown(
        "<p class='sajag-page-intro'>Who is at risk, and what is driving the priority?</p>",
        unsafe_allow_html=True,
    )
    weights = config["priority"]["weights"]
    if df.empty:
        st.info("No borrowers match the current filters.")
        return

    _render_priority_overview(df, weights)
    sorted_df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    st.markdown("<div class='sajag-section-head'>BORROWER REVIEW QUEUE</div>", unsafe_allow_html=True)
    queue_col, download_col = st.columns([1, 5], vertical_alignment="center")
    with queue_col:
        st.caption(f"{len(sorted_df):,} borrowers in current view")
    with download_col:
        csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered queue as CSV",
            data=csv_bytes,
            file_name="review_queue.csv",
            mime="text/csv",
        )

    display = pd.DataFrame(
        {
            "Tier": sorted_df["tier"].map(lambda t: TIER_LABELS.get(t, t)),
            "Borrower ID": sorted_df["borrower_id"],
            "District": sorted_df["district"],
            "Branch": sorted_df["branch_id"],
            "Occupation": sorted_df["occupation"].map(lambda o: OCCUPATION_LABELS.get(o, o)),
            "Priority": sorted_df["priority_score"],
            "Outstanding": sorted_df["outstanding"].map(format_inr),
            "Action": sorted_df["suggested_action"],
        }
    )

    event = st.dataframe(
        _style_tier_column(display),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="review_queue_table",
        column_config={
            "Priority": st.column_config.ProgressColumn("Priority", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    selected_row = None
    village_proxy = None
    selected_rows = event.selection.rows if event is not None else []
    if selected_rows:
        selected_row = sorted_df.iloc[selected_rows[0]]

    map_state = None
    with st.expander("Spatial exposure and borrower map", expanded=False):
        lead, scenario = current_lead_and_scenario()
        show_individual = st.checkbox("Show individual borrowers")
        if show_individual and len(df) > MAP_INDIVIDUAL_BORROWER_CAP:
            st.caption(
                f"Narrow the filters to {MAP_INDIVIDUAL_BORROWER_CAP:,} borrowers or fewer to show individual "
                f"markers (currently {len(df):,}). Showing village-level markers instead."
            )
            show_individual = False

        m = _folium_base_map()
        districts = sorted(df["district"].unique())
        folium.GeoJson(
            _subdivision_outline_geojson(districts),
            style_function=_geojson_style_fn(MUTED, weight=1),
        ).add_to(m)
        folium.GeoJson(
            _exposure_zone_geojson("medium"),
            style_function=_geojson_style_fn(RISK_PALETTE["medium"], fill=True, fill_opacity=0.12),
        ).add_to(m)
        folium.GeoJson(
            _exposure_zone_geojson("high"),
            style_function=_geojson_style_fn(RISK_PALETTE["high"], fill=True, fill_opacity=0.14),
        ).add_to(m)
        _add_countdown_track_layers(m, lead, scenario)

        if show_individual:
            jitter = _borrower_jitter_offsets(tuple(sorted(df["borrower_id"])))
            for row in _borrower_jittered_points(df, jitter).itertuples():
                _add_tier_marker(m, row.lat, row.lon, row.tier, 5, row.borrower_id, _borrower_popup_html(row._asdict()))
        else:
            village_df = _village_agg(df)
            max_outstanding = village_df["outstanding"].max() or 1
            for row in village_df.itertuples():
                tier = "high" if row.high_count > 0 else ("medium" if row.medium_count > 0 else "low")
                radius = 6 + 12 * (row.outstanding / max_outstanding) ** 0.5
                _add_tier_marker(m, row.lat, row.lon, tier, radius, f"Village {row.village_id}", row.tooltip)

        branch_df = _branch_agg(df)
        for row in branch_df.itertuples():
            folium.Marker(
                location=[row.lat, row.lon],
                icon=folium.DivIcon(html=f'<div style="color:{TEXT};font-size:16px;font-weight:700;">^</div>'),
                tooltip=f"Branch {row.branch_id}",
                popup=folium.Popup(row.tooltip, max_width=260),
            ).add_to(m)

        map_col, legend_col = st.columns([6, 1])
        with map_col:
            map_state = st_folium(
                m,
                height=500,
                use_container_width=True,
                returned_objects=["last_object_clicked_tooltip"],
            )
        with legend_col:
            render_legend()

    click_result = _resolve_map_click(map_state, df)
    if click_result is not None:
        selected_row, village_proxy = click_result

    if selected_row is not None:
        if village_proxy:
            st.markdown(f"**Top-priority borrower in this village** ({village_proxy})")
        render_detail_panel(selected_row)


def _countdown_track_geojson(lead, sign, offset_km, error_km, scenario_tracks, landfall_time):
    """observed/forecast/cone GeoJSON (EPSG:4326) for one countdown frame."""
    current_time = landfall_time - pd.Timedelta(hours=lead)

    true_points = load_track_points().to_crs("EPSG:4326")
    observed = true_points[pd.to_datetime(true_points["ISO_TIME"]) <= current_time]
    if len(observed) >= 2:
        observed_line = LineString([(p.x, p.y) for p in observed.geometry])
        observed_geojson = json.loads(gpd.GeoDataFrame(geometry=[observed_line], crs="EPSG:4326").to_json())
    else:
        observed_geojson = {"type": "FeatureCollection", "features": []}

    forecast_points = shifted_track_points(scenario_tracks, sign, offset_km)
    forecast_line_metric = LineString([(p.x, p.y) for p in forecast_points.geometry])
    forecast_geojson = json.loads(
        gpd.GeoDataFrame(geometry=[forecast_line_metric], crs=METRIC_CRS).to_crs("EPSG:4326").to_json()
    )

    if error_km > 0:
        cone_geojson = json.loads(
            gpd.GeoDataFrame(geometry=[forecast_line_metric.buffer(error_km * 1000)], crs=METRIC_CRS)
            .to_crs("EPSG:4326")
            .to_json()
        )
    else:
        cone_geojson = {"type": "FeatureCollection", "features": []}

    return observed_geojson, forecast_geojson, cone_geojson


def _render_sense_map(joined_df, lead, scenario, map_key):
    """Sense's overview map: folium, on the same shared dark basemap
    Prioritize's map uses. Bird's-eye only - no individual-borrower toggle,
    no click handling (Prioritize already owns "why this borrower")."""
    m = _folium_base_map()
    districts = sorted(joined_df["district"].unique())
    folium.GeoJson(_subdivision_outline_geojson(districts), style_function=_geojson_style_fn(MUTED, weight=1)).add_to(m)
    folium.GeoJson(
        _exposure_zone_geojson("medium"), style_function=_geojson_style_fn(RISK_PALETTE["medium"], fill=True, fill_opacity=0.12)
    ).add_to(m)
    folium.GeoJson(
        _exposure_zone_geojson("high"), style_function=_geojson_style_fn(RISK_PALETTE["high"], fill=True, fill_opacity=0.14)
    ).add_to(m)
    _add_countdown_track_layers(m, lead, scenario)

    village_df = _village_agg(joined_df)
    max_outstanding = village_df["outstanding"].max() or 1
    for row in village_df.itertuples():
        tier = "high" if row.high_count > 0 else ("medium" if row.medium_count > 0 else "low")
        radius = 5 + 11 * (row.outstanding / max_outstanding) ** 0.5
        _add_tier_marker(m, row.lat, row.lon, tier, radius, f"Village {row.village_id}", row.tooltip)

    st_folium(m, height=460, use_container_width=True, returned_objects=[], key=map_key)


def _kpi_row(cards):
    """cards: [(label, value, primary)] rendered as one equal-width row."""
    cells = "".join(
        f"<div class='sajag-kpi{' sajag-kpi--primary' if primary else ''}'>"
        f"<div class='sajag-kpi-value'>{value}</div>"
        f"<div class='sajag-kpi-label'>{label}</div>"
        f"</div>"
        for label, value, primary in cards
    )
    st.markdown(f"<div class='sajag-kpi-row'>{cells}</div>", unsafe_allow_html=True)


def _render_sense_frame(lead, scenario, config, landfall_time):
    event_name = config["event"]["name"]
    current_time = landfall_time - pd.Timedelta(hours=lead)
    date_str = f"{current_time.day} {current_time.strftime('%b %Y %H:%M')}"

    joined = get_current_frame(lead, scenario)
    exposed = joined[joined["tier"] != "not_exposed"]
    high = joined[joined["tier"] == "high"]

    _kpi_row([
        ("Borrowers exposed", f"{len(exposed):,}", False),
        ("Flagged for early outreach", f"{len(high):,}", True),
        ("Outstanding in High tier", format_inr_lakh_cr(high["outstanding"].sum()), False),
    ])

    # Key includes lead: Play redraws this frame once per stop within a single run.
    with st.container(key=f"sajag_risk_panel_{lead}"):
        map_col, side_col = st.columns([13, 7], gap="large")
        with map_col:
            st.markdown(
                "<div class='sajag-panel-head'><span class='sajag-section-label'>EXPOSURE MAP</span></div>",
                unsafe_allow_html=True,
            )
            _render_sense_map(joined, lead, scenario, map_key=f"sense_map_{lead}_{scenario}")

        with side_col:
            st.markdown(
                f"<div class='sajag-panel-head'><span class='sajag-section-label'>WHERE TO ACT FIRST</span>"
                f"<span class='sajag-panel-context'><span class='sajag-mono'>T&minus;{lead}h</span>"
                f" &middot; {html.escape(event_name)} &middot; {date_str}</span></div>",
                unsafe_allow_html=True,
            )
            district_fig = _district_high_bar_chart(joined)
            if district_fig is not None:
                _plot(district_fig)
            else:
                st.caption("No High-tier borrowers at this lead time/scenario.")

    st.markdown("<div class='sajag-section-head'>NEXT ACTIONS</div>", unsafe_allow_html=True)
    _render_next_actions(joined, config)


def render_sense(config, play_container=st):
    landfall_time = _landfall_time()
    current_lead, current_scenario = current_lead_and_scenario()
    st.subheader("Sense")
    st.markdown(
        "<p class='sajag-page-intro'>Hazard position, forecast uncertainty, and borrower exposure for the selected scenario.</p>",
        unsafe_allow_html=True,
    )
    play_clicked = play_container.button("▶ Play")

    placeholder = st.empty()

    def draw(lead_value, scenario_value):
        with placeholder.container():
            _render_sense_frame(lead_value, scenario_value, config, landfall_time)

    if play_clicked:
        for h in TIMELINE_STOPS:
            draw(h, current_scenario)
            time.sleep(1.5)
        # countdown_lead is never a live widget key (the timeline is plain
        # st.buttons), so it's always safe to reassign it here directly.
        st.session_state["countdown_lead"] = TIMELINE_STOPS[-1]
        st.rerun()
    else:
        draw(current_lead, current_scenario)


def _branch_team_sheet_csv(branch_df):
    actionable = branch_df[branch_df["tier"] != "not_exposed"].sort_values("priority_score", ascending=False)
    sheet = pd.DataFrame(
        {
            "borrower_id": actionable["borrower_id"],
            "village_id": actionable["village_id"],
            "subdivision": actionable["subdivision"],
            "occupation": actionable["occupation"].map(lambda o: OCCUPATION_LABELS.get(o, o)),
            "tier": actionable["tier"].map(lambda t: TIER_LABELS.get(t, t)),
            "missed_emis": actionable["missed_emis"],
            "outstanding_inr": actionable["outstanding"],
            "priority_score": actionable["priority_score"],
            "suggested_action": actionable["suggested_action"],
            "reasons": actionable["reasons"],
        }
    )
    return sheet.to_csv(index=False).encode("utf-8")


def render_field_plan(df, config):
    with st.container(key="sajag_act_page"):
        _render_field_plan_content(df, config)


def _render_field_plan_content(df, config):
    st.subheader("Field Plan")
    st.markdown(
        "<p class='sajag-act-intro'>Prioritized branch actions based on borrower risk and exposure.</p>",
        unsafe_allow_html=True,
    )
    if df.empty:
        st.info("No borrowers match the current filters.")
        return

    branch_rank = (
        _branch_agg(df).sort_values(["high_count", "medium_count", "borrowers"], ascending=False).reset_index(drop=True)
    )
    branch_options = branch_rank["branch_id"].tolist()
    header_col, branch_col = st.columns([1.5, 1], vertical_alignment="bottom")
    with header_col:
        st.markdown("<div class='sajag-act-eyebrow'>SELECTED BRANCH</div>", unsafe_allow_html=True)
    with branch_col:
        branch_id = st.selectbox("Branch", branch_options, index=0, key="branch_actions_branch")

    branch_df = df[df["branch_id"] == branch_id]
    actions_cfg = config["priority"]["actions"]
    facts = build_branch_facts(df, branch_id)

    summary_html = (
        "<div class='sajag-act-summary'>"
        "<div class='sajag-act-stat sajag-act-stat--high'>"
        f"<div class='sajag-act-stat-value'>{facts['urgent_count']:,}</div>"
        "<div class='sajag-act-stat-label'>High priority</div></div>"
        "<div class='sajag-act-stat'>"
        f"<div class='sajag-act-stat-value'>{facts['watch_count']:,}</div>"
        "<div class='sajag-act-stat-label'>Watch</div></div>"
        "<div class='sajag-act-stat'>"
        f"<div class='sajag-act-stat-value'>{format_inr_lakh_cr(facts['exposed_outstanding_inr'])}</div>"
        "<div class='sajag-act-stat-label'>Exposed</div></div>"
        "<a class='sajag-act-jump' href='#act-urgent-borrowers'>View urgent borrowers &rarr;</a>"
        "</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    st.markdown(
        "<div class='sajag-act-section'>Branch priority list</div>",
        unsafe_allow_html=True,
    )
    branch_table = _branch_alert_table(df)
    _render_act_branch_table(branch_table)

    st.markdown(
        f"<div class='sajag-act-section'>Branch intelligence · {html.escape(str(branch_id))} · "
        f"{html.escape(str(facts['district']))}</div>",
        unsafe_allow_html=True,
    )
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("<div class='sajag-act-eyebrow'>TOP OCCUPATIONS</div>", unsafe_allow_html=True)
        if facts["top_occupations"]:
            occupation_rows = "".join(
                f"<li><span>{html.escape(str(item['occupation']))}</span>"
                f"<span class='sajag-act-list-count'>{item['borrower_count']:,}</span></li>"
                for item in facts["top_occupations"]
            )
            st.markdown(f"<ul class='sajag-act-list'>{occupation_rows}</ul>", unsafe_allow_html=True)
        else:
            st.caption("No High-tier borrowers at this branch under the current filters.")
    with detail_col2:
        st.markdown("<div class='sajag-act-eyebrow'>WHY BORROWERS ARE FLAGGED</div>", unsafe_allow_html=True)
        if facts["top_reasons"]:
            reason_labels = {
                "High/medium exposure to the cyclone track": "Cyclone exposure",
                "Already missed EMIs before the event": "Missed EMIs",
                "Livelihood highly sensitive to the hazard": "Livelihood sensitivity",
            }
            reason_rows = "".join(
                f"<li><span>{html.escape(reason_labels.get(item['reason'], item['reason']))}</span>"
                f"<span class='sajag-act-list-count'>{item['borrower_count']:,}</span></li>"
                for item in facts["top_reasons"]
            )
            st.markdown(f"<ul class='sajag-act-list'>{reason_rows}</ul>", unsafe_allow_html=True)
        else:
            st.caption("No High-tier borrowers at this branch under the current filters.")

    action_items = "".join(
        f"<li>{html.escape(action.strip()[:1].upper() + action.strip()[1:])}</li>"
        for action in actions_cfg.get("high", "").split(" + ")
        if action.strip()
    )
    st.markdown("<div class='sajag-act-section'>Recommended action</div>", unsafe_allow_html=True)
    st.markdown(f"<ul class='sajag-act-recommendation'>{action_items}</ul>", unsafe_allow_html=True)

    st.markdown("<div class='sajag-act-section'>Field briefing</div>", unsafe_allow_html=True)
    if facts["top_villages"]:
        priority_villages = ", ".join(dict.fromkeys(
            str(village["subdivision"] or village["village_id"]) for village in facts["top_villages"]
        ))
        briefing_preview = (
            f"Prioritize borrower contact in {priority_villages} and verify safety, livelihood "
            "disruption, and ability to reach the branch."
        )
    else:
        briefing_preview = (
            "Prioritize contact with high-priority borrowers and verify safety, livelihood "
            "disruption, and ability to reach the branch."
        )
    st.markdown(
        f"<p class='sajag-act-briefing'>{facts['urgent_count']:,} urgent · {facts['watch_count']:,} watch. "
        f"{html.escape(briefing_preview)}</p>",
        unsafe_allow_html=True,
    )
    download_col, briefing_col = st.columns([1, 3], vertical_alignment="center")
    with download_col:
        st.download_button(
            "Download team sheet",
            data=_branch_team_sheet_csv(branch_df),
            file_name=f"{branch_id}_team_sheet.csv",
            mime="text/csv",
        )
    with briefing_col:
        with st.expander("View full briefing", expanded=False):
            st.markdown(template_branch_briefing(facts))

    urgent_df = branch_df[branch_df["tier"] == "high"].sort_values("priority_score", ascending=False).reset_index(drop=True)
    st.markdown(
        f"<div class='sajag-act-section' id='act-urgent-borrowers'>Urgent borrowers · "
        f"{len(urgent_df):,} requiring attention</div>",
        unsafe_allow_html=True,
    )
    if urgent_df.empty:
        st.caption("No urgent (high-priority) borrowers at this branch under the current filters.")
        return

    filter_col1, filter_tools = st.columns([1.4, 3.6])
    with filter_col1:
        borrower_query = st.text_input(
            "Search borrowers",
            placeholder="Borrower ID, village, or occupation",
            key=f"act_borrower_search_{branch_id}",
        ).strip()
    with filter_tools:
        with st.expander("Filters", expanded=False):
            village_col, occupation_col = st.columns(2)
            with village_col:
                village_filter = st.selectbox(
                    "Village",
                    ["All"] + sorted(urgent_df["subdivision"].dropna().unique().tolist()),
                    key=f"act_borrower_village_{branch_id}",
                )
            with occupation_col:
                occupation_filter = st.selectbox(
                    "Occupation",
                    ["All"] + sorted(
                        urgent_df["occupation"].map(
                            lambda value: OCCUPATION_LABELS.get(value, value)
                        ).dropna().unique().tolist()
                    ),
                    key=f"act_borrower_occupation_{branch_id}",
                )

    visible_urgent_df = urgent_df
    if borrower_query:
        search_fields = (
            visible_urgent_df["borrower_id"].astype(str)
            + " "
            + visible_urgent_df["subdivision"].astype(str)
            + " "
            + visible_urgent_df["occupation"].map(lambda value: OCCUPATION_LABELS.get(value, value)).astype(str)
        )
        visible_urgent_df = visible_urgent_df[search_fields.str.contains(borrower_query, case=False, regex=False)]
    if village_filter != "All":
        visible_urgent_df = visible_urgent_df[visible_urgent_df["subdivision"] == village_filter]
    if occupation_filter != "All":
        visible_urgent_df = visible_urgent_df[
            visible_urgent_df["occupation"].map(lambda value: OCCUPATION_LABELS.get(value, value)) == occupation_filter
        ]
    if visible_urgent_df.empty:
        st.caption("No urgent borrowers match the current search and filters.")
        return

    display = pd.DataFrame(
        {
            "Tier": visible_urgent_df["tier"].map(lambda t: TIER_LABELS.get(t, t)),
            "Borrower ID": visible_urgent_df["borrower_id"],
            "Village": visible_urgent_df["subdivision"],
            "Occupation": visible_urgent_df["occupation"].map(lambda o: OCCUPATION_LABELS.get(o, o)),
            "Outstanding": visible_urgent_df["outstanding"].map(format_inr),
        }
    )
    event = st.dataframe(
        _style_act_tier_column(display),
        hide_index=True,
        width="stretch",
        height=min(560, 42 + 39 * len(display)),
        on_select="rerun", selection_mode="single-row", key=f"branch_actions_table_{branch_id}",
        column_config={
            "Borrower ID": st.column_config.TextColumn("Borrower ID", width="small"),
            "Village": st.column_config.TextColumn("Village"),
            "Occupation": st.column_config.TextColumn("Occupation"),
            "Outstanding": st.column_config.TextColumn("Outstanding", alignment="right"),
        },
    )

    selected_rows = event.selection.rows if event is not None else []
    if not selected_rows:
        return

    borrower = visible_urgent_df.iloc[selected_rows[0]]
    st.markdown(
        f"<div class='sajag-act-section'>Outreach message · {html.escape(str(borrower['borrower_id']))}</div>",
        unsafe_allow_html=True,
    )
    messages, msg_source = get_borrower_outreach(borrower)
    render_outreach_widget(borrower["borrower_id"], messages, msg_source, location="branch")


LANDING_VIEWBOX_W, LANDING_VIEWBOX_H = 800, 500
LANDING_HERO_STRIP_H = 94  # result-metrics strip under the visualization


@st.cache_data
def _landing_geometry():
    """Real-data prep for the landing hero SVG: a simplified district
    silhouette, Fani's actual recorded track (its >=64kt final approach), and
    village dots colored by their real final tier - all projected into one
    fixed SVG viewBox. Computed once."""
    config = load_countdown_config()
    districts = config["region"]["affected_districts"] + config["region"]["reference_districts"]
    tehsils = gpd.read_file(PROCESSED / "odisha_tehsils.geojson")
    tehsils = tehsils[tehsils["NAME_2"].isin(districts)]
    coast_geom = tehsils.geometry.union_all().simplify(0.02)

    track_points = load_track_points().to_crs("EPSG:4326")
    # The >=64kt track spans equator-to-landfall; keep just the final
    # approach toward Odisha so the coastline isn't squeezed into a sliver.
    track_coords = [(p.x, p.y) for p in track_points.geometry if p.y >= 16.0]

    final = get_current_frame(0, "inland")
    villages = _village_agg(final)

    cminx, cminy, cmaxx, cmaxy = coast_geom.bounds
    lons = [c[0] for c in track_coords] + list(villages["lon"]) + [cminx, cmaxx]
    lats = [c[1] for c in track_coords] + list(villages["lat"]) + [cminy, cmaxy]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    def project(lon, lat):
        x = 40 + (lon - min_lon) / (max_lon - min_lon) * (LANDING_VIEWBOX_W - 80)
        y = 30 + (max_lat - lat) / (max_lat - min_lat) * (LANDING_VIEWBOX_H - 60)
        return x, y

    def polygon_to_path(geom):
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        parts = []
        for poly in polys:
            pts = [project(lon, lat) for lon, lat in poly.exterior.coords]
            parts.append("M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z")
        return " ".join(parts)

    coast_path = polygon_to_path(coast_geom)
    track = [{"x": round(x, 1), "y": round(y, 1)} for x, y in (project(lon, lat) for lon, lat in track_coords)]

    track_xy = np.array([[t["x"], t["y"]] for t in track]) if track else np.empty((0, 2))
    village_list = []
    for row in villages.itertuples():
        vx, vy = project(row.lon, row.lat)
        dist = float(np.min(np.hypot(track_xy[:, 0] - vx, track_xy[:, 1] - vy))) if len(track_xy) else 0.0
        tier = "high" if row.high_count > 0 else ("medium" if row.medium_count > 0 else "low")
        village_list.append({"x": round(vx, 1), "y": round(vy, 1), "tier": tier, "_dist": dist})
    village_list.sort(key=lambda v: v["_dist"])
    for i, v in enumerate(village_list):
        v["order"] = i
        del v["_dist"]

    return {"coast_path": coast_path, "track": track, "villages": village_list}


def _landing_hero_html(geometry, stats):
    """Build the landing hero as a static inline SVG with its final values."""
    coast_path = geometry["coast_path"]
    track_pts = geometry["track"]
    track_path_d = ("M " + " L ".join(f"{p['x']:.1f},{p['y']:.1f}" for p in track_pts)) if track_pts else ""
    # Exact polyline length, so the draw-in tip moves in step with the glyph.
    track_len = sum(
        float(np.hypot(b["x"] - a["x"], b["y"] - a["y"])) for a, b in zip(track_pts, track_pts[1:])
    ) + 1
    cyclone_tip = track_pts[-1] if track_pts else {"x": 0, "y": 0}

    # Reveal order only (positions/tiers untouched): within each tier, nearest-to-track
    # first, in groups of 6. (start, max stagger) per tier, in seconds.
    tier_timing = {"low": (0.5, 0.3), "high": (1.0, 0.45), "medium": (1.4, 0.3)}
    seen = {t: 0 for t in tier_timing}
    dot_delays = []
    for v in geometry["villages"]:  # already sorted by distance to track
        start, span = tier_timing[v["tier"]]
        dot_delays.append(start + min((seen[v["tier"]] // 6) * 0.05, span))
        seen[v["tier"]] += 1
    villages_html = "".join(
        f'<circle class="sajag-vdot" cx="{v["x"]:.1f}" cy="{v["y"]:.1f}" r="3" data-tier="{v["tier"]}" '
        f'style="animation-delay:{d:.2f}s"></circle>'
        for v, d in zip(geometry["villages"], dot_delays)
    )
    screened, flagged72, confirmed0 = stats

    return f"""
    <div class="sajag-hero-wrap sajag-hero-static">
      <style>
        .sajag-hero-wrap {{
          width: 100%; background: {BG}; container-type: inline-size;
          overflow: hidden; font-family: 'IBM Plex Sans', sans-serif;
        }}
        /* Height follows the hero's own width (40%): ~500px at the normal 1240px
           content width, taller when the layout widens. Type uses cqw (hero width). */
        .sajag-hero-stage {{
          position: relative; width: 100%; aspect-ratio: 5 / 2; min-height: 420px; max-height: 860px; overflow: hidden;
        }}
        .sajag-hero-bg {{
          position:absolute; inset:0;
          background-image: repeating-radial-gradient(circle at 85% 18%, rgba(76,195,217,0.05) 0, rgba(76,195,217,0.05) 1px, transparent 1px, transparent 46px);
        }}
        /* Base styles below ARE the final resting state; each entrance animation
           runs from an offset to that base (fill-mode: backwards, so once
           finished it releases entirely and the plain base styles apply), so disabling the
           animations shows the finished hero immediately. Background rings stay static. */
        .sajag-hero-svg {{
          position:relative; width:100%; height:100%; display:block; transform-origin: 50% 50%;
          animation: sajag-from-transparent 0.7s ease-out backwards, sajag-map-settle 1.5s cubic-bezier(.22,.61,.36,1) 0.2s backwards;
        }}
        .sajag-coast {{ fill:{SURFACE}; stroke:{DIVIDER}; stroke-width:1.2; animation: sajag-from-transparent 0.8s ease 0.1s backwards; }}
        .sajag-cone {{ fill:none; stroke:{ACCENT}; stroke-width:30; stroke-linecap:round; stroke-opacity:0.14; animation: sajag-from-transparent 0.9s ease 0.6s backwards; }}
        .sajag-track {{
          fill:none; stroke:{TEXT}; stroke-width:2.2; stroke-linecap:round;
          stroke-dasharray: 3000;  /* resting style, unchanged; the draw-in uses the exact length */
          animation: sajag-draw 1.5s cubic-bezier(.45,0,.25,1) 0.6s backwards;
        }}
        .sajag-cyclone circle, .sajag-cyclone path {{ vector-effect: non-scaling-stroke; }}
        .sajag-cyclone {{ animation: sajag-spin 2.6s linear infinite, sajag-from-transparent 0.25s ease 0.6s backwards; }}
        .sajag-vdot {{
          fill: var(--tc, #3A4653); transform-box: fill-box; transform-origin: center;
          animation: sajag-dot-in 0.4s cubic-bezier(.2,.7,.3,1) backwards, sajag-dot-color 0.5s ease backwards;
        }}
        .sajag-vdot[data-tier="high"] {{ --tc:{RISK_PALETTE['high']}; }}
        .sajag-vdot[data-tier="medium"] {{ --tc:{RISK_PALETTE['medium']}; }}
        .sajag-vdot[data-tier="low"] {{ --tc:{RISK_PALETTE['low']}; }}
        .sajag-counter-value {{ animation: sajag-from-transparent 0.3s ease 1.7s backwards; }}
        @keyframes sajag-from-transparent {{ from {{ opacity:0; }} }}
        @keyframes sajag-map-settle {{ from {{ transform: scale(1.06); }} to {{ transform: scale(1); }} }}
        @keyframes sajag-draw {{
          from {{ stroke-dasharray:{track_len:.1f} {track_len:.1f}; stroke-dashoffset:{track_len:.1f}; }}
          to {{ stroke-dasharray:{track_len:.1f} {track_len:.1f}; stroke-dashoffset:0; }}
        }}
        @keyframes sajag-spin {{ from {{ transform:rotate(0deg);}} to {{ transform:rotate(360deg);}} }}
        @keyframes sajag-dot-in {{ from {{ opacity:0; transform: scale(0.4); }} to {{ opacity:1; transform: scale(1); }} }}
        @keyframes sajag-dot-color {{ from {{ fill:#3A4653; }} }}

        .sajag-hero-text {{ position:absolute; top:30px; left:40px; z-index:2; max-width:420px; }}
        .sajag-hero-word {{ font-family:'IBM Plex Mono', monospace; font-size:clamp(1.6rem, 1.2rem + 1.1cqw, 2rem); font-weight:600; color:{TEXT}; letter-spacing:0.04em; line-height:1.15; }}
        /* Home link: reloads the landing page; looks identical to the plain wordmark
           (same hover cue as the dashboard logo). */
        .sajag-hero-word a.sajag-hero-home,
        .sajag-hero-word a.sajag-hero-home:visited,
        .sajag-hero-word a.sajag-hero-home:hover {{ color: inherit !important; text-decoration: none !important; }}
        .sajag-hero-word a.sajag-hero-home {{ cursor: pointer; transition: opacity 150ms ease; }}
        .sajag-hero-word a.sajag-hero-home:hover {{ opacity: 0.75; }}
        .sajag-hero-tag {{ font-size:clamp(0.84rem, 0.78rem + 0.25cqw, 0.92rem); color:{MUTED}; margin-top:6px; line-height:1.45; }}
        /* Result metrics: one grouped strip directly under the visualization */
        /* Same 94px strip height as before, so nothing above/below moves; the
           three KPIs sit in it as quiet surface cells. */
        .sajag-hero-counters {{
          height: {LANDING_HERO_STRIP_H}px; box-sizing: border-box; padding: 4px 0 10px 0;
          display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; align-items: stretch;
        }}
        .sajag-kpi-cell {{
          position: relative; display: flex; flex-direction: column; justify-content: center;
          padding: 0 24px; background: {LANDING_SURFACE}; border: 1px solid {LANDING_BORDER}; border-radius: 2px;
        }}
        /* Restrained accent: a short hairline on each cell's top edge */
        .sajag-kpi-cell::before {{
          content: ""; position: absolute; top: -1px; left: 24px; width: 20px; height: 1px;
          background: rgba(53, 197, 229, 0.45);
        }}
        .sajag-kpi-cell--primary {{ border-color: rgba(53, 197, 229, 0.38); }}
        .sajag-kpi-cell--primary::before {{ width: 36px; height: 2px; top: -1px; background: {LANDING_CYAN}; }}
        .sajag-kpi-cell--primary .sajag-counter-label {{ color: {LANDING_CYAN}; opacity: 0.9; }}
        .sajag-counter-value {{
          font-family:'IBM Plex Mono', monospace; font-size:clamp(1.6rem, 1.25rem + 1cqw, 2.6rem); font-weight:600; color:{LANDING_TEXT};
          line-height:1.1; font-variant-numeric: tabular-nums;
        }}
        .sajag-counter-label {{ font-size:clamp(0.58rem, 0.55rem + 0.12cqw, 0.62rem); color:{LANDING_MUTED}; text-transform:uppercase; letter-spacing:0.06em; margin-top:5px; line-height:1.4; }}

        /* No entrance: reduced motion (no animation at all, as before), or a
           repeat render this session (final state; the glyph keeps its spin). */
        @media (prefers-reduced-motion: reduce) {{
          .sajag-hero-wrap * {{ animation: none !important; }}
        }}
        .sajag-hero-static * {{ animation: none !important; }}
      </style>
      <div class="sajag-hero-stage">
      <div class="sajag-hero-bg"></div>
      <svg class="sajag-hero-svg" viewBox="0 0 {LANDING_VIEWBOX_W} {LANDING_VIEWBOX_H}" preserveAspectRatio="xMidYMid meet">
        <path class="sajag-coast" d="{coast_path}"></path>
        <path class="sajag-cone" d="{track_path_d}"></path>
        <path class="sajag-track" d="{track_path_d}"></path>
        {villages_html}
        <g class="sajag-cyclone" transform="translate({cyclone_tip['x']:.1f} {cyclone_tip['y']:.1f})">
          <circle cx="0" cy="0" r="7" fill="none" stroke="{TEXT}" stroke-width="1.4"></circle>
          <path d="M -7 0 A 7 7 0 0 1 0 -7" fill="none" stroke="{ACCENT}" stroke-width="1.6"></path>
        </g>
      </svg>
      <div class="sajag-hero-text">
        <div class="sajag-hero-word"><a class="sajag-hero-home" href="./" target="_self">SAJAG</a></div>
        <div class="sajag-hero-tag">72 hours before a cyclone lands, know which borrowers to reach first &mdash; and why.</div>
      </div>
      </div>
      <div class="sajag-hero-counters">
        <div class="sajag-kpi-cell"><div class="sajag-counter-value" data-target="{screened}">{screened:,}</div><div class="sajag-counter-label">Screened</div></div>
        <div class="sajag-kpi-cell sajag-kpi-cell--primary"><div class="sajag-counter-value" data-target="{flagged72}">{flagged72:,}</div><div class="sajag-counter-label">Flagged at T&minus;72h</div></div>
        <div class="sajag-kpi-cell"><div class="sajag-counter-value" data-target="{confirmed0}">{confirmed0:,}</div><div class="sajag-counter-label">Confirmed at landfall</div></div>
      </div>
    </div>
    """


# Stage colours: one coral family, a subtle step darker per stage.
LANDING_FUNNEL_COLORS = ["#563138", "#503039", "#492F38", "#432D37", "#3D2B35"]


def _landing_funnel_html(rows):
    """Landing "How the list narrows" funnel as plain HTML/CSS: label | bar | spacer
    per stage, bar width proportional to the High-tier count, thin trapezoid
    connectors between stages. Values come straight from `rows`."""
    max_v = max(v for _, v, _ in rows) or 1
    widths = [v / max_v * 100 for _, v, _ in rows]
    colors = [LANDING_FUNNEL_COLORS[min(i, len(LANDING_FUNNEL_COLORS) - 1)] for i in range(len(rows))]
    parts = []
    for i, (label, v, prec) in enumerate(rows):
        shown = "LANDFALL" if label == "Landfall" else label
        parts.append(
            f'<div class="sajag-fstage" style="--i:{i}; --w:{widths[i]:.2f}%; --c:{colors[i]};">'
            f'<div class="sajag-flabel"><span class="sajag-fidx">{i + 1:02d}</span><span class="sajag-ftime">{shown}</span></div>'
            f'<div class="sajag-ftrack"><div class="sajag-fbar">'
            f'<span class="sajag-fnum">{v:,}</span><span class="sajag-fprec">{prec:.0f}% precision</span>'
            f'</div></div><div></div></div>'
        )
        if i + 1 < len(rows):
            top, bot = widths[i], widths[i + 1]
            clip = (f"polygon({50 - top / 2:.2f}% 0, {50 + top / 2:.2f}% 0, "
                    f"{50 + bot / 2:.2f}% 100%, {50 - bot / 2:.2f}% 100%)")
            parts.append(
                f'<div class="sajag-fconn" style="--i:{i}; --c:{colors[i + 1]}; --clip:{clip};">'
                f'<div></div><div></div><div></div></div>'
            )
    summary = "; ".join(f"{label} {v:,} borrowers, {prec:.0f}% precision" for label, v, prec in rows)
    return f'<div class="sajag-funnel" role="img" aria-label="High-tier list by lead time: {summary}">{"".join(parts)}</div>'


def render_landing(config):
    geometry = _landing_geometry()
    df72 = get_current_frame(72, "inland")
    df0 = get_current_frame(0, "inland")
    screened = len(df72)
    flagged72 = int((df72["tier"] == "high").sum())
    confirmed0 = int((df0["tier"] == "high").sum())

    # Rendered inline (not in an iframe) so its height follows its responsive width
    # naturally. st.markdown keeps the SVG (st.html strips it); lines are stripped so
    # markdown doesn't read indented HTML as a code block.
    # Render the complete final state immediately; the visualization is
    # informative without an entrance sequence or delayed counter updates.
    hero_html = _landing_hero_html(geometry, (screened, flagged72, confirmed0))
    with st.container(key="sajag_landing_hero", gap=None):
        st.markdown(" ".join(line.strip() for line in hero_html.splitlines()), unsafe_allow_html=True)

    summary_table = summarize_countdown(load_countdown())
    funnel_rows = _precision_funnel_rows(summary_table, scenario="inland")
    if funnel_rows:
        st.markdown("<div class='sajag-landing-head'>HOW THE LIST NARROWS</div>", unsafe_allow_html=True)
        with st.container(key="sajag_landing_funnel", gap=None):
            st.markdown(_landing_funnel_html(funnel_rows), unsafe_allow_html=True)

    st.markdown("<div class='sajag-landing-head sajag-landing-head--divided'>HOW IT WORKS</div>", unsafe_allow_html=True)
    steps = [
        ("01", "SENSE", "Hazard track and forecast uncertainty, replayed hour by hour."),
        ("02", "PRIORITIZE", "Explainable borrower risk - exposure, vulnerability, and repayment stress in one score."),
        ("03", "ACT", "Branch-level field plans, ranked by who needs outreach first."),
        ("04", "MEASURE", "Break-even economics for targeted outreach vs. doing nothing."),
    ]
    cols = st.columns(4, gap="large")
    for col, (num, name, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div class='sajag-howstep'><span class='sajag-howstep-num'>{num}</span>"
                f"<span class='sajag-howstep-name'>{name}</span><p>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    with st.container(key="sajag_landing_cta"):
        if st.button("Replay Cyclone Fani →", type="primary", key="landing_cta"):
            countdown_df = load_countdown()
            lead_options = sorted(countdown_df["timestep"].unique(), reverse=True)
            _init_countdown_state(lead_options, list(COUNTDOWN_SCENARIO_LABELS.keys()))
            st.session_state["countdown_lead"] = 72
            st.session_state["countdown_scenario_label"] = next(
                k for k, v in COUNTDOWN_SCENARIO_LABELS.items() if v == "inland"
            )
            st.session_state["route"] = "sense"
            st.rerun()

        st.markdown(
            "<p class='sajag-footer-note'>Recommendations support credit-team review; they do not make credit "
            "decisions.</p>",
            unsafe_allow_html=True,
        )


def render_topbar(config):
    """Persistent bar on every app screen: wordmark, event/date, big mono
    T-Xh, scenario segmented control, and the 5-stop timeline (navigation;
    Play lives on Sense itself, see render_sense)."""
    current_lead, current_scenario = current_lead_and_scenario()
    landfall_time = _landfall_time()
    current_time = landfall_time - pd.Timedelta(hours=current_lead)
    date_str = _format_ist(current_time)
    event_name = config["event"]["name"]

    with st.container(key="sajag_topbar"):
        # Row 1: brand + event | countdown | empty. CSS turns this into a 1fr-auto-1fr grid so the
        # countdown sits on the true centre.
        with st.container(key="sajag_header"):
            brand_col, stage_col, _ = st.columns(3)
            with brand_col:
                if st.button("SAJAG", key="brand_home", help="Back to the SAJAG home page"):
                    st.session_state["route"] = "landing"
                    st.rerun()
                st.markdown(
                    f'<div class="sajag-event">{html.escape(event_name)} &middot; {date_str} IST</div>',
                    unsafe_allow_html=True,
                )
            status_sub = "At landfall" if current_lead == 0 else f"{current_lead} hours before landfall"
            stage_col.markdown(
                f'<div class="sajag-status"><div class="sajag-countdown">T&minus;{current_lead}h</div>'
                f'<div class="sajag-status-sub">{status_sub}</div></div>',
                unsafe_allow_html=True,
            )

        # Row 2: vertical scenario control | timeline. CSS turns this into an auto-1fr grid.
        with st.container(key="sajag_row2"):
            controls_col, timeline_col = st.columns([1, 4])
            with controls_col:
                scenario_label = st.radio(
                    "Scenario", list(COUNTDOWN_SCENARIO_LABELS.keys()),
                    key="countdown_scenario_label", label_visibility="collapsed",
                )
            with timeline_col, st.container(key="sajag_timeline"):
                tl_cols = st.columns(len(TIMELINE_STOPS))
                for col, hour in zip(tl_cols, TIMELINE_STOPS):
                    label = "LANDFALL" if hour == 0 else f"T−{hour}h"
                    with col:
                        if st.button(
                            label, key=f"tl_{hour}", type="primary" if hour == current_lead else "secondary",
                            width="stretch",
                        ):
                            st.session_state["countdown_lead"] = hour
                            st.rerun()

    return current_lead, COUNTDOWN_SCENARIO_LABELS[scenario_label]


def _render_step_nav():
    route = st.session_state["route"]
    steps = [("01  SENSE", "sense"), ("02  PRIORITIZE", "prioritize"), ("03  ACT", "act"), ("04  MEASURE", "measure")]
    with st.container(key="sajag_step_nav"):
        cols = st.columns([1, 1.3, 1, 1.2])
        for col, (label, key) in zip(cols, steps):
            with col:
                if st.button(label, key=f"nav_{key}", type="primary" if route == key else "secondary", width="stretch"):
                    st.session_state["route"] = key
                    st.rerun()


def main():
    config = load_countdown_config()
    project_name = config["app"]["project_name"]

    st.set_page_config(page_title=project_name, layout="wide", initial_sidebar_state="collapsed")
    inject_custom_css()

    st.session_state.setdefault("route", "landing")

    if st.session_state["route"] == "landing":
        render_landing(config)
        return

    countdown_df = load_countdown()
    lead_options = sorted(countdown_df["timestep"].unique(), reverse=True)
    _init_countdown_state(lead_options, list(COUNTDOWN_SCENARIO_LABELS.keys()))

    current_lead, current_scenario = render_topbar(config)
    _render_step_nav()

    current_frame = get_current_frame(current_lead, current_scenario)
    # Filters and (on Sense) Play share one compact row; Play's column stays empty elsewhere.
    with st.container(key="sajag_controls"):
        filters_col, play_col = st.columns(2)
        with filters_col:
            filtered = apply_filters(current_frame)

    route = st.session_state["route"]
    if route == "sense":
        render_sense(config, play_container=play_col)
    elif route == "prioritize":
        render_prioritize(filtered, config)
    elif route == "act":
        render_field_plan(filtered, config)
    elif route == "measure":
        render_measure(filtered, config)


if __name__ == "__main__":
    main()
