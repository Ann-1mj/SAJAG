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
import streamlit.components.v1 as components
from shapely.geometry import LineString, Point
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hazard import METRIC_CRS, load_track_points  # noqa: E402
from src.labels import EXPOSURE_BAND_LABELS, OCCUPATION_LABELS, TIER_COLORS, TIER_LABELS  # noqa: E402
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

COUNTDOWN_SCENARIO_LABELS = {"Inland": "inland", "Seaward": "seaward", "Forecast missed": "forecast_miss"}
COUNTDOWN_LINE_STYLE = {
    "inland": (ACCENT, "Inland"),
    "seaward": ("#3E9C7D", "Seaward"),
    "forecast_miss": (TIER_COLORS["high"], "Forecast missed"),
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
            padding-bottom: 8px; margin-bottom: 4px;
            background-image: repeating-radial-gradient(circle at 92% -30%, rgba(76,195,217,0.06) 0, rgba(76,195,217,0.06) 1px, transparent 1px, transparent 42px);
        }}
        .sajag-wordmark {{
            font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 1.25rem;
            letter-spacing: 0.06em; color: {TEXT}; padding-top: 6px;
        }}
        .sajag-event {{ color: {MUTED}; font-size: 0.85rem; padding-top: 11px; }}
        .sajag-countdown {{
            font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 1.5rem;
            color: {TEXT}; padding-top: 3px;
        }}
        .sajag-demo-label {{ color: {MUTED}; font-size: 0.7rem; text-align: right; padding-top: 12px; }}

        [class*="st-key-sajag_topbar"] div[role="radiogroup"] {{ gap: 4px; padding-top: 2px; }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label {{
            background: {SURFACE}; border: 1px solid {DIVIDER}; border-radius: 6px;
            padding: 2px 10px !important; margin-right: 0 !important;
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label:has(input:checked) {{
            border-color: {ACCENT};
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label:has(input:checked) p {{
            color: {ACCENT} !important;
        }}
        [class*="st-key-sajag_topbar"] div[role="radiogroup"] label div:first-child {{ display: none; }}

        /* Timeline + step-nav pills, rendered as minimal st.buttons */
        [class*="st-key-sajag_topbar"] [data-testid="stButton"] button,
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
        [class*="st-key-sajag_topbar"] [data-testid="stButton"] button[kind="primary"],
        [class*="st-key-sajag_step_nav"] [data-testid="stButton"] button[kind="primary"] {{
            color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important;
        }}
        [class*="st-key-sajag_topbar"] [data-testid="stButton"] button:hover,
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
        .sajag-hero-summary {{ font-size: 1.0rem; color: {MUTED}; margin: 14px 0 4px 0; }}
        .sajag-howstep {{ border-top: 1px solid {DIVIDER}; padding-top: 10px; }}
        .sajag-howstep-num {{ font-family:"IBM Plex Mono", monospace; color:{ACCENT}; font-weight:600; }}
        .sajag-howstep-name {{ font-weight:600; letter-spacing:0.03em; margin-left:6px; }}
        .sajag-howstep p {{ color: {MUTED}; font-size: 0.85rem; margin-top: 6px; line-height:1.4; }}
        .sajag-footer-note {{ color: {MUTED}; font-size: 0.78rem; margin-top: 24px; }}
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
    label_to_tier = {v: k for k, v in TIER_LABELS.items()}

    def _cell_style(value):
        color = TIER_COLORS.get(label_to_tier.get(value))
        return f"background-color: {color}; color: white; font-weight: 600" if color else ""

    return display_df.style.map(_cell_style, subset=["Tier"])


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
    tier_color = TIER_COLORS.get(tier, MUTED)
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
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=TIER_COLORS["high"]))
    fig.update_layout(_plotly_layout(height=220, xaxis=dict(title=None), yaxis=dict(color=TEXT)))
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


def _precision_funnel_chart(summary_table, scenario="inland", stops=TIMELINE_STOPS):
    """The "list narrowing" visual: a native Plotly funnel (decreasing-width
    by construction, no custom animation needed) of High-tier count at each
    canonical stop, strict precision labeled beside each one."""
    rows = []
    for h in stops:
        row = summary_table[(summary_table["lead_time_h"] == h) & (summary_table["scenario"] == scenario)]
        if row.empty:
            continue
        r = row.iloc[0]
        label = "Landfall" if h == 0 else f"T-{h}h"
        rows.append((label, int(r["high"]), float(r["strict_precision_pct"])))
    if not rows:
        return None
    fig = go.Figure(
        go.Funnel(
            y=[r[0] for r in rows],
            x=[r[1] for r in rows],
            text=[f"{v:,} · {p:.0f}% precision" for _, v, p in rows],
            textinfo="text",
            textposition="inside",
            textfont=dict(color=BG, size=13),
            marker=dict(color=TIER_COLORS["high"]),
            connector=dict(line=dict(color=GRID, width=1)),
        )
    )
    fig.update_layout(_plotly_layout(height=320))
    return fig


def render_measure(df, config):
    st.subheader("Measure")
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
        st.markdown("## Outreach pays for itself: n/a — no default-probability uplift is assumed")
        st.caption("No PD uplift is assumed, so outreach has nothing to offset - this isn't a claim that outreach is worthless.")
    else:
        pct_display = f"{min(breakeven, 999):.1f}%" if breakeven <= 100 else f"{breakeven:.0f}%"
        st.markdown(
            f"## Outreach pays for itself if it prevents "
            f"<span class='sajag-mono' style='color:{ACCENT};'>{pct_display}</span> "
            f"of excess (post-disaster) defaults",
            unsafe_allow_html=True,
        )
        if breakeven > 100:
            st.caption("Exceeds 100% - under these assumptions, no achievable effectiveness makes outreach's benefit exceed its cost.")
        else:
            st.caption("Below this share, targeted outreach's cost exceeds its expected benefit; above it, benefit exceeds cost.")

    st.markdown("<div class='sajag-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sajag-section-label'>STRATEGY COMPARISON</div>", unsafe_allow_html=True)
    st.caption(
        "Do nothing = reactive baseline, zero cost/zero avoided loss by definition. "
        "Targeted = outreach to High + Medium tier only (current policy, single point in time). "
        "Blanket = moratorium offered to every exposed borrower (High + Medium + Low). "
        "Staged = outreach-only at T-72h to High + Medium, then deferral + review only for whoever is still "
        f"High at T-24h under the {COUNTDOWN_LINE_STYLE.get(scenario, (None, scenario))[1]} scenario. "
        "Deferred cash is money whose repayment is paused, not a cost - only outreach and liquidity cost are "
        "actual costs. Net vs. do-nothing shows each strategy's outcome relative to that baseline, not a ranking."
    )
    st.dataframe(_strategy_comparison_table(result), width="stretch")

    st.markdown("<div class='sajag-section-label'>SENSITIVITY</div>", unsafe_allow_html=True)
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

    st.markdown("<div class='sajag-section-label'>FORECAST QUALITY: CONSISTENCY WITH LANDFALL ASSESSMENT</div>", unsafe_allow_html=True)
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


def render_legend():
    st.markdown("**Legend**")
    for tier in TIER_ORDER:
        color = TIER_COLORS[tier]
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
    color = TIER_COLORS.get(tier, TIER_COLORS["not_exposed"])
    if tier == "high":
        folium.CircleMarker(
            location=[lat, lon], radius=radius * 2.3, color=color, weight=0,
            fill=True, fill_color=color, fill_opacity=0.16,
        ).add_to(m)
    popup = folium.Popup(popup_html, max_width=260) if popup_html else None
    fill_opacity = 0.92 if tier == "high" else (0.85 if tier == "medium" else 0.55)
    folium.CircleMarker(
        location=[lat, lon], radius=radius, color=color, fill=True, fill_color=color,
        fill_opacity=fill_opacity, weight=1, tooltip=tooltip, popup=popup,
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


def render_prioritize(df, config):
    st.subheader("Prioritize")
    weights = config["priority"]["weights"]
    st.caption(
        f"Borrowers are flagged by combining hazard exposure ({weights['exposure'] * 100:.0f}%), "
        f"livelihood vulnerability ({weights['vulnerability'] * 100:.0f}%), and repayment stress "
        f"({weights['stress'] * 100:.0f}%) into one priority score."
    )
    if df.empty:
        st.info("No borrowers match the current filters.")
        return

    sorted_df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

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

    csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered queue as CSV", data=csv_bytes, file_name="review_queue.csv", mime="text/csv")

    selected_row = None
    village_proxy = None
    selected_rows = event.selection.rows if event is not None else []
    if selected_rows:
        selected_row = sorted_df.iloc[selected_rows[0]]

    st.markdown("<div class='sajag-section-label'>MAP</div>", unsafe_allow_html=True)
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
    folium.GeoJson(_subdivision_outline_geojson(districts), style_function=_geojson_style_fn(MUTED, weight=1)).add_to(m)
    folium.GeoJson(
        _exposure_zone_geojson("medium"), style_function=_geojson_style_fn(TIER_COLORS["medium"], fill=True, fill_opacity=0.2)
    ).add_to(m)
    folium.GeoJson(
        _exposure_zone_geojson("high"), style_function=_geojson_style_fn(TIER_COLORS["high"], fill=True, fill_opacity=0.22)
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
        map_state = st_folium(m, height=560, use_container_width=True, returned_objects=["last_object_clicked_tooltip"])
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
        _exposure_zone_geojson("medium"), style_function=_geojson_style_fn(TIER_COLORS["medium"], fill=True, fill_opacity=0.2)
    ).add_to(m)
    folium.GeoJson(
        _exposure_zone_geojson("high"), style_function=_geojson_style_fn(TIER_COLORS["high"], fill=True, fill_opacity=0.22)
    ).add_to(m)
    _add_countdown_track_layers(m, lead, scenario)

    village_df = _village_agg(joined_df)
    max_outstanding = village_df["outstanding"].max() or 1
    for row in village_df.itertuples():
        tier = "high" if row.high_count > 0 else ("medium" if row.medium_count > 0 else "low")
        radius = 5 + 11 * (row.outstanding / max_outstanding) ** 0.5
        _add_tier_marker(m, row.lat, row.lon, tier, radius, f"Village {row.village_id}", row.tooltip)

    st_folium(m, height=560, use_container_width=True, returned_objects=[], key=map_key)


def _stacked_number(label, value):
    st.markdown(
        f"<div style='margin-bottom:16px;'>"
        f"<div class='sajag-section-label' style='margin:0 0 2px 0;'>{label.upper()}</div>"
        f"<div class='sajag-mono' style='font-size:1.55rem; font-weight:600; color:{TEXT};'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_sense_frame(lead, scenario, config, landfall_time):
    event_name = config["event"]["name"]
    current_time = landfall_time - pd.Timedelta(hours=lead)
    date_str = f"{current_time.day} {current_time.strftime('%b %Y %H:%M')}"

    joined = get_current_frame(lead, scenario)
    exposed = joined[joined["tier"] != "not_exposed"]
    high = joined[joined["tier"] == "high"]

    map_col, side_col = st.columns([13, 7])
    with map_col:
        _render_sense_map(joined, lead, scenario, map_key=f"sense_map_{lead}_{scenario}")

    with side_col:
        st.markdown(f"<div class='sajag-countdown' style='font-size:2.1rem;'>T&minus;{lead}h</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{MUTED}; margin-bottom:18px;'>{html.escape(event_name)} &middot; {date_str}</div>", unsafe_allow_html=True)

        _stacked_number("Borrowers exposed", f"{len(exposed):,}")
        _stacked_number("Flagged for early outreach", f"{len(high):,}")
        _stacked_number("₹ outstanding in High tier", format_inr_lakh_cr(high["outstanding"].sum()))

        st.markdown("<div class='sajag-section-label'>WHERE TO ACT FIRST</div>", unsafe_allow_html=True)
        district_fig = _district_high_bar_chart(joined)
        if district_fig is not None:
            _plot(district_fig)
        else:
            st.caption("No High-tier borrowers at this lead time/scenario.")

    st.markdown("<div class='sajag-section-label'>NEXT ACTIONS</div>", unsafe_allow_html=True)
    st.dataframe(_style_tier_column(_next_actions_table(joined, config)), hide_index=True, width="stretch")


def render_sense(config):
    landfall_time = _landfall_time()
    current_lead, current_scenario = current_lead_and_scenario()
    play_clicked = st.button("▶ Play")

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
    st.subheader("Field Plan")
    if df.empty:
        st.info("No borrowers match the current filters.")
        return

    st.markdown("<div class='sajag-section-label'>BRANCHES RANKED BY HIGH COUNT</div>", unsafe_allow_html=True)
    st.dataframe(_branch_alert_table(df), hide_index=True, width="stretch")

    branch_rank = (
        _branch_agg(df).sort_values(["high_count", "medium_count", "borrowers"], ascending=False).reset_index(drop=True)
    )
    branch_options = branch_rank["branch_id"].tolist()
    branch_id = st.selectbox("Branch", branch_options, index=0, key="branch_actions_branch")

    branch_df = df[df["branch_id"] == branch_id]
    actions_cfg = config["priority"]["actions"]
    facts = build_branch_facts(df, branch_id)

    st.markdown(f"#### {branch_id} · {facts['district']}")
    tier_agg = (
        branch_df.groupby("tier")
        .agg(borrowers=("borrower_id", "count"), outstanding=("outstanding", "sum"))
        .reindex(TIER_ORDER, fill_value=0)
        .reset_index()
    )
    tier_display = pd.DataFrame(
        {
            "Tier": tier_agg["tier"].map(lambda t: TIER_LABELS.get(t, t)),
            "Borrowers": tier_agg["borrowers"],
            "₹ Outstanding": tier_agg["outstanding"].map(format_inr_lakh_cr),
        }
    )
    st.dataframe(_style_tier_column(tier_display), hide_index=True, width="stretch")

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("**Top occupations (High tier)**")
        if facts["top_occupations"]:
            st.markdown("\n".join(f"- {o['occupation']} ({o['borrower_count']})" for o in facts["top_occupations"]))
        else:
            st.caption("No High-tier borrowers at this branch under the current filters.")
    with detail_col2:
        st.markdown("**Top reasons flagged (High tier)**")
        if facts["top_reasons"]:
            st.markdown("\n".join(f"- {r['reason']} ({r['borrower_count']})" for r in facts["top_reasons"]))
        else:
            st.caption("No High-tier borrowers at this branch under the current filters.")

    st.markdown(
        f"**₹ exposed:** {format_inr_lakh_cr(facts['exposed_outstanding_inr'])} &nbsp;&nbsp;&nbsp; "
        f"**Recommended action (High tier):** {actions_cfg.get('high', '')}",
        unsafe_allow_html=True,
    )

    st.markdown("#### Field briefing")
    st.info(template_branch_briefing(facts))

    st.download_button(
        "Download team sheet (CSV)", data=_branch_team_sheet_csv(branch_df),
        file_name=f"{branch_id}_team_sheet.csv", mime="text/csv",
    )

    urgent_df = branch_df[branch_df["tier"] == "high"].sort_values("priority_score", ascending=False).reset_index(drop=True)
    st.markdown(f"#### Urgent borrowers ({len(urgent_df)})")
    if urgent_df.empty:
        st.caption("No urgent (high-priority) borrowers at this branch under the current filters.")
        return

    display = pd.DataFrame(
        {
            "Tier": urgent_df["tier"].map(lambda t: TIER_LABELS.get(t, t)),
            "Borrower ID": urgent_df["borrower_id"],
            "Village": urgent_df["subdivision"],
            "Occupation": urgent_df["occupation"].map(lambda o: OCCUPATION_LABELS.get(o, o)),
            "Outstanding": urgent_df["outstanding"].map(format_inr),
        }
    )
    event = st.dataframe(
        _style_tier_column(display), hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key=f"branch_actions_table_{branch_id}",
    )

    selected_rows = event.selection.rows if event is not None else []
    if not selected_rows:
        return

    borrower = urgent_df.iloc[selected_rows[0]]
    st.markdown(f"#### Outreach message · {borrower['borrower_id']}")
    messages, msg_source = get_borrower_outreach(borrower)
    render_outreach_widget(borrower["borrower_id"], messages, msg_source, location="branch")


LANDING_VIEWBOX_W, LANDING_VIEWBOX_H = 800, 500


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
    """Self-contained HTML/SVG/CSS/JS hero: track draws in over ~4s with a
    rotating cyclone glyph, villages fade in and light up in real tier
    colors, counters count up to the real stats. Respects
    prefers-reduced-motion (jumps straight to the end state); the end state
    itself is the same regardless, so it also holds up as a still frame."""
    coast_path = geometry["coast_path"]
    track_pts = geometry["track"]
    track_path_d = ("M " + " L ".join(f"{p['x']:.1f},{p['y']:.1f}" for p in track_pts)) if track_pts else ""
    track_len = 3000  # generous overestimate of path length - see the draw-in comment below

    villages_html = "".join(
        f'<circle class="sajag-vdot" cx="{v["x"]:.1f}" cy="{v["y"]:.1f}" r="3" data-tier="{v["tier"]}" '
        f'style="animation-delay:{0.6 + min(v["order"], 70) * 0.028:.2f}s"></circle>'
        for v in geometry["villages"]
    )
    screened, flagged72, confirmed0 = stats

    return f"""
    <div class="sajag-hero-wrap">
      <style>
        .sajag-hero-wrap {{
          position: relative; width: 100%; height: 500px; background: {BG};
          overflow: hidden; font-family: 'IBM Plex Sans', sans-serif;
        }}
        .sajag-hero-bg {{
          position:absolute; inset:0;
          background-image: repeating-radial-gradient(circle at 85% 18%, rgba(76,195,217,0.05) 0, rgba(76,195,217,0.05) 1px, transparent 1px, transparent 46px);
        }}
        .sajag-hero-svg {{ position:relative; width:100%; height:100%; display:block; }}
        .sajag-coast {{ fill:{SURFACE}; stroke:{DIVIDER}; stroke-width:1.2; opacity:0; animation: sajag-fadein 1.1s ease forwards; }}
        .sajag-cone {{ fill:none; stroke:{ACCENT}; stroke-width:30; stroke-linecap:round; opacity:0; animation: sajag-fadein 1.4s ease 0.9s forwards; stroke-opacity:0.14; }}
        .sajag-track {{
          fill:none; stroke:{TEXT}; stroke-width:2.2; stroke-linecap:round;
          stroke-dasharray: {track_len}; stroke-dashoffset: {track_len};
          animation: sajag-draw 4s ease-out 0.4s forwards;
        }}
        .sajag-cyclone circle, .sajag-cyclone path {{ vector-effect: non-scaling-stroke; }}
        .sajag-cyclone {{ opacity:0; animation: sajag-spin 2.6s linear infinite, sajag-fadein 0.3s ease 4.1s forwards; }}
        .sajag-vdot {{
          fill:#3A4653; opacity:0;
          animation: sajag-dotfade 0.5s ease forwards, sajag-dotcolor 0.6s ease forwards;
        }}
        .sajag-vdot[data-tier="high"] {{ --tc:{TIER_COLORS['high']}; }}
        .sajag-vdot[data-tier="medium"] {{ --tc:{TIER_COLORS['medium']}; }}
        .sajag-vdot[data-tier="low"] {{ --tc:{TIER_COLORS['low']}; }}
        @keyframes sajag-fadein {{ to {{ opacity:1; }} }}
        @keyframes sajag-draw {{ to {{ stroke-dashoffset:0; }} }}
        @keyframes sajag-spin {{ from {{ transform:rotate(0deg);}} to {{ transform:rotate(360deg);}} }}
        @keyframes sajag-dotfade {{ to {{ opacity:1; }} }}
        @keyframes sajag-dotcolor {{ from {{ fill:#3A4653; }} to {{ fill: var(--tc, #3A4653); }} }}

        .sajag-hero-text {{ position:absolute; top:34px; left:40px; z-index:2; max-width:440px; }}
        .sajag-hero-word {{ font-family:'IBM Plex Mono', monospace; font-size:2.3rem; font-weight:600; color:{TEXT}; letter-spacing:0.04em; }}
        .sajag-hero-tag {{ font-size:1.02rem; color:{MUTED}; margin-top:8px; line-height:1.5; }}
        .sajag-hero-counters {{ position:absolute; bottom:28px; left:40px; z-index:2; display:flex; gap:32px; }}
        .sajag-counter-value {{ font-family:'IBM Plex Mono', monospace; font-size:1.6rem; font-weight:600; color:{TEXT}; }}
        .sajag-counter-label {{ font-size:0.68rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.05em; margin-top:2px; }}

        @media (prefers-reduced-motion: reduce) {{
          .sajag-coast, .sajag-cone {{ animation:none !important; opacity:1 !important; }}
          .sajag-track {{ animation:none !important; stroke-dashoffset:0 !important; }}
          .sajag-cyclone {{ animation:none !important; opacity:1 !important; }}
          .sajag-vdot {{ animation:none !important; opacity:1 !important; }}
          .sajag-vdot[data-tier="high"] {{ fill:{TIER_COLORS['high']} !important; }}
          .sajag-vdot[data-tier="medium"] {{ fill:{TIER_COLORS['medium']} !important; }}
          .sajag-vdot[data-tier="low"] {{ fill:{TIER_COLORS['low']} !important; }}
        }}
      </style>
      <div class="sajag-hero-bg"></div>
      <svg class="sajag-hero-svg" viewBox="0 0 {LANDING_VIEWBOX_W} {LANDING_VIEWBOX_H}" preserveAspectRatio="xMidYMid meet">
        <path class="sajag-coast" d="{coast_path}"></path>
        <path class="sajag-cone" d="{track_path_d}"></path>
        <path class="sajag-track" d="{track_path_d}"></path>
        {villages_html}
        <g class="sajag-cyclone">
          <circle cx="0" cy="0" r="7" fill="none" stroke="{TEXT}" stroke-width="1.4"></circle>
          <path d="M -7 0 A 7 7 0 0 1 0 -7" fill="none" stroke="{ACCENT}" stroke-width="1.6"></path>
          <animateMotion dur="4s" begin="0.4s" fill="freeze" path="{track_path_d}"></animateMotion>
        </g>
      </svg>
      <div class="sajag-hero-text">
        <div class="sajag-hero-word">SAJAG</div>
        <div class="sajag-hero-tag">72 hours before a cyclone lands, know which borrowers to reach first &mdash; and why.</div>
      </div>
      <div class="sajag-hero-counters">
        <div><div class="sajag-counter-value" data-target="{screened}">0</div><div class="sajag-counter-label">Screened</div></div>
        <div><div class="sajag-counter-value" data-target="{flagged72}">0</div><div class="sajag-counter-label">Flagged at T&minus;72h</div></div>
        <div><div class="sajag-counter-value" data-target="{confirmed0}">0</div><div class="sajag-counter-label">Confirmed at landfall</div></div>
      </div>
      <script>
        (function() {{
          var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
          document.querySelectorAll('.sajag-counter-value').forEach(function(el) {{
            var target = parseInt(el.getAttribute('data-target'), 10) || 0;
            if (reduce) {{ el.textContent = target.toLocaleString(); return; }}
            var start = null;
            var duration = 1100;
            function step(ts) {{
              if (!start) start = ts;
              var progress = Math.min((ts - start) / duration, 1);
              el.textContent = Math.round(progress * target).toLocaleString();
              if (progress < 1) requestAnimationFrame(step);
            }}
            requestAnimationFrame(step);
          }});
        }})();
      </script>
    </div>
    """


def render_landing(config):
    geometry = _landing_geometry()
    df72 = get_current_frame(72, "inland")
    df0 = get_current_frame(0, "inland")
    screened = len(df72)
    flagged72 = int((df72["tier"] == "high").sum())
    confirmed0 = int((df0["tier"] == "high").sum())

    components.html(_landing_hero_html(geometry, (screened, flagged72, confirmed0)), height=500, scrolling=False)

    st.markdown(
        f"<p class='sajag-hero-summary'>Cyclone Fani replay: {screened:,} borrowers screened &middot; "
        f"{flagged72:,} flagged at T&minus;72h &middot; {confirmed0:,} confirmed high-risk at landfall.</p>",
        unsafe_allow_html=True,
    )

    summary_table = summarize_countdown(load_countdown())
    funnel_fig = _precision_funnel_chart(summary_table, scenario="inland")
    if funnel_fig is not None:
        st.markdown("<div class='sajag-section-label'>HOW THE LIST NARROWS</div>", unsafe_allow_html=True)
        _plot(funnel_fig)

    st.markdown("<div class='sajag-section-label'>HOW IT WORKS</div>", unsafe_allow_html=True)
    steps = [
        ("01", "SENSE", "Hazard track and forecast uncertainty, replayed hour by hour."),
        ("02", "PRIORITIZE", "Explainable borrower risk - exposure, vulnerability, and repayment stress in one score."),
        ("03", "ACT", "Branch-level field plans, ranked by who needs outreach first."),
        ("04", "MEASURE", "Break-even economics for targeted outreach vs. doing nothing."),
    ]
    cols = st.columns(4)
    for col, (num, name, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div class='sajag-howstep'><span class='sajag-howstep-num'>{num}</span>"
                f"<span class='sajag-howstep-name'>{name}</span><p>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.write("")
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
        top_cols = st.columns([1.1, 2.4, 1.3, 3.4, 1.6])
        top_cols[0].markdown('<div class="sajag-wordmark">SAJAG</div>', unsafe_allow_html=True)
        top_cols[1].markdown(
            f'<div class="sajag-event">{html.escape(event_name)} &middot; {date_str} IST</div>', unsafe_allow_html=True
        )
        top_cols[2].markdown(f'<div class="sajag-countdown">T&minus;{current_lead}h</div>', unsafe_allow_html=True)
        with top_cols[3]:
            scenario_label = st.radio(
                "Scenario", list(COUNTDOWN_SCENARIO_LABELS.keys()), horizontal=True,
                key="countdown_scenario_label", label_visibility="collapsed",
            )
        top_cols[4].markdown(
            '<div class="sajag-demo-label">Demo &middot; synthetic borrower data</div>', unsafe_allow_html=True
        )

        tl_cols = st.columns(len(TIMELINE_STOPS))
        for col, hour in zip(tl_cols, TIMELINE_STOPS):
            label = "LANDFALL" if hour == 0 else f"T−{hour}h"
            with col:
                if st.button(
                    label, key=f"tl_{hour}", type="primary" if hour == current_lead else "secondary", width="stretch"
                ):
                    st.session_state["countdown_lead"] = hour
                    st.rerun()

    return current_lead, COUNTDOWN_SCENARIO_LABELS[scenario_label]


def _render_step_nav():
    route = st.session_state["route"]
    steps = [("01  SENSE", "sense"), ("02  PRIORITIZE", "prioritize"), ("03  ACT", "act"), ("04  MEASURE", "measure")]
    with st.container(key="sajag_step_nav"):
        cols = st.columns([1, 1.3, 1, 1.2, 4])
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
    filtered = apply_filters(current_frame)

    route = st.session_state["route"]
    if route == "sense":
        render_sense(config)
    elif route == "prioritize":
        render_prioritize(filtered, config)
    elif route == "act":
        render_field_plan(filtered, config)
    elif route == "measure":
        render_measure(filtered, config)


if __name__ == "__main__":
    main()
