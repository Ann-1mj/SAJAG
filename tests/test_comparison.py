import pandas as pd
import pytest

from src.comparison import (
    DEFAULT_ASSUMPTIONS,
    breakeven_effectiveness_pct,
    compare_strategies,
    sensitivity_table,
    staged_strategy_summary,
)


def _portfolio():
    return pd.DataFrame(
        {
            "borrower_id": ["B1", "B2", "B3", "B4"],
            "tier": ["high", "medium", "low", "not_exposed"],
            "outstanding": [100_000, 200_000, 300_000, 400_000],
            "emi": [5_000, 8_000, 10_000, 12_000],
        }
    )


def test_targeted_contacts_fewer_borrowers_than_blanket():
    result = compare_strategies(_portfolio(), DEFAULT_ASSUMPTIONS)
    assert result["targeted"]["contacted"] == 2  # high + medium
    assert result["blanket"]["contacted"] == 3  # high + medium + low
    assert result["targeted"]["contacted"] < result["blanket"]["contacted"]


def test_do_nothing_baseline_is_all_zero():
    result = compare_strategies(_portfolio(), DEFAULT_ASSUMPTIONS)
    baseline = result["do_nothing"]
    assert baseline["contacted"] == 0
    assert baseline["deferred_inr"] == 0
    assert baseline["total_cost_inr"] == 0
    assert baseline["expected_loss_avoided_inr"] == 0
    assert baseline["net_vs_do_nothing_inr"] == 0


def test_net_vs_do_nothing_matches_net_benefit():
    """do_nothing's own net_benefit is 0 by construction, so each strategy's
    outcome relative to it is just its own net_benefit_inr."""
    result = compare_strategies(_portfolio(), DEFAULT_ASSUMPTIONS)
    assert result["targeted"]["net_vs_do_nothing_inr"] == pytest.approx(result["targeted"]["net_benefit_inr"])
    assert result["blanket"]["net_vs_do_nothing_inr"] == pytest.approx(result["blanket"]["net_benefit_inr"])


def test_deferred_amount_is_emi_times_moratorium_months_capped_at_outstanding():
    """Deferred EMIs, NOT the full outstanding loan - and never more than a
    borrower actually owes."""
    assumptions = {**DEFAULT_ASSUMPTIONS, "moratorium_months": 3}
    result = compare_strategies(_portfolio(), assumptions)

    # targeted = B1 (high, emi 5000) + B2 (medium, emi 8000)
    expected_targeted_deferred = min(5_000 * 3, 100_000) + min(8_000 * 3, 200_000)
    assert result["targeted"]["deferred_inr"] == pytest.approx(expected_targeted_deferred)
    assert result["targeted"]["deferred_inr"] < _portfolio().loc[_portfolio()["tier"].isin(["high", "medium"]), "outstanding"].sum()


def test_liquidity_cost_prorates_by_moratorium_length_not_a_full_year():
    """3 months of deferral should cost 1/4 of what 12 months would, all
    else equal - the bug being fixed here charged a full year's cost of
    funds on the whole outstanding balance regardless of moratorium length."""
    short = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "moratorium_months": 3})
    full_year = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "moratorium_months": 12})
    short_rate = short["blanket"]["liquidity_cost_inr"] / short["blanket"]["deferred_inr"]
    full_year_rate = full_year["blanket"]["liquidity_cost_inr"] / full_year["blanket"]["deferred_inr"]
    assert short_rate == pytest.approx(DEFAULT_ASSUMPTIONS["cost_of_funds_pct"] / 100.0 * 3 / 12)
    assert full_year_rate == pytest.approx(DEFAULT_ASSUMPTIONS["cost_of_funds_pct"] / 100.0 * 12 / 12)


def test_every_strategy_that_defers_pays_the_same_liquidity_cost_formula():
    """Consistency fix: Targeted also defers EMIs for its contacted
    borrowers, so it must be charged the same liquidity-cost formula as
    Blanket - not exempted from it."""
    result = compare_strategies(_portfolio(), DEFAULT_ASSUMPTIONS)
    assert result["targeted"]["liquidity_cost_inr"] > 0
    assert result["blanket"]["liquidity_cost_inr"] > 0

    rate = DEFAULT_ASSUMPTIONS["cost_of_funds_pct"] / 100.0 * DEFAULT_ASSUMPTIONS["moratorium_months"] / 12.0
    assert result["targeted"]["liquidity_cost_inr"] == pytest.approx(result["targeted"]["deferred_inr"] * rate)
    assert result["blanket"]["liquidity_cost_inr"] == pytest.approx(result["blanket"]["deferred_inr"] * rate)
    assert result["targeted"]["total_cost_inr"] == pytest.approx(
        result["targeted"]["outreach_cost_inr"] + result["targeted"]["liquidity_cost_inr"]
    )


def test_raising_outreach_cost_lowers_net_benefit():
    cheap = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "outreach_cost_inr": 0})
    expensive = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "outreach_cost_inr": 100_000})
    assert expensive["targeted"]["net_benefit_inr"] < cheap["targeted"]["net_benefit_inr"]


def test_raising_cost_of_funds_hurts_every_deferring_strategy():
    low = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "cost_of_funds_pct": 0})
    high = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "cost_of_funds_pct": 50})
    assert high["blanket"]["net_benefit_inr"] < low["blanket"]["net_benefit_inr"]
    assert high["targeted"]["net_benefit_inr"] < low["targeted"]["net_benefit_inr"]


def test_zero_effectiveness_means_zero_loss_avoided():
    assumptions = {**DEFAULT_ASSUMPTIONS, "effectiveness_pct": 0}
    result = compare_strategies(_portfolio(), assumptions)
    assert result["targeted"]["expected_loss_avoided_inr"] == 0
    assert result["blanket"]["expected_loss_avoided_inr"] == 0


def test_breakeven_effectiveness_is_where_net_benefit_crosses_zero():
    breakeven = breakeven_effectiveness_pct(_portfolio(), DEFAULT_ASSUMPTIONS)
    assert breakeven is not None

    at_breakeven = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "effectiveness_pct": breakeven})
    assert at_breakeven["targeted"]["net_benefit_inr"] == pytest.approx(0, abs=1e-6)

    below = compare_strategies(_portfolio(), {**DEFAULT_ASSUMPTIONS, "effectiveness_pct": max(breakeven - 5, 0)})
    assert below["targeted"]["net_benefit_inr"] <= at_breakeven["targeted"]["net_benefit_inr"]


def test_breakeven_is_none_when_pd_uplift_is_zero():
    assumptions = {**DEFAULT_ASSUMPTIONS, "pd_uplift_pp": {"high": 0, "medium": 0, "low": 0}}
    assert breakeven_effectiveness_pct(_portfolio(), assumptions) is None


def test_sensitivity_table_scales_pd_uplift_and_preserves_targeted_lt_blanket_cost():
    rows = sensitivity_table(_portfolio(), DEFAULT_ASSUMPTIONS)
    assert [r["multiplier"] for r in rows] == [0.5, 1.0, 1.5]
    assert rows[0]["targeted_net_inr"] <= rows[1]["targeted_net_inr"] <= rows[2]["targeted_net_inr"]


# --- Staged strategy ---

def _t72_frame():
    return pd.DataFrame(
        {
            "borrower_id": ["B1", "B2", "B3", "B4"],
            "tier": ["high", "medium", "medium", "not_exposed"],  # B1,B2,B3 get the early outreach touch
            "outstanding": [100_000, 200_000, 150_000, 400_000],
            "emi": [5_000, 8_000, 6_000, 12_000],
        }
    )


def _t24_frame():
    return pd.DataFrame(
        {
            "borrower_id": ["B1", "B2", "B3", "B4"],
            # only B1 is still High at T-24h; B2 cooled to medium, B3 to low,
            # B4 newly not_exposed -> high (edge case: never touched at T-72h)
            "tier": ["high", "medium", "low", "high"],
            "outstanding": [100_000, 200_000, 150_000, 400_000],
            "emi": [5_000, 8_000, 6_000, 12_000],
        }
    )


def test_staged_only_defers_borrowers_still_high_at_t24h():
    result = staged_strategy_summary(_t72_frame(), _t24_frame(), DEFAULT_ASSUMPTIONS)
    # deferred = B1 (still high) + B4 (newly high at T-24h) = min(5000*3,100000) + min(12000*3,400000)
    expected_deferred = min(5_000 * 3, 100_000) + min(12_000 * 3, 400_000)
    assert result["deferred_inr"] == pytest.approx(expected_deferred)


def test_staged_contacted_is_union_of_outreach_and_deferred_no_double_count():
    result = staged_strategy_summary(_t72_frame(), _t24_frame(), DEFAULT_ASSUMPTIONS)
    # outreach (T-72h high/medium) = B1,B2,B3; deferred (T-24h high) = B1,B4 -> union = B1,B2,B3,B4 = 4
    assert result["contacted"] == 4
    assert result["outreach_cost_inr"] == pytest.approx(4 * DEFAULT_ASSUMPTIONS["outreach_cost_inr"])


def test_staged_uses_lower_effectiveness_for_outreach_only_borrowers():
    """B2 and B3 got the T-72h touch but were never deferred - they should
    be discounted by outreach_only_effectiveness_pct, not the main
    (higher) effectiveness_pct."""
    full_effectiveness_only = {**DEFAULT_ASSUMPTIONS, "outreach_only_effectiveness_pct": DEFAULT_ASSUMPTIONS["effectiveness_pct"]}
    zero_outreach_only = {**DEFAULT_ASSUMPTIONS, "outreach_only_effectiveness_pct": 0}

    higher = staged_strategy_summary(_t72_frame(), _t24_frame(), full_effectiveness_only)
    lower = staged_strategy_summary(_t72_frame(), _t24_frame(), zero_outreach_only)
    assert lower["expected_loss_avoided_inr"] < higher["expected_loss_avoided_inr"]


def test_compare_strategies_includes_staged_only_when_both_frames_given():
    result_without = compare_strategies(_portfolio(), DEFAULT_ASSUMPTIONS)
    assert "staged" not in result_without

    result_with = compare_strategies(_portfolio(), DEFAULT_ASSUMPTIONS, df_t72=_t72_frame(), df_t24=_t24_frame())
    assert "staged" in result_with
    assert result_with["staged"]["net_vs_do_nothing_inr"] == pytest.approx(result_with["staged"]["net_benefit_inr"])
