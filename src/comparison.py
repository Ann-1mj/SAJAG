"""Response strategies compared: Do nothing, Targeted outreach, Blanket
moratorium, and Staged (outreach early, defer only who's still High close to
landfall). Contacted counts, ₹ deferred, cost, and expected loss avoided
under editable assumptions. Pure functions (no Streamlit) so the math is
directly testable; app.py wires these into an "Assumptions" expander.

Every number here is a PROTOTYPE ASSUMPTION, not fitted to real lender data -
see DEFAULT_ASSUMPTIONS below and config.yaml's own comparison: block.

Cost model - ANY strategy that defers EMIs pays the same liquidity-cost
formula on its own deferred amount (this used to only apply to Blanket;
that asymmetry is gone - Targeted's credit-team review can and does defer
payments for the borrowers it reaches, so it's costed the same way):
  - Admin/outreach cost: outreach_cost_inr per borrower actually contacted
    (call/SMS/visit/review) - the same rate for every strategy.
  - Liquidity cost: a moratorium only pauses the EMIs that actually fall due
    during it, not the whole outstanding loan. Deferred amount per borrower
    = min(emi * moratorium_months, outstanding) (can't defer more than what's
    actually owed). Liquidity cost = that deferred amount x annual
    cost_of_funds_pct x (moratorium_months / 12) - the lender's cost of
    funds, prorated to however long the money is actually held up.
  - Total cost = admin cost + liquidity cost, for every strategy alike.

₹ deferred (the table figure) is this same per-borrower EMI-based amount for
every strategy, since "how much repayment is being paused" is a factual
question independent of which strategy is doing the deferring.

Staged is the odd one out: it spans two different points in time (an early,
cheap outreach touch at T-72h, followed by an actual deferral only for
whoever is still High close to landfall, at T-24h), so it needs both frames
passed in explicitly rather than a single scored frame - see
staged_strategy_summary().
"""

# Sensible illustrative defaults; all overridable in the UI, none hidden.
DEFAULT_ASSUMPTIONS = {
    "outreach_cost_inr": 150,  # per borrower actually contacted (call/SMS/visit/review)
    "pd_uplift_pp": {"high": 25, "medium": 12, "low": 3},  # post-disaster default-probability increase, no intervention
    "lgd_pct": 60,  # loss given default - no collateral assumed
    "effectiveness_pct": 40,  # % of PD uplift offset for borrowers who get a real deferral + review
    "outreach_only_effectiveness_pct": 10,  # % of PD uplift offset for a light-touch call/SMS with no deferral
    "cost_of_funds_pct": 8,  # ANNUAL cost of funds, applied only to the EMIs actually deferred
    "moratorium_months": 3,  # how many months of EMIs a moratorium actually pauses
}

# Tiers contacted under each response strategy.
TARGETED_TIERS = {"high", "medium"}
BLANKET_TIERS = {"high", "medium", "low"}  # every borrower with tier != not_exposed
DO_NOTHING_TIERS = set()  # the reactive baseline: nobody is contacted
STAGED_OUTREACH_TIERS = {"high", "medium"}  # T-72h: who gets the early, cheap touch

SENSITIVITY_MULTIPLIERS = (0.5, 1.0, 1.5)


def _potential_loss_base(df, assumptions):
    """sum(outstanding * pd_uplift_frac * lgd_frac) - the loss that would be
    avoided at 100% intervention effectiveness. expected_loss_avoided = this
    * effectiveness; break-even effectiveness solves this equation for
    effectiveness at net_benefit == 0."""
    pd_uplift = assumptions["pd_uplift_pp"]
    lgd = assumptions["lgd_pct"] / 100.0
    # astype(float): on an empty frame (e.g. do-nothing) mapping an Arrow-backed
    # str column can't infer a numeric dtype and would stay str.
    uplift_frac = df["tier"].map(lambda t: pd_uplift.get(t, 0) / 100.0).astype(float)
    return float((df["outstanding"] * uplift_frac * lgd).sum())


def _deferred_emi_inr(df, assumptions):
    """Per borrower: min(emi * moratorium_months, outstanding) - a
    moratorium only pauses the EMIs falling due in that window, never more
    than the loan actually still owes. Summed over the given (already
    tier-filtered) borrowers."""
    months = assumptions["moratorium_months"]
    capped = (df["emi"] * months).clip(upper=df["outstanding"])
    return float(capped.sum())


def _liquidity_cost_inr(deferred_emi_inr, assumptions):
    return deferred_emi_inr * (assumptions["cost_of_funds_pct"] / 100.0) * (assumptions["moratorium_months"] / 12.0)


def _strategy_summary(df, tiers, assumptions):
    contacted = df[df["tier"].isin(tiers)]
    deferred = _deferred_emi_inr(contacted, assumptions)
    outreach_cost = len(contacted) * assumptions["outreach_cost_inr"]
    liquidity_cost = _liquidity_cost_inr(deferred, assumptions)
    total_cost = outreach_cost + liquidity_cost
    loss_avoided = _potential_loss_base(contacted, assumptions) * (assumptions["effectiveness_pct"] / 100.0)
    return {
        "contacted": len(contacted),
        "deferred_inr": deferred,
        "outreach_cost_inr": float(outreach_cost),
        "liquidity_cost_inr": float(liquidity_cost),
        "total_cost_inr": float(total_cost),
        "expected_loss_avoided_inr": loss_avoided,
        "net_benefit_inr": loss_avoided - total_cost,
    }


def staged_strategy_summary(df_t72, df_t24, assumptions=None):
    """Staged: outreach-only (admin cost, no deferral) to High+Medium at
    T-72h; deferral + credit review (liquidity cost) only for whoever is
    still High at T-24h. Loss avoided uses the main effectiveness_pct for
    the T-24h deferred group, and the separate (lower)
    outreach_only_effectiveness_pct for whoever only got the T-72h touch and
    was never deferred. df_t72/df_t24 must be the same scenario's scored
    frame at those two lead times (borrower_id, tier, outstanding, emi).
    """
    assumptions = assumptions or DEFAULT_ASSUMPTIONS

    outreach_ids = set(df_t72.loc[df_t72["tier"].isin(STAGED_OUTREACH_TIERS), "borrower_id"])
    deferred_ids = set(df_t24.loc[df_t24["tier"] == "high", "borrower_id"])
    outreach_only_ids = outreach_ids - deferred_ids
    contacted_ids = outreach_ids | deferred_ids

    deferred_rows = df_t24[df_t24["borrower_id"].isin(deferred_ids)]
    outreach_only_rows = df_t72[df_t72["borrower_id"].isin(outreach_only_ids)]

    deferred_emi = _deferred_emi_inr(deferred_rows, assumptions)
    outreach_cost = len(contacted_ids) * assumptions["outreach_cost_inr"]
    liquidity_cost = _liquidity_cost_inr(deferred_emi, assumptions)
    total_cost = outreach_cost + liquidity_cost

    loss_avoided_deferred = _potential_loss_base(deferred_rows, assumptions) * (assumptions["effectiveness_pct"] / 100.0)
    loss_avoided_outreach_only = _potential_loss_base(outreach_only_rows, assumptions) * (
        assumptions["outreach_only_effectiveness_pct"] / 100.0
    )
    loss_avoided = loss_avoided_deferred + loss_avoided_outreach_only

    return {
        "contacted": len(contacted_ids),
        "deferred_inr": deferred_emi,
        "outreach_cost_inr": float(outreach_cost),
        "liquidity_cost_inr": float(liquidity_cost),
        "total_cost_inr": float(total_cost),
        "expected_loss_avoided_inr": loss_avoided,
        "net_benefit_inr": loss_avoided - total_cost,
    }


def compare_strategies(df, assumptions=None, df_t72=None, df_t24=None):
    """(do_nothing, targeted, blanket[, staged]) summary dicts for the given
    (already lead/scenario/filter-scoped) scored frame. do_nothing is the
    reactive baseline (zero cost, zero loss avoided by construction) -
    net_vs_do_nothing_inr makes each strategy's outcome relative to it
    explicit. Pass df_t72/df_t24 (same scenario's T-72h/T-24h frames) to
    also include "staged"; omitted if either is None, since Staged needs
    both time points and can't be derived from a single frame."""
    assumptions = assumptions or DEFAULT_ASSUMPTIONS
    result = {
        "do_nothing": _strategy_summary(df, DO_NOTHING_TIERS, assumptions),
        "targeted": _strategy_summary(df, TARGETED_TIERS, assumptions),
        "blanket": _strategy_summary(df, BLANKET_TIERS, assumptions),
    }
    if df_t72 is not None and df_t24 is not None:
        result["staged"] = staged_strategy_summary(df_t72, df_t24, assumptions)

    baseline_net = result["do_nothing"]["net_benefit_inr"]
    for key, summary in result.items():
        summary["net_vs_do_nothing_inr"] = summary["net_benefit_inr"] - baseline_net
    return result


def breakeven_effectiveness_pct(df, assumptions=None, tiers=TARGETED_TIERS):
    """Minimum intervention effectiveness (%) at which net benefit >= 0 for
    the given tiers, holding every other assumption fixed. Solves
    total_cost = potential_loss_base * effectiveness/100 for effectiveness.
    Returns None if there's nothing to avoid (zero contacted, or zero
    potential loss base - e.g. pd_uplift_pp is all zero) - in that case no
    finite effectiveness makes the strategy pay for itself. The result can
    exceed 100, meaning the strategy cannot break even even at perfect (100%)
    effectiveness under the current cost assumptions.
    """
    assumptions = assumptions or DEFAULT_ASSUMPTIONS
    contacted = df[df["tier"].isin(tiers)]
    if contacted.empty:
        return None

    deferred = _deferred_emi_inr(contacted, assumptions)
    outreach_cost = len(contacted) * assumptions["outreach_cost_inr"]
    liquidity_cost = _liquidity_cost_inr(deferred, assumptions)
    fixed_cost = outreach_cost + liquidity_cost

    base = _potential_loss_base(contacted, assumptions)
    if base <= 0:
        return None
    return 100.0 * fixed_cost / base


def sensitivity_table(df, assumptions=None, multipliers=SENSITIVITY_MULTIPLIERS):
    """Targeted vs. blanket net benefit as pd_uplift_pp is scaled by each
    multiplier (0.5x/1x/1.5x by default) - a quick read on how sensitive
    Targeted's net-benefit margin over Blanket is to how bad the actual
    default-rate impact turns out to be, without touching any other
    assumption."""
    assumptions = assumptions or DEFAULT_ASSUMPTIONS
    rows = []
    for m in multipliers:
        scaled = {
            **assumptions,
            "pd_uplift_pp": {tier: value * m for tier, value in assumptions["pd_uplift_pp"].items()},
        }
        result = compare_strategies(df, scaled)
        rows.append(
            {
                "multiplier": m,
                "targeted_net_inr": result["targeted"]["net_benefit_inr"],
                "blanket_net_inr": result["blanket"]["net_benefit_inr"],
            }
        )
    return rows
