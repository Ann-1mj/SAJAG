import pandas as pd

from src.countdown import PROCESSED, load_config, run_countdown, summarize


def test_lead_zero_matches_scored_portfolio_in_all_scenarios():
    config = load_config()
    countdown_df = run_countdown(config)
    final = pd.read_csv(PROCESSED / "scored_portfolio.csv").set_index("borrower_id").sort_index()

    for scenario in config["hazard"]["scenarios"]:
        at_zero = (
            countdown_df[(countdown_df["timestep"] == 0) & (countdown_df["scenario"] == scenario)]
            .set_index("borrower_id")
            .sort_index()
        )
        expected = final.loc[at_zero.index]

        assert (at_zero["tier"] == expected["tier"]).all(), scenario
        pd.testing.assert_series_equal(
            at_zero["priority_score"], expected["priority_score"], check_names=False, atol=1e-9
        )


def test_recall_is_100_percent_for_inland_and_seaward_at_every_lead_time():
    config = load_config()
    countdown_df = run_countdown(config)
    table = summarize(countdown_df)

    guaranteed = table[table["scenario"].isin(["inland", "seaward"])]
    assert (guaranteed["recall_pct"] == 100.0).all()


def test_forecast_miss_can_drop_below_full_recall():
    config = load_config()
    countdown_df = run_countdown(config)
    table = summarize(countdown_df)

    miss = table[table["scenario"] == "forecast_miss"]
    assert (miss["recall_pct"] < 100.0).any()
