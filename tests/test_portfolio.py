import pandas as pd

from src.portfolio import generate_portfolio, load_config, load_tehsils


def test_same_seed_reproduces_portfolio():
    config = load_config()
    tehsils = load_tehsils(config)

    _, _, borrowers_1 = generate_portfolio(config, tehsils)
    _, _, borrowers_2 = generate_portfolio(config, tehsils)

    pd.testing.assert_frame_equal(borrowers_1, borrowers_2)
