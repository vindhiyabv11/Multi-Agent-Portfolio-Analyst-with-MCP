import math

import pandas as pd

from finagent.indicators import annualized_volatility, max_drawdown, rsi, sma


def test_sma_last_value():
    close = pd.Series(range(1, 61), dtype=float)
    assert sma(close, 20).iloc[-1] == 50.5  # mean of 41..60


def test_max_drawdown():
    close = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert math.isclose(max_drawdown(close), -0.5)


def test_rsi_all_gains_is_100():
    close = pd.Series(range(1, 40), dtype=float)
    assert rsi(close).iloc[-1] == 100


def test_volatility_of_flat_price_is_zero():
    close = pd.Series([50.0] * 30)
    assert annualized_volatility(close) == 0.0