import math

import numpy as np
import pandas as pd


def periods_for(ticker: str) -> int:
    """Crypto trades every day (365). Stocks have about 252 sessions a year."""
    return 365 if ticker.upper().endswith("-USD") else 252


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_volatility(close: pd.Series, periods_per_year: int = 252) -> float:
    return float(daily_returns(close).std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(close: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns(close) - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if not sd or math.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def max_drawdown(close: pd.Series) -> float:
    return float((close / close.cummax() - 1).min())


def cross_state(close: pd.Series) -> str:
    diff = (sma(close, 20) - sma(close, 50)).dropna()
    if diff.empty:
        return "insufficient_data"
    recent = diff.tail(10)
    now = recent.iloc[-1]
    if now > 0 and (recent < 0).any():
        return "golden_cross_recent"
    if now < 0 and (recent > 0).any():
        return "death_cross_recent"
    return "bullish_alignment" if now > 0 else "bearish_alignment"


def _clean(x, ndigits: int = 4):
    if x is None:
        return None
    x = float(x)
    return round(x, ndigits) if math.isfinite(x) else None


def compute_metrics(df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """df needs 'date' and 'close' columns. Returns JSON-safe metrics."""
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    n = len(close)
    return {
        "as_of": str(df["date"].iloc[-1])[:10],
        "n_candles": n,
        "last_close": _clean(close.iloc[-1]),
        "sma20": _clean(sma(close, 20).iloc[-1]) if n >= 20 else None,
        "sma50": _clean(sma(close, 50).iloc[-1]) if n >= 50 else None,
        "ema20": _clean(ema(close, 20).iloc[-1]) if n >= 20 else None,
        "rsi14": _clean(rsi(close).iloc[-1], 2) if n >= 15 else None,
        "volatility": _clean(annualized_volatility(close, periods_per_year)) if n >= 3 else None,
        "sharpe": _clean(sharpe_ratio(close, periods_per_year), 3) if n >= 3 else None,
        "max_drawdown": _clean(max_drawdown(close)),
        "return_1m": _clean(close.iloc[-1] / close.iloc[-22] - 1) if n > 22 else None,
        "cross_state": cross_state(close),
    }