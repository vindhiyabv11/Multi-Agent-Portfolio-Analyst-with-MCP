import json

import pandas as pd

from ..guardrails import data_quality_warnings, with_retry
from ..indicators import compute_metrics, periods_for
from ..llm import get_llm
from ..mcp_client import call_json, get_tools
from ..schemas import QuantView
from ..telemetry import traced

SYSTEM = """You are a quantitative/technical analyst.
You receive pre-computed metrics. NEVER recompute or invent numbers; cite only the provided values.
- trend: use price vs SMA20/SMA50 and cross_state.
- momentum: RSI above 70 = overbought, below 30 = oversold, else neutral.
- risk_level: judge from annualized volatility and max drawdown (fractions, e.g. 0.25 = 25%).
- bias: bullish, neutral or bearish net technical view.
- signals: 2-5 short bullet points that quote the actual numbers.
- confidence: 0.0-1.0; lower it if many metrics are null (short history)."""


def _clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


@traced("quant")
async def quant_node(state):
    ticker, run_id = state["ticker"], state["run_id"]
    tools = await get_tools()

    async def fetch():
        d = await call_json(tools["get_price_history"], ticker=ticker, period="1y", interval="1d")
        if "error" in d:
            raise RuntimeError(d["error"])
        return d

    data = await with_retry(fetch, attempts=2)

    df = pd.DataFrame(data["candles"])
    df["date"] = pd.to_datetime(df["date"])
    metrics = compute_metrics(df, periods_per_year=periods_for(ticker))  # real math, in code
    warns = data_quality_warnings(metrics)

    errors = []
    try:
        await call_json(
            tools["save_indicators"], run_id=run_id, ticker=ticker, metrics_json=json.dumps(metrics)
        )
    except Exception as e:
        errors.append(f"quant: could not save indicators ({e})")

    llm = get_llm().with_structured_output(QuantView)
    human = f"Ticker: {ticker}\nMetrics:\n{json.dumps(metrics, indent=2)}"
    view = await with_retry(lambda: llm.ainvoke([("system", SYSTEM), ("human", human)]))
    out = view.model_dump()
    out["confidence"] = _clamp(out["confidence"], 0, 1)
    if warns:
        out["confidence"] = min(out["confidence"], 0.6)

    candles = df.tail(250).copy()
    candles["date"] = candles["date"].dt.strftime("%Y-%m-%d")
    result = {"quant": out, "metrics": metrics, "candles": candles.to_dict("records")}
    if errors:
        result["errors"] = errors
    if warns:
        result["warnings"] = warns
    return result