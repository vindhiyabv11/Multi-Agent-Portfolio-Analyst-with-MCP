"""Market-data MCP server (wraps yfinance). Talks to the agent over stdio.
Never print() to stdout here: stdout carries the MCP protocol."""
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("market-data")


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


@mcp.tool()
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """Get daily OHLCV candles for a ticker (e.g. TCS.NS, AAPL, BTC-USD).
    period: 1mo, 3mo, 6mo, 1y, 2y. interval: 1d. Returns JSON with a 'candles' list."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception as e:  # network / rate limit
        return _err(f"Failed to fetch history for {ticker}: {e}")
    if df is None or df.empty:
        return _err(f"No price data found for {ticker}. Check the symbol (NSE needs .NS).")

    df = df.dropna(subset=["Close"]).copy()
    df["Volume"] = df["Volume"].fillna(0)
    df = df.reset_index()
    date_col = df.columns[0]
    candles = [
        {
            "date": str(row[date_col])[:10],
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        }
        for _, row in df.iterrows()
    ]
    return json.dumps({"ticker": ticker, "candles": candles})


def _parse_news_item(item: dict) -> dict:
    """Yahoo changed its news format, so handle both the new and the old shape."""
    content = item.get("content")
    if isinstance(content, dict):
        return {
            "title": content.get("title"),
            "summary": content.get("summary") or "",
            "publisher": (content.get("provider") or {}).get("displayName"),
            "published": content.get("pubDate"),
            "link": (content.get("canonicalUrl") or {}).get("url"),
        }
    ts = item.get("providerPublishTime")
    published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None
    return {
        "title": item.get("title"),
        "summary": "",
        "publisher": item.get("publisher"),
        "published": published,
        "link": item.get("link"),
    }

def _news_query(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith("-USD"):
        return f"{t[:-4]} crypto price when:7d"
    base = re.split(r"[.\-]", t)[0]
    if t.endswith((".NS", ".BO")):
        return f"{base} share stock India when:7d"
    return f"{base} stock when:7d"


def _google_news(query: str, limit: int) -> list:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        root = ET.fromstring(resp.read())

    items = []
    for it in root.iter("item"):
        title = it.findtext("title")
        if not title:
            continue
        pub = it.findtext("pubDate")
        try:
            published = parsedate_to_datetime(pub).isoformat() if pub else None
        except Exception:
            published = pub
        source = it.find("source")
        items.append(
            {
                "title": title,
                "summary": "",
                "publisher": source.text if source is not None else None,
                "published": published,
                "link": it.findtext("link"),
            }
        )
        if len(items) >= limit:
            break
    return items


@mcp.tool()
def get_news(ticker: str, limit: int = 10) -> str:
    """Get recent news headlines for a ticker. Tries Yahoo Finance first, then Google News.
    Returns a JSON list of {title, summary, publisher, published, link}."""
    limit = max(1, int(limit))
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
        items = [_parse_news_item(n) for n in raw if isinstance(n, dict)]
        items = [i for i in items if i.get("title")][:limit]
    except Exception:
        items = []

    if not items:  # Yahoo returned nothing, so try the fallback source
        try:
            items = _google_news(_news_query(ticker), limit)
        except Exception as e:
            return _err(f"No news from Yahoo or Google News for {ticker}: {e}")
    return json.dumps(items)


@mcp.tool()
def get_quote(ticker: str) -> str:
    """Get the latest price and currency for a ticker."""
    t = yf.Ticker(ticker)
    try:
        fi = t.fast_info
        return json.dumps(
            {"ticker": ticker, "price": float(fi["last_price"]), "currency": fi["currency"]}
        )
    except Exception:
        try:
            hist = t.history(period="5d")
            if hist.empty:
                return _err(f"No quote for {ticker}")
            return json.dumps(
                {"ticker": ticker, "price": float(hist["Close"].iloc[-1]), "currency": None}
            )
        except Exception as e:
            return _err(f"Failed to fetch quote for {ticker}: {e}")


if __name__ == "__main__":
    mcp.run()  # stdio transport
    