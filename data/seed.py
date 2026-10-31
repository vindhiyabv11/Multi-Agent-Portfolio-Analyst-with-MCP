import sqlite3
from contextlib import closing
from pathlib import Path

DB = Path(__file__).resolve().parent / "portfolio.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    ticker        TEXT PRIMARY KEY,
    quantity      REAL NOT NULL,
    avg_buy_price REAL NOT NULL,
    asset_class   TEXT,
    sector        TEXT,
    target_weight REAL
);

CREATE TABLE IF NOT EXISTS indicator_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT,
    ticker       TEXT,
    as_of        TEXT,
    last_close   REAL,
    sma20        REAL,
    sma50        REAL,
    rsi14        REAL,
    volatility   REAL,
    sharpe       REAL,
    max_drawdown REAL
);

CREATE TABLE IF NOT EXISTS analysis_reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    ticker      TEXT,
    created_at  TEXT,
    verdict     TEXT,
    confidence  REAL,
    report_json TEXT
);

CREATE TABLE IF NOT EXISTS run_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    node        TEXT,
    status      TEXT,
    started_at  TEXT,
    duration_ms REAL,
    detail      TEXT
);
"""

HOLDINGS = [
    # ticker,        quantity, avg_buy_price, asset_class, sector,          target_weight
    ("WIPRO.NS",     50,       210.0,         "Equity",    "IT",            0.25),
    ("LT.NS",        15,       3500.0,        "Equity",    "Infrastructure", 0.25),
    ("ICICIBANK.NS", 20,       1300.0,        "Equity",    "Banking",       0.25),
    ("ITC.NS",       100,      280.0,         "Equity",    "FMCG",          0.25),
]

if __name__ == "__main__":
    with closing(sqlite3.connect(DB)) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT OR REPLACE INTO holdings VALUES (?, ?, ?, ?, ?, ?)", HOLDINGS
        )
        conn.commit()
    print(f"Database ready at {DB}")