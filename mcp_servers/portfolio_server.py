"""Portfolio/SQLite MCP server: reads holdings, writes indicator frames and reports."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio.db"
mcp = FastMCP("portfolio")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def get_holdings() -> str:
    """List every holding in the user's portfolio as a JSON list."""
    with closing(_conn()) as conn:
        rows = conn.execute("SELECT * FROM holdings ORDER BY ticker").fetchall()
    return json.dumps([dict(r) for r in rows])


@mcp.tool()
def get_holding(ticker: str) -> str:
    """Get one holding by ticker. Returns {held: false} if the user doesn't own it."""
    with closing(_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM holdings WHERE ticker = ?", (ticker.upper(),)
        ).fetchone()
    if row is None:
        return json.dumps({"held": False, "ticker": ticker.upper()})
    return json.dumps({"held": True, **dict(row)})


@mcp.tool()
def save_indicators(run_id: str, ticker: str, metrics_json: str) -> str:
    """Write one row of computed indicators (as a DataFrame) to indicator_snapshots."""
    m = json.loads(metrics_json)
    df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "ticker": ticker.upper(),
                "as_of": m.get("as_of"),
                "last_close": m.get("last_close"),
                "sma20": m.get("sma20"),
                "sma50": m.get("sma50"),
                "rsi14": m.get("rsi14"),
                "volatility": m.get("volatility"),
                "sharpe": m.get("sharpe"),
                "max_drawdown": m.get("max_drawdown"),
            }
        ]
    )
    with closing(sqlite3.connect(DB_PATH, timeout=15)) as conn:
        df.to_sql("indicator_snapshots", conn, if_exists="append", index=False)
        conn.commit()
    return json.dumps({"ok": True, "rows": len(df)})


@mcp.tool()
def save_report(run_id: str, ticker: str, report_json: str) -> str:
    """Save the final analysis report for a run."""
    report = json.loads(report_json)
    with closing(_conn()) as conn:
        conn.execute(
            "INSERT INTO analysis_reports (run_id, ticker, created_at, verdict, confidence, report_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                ticker.upper(),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                report.get("verdict"),
                report.get("overall_confidence"),
                report_json,
            ),
        )
        conn.commit()
    return json.dumps({"ok": True})


@mcp.tool()
def save_review(run_id: str, ticker: str, agreement: str, choice: str, note: str = "") -> str:
    """Record the human decision made when the analysts disagreed."""
    with closing(_conn()) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS human_reviews ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ticker TEXT, agreement TEXT,"
            " choice TEXT, note TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO human_reviews (run_id, ticker, agreement, choice, note, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, ticker.upper(), agreement, choice, note,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()
    return json.dumps({"ok": True})

@mcp.tool()
def save_rebalance(batch_id: str, decisions_json: str) -> str:
    """Save rebalancing proposals with the human's approve/reject decision."""
    decisions = json.loads(decisions_json)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_conn()) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rebalance_decisions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, ticker TEXT, action TEXT,"
            " shares INTEGER, est_value REAL, approved INTEGER, created_at TEXT)"
        )
        conn.executemany(
            "INSERT INTO rebalance_decisions (batch_id, ticker, action, shares, est_value, approved, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (batch_id, d["ticker"], d["action"], int(d["shares"]), float(d["est_value"]),
                 1 if d["approve"] else 0, now)
                for d in decisions
            ],
        )
        conn.commit()
    return json.dumps({"ok": True, "saved": len(decisions)})

if __name__ == "__main__":
    mcp.run()

