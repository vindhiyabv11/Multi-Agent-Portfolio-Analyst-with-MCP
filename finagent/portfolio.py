import json
import uuid

import pandas as pd

from .graph import run_many
from .mcp_client import call_json, get_tools

DRIFT_THRESHOLD = 0.05  # 5 percentage points

PROPOSAL_COLUMNS = ["approve", "ticker", "action", "shares", "est_value", "drift_pp", "caution"]


async def load_holdings() -> list:
    tools = await get_tools()
    data = await call_json(tools["get_holdings"])
    return data if isinstance(data, list) else []


def build_table(holdings: list, results: dict) -> pd.DataFrame:
    rows = []
    for h in holdings:
        t = h["ticker"]
        res = results.get(t, {})
        state = res.get("state") or {}
        final = state.get("final") or {}
        metrics = state.get("metrics") or {}
        last = metrics.get("last_close")
        qty, cost = h["quantity"], h["avg_buy_price"]
        rows.append(
            {
                "ticker": t,
                "verdict": final.get("verdict"),
                "action": final.get("suggested_action"),
                "agreement": final.get("agreement"),
                "confidence": final.get("overall_confidence"),
                "quantity": qty,
                "avg_buy_price": cost,
                "last_close": last,
                "market_value": round(qty * last, 2) if last else None,
                "pnl_pct": round((last / cost - 1) * 100, 2) if last else None,
                "target_weight": h.get("target_weight"),
                "run_id": res.get("run_id"),
                "note": res.get("error") or "",
            }
        )
    df = pd.DataFrame(rows)
    total = df["market_value"].sum(skipna=True)
    df["actual_weight"] = df["market_value"] / total if total else float("nan")
    df["drift"] = df["actual_weight"] - df["target_weight"]
    return df


def build_proposals(table: pd.DataFrame) -> pd.DataFrame:
    total = table["market_value"].sum(skipna=True)
    rows = []
    for r in table.itertuples():
        if pd.isna(r.drift) or pd.isna(r.last_close) or abs(r.drift) < DRIFT_THRESHOLD:
            continue
        action = "trim" if r.drift > 0 else "add"
        shares = int((abs(r.drift) * total) // r.last_close)
        if action == "trim":
            shares = min(shares, int(r.quantity))
        if shares < 1:
            continue
        caution = []
        if action == "add" and r.verdict == "at_risk":
            caution.append("verdict is at_risk: consider holding off")
        if r.agreement in ("conflict", "incomplete"):
            caution.append("analysts disagree or report incomplete")
        rows.append(
            {
                "approve": False,
                "ticker": r.ticker,
                "action": action,
                "shares": shares,
                "est_value": round(shares * r.last_close, 2),
                "drift_pp": round(r.drift * 100, 1),
                "caution": "; ".join(caution),
            }
        )
    return pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)


async def analyze_portfolio(on_update=None, concurrency: int = 2) -> dict:
    """on_update(ticker, node, update). Returns table, proposals and raw results."""
    holdings = await load_holdings()
    if not holdings:
        raise RuntimeError("No holdings found. Run: uv run python data\\seed.py")
    results = await run_many([h["ticker"] for h in holdings], concurrency=concurrency, on_update=on_update)
    table = build_table(holdings, results)
    return {
        "batch_id": uuid.uuid4().hex[:8],
        "table": table,
        "proposals": build_proposals(table),
        "results": results,
    }


async def save_decisions(batch_id: str, proposals: pd.DataFrame):
    tools = await get_tools()
    payload = [
        {
            "ticker": str(r["ticker"]),
            "action": str(r["action"]),
            "shares": int(r["shares"]),
            "est_value": float(r["est_value"]),
            "approve": bool(r["approve"]),
        }
        for r in proposals.to_dict("records")
    ]
    return await call_json(tools["save_rebalance"], batch_id=batch_id, decisions_json=json.dumps(payload))