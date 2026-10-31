import asyncio
import sqlite3
from contextlib import closing

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from finagent.config import DB_PATH, OPENAI_API_KEY
from finagent.graph import resume_analysis, start_analysis
from finagent.indicators import rsi, sma
from finagent.portfolio import analyze_portfolio, save_decisions

st.set_page_config(page_title="Fin Agent Desk", page_icon="📈", layout="wide")
st.title("📈 Fin Agent Desk")
st.caption("Macro analyst + Quant analyst → Aggregator, with guardrails, human review and live telemetry. Educational use only.")

CHOICES = {
    "weigh_news": "Weigh the NEWS more",
    "weigh_technicals": "Weigh the TECHNICALS more",
    "neither": "Neither: stay cautious",
}


def query(sql: str, params=()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with closing(sqlite3.connect(DB_PATH, timeout=15)) as conn:
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except Exception:
            return pd.DataFrame()


def fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:,.{nd}f}"


def store_result(result: dict):
    st.session_state["state"] = result["state"]
    st.session_state["pending"] = result if result["status"] == "waiting" else None


# ---------- sidebar ----------
holdings = query("SELECT * FROM holdings ORDER BY ticker")
with st.sidebar:
    st.header("Analyze an asset")
    options = ["(none)"] + (holdings["ticker"].tolist() if not holdings.empty else [])
    picked = st.selectbox("Pick a holding", options)
    custom = st.text_input("...or type any ticker", placeholder="AAPL, BTC-USD, TCS.NS")
    ticker = custom.strip().upper() or ("" if picked == "(none)" else picked)
    hitl = st.checkbox("Ask me when the analysts disagree", value=True)
    run_clicked = st.button("Run analysis", type="primary", disabled=not ticker)
    st.caption("NSE tickers need the .NS suffix, e.g. INFY.NS")

if run_clicked:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY missing. Add it to .env and restart.")
        st.stop()
    with st.status(f"Running agents for {ticker}...", expanded=True) as status:

        def on_update(node, update):
            status.write(f"✅ **{node}** finished")

        try:
            result = asyncio.run(start_analysis(ticker, hitl=hitl, on_update=on_update))
            store_result(result)
            if result["status"] == "waiting":
                status.update(label="Paused: your input is needed", state="complete")
            else:
                status.update(label="Analysis complete", state="complete")
        except Exception as e:
            status.update(label="Run failed", state="error")
            st.error(str(e))

# ---------- human review form ----------
pending = st.session_state.get("pending")
if pending:
    info = pending.get("interrupt") or {}
    macro, quant = info.get("macro") or {}, info.get("quant") or {}
    st.warning("⏸ The analysts disagree. The final report is waiting for your decision.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Macro (news): {macro.get('sentiment')}**")
        st.write(", ".join(macro.get("key_themes", [])))
        for e in macro.get("evidence", []):
            st.caption(f"• {e}")
    with c2:
        st.markdown(f"**Quant (technicals): {quant.get('bias')}**")
        st.write(quant.get("summary", ""))
    with st.form("review_form"):
        choice = st.radio("How should the final report weigh the evidence?", list(CHOICES), format_func=lambda k: CHOICES[k])
        note = st.text_input("Optional note")
        submitted = st.form_submit_button("Continue")
    if submitted:
        resumed = False
        with st.status("Finishing the report...", expanded=True) as status:

            def on_update2(node, update):
                status.write(f"✅ **{node}** finished")

            try:
                result = asyncio.run(resume_analysis(pending["run_id"], choice, note, on_update=on_update2))
                store_result(result)
                status.update(label="Analysis complete", state="complete")
                resumed = True
            except Exception as e:
                status.update(label="Resume failed", state="error")
                st.error(str(e))
        if resumed:
            st.rerun()

state = st.session_state.get("state")
tab_report, tab_charts, tab_tele, tab_port = st.tabs(["Report", "Charts", "Telemetry", "Portfolio"])

# ---------- report ----------
with tab_report:
    if not state:
        st.info("Pick a ticker in the sidebar and press Run analysis.")
    else:
        final = state.get("final")
        if state.get("errors"):
            st.warning("Some steps had problems:\n\n" + "\n".join(f"- {e}" for e in state["errors"]))
        if not final:
            if pending:
                st.info("Waiting for your review above before the report can be written.")
            else:
                st.error("No report was produced.")
        else:
            # guardrail messages
            for w in final.get("warnings", []):
                st.warning(f"Data/safety warning: {w}")
            if final.get("banned_phrases"):
                st.error("Language guardrail triggered (confidence reduced): " + ", ".join(final["banned_phrases"]))
            if final.get("ungrounded_numbers"):
                st.warning("These numbers in the text could not be traced to computed data: " + ", ".join(final["ungrounded_numbers"]))
            if final.get("unsupported_claims"):
                st.warning("The text makes claims the system has no data to support: " + ", ".join(final["unsupported_claims"]))
            if final.get("consistency_note"):
                st.info(final["consistency_note"])
            if final.get("human_guidance"):
                msg = f"Human review: you chose **{CHOICES.get(final['human_guidance'], final['human_guidance'])}**"
                if final.get("human_note"):
                    msg += f" (note: {final['human_note']})"
                st.info(msg)

            colors = {"healthy": "green", "watch": "orange", "at_risk": "red"}
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"### {final['ticker']}")
            c2.markdown(f"### :{colors.get(final['verdict'], 'gray')}[{final['verdict'].upper()}]")
            c3.metric("Confidence", f"{final['overall_confidence']:.0%}")
            st.write(f"**Suggested action:** {final['suggested_action']}")

            agree = final["agreement"]
            msg = f"Macro: **{final['macro_sentiment']}** vs Quant: **{final['quant_bias']}**"
            if agree == "agree":
                st.success(f"Analysts agree. {msg}")
            elif agree == "partial":
                st.warning(f"Partial agreement. {msg}")
            elif agree == "conflict":
                st.warning(f"Analysts disagree. {msg}. Treat this report with caution.")
            else:
                st.warning("Incomplete: one analyst failed.")

            st.subheader("Summary")
            st.write(final["summary"])
            st.subheader("Where the analysts differ")
            st.write(final["conflict_explanation"])
            st.subheader("Impact on your portfolio")
            st.write(final["portfolio_impact"])
            if final.get("position"):
                p = final["position"]
                a, b, c = st.columns(3)
                a.metric("Market value", fmt(p["market_value"]))
                b.metric("Unrealized P&L", fmt(p["unrealized_pnl"]), f"{p['unrealized_pnl_pct']}%")
                c.metric("Target weight", fmt(p["target_weight"], pct=True, nd=0))
            st.subheader("Key risks")
            for r in final["risks"]:
                st.write(f"- {r}")

            m = state.get("metrics") or {}
            st.subheader("Computed metrics (pandas, not the LLM)")
            k = st.columns(5)
            k[0].metric("Last close", fmt(m.get("last_close")))
            k[1].metric("RSI 14", fmt(m.get("rsi14")))
            k[2].metric("Volatility (ann.)", fmt(m.get("volatility"), pct=True))
            k[3].metric("Sharpe", fmt(m.get("sharpe")))
            k[4].metric("Max drawdown", fmt(m.get("max_drawdown"), pct=True))

            with st.expander("Macro analyst output"):
                st.json(state.get("macro") or {})
            with st.expander("Quant analyst output"):
                st.json(state.get("quant") or {})
            st.caption(final["disclaimer"])

# ---------- charts ----------
with tab_charts:
    if not state or not state.get("candles"):
        st.info("Run an analysis to see charts.")
    else:
        df = pd.DataFrame(state["candles"])
        df["date"] = pd.to_datetime(df["date"])
        df["sma20"] = sma(df["close"], 20)
        df["sma50"] = sma(df["close"], 50)
        df["rsi"] = rsi(df["close"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
        fig.add_trace(
            go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"],
                           close=df["close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["sma20"], name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["sma50"], name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI 14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.update_layout(height=650, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

# ---------- telemetry ----------
with tab_tele:
    if state and state.get("run_id"):
        logs = query(
            "SELECT node, status, started_at, duration_ms, detail FROM run_logs WHERE run_id = ? ORDER BY id",
            (state["run_id"],),
        )
        st.subheader(f"Run {state['run_id']}")
        if logs.empty:
            st.info("No logs for this run.")
        else:
            st.plotly_chart(px.bar(logs, x="node", y="duration_ms", color="status", title="Node duration (ms)"))
            st.dataframe(logs)
    st.subheader("Recent runs")
    st.dataframe(query("SELECT run_id, node, status, duration_ms, started_at FROM run_logs ORDER BY id DESC LIMIT 40"))
    st.subheader("Human reviews")
    st.dataframe(query("SELECT created_at, ticker, choice, note FROM human_reviews ORDER BY id DESC LIMIT 10"))

# ---------- portfolio ----------
with tab_port:
    if holdings.empty:
        st.warning("No holdings found. Run: uv run python data\\seed.py")
    else:
        h = holdings.copy()
        h["cost_basis"] = h["quantity"] * h["avg_buy_price"]
        st.dataframe(h)
        st.plotly_chart(px.pie(h, names="ticker", values="cost_basis", title="Allocation by cost basis (INR)"))

        st.subheader("Whole-portfolio analysis")
        st.caption("Runs every holding through the agents, then compares live weights to your targets.")
        if st.button("Analyze whole portfolio"):
            if not OPENAI_API_KEY:
                st.error("OPENAI_API_KEY missing.")
            else:
                with st.status("Analyzing all holdings (about 1-2 minutes)...", expanded=True) as status:

                    def on_p(tkr, node, update):
                        status.write(f"✅ {tkr}: {node}")

                    try:
                        st.session_state["portfolio"] = asyncio.run(analyze_portfolio(on_update=on_p))
                        status.update(label="Portfolio analysis complete", state="complete")
                    except Exception as e:
                        status.update(label="Failed", state="error")
                        st.error(str(e))

        pf = st.session_state.get("portfolio")
        if pf:
            table = pf["table"]
            st.dataframe(
                table[["ticker", "verdict", "action", "agreement", "confidence", "pnl_pct",
                       "market_value", "actual_weight", "target_weight", "drift", "note"]]
            )
            fig = go.Figure()
            fig.add_bar(x=table["ticker"], y=table["actual_weight"] * 100, name="Actual weight %")
            fig.add_bar(x=table["ticker"], y=table["target_weight"] * 100, name="Target weight %")
            fig.update_layout(barmode="group", title="Actual vs target weight")
            st.plotly_chart(fig)

            st.subheader("Rebalancing proposals (you decide)")
            st.caption("Rule-based: holdings more than 5 percentage points off target. Educational only, not advice.")
            props = pf["proposals"]
            if props.empty:
                st.success("All holdings are within 5 percentage points of target.")
            else:
                edited = st.data_editor(
                    props,
                    column_config={"approve": st.column_config.CheckboxColumn("Approve?")},
                    disabled=["ticker", "action", "shares", "est_value", "drift_pp", "caution"],
                    hide_index=True,
                    key="proposal_editor",
                )
                if st.button("Save my decisions"):
                    try:
                        res = asyncio.run(save_decisions(pf["batch_id"], edited))
                        st.success(f"Saved {res.get('saved', 0)} decision(s).")
                    except Exception as e:
                        st.error(str(e))

    st.subheader("Past reports")
    st.dataframe(query("SELECT created_at, ticker, verdict, confidence FROM analysis_reports ORDER BY id DESC LIMIT 20"))