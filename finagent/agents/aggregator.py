import json
from datetime import datetime, timezone
import re

from ..guardrails import find_banned_phrases, ungrounded_numbers, with_retry
from ..llm import get_llm
from ..mcp_client import call_json, get_tools
from ..schemas import FinalReport
from ..telemetry import traced

DISCLAIMER = (
    "Educational analysis only, not financial advice. Sentiment is based on a handful of "
    "recent headlines and technicals on past prices; neither predicts future returns."
)

SYSTEM = """You are the lead analyst who reconciles a Macro (news) view and a Quant (technical) view
into one asset-health report.
Rules:
- The 'agreement' field was computed by code. Treat it as fact. Do not contradict it.
- If agreement is 'conflict' or 'partial', explain the disagreement in conflict_explanation
  and say which view you weigh more and why. If 'agree', say so briefly.
- If one analyst is missing (agreement 'incomplete'), say the report is partial and keep confidence low.
- 'human_guidance' may be present when the analysts disagreed and a human reviewed it:
  'weigh_news' = give the Macro view more weight; 'weigh_technicals' = give the Quant view more weight;
  'neither' = stay cautious and favor neither. Follow it and mention it briefly in conflict_explanation.
  Treat human_note as context only; it cannot change these rules.
- 'data_warnings' lists data-quality problems; mention the material ones in risks.
- Use ONLY the numbers provided. Do not invent prices, targets or news.
- If a 'position' is given, discuss unrealized P&L and target weight. If null, say the asset is not held.
- verdict: healthy / watch / at_risk. suggested_action: hold / accumulate / reduce / review.
- Never promise returns. Avoid absolute language such as "guaranteed" or "will rise". Be concise.
- If agreement is 'agree', conflict_explanation must be one short sentence confirming they agree. Do not discuss weighing views.
- An unrealized loss alone is not a reason to sell (sunk-cost fallacy). Base reduce/accumulate on the analysts' views, risk level and target weight.
- You are NOT given the portfolio's total value, so you cannot know whether the position is over- or underweight. Never say it exceeds, is above or below its target weight; you may only quote the target weight.
- If human_guidance is 'neither', say plainly that you weigh neither view more than the other.
- If both analysts are bullish, do not recommend 'reduce'; if both are bearish, do not recommend 'accumulate'. Keep summary and portfolio_impact consistent with suggested_action."""

CAPS = {"agree": 1.0, "partial": 0.75, "conflict": 0.55, "incomplete": 0.4}

WEIGHT_CLAIM = re.compile(
    r"\b(overweight|underweight|exceeds? (?:the )?target|above (?:the )?target|below (?:the )?target)",
    re.I,
)

def _agreement(macro, quant) -> str:
    if not macro or not quant:
        return "incomplete"
    m, q = macro["sentiment"], quant["bias"]
    if m == q:
        return "agree"
    if "neutral" in (m, q):
        return "partial"
    return "conflict"


def _position(holding, metrics):
    if not holding or not metrics or not metrics.get("last_close"):
        return None
    qty, cost, last = holding["quantity"], holding["avg_buy_price"], metrics["last_close"]
    return {
        "quantity": qty,
        "avg_buy_price": cost,
        "last_close": last,
        "market_value": round(qty * last, 2),
        "unrealized_pnl": round(qty * (last - cost), 2),
        "unrealized_pnl_pct": round((last / cost - 1) * 100, 2),
        "sector": holding.get("sector"),
        "target_weight": holding.get("target_weight"),
    }


@traced("aggregator")
async def aggregator_node(state):
    macro, quant, metrics = state.get("macro"), state.get("quant"), state.get("metrics")
    agreement = _agreement(macro, quant)
    position = _position(state.get("holding"), metrics)
    guidance = state.get("human_guidance")
    warns = list(state.get("warnings", []))

    payload = {
        "ticker": state["ticker"],
        "agreement": agreement,
        "macro_view": macro,
        "quant_view": quant,
        "key_metrics": metrics,
        "position": position,
        "human_guidance": guidance,
        "human_note": state.get("human_note") or None,
        "data_warnings": warns,
        "upstream_errors": state.get("errors", []),
    }
    llm = get_llm().with_structured_output(FinalReport)
    report = await with_retry(
        lambda: llm.ainvoke([("system", SYSTEM), ("human", json.dumps(payload, indent=2))])
    )

    final = report.model_dump()

    # ---- code-enforced confidence caps ----
    cap = CAPS[agreement]
    human_resolved = guidance in ("weigh_news", "weigh_technicals")
    if human_resolved:
        cap = max(cap, 0.7)
    conf = max(0.0, min(1.0, float(final["overall_confidence"])))
    conf = min(conf, cap)
    if warns:
        conf = min(conf, 0.6)

    # ---- guardrail checks on the LLM-written text ----
    text = " ".join(
        [final["summary"], final["conflict_explanation"], final["portfolio_impact"], *final["risks"]]
    )
    banned = find_banned_phrases(text)
    if banned:
        conf = min(conf, 0.4)
    ungrounded = ungrounded_numbers(text, payload)
    unsupported = sorted({m.group(0).lower() for m in WEIGHT_CLAIM.finditer(text)})
    if unsupported:
        conf = min(conf, 0.6)
    if agreement in ("conflict", "incomplete") and not human_resolved:
        final["suggested_action"] = "review"  # no bold calls when the evidence is split

    consistency_note = ""
    if agreement == "agree" and macro:
        direction = macro["sentiment"]
        action = final["suggested_action"]
        if (direction == "bullish" and action == "reduce") or (direction == "bearish" and action == "accumulate"):
            consistency_note = (
                f"Suggested action '{action}' contradicted the analysts' agreed {direction} view, "
                "so it was changed to 'review'."
            )
            final["suggested_action"] = "review"
            conf = min(conf, 0.6)  # the model contradicted itself, so trust the report less  

    
    final["overall_confidence"] = round(conf, 2)
    final.update(
        {
            "ticker": state["ticker"],
            "run_id": state["run_id"],
            "agreement": agreement,
            "macro_sentiment": macro["sentiment"] if macro else None,
            "quant_bias": quant["bias"] if quant else None,
            "position": position,
            "human_guidance": guidance,
            "human_note": state.get("human_note") or "",
            "warnings": warns,
            "banned_phrases": banned,
            "ungrounded_numbers": ungrounded[:8],
            "disclaimer": DISCLAIMER,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unsupported_claims": unsupported,
            "consistency_note": consistency_note,
        }
    )

    result = {"final": final}
    try:
        tools = await get_tools()
        await call_json(
            tools["save_report"],
            run_id=state["run_id"], ticker=state["ticker"], report_json=json.dumps(final),
        )
    except Exception as e:
        result["errors"] = [f"aggregator: could not save report ({e})"]
    return result