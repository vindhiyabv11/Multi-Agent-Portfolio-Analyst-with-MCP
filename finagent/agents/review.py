from langgraph.types import interrupt

from ..mcp_client import call_json, get_tools
from ..telemetry import traced
from .aggregator import _agreement

VALID = ("weigh_news", "weigh_technicals", "neither")


@traced("review_gate")
async def review_gate_node(state):
    macro, quant = state.get("macro"), state.get("quant")

    # Pass straight through unless review is enabled AND the analysts truly conflict
    if not state.get("hitl") or _agreement(macro, quant) != "conflict":
        return {"human_guidance": None, "human_note": ""}

    # PAUSE: the graph state is checkpointed; this call returns when the run is resumed.
    # Note: on resume this node re-runs from the top, so side effects go AFTER interrupt().
    answer = interrupt(
        {
            "type": "analyst_conflict",
            "ticker": state["ticker"],
            "macro": macro,
            "quant": quant,
            "question": "The news and technical analysts disagree. How should the final report weigh them?",
            "options": list(VALID),
        }
    )

    choice = answer.get("choice") if isinstance(answer, dict) else None
    if choice not in VALID:
        choice = "neither"
    note = ((answer.get("note") if isinstance(answer, dict) else "") or "")[:300]

    try:
        tools = await get_tools()
        await call_json(
            tools["save_review"],
            run_id=state["run_id"], ticker=state["ticker"],
            agreement="conflict", choice=choice, note=note,
        )
    except Exception:
        pass  # saving the decision is best-effort

    return {"human_guidance": choice, "human_note": note}