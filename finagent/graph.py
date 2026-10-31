import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .agents.aggregator import aggregator_node
from .agents.context import load_context_node
from .agents.macro import macro_node
from .agents.quant import quant_node
from .agents.review import review_gate_node
from .config import CHECKPOINT_DB
from .guardrails import validate_ticker
from .state import AgentState


def build_graph(checkpointer=None):
    b = StateGraph(AgentState)
    b.add_node("load_context", load_context_node)
    b.add_node("macro", macro_node)
    b.add_node("quant", quant_node)
    b.add_node("review_gate", review_gate_node)
    b.add_node("aggregator", aggregator_node)

    b.add_edge(START, "load_context")
    b.add_edge("load_context", "macro")                 # fan-out (parallel)
    b.add_edge("load_context", "quant")
    b.add_edge(["macro", "quant"], "review_gate")       # fan-in
    b.add_edge("review_gate", "aggregator")
    b.add_edge("aggregator", END)
    return b.compile(checkpointer=checkpointer)


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _initial(ticker: str, run_id: str, hitl: bool) -> dict:
    return {"ticker": validate_ticker(ticker), "run_id": run_id, "hitl": hitl, "errors": [], "warnings": []}


async def _drive(graph, graph_input, run_id: str, on_update) -> dict:
    """Run (or resume) the graph. Returns status 'done' or 'waiting' (paused for a human)."""
    config = _config(run_id)
    async for chunk in graph.astream(graph_input, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node.startswith("__"):  # e.g. "__interrupt__"
                continue
            if on_update:
                on_update(node, update)

    snap = await graph.aget_state(config)
    interrupt_value = None
    for task in snap.tasks:
        if task.interrupts:
            interrupt_value = task.interrupts[0].value
            break
    return {
        "status": "waiting" if snap.next else "done",
        "run_id": run_id,
        "state": dict(snap.values),
        "interrupt": interrupt_value,
    }


async def start_analysis(ticker: str, run_id: str | None = None, hitl: bool = False, on_update=None) -> dict:
    run_id = run_id or uuid.uuid4().hex[:8]
    graph_input = _initial(ticker, run_id, hitl)  # validates the ticker
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(saver)
        return await _drive(graph, graph_input, run_id, on_update)


async def resume_analysis(run_id: str, choice: str, note: str = "", on_update=None) -> dict:
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(saver)
        return await _drive(graph, Command(resume={"choice": choice, "note": note}), run_id, on_update)


async def run_analysis(ticker: str, run_id: str | None = None, on_update=None) -> dict:
    """Simple one-shot run (no human review). Returns the final state."""
    result = await start_analysis(ticker, run_id=run_id, hitl=False, on_update=on_update)
    return result["state"]


async def run_many(tickers: list[str], concurrency: int = 2, on_update=None) -> dict:
    """Analyze several tickers with one shared checkpointer. on_update(ticker, node, update)."""
    results: dict = {}
    sem = asyncio.Semaphore(concurrency)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(saver)

        async def one(t: str):
            run_id = uuid.uuid4().hex[:8]
            async with sem:
                try:
                    cb = (lambda n, u, t=t: on_update(t, n, u)) if on_update else None
                    results[t] = await _drive(graph, _initial(t, run_id, False), run_id, cb)
                except Exception as e:
                    results[t] = {"status": "error", "error": str(e), "state": {}, "run_id": run_id, "interrupt": None}

        await asyncio.gather(*(one(t) for t in tickers))
    return results

    