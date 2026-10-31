import functools
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone

from langgraph.errors import GraphBubbleUp

from .config import DB_PATH


def _log(run_id, node, status, started_at, duration_ms, detail=""):
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=15)) as conn:
            conn.execute(
                "INSERT INTO run_logs (run_id, node, status, started_at, duration_ms, detail)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, node, status, started_at, round(duration_ms, 1), detail[:500]),
            )
            conn.commit()
    except Exception:
        pass  # telemetry must never crash an agent


def traced(node_name: str):
    """Log each node's duration/status, and turn exceptions into state errors
    so one failing analyst doesn't kill the whole run."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state):
            started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            t0 = time.perf_counter()
            run_id = state.get("run_id")
            try:
                out = await fn(state)
                _log(run_id, node_name, "ok", started_at, (time.perf_counter() - t0) * 1000)
                return out
            except GraphBubbleUp:
                raise  # a human-in-the-loop pause is not an error
            except Exception as e:
                _log(run_id, node_name, "error", started_at,
                     (time.perf_counter() - t0) * 1000, str(e))
                return {"errors": [f"{node_name}: {e}"]}

        return wrapper

    return decorator