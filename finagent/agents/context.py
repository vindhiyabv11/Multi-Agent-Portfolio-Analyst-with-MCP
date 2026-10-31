from ..mcp_client import call_json, get_tools
from ..telemetry import traced


@traced("load_context")
async def load_context_node(state):
    tools = await get_tools()
    res = await call_json(tools["get_holding"], ticker=state["ticker"])
    holding = res if isinstance(res, dict) and res.get("held") else None
    return {"holding": holding}

    