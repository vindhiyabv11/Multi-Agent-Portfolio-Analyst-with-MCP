import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import ROOT

_TOOLS = None


def _client() -> MultiServerMCPClient:
    # sys.executable = the uv venv's python, so the servers get the same packages
    return MultiServerMCPClient(
        {
            "market": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(ROOT / "mcp_servers" / "market_server.py")],
            },
            "portfolio": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(ROOT / "mcp_servers" / "portfolio_server.py")],
            },
        }
    )


async def get_tools() -> dict:
    """Return {tool_name: tool} from both MCP servers (cached per process)."""
    global _TOOLS
    if _TOOLS is None:
        tools = await _client().get_tools()
        _TOOLS = {t.name: t for t in tools}
    return _TOOLS


def _to_text(result) -> str:
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "content"):
        result = result.content
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(getattr(item, "text", ""))
        return "".join(parts)
    return str(result)


async def call_json(tool, **kwargs):
    """Call an MCP tool and parse its JSON output."""
    raw = await tool.ainvoke(kwargs)
    text = _to_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"Unparseable tool output: {text[:200]}"}