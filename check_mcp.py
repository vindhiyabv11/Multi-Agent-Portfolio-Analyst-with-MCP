import asyncio

from finagent.mcp_client import call_json, get_tools


async def main():
    tools = await get_tools()
    print("Tools:", sorted(tools))
    print(await call_json(tools["get_quote"], ticker="TCS.NS"))
    print(await call_json(tools["get_holding"], ticker="TCS.NS"))


asyncio.run(main())