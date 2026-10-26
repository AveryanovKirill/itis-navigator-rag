import os
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ROOT


@asynccontextmanager
async def connect_mcp(args=None):
    # API-ключ не передаётся процессу, которому он не нужен.
    env = {k: v for k, v in os.environ.items() if k not in {"OPENROUTER_API_KEY"}}
    env["PYTHONUTF8"] = "1"
    env["ITIS_MCP_SERVER"] = "1"
    parameters = StdioServerParameters(command=sys.executable, args=args or ["-m", "src.mcp_server"], cwd=str(ROOT), env=env)
    if os.getenv("ITIS_TRACE_MCP") == "1":
        print("MCP: запускается stdio subprocess", file=sys.stderr)
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=180)) as session:
            await session.initialize()
            if os.getenv("ITIS_TRACE_MCP") == "1":
                print("MCP: initialize выполнен", file=sys.stderr)
            tools = await session.list_tools()
            if "search_docs" not in {t.name for t in tools.tools}:
                raise ValueError("MCP-сервер не объявил search_docs.")
            yield session


async def call_search(session, question):
    import json
    if os.getenv("ITIS_TRACE_MCP") == "1":
        print("MCP: call_tool(search_docs)", file=sys.stderr)
    result = await session.call_tool("search_docs", {"query": question})
    if result.isError:
        raise ValueError("MCP search_docs вернул ошибку: " + " ".join(c.text for c in result.content if c.type == "text"))
    data = result.structuredContent
    if not data:
        data = json.loads("\n".join(c.text for c in result.content if c.type == "text"))
    if data.get("status") not in {"found", "not_found"} or not isinstance(data.get("fragments"), list):
        raise ValueError("Неправильный ответ search_docs.")
    return data
