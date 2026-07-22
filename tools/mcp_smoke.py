"""End-to-end read-only MCP stdio smoke test for an indexed project."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def smoke(project_root: Path) -> None:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("install hlsgraph[mcp] to run this smoke test") from exc

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hlsgraph.mcp.server", str(project_root)],
        env=dict(os.environ),
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            initialized = await session.initialize()
            listed = await session.list_tools()
            names = sorted(tool.name for tool in listed.tools)
            required = {"explore"}
            missing = sorted(required - set(names))
            if missing:
                raise RuntimeError(f"MCP server is missing tools: {', '.join(missing)}")
            explored = await session.call_tool(
                "explore", {"query": "top kernel architecture", "max_chars": 13_000},
            )
            if explored.isError:
                raise RuntimeError("explore returned an MCP tool error")
            print(json.dumps({
                "server": initialized.serverInfo.name,
                "tools": names,
                "explore_content_blocks": len(explored.content),
                "status": "MCP_SMOKE_OK",
            }, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path,
                        help="project containing an indexed .hlsgraph bundle")
    args = parser.parse_args()
    asyncio.run(smoke(args.project_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
