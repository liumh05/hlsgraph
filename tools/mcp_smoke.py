"""End-to-end read-only MCP stdio smoke test for an indexed project."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


async def _smoke_session(project_root: Path, mode: str) -> dict[str, Any]:
    try:
        from mcp import Client, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("install hlsgraph[mcp] to run this smoke test") from exc

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hlsgraph.mcp.server", str(project_root)],
        env=dict(os.environ),
    )
    async with Client(stdio_client(parameters), mode=mode) as client:
        listed = await client.list_tools()
        names = sorted(tool.name for tool in listed.tools)
        required = {"explore"}
        missing = sorted(required - set(names))
        if missing:
            raise RuntimeError(f"MCP server is missing tools: {', '.join(missing)}")
        explored = await client.call_tool(
            "explore", {"query": "top kernel architecture", "max_chars": 13_000},
        )
        if explored.is_error:
            raise RuntimeError("explore returned an MCP tool error")
        if explored.structured_content is None:
            raise RuntimeError("explore did not return structuredContent")
        return {
            "mode": mode,
            "protocol_version": str(client.protocol_version),
            "tools": names,
            "explore_content_blocks": len(explored.content),
            "explore_chars": len(explored.content[0].text),
        }


async def smoke(project_root: Path) -> None:
    sessions = [
        await _smoke_session(project_root, "auto"),
        await _smoke_session(project_root, "legacy"),
    ]
    print(json.dumps({
        "server": "hlsgraph",
        "sessions": sessions,
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
