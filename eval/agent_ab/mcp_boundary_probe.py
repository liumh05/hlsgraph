"""Tiny dependency-free MCP server used only for the official boundary canary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from typing import Any


def _read(path: str) -> bool:
    try:
        candidate = Path(path)
        if candidate.is_dir():
            next(os.scandir(candidate), None)
        else:
            candidate.open("rb").read(1)
        return True
    except (OSError, RuntimeError):
        return False


def _call(arguments: dict[str, Any]) -> dict[str, Any]:
    allowed = [str(item) for item in arguments.get("allowed", [])]
    denied = [str(item) for item in arguments.get("denied", [])]
    port = int(arguments.get("port", 0))
    network = False
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=1)
        connection.close()
        network = True
    except OSError:
        pass
    value = {
        "allowed": [_read(path) for path in allowed],
        "denied": [_read(path) for path in denied],
        "network": network,
        "home": os.environ.get("HOME"),
        "proxy_present": any("proxy" in key.casefold() for key in os.environ),
        "codex_home_present": "CODEX_HOME" in os.environ,
    }
    return {
        "content": [{"type": "text", "text": json.dumps(value, sort_keys=True)}],
        "structuredContent": value,
        "isError": False,
    }


def main() -> int:
    for raw in sys.stdin:
        message: Any = {}
        try:
            message = json.loads(raw)
            request_id = message.get("id")
            method = message.get("method")
            if request_id is None:
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hlsgraph-boundary-probe", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [{
                    "name": "boundary_probe", "description": "Probe the frozen MCP boundary",
                    "inputSchema": {"type": "object"},
                }]}
            elif method == "tools/call":
                result = _call(dict(message.get("params", {}).get("arguments", {})))
            else:
                raise ValueError(f"unsupported method: {method}")
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:  # fail visibly through JSON-RPC, never traceback to stdout
            response = {
                "jsonrpc": "2.0", "id": message.get("id") if isinstance(message, dict) else None,
                "error": {"code": -32603, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
