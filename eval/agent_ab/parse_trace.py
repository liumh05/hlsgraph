"""Normalize Codex ``--json`` output without depending on a private event schema."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


READ_COMMAND = re.compile(
    r"(?:^|[;&|\s])(rg|grep|cat|type|get-content|select-string|findstr|sed|head|tail)\b",
    re.IGNORECASE,
)

# ``file_reads`` is retained as the public v1 column name, but its semantics are
# deliberately broader than shell invocations: one source-bearing MCP retrieval
# is one source-access call too.  This avoids granting either graph arm a free
# "zero reads" merely because source arrived inside a tool result.
SOURCE_BEARING_MCP_TOOLS = {
    "codegraph_explore", "explore", "context", "module_or_region", "evidence",
    "search",
}

_LEGACY_HLSGRAPH_TOOLS = {
    "overview", "search", "context", "module_or_region", "traverse", "impact",
    "evidence", "feature_evidence", "correspondences", "compare", "health",
    "runs", "predictions", "variants", "render", "knowledge",
}
_NETWORK_COMMAND = re.compile(
    r"(?:^|[;&|\s])(?:curl|wget|ssh|scp|sftp|ftp|ping|nslookup|telnet|nc|ncat)\b"
    r"|\b(?:invoke-webrequest|invoke-restmethod|start-bitstransfer|test-netconnection)\b"
    r"|\b(?:requests|urllib|http\.client|socket|webclient|tcpclient)\b"
    r"|https?://|\bgit\s+(?:clone|fetch|pull|ls-remote)\b",
    re.IGNORECASE,
)
_ESCAPE_COMMAND = re.compile(
    r"(?:^|[\\/\s'\"])[.][.](?:$|[\\/\s'\"])"
    r"|(?:^|[\s'\"])(?:[A-Za-z]:[\\/]|\\\\)"
    r"|(?:^|[\s'\"=(:,])/(?!/)"
    r"|(?:^|[;&|\s])(?:cd|chdir|set-location|push-location|pop-location|resolve-path|split-path)\b"
    r"|\.(?:parent|parents)\b|\$env:|\$\{|\$\(|\$[A-Za-z_]|`|%[A-Za-z_][A-Za-z0-9_]*%|~[\\/]"
    r"|(?:^|[;&|])\s*(?:env|printenv|set)(?:\s|$)|\benv:"
    r"|(?:^|[;&|\s])(?:python(?:3(?:\.\d+)?)?|py|node|ruby|perl|bash|sh|cmd|powershell|pwsh)\b",
    re.IGNORECASE,
)
_GOLD_COMMAND = re.compile(
    r"questions\.jsonl|static_cases\.jsonl|answer\.schema\.json|"
    r"(?:^|[\\/])(?:score|bootstrap|runner|common)\.py\b",
    re.IGNORECASE,
)
_GOLD_OUTPUT_MARKERS = (
    '"schema_version":"hlsgraph.agent_eval.question.v1"',
    '"schema_version": "hlsgraph.agent_eval.question.v1"',
    '"evidence_selectors"',
    '"forbidden_claims"',
    '"schema_version":"hlsgraph.agent_eval.static_case.v1"',
    '"schema_version": "hlsgraph.agent_eval.static_case.v1"',
)


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Codex JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            yield value


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("output_text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces) if pieces else None
    if isinstance(value, dict):
        for key in ("output_text", "text", "content"):
            text = _content_text(value.get(key))
            if text:
                return text
    return None


def extract_final_text(events: Iterable[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for event in events:
        event_type = str(event.get("type", ""))
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type", ""))
        if item_type in {"agent_message", "message", "assistant_message"}:
            text = _content_text(item)
            if text:
                candidates.append(text)
        elif event_type in {"agent_message", "message.completed", "response.completed"}:
            text = _content_text(event.get("message") or event.get("response") or event)
            if text:
                candidates.append(text)
    if not candidates:
        raise ValueError("Codex trace contains no completed assistant message")
    return candidates[-1].strip()


def parse_answer_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("final answer must be a JSON object")
    return value


def _tool_identity(
    event: dict[str, Any], *, include_started: bool = False,
) -> tuple[str, str, str, str, str] | None:
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    item_type = str(item.get("type", ""))
    event_type = str(event.get("type", ""))
    toolish = {
        "command_execution", "mcp_tool_call", "tool_call", "file_search",
        "computer_tool_call", "function_call",
    }
    if item_type not in toolish and "tool" not in event_type and "command" not in event_type:
        return None
    if event_type.endswith(".started") and not include_started:
        return None
    call_id = str(item.get("id") or item.get("call_id") or event.get("call_id") or "")
    name = str(
        item.get("name") or item.get("tool_name") or item.get("tool")
        or item_type or event_type
    )
    command = item.get("command") or item.get("arguments") or ""
    if isinstance(command, (dict, list)):
        command = json.dumps(command, sort_keys=True)
    server = str(item.get("server") or item.get("server_name") or event.get("server") or "")
    return call_id, name, str(command), item_type, server


def _tool_output_text(event: dict[str, Any]) -> str:
    """Return only tool-result material, never prompts or assistant prose."""

    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    values = [
        item.get("aggregated_output"), item.get("output"), item.get("result"),
        item.get("content"), event.get("tool_output"), event.get("result"),
    ]
    pieces: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            pieces.append(value)
        else:
            try:
                pieces.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
            except (TypeError, ValueError):
                pieces.append(str(value))
    return "\n".join(pieces)


def _tool_leaf(name: str) -> str:
    lowered = name.casefold().replace("-", "_")
    for separator in ("__", ".", "/", ":"):
        if separator in lowered:
            lowered = lowered.split(separator)[-1]
    return lowered


def _is_mcp_tool(item_type: str, name: str, server: str) -> bool:
    lowered = name.casefold()
    return item_type == "mcp_tool_call" or bool(server) or lowered.startswith("mcp_")


def _mcp_allowed(arm: str, name: str, server: str) -> bool:
    leaf = _tool_leaf(name)
    lowered_name = name.casefold()
    lowered_server = server.casefold()
    if arm == "native":
        return False
    if arm == "codegraph":
        server_ok = lowered_server == "codegraph" or (
            not lowered_server and lowered_name in {
                "codegraph_explore", "codegraph.explore",
                "mcp__codegraph__codegraph_explore",
            }
        )
        name_ok = leaf in {"explore", "codegraph_explore"}
        return server_ok and name_ok
    if arm == "hlsgraph-v03":
        server_ok = lowered_server == "hlsgraph" or (
            not lowered_server and lowered_name in {
                "hlsgraph.explore", "mcp__hlsgraph__explore",
            }
        )
        return server_ok and leaf == "explore"
    if arm == "hlsgraph-v02":
        server_ok = lowered_server == "hlsgraph" or (
            not lowered_server and any(
                lowered_name == f"hlsgraph.{tool}"
                or lowered_name == f"mcp__hlsgraph__{tool}"
                for tool in _LEGACY_HLSGRAPH_TOOLS
            )
        )
        return server_ok and leaf in _LEGACY_HLSGRAPH_TOOLS
    return False


def _tool_outcome(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "")).casefold()
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    status = str(item.get("status") or event.get("status") or "").casefold()
    error = item.get("error") or event.get("error")
    if error or status in {"failed", "error", "cancelled", "canceled"}:
        return "failed"
    if event_type.endswith(".completed") or status in {"completed", "success", "succeeded"}:
        return "completed"
    return "incomplete"


def validate_trace_policy(
    events: Iterable[dict[str, Any]], *, arm: str, workspace: Path,
    boundary_canary: bytes | None = None,
) -> dict[str, Any]:
    """Fail closed on tool, network, and readable-boundary violations.

    The named Codex permission profile is the primary boundary.  This trace
    check is an independent second gate: even a successful command is unusable
    for the public result when its request escaped the corpus, invoked the web,
    or used an MCP server/tool outside the frozen arm.
    """

    root = workspace.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("trace policy workspace is missing or linked")
    canary_text: str | None = None
    if boundary_canary:
        try:
            canary_text = boundary_canary.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("boundary canary must be UTF-8") from exc
        if not canary_text:
            raise ValueError("boundary canary must not be empty")

    violations: list[str] = []
    completed_tools = 0
    logical_tools: list[dict[str, Any]] = []
    logical_by_id: dict[str, dict[str, Any]] = {}
    for ordinal, event in enumerate(events):
        event_type = str(event.get("type", ""))
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type", ""))
        name = str(
            item.get("name") or item.get("tool_name") or item.get("tool")
            or item_type or event_type
        )
        lowered = f"{event_type} {item_type} {name}".casefold()
        if any(token in lowered for token in (
            "web_search", "browser", "computer_tool", "image_generation",
        )):
            violations.append(f"forbidden-tool:{name}")
        if item_type in {"agent_message", "message", "assistant_message"}:
            assistant_text = _content_text(item) or ""
            compact_assistant = re.sub(r"\s+", "", assistant_text)
            if any(re.sub(r"\s+", "", marker) in compact_assistant
                   for marker in _GOLD_OUTPUT_MARKERS):
                violations.append("gold-output:assistant")
            if canary_text is not None and canary_text in assistant_text:
                violations.append("boundary canary disclosed by assistant")
        identity = _tool_identity(event, include_started=True)
        if identity is None:
            continue
        call_id, tool_name, command, tool_type, server = identity
        logical_id = call_id or f"event-{ordinal}"
        logical = logical_by_id.get(logical_id)
        if logical is None:
            logical = {
                "name": tool_name, "item_type": tool_type, "server": server,
                "outcome": _tool_outcome(event),
            }
            logical_by_id[logical_id] = logical
            logical_tools.append(logical)
        else:
            outcome = _tool_outcome(event)
            if outcome != "incomplete":
                logical["outcome"] = outcome
        if not str(event.get("type", "")).endswith(".started"):
            completed_tools += 1
        if _is_mcp_tool(tool_type, tool_name, server):
            if not _mcp_allowed(arm, tool_name, server):
                violations.append(f"unexpected-mcp:{server or '?'}:{tool_name}")
        if command:
            if _NETWORK_COMMAND.search(command):
                violations.append(f"network-command:{tool_name}")
            if _ESCAPE_COMMAND.search(command):
                violations.append(f"workspace-escape:{tool_name}")
            if _GOLD_COMMAND.search(command):
                violations.append(f"gold-command:{tool_name}")
            if canary_text is not None and canary_text in command:
                violations.append(f"boundary canary requested by {tool_name}")
        output = _tool_output_text(event)
        compact_output = re.sub(r"\s+", "", output)
        if any(re.sub(r"\s+", "", marker) in compact_output for marker in _GOLD_OUTPUT_MARKERS):
            violations.append(f"gold-output:{tool_name}")
        if canary_text is not None and canary_text in output:
            violations.append(f"boundary canary disclosed by {tool_name}")
    graph_arm = arm != "native"
    treatment_calls = [
        item for item in logical_tools
        if _is_mcp_tool(item["item_type"], item["name"], item["server"])
        and _mcp_allowed(arm, item["name"], item["server"])
    ]
    first_is_treatment = bool(logical_tools and treatment_calls
                              and logical_tools[0] is treatment_calls[0])
    if graph_arm and not treatment_calls:
        violations.append("missing-treatment-mcp")
    if graph_arm and not first_is_treatment:
        violations.append("first-call-not-treatment-mcp")
    if violations:
        raise ValueError("trace policy violation: " + ", ".join(sorted(set(violations))))
    return {
        "schema_version": "hlsgraph.agent_eval.trace_policy.v1",
        "passed": True,
        "arm": arm,
        "workspace": "$CORPUS_WORKSPACE",
        "completed_tools": completed_tools,
        "treatment_mcp_required": graph_arm,
        "treatment_mcp_calls": len(treatment_calls),
        "first_call_treatment_mcp": first_is_treatment,
        "treatment_mcp_first_outcome": (
            treatment_calls[0]["outcome"] if treatment_calls else "not_applicable"
        ),
        "treatment_mcp_outcomes": [item["outcome"] for item in treatment_calls],
    }


def _last_usage(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return one atomic usage object from the last terminal event.

    A terminal event may expose usage either directly or below ``response``.
    Never merge fields across objects or events: doing so could synthesize a
    seemingly complete counter set from multiple incomplete observations.
    """

    usage: dict[str, Any] = {}
    for event in events:
        if str(event.get("type", "")) not in {"turn.completed", "response.completed"}:
            continue
        candidates = [event.get("usage")]
        response = event.get("response")
        if isinstance(response, dict):
            candidates.append(response.get("usage"))
        objects = [candidate for candidate in candidates if isinstance(candidate, dict)]
        if len(objects) > 1 and any(candidate != objects[0] for candidate in objects[1:]):
            raise ValueError("Codex terminal event contains conflicting usage objects")
        # A later terminal event without usage invalidates an earlier observation;
        # the last completion is the authoritative terminal state.
        usage = dict(objects[-1]) if objects else {}
    return usage


def normalize_trace(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    event_list = list(events)
    thread_ids = sorted({
        str(event.get("thread_id")) for event in event_list
        if event.get("type") == "thread.started"
        and isinstance(event.get("thread_id"), str)
        and event.get("thread_id")
    })
    tools: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for ordinal, event in enumerate(event_list):
        identity = _tool_identity(event)
        if identity is None:
            continue
        call_id, name, command, item_type, server = identity
        dedupe = (call_id, name, command) if call_id else (f"event-{ordinal}", name, command)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        tools.append({
            "call_id": call_id, "name": name, "command": command,
            "item_type": item_type, "server": server,
        })
    reads = sum(
        1 for item in tools
        if READ_COMMAND.search(item["command"])
        or item["name"].casefold() in {"read", "grep", "search", "file_search"}
        or (
            _is_mcp_tool(item["item_type"], item["name"], item["server"])
            and _tool_leaf(item["name"]) in SOURCE_BEARING_MCP_TOOLS
        )
    )
    final_text = extract_final_text(event_list)
    return {
        "schema_version": "hlsgraph.agent_eval.normalized_trace.v1",
        "answer": parse_answer_text(final_text),
        "tool_calls": len(tools),
        "file_reads": reads,
        "file_read_semantics": "source_access_tool_calls",
        "tools": tools,
        "usage": _last_usage(event_list),
        "thread_ids": thread_ids,
    }


def parse_run(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    normalized = normalize_trace(iter_events(run_dir / "codex.jsonl"))
    normalized["run"] = metadata
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = normalize_trace(iter_events(args.trace))
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
