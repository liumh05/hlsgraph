"""Sanitize ignored raw A/B artifacts before copying them into a public result set."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


REDACTIONS = (
    (re.compile(r"(?i)\b(?:sk|ghp|github_pat)-?[A-Za-z0-9_-]{12,}\b"), "$TOKEN"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"), "Bearer $TOKEN"),
    (re.compile(r"(?i)\b[A-Z]:[\\/][^\s\"']*"), "$ABS_PATH"),
    (re.compile(r"\\\\[^\\\s\"']+\\[^\s\"']+"), "$ABS_PATH"),
    (re.compile(r"/(?:home|Users)/[^/\s\"']+(?:/[^\s\"']*)?"), "$HOME_PATH"),
    (re.compile(r"(?<![A-Za-z0-9:/])/(?!/)[^\s\"']+"), "$ABS_PATH"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "$IP"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "$EMAIL"),
)

TOOL_ITEM_TYPES = {
    "command_execution", "mcp_tool_call", "tool_call", "file_search",
    "computer_tool_call", "function_call",
}
TOOL_PAYLOAD_FIELDS = {
    "aggregated_output", "arguments", "command", "content", "output",
    "result", "stderr", "stdout", "text",
}
TOOL_PAYLOAD_REDACTION = "$TOOL_PAYLOAD_REDACTED"


def sanitize_text(text: str, *, workspace: str | None = None) -> str:
    output = text
    if workspace:
        output = output.replace(workspace, "$WORKSPACE")
        output = output.replace(workspace.replace("\\", "/"), "$WORKSPACE")
    for pattern, replacement in REDACTIONS:
        output = pattern.sub(replacement, output)
    return output


def sanitize_value(value: Any, *, workspace: str | None = None) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, workspace=workspace)
    if isinstance(value, list):
        return [sanitize_value(item, workspace=workspace) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_value(item, workspace=workspace) for key, item in value.items()}
    return value


def sanitize_public_record(value: Any, *, workspace: str | None = None) -> Any:
    """Sanitize an event and remove raw command/tool request and response bytes."""
    clean = sanitize_value(value, workspace=workspace)
    return _redact_tool_payloads(clean)


def _redact_tool_payloads(clean: Any) -> Any:
    if isinstance(clean, list):
        return [_redact_tool_payloads(item) for item in clean]
    if not isinstance(clean, dict):
        return clean
    item = clean.get("item")
    event_type = str(clean.get("type", "")).casefold()
    if isinstance(item, dict) and str(item.get("type", "")).casefold() in TOOL_ITEM_TYPES:
        for key in TOOL_PAYLOAD_FIELDS & set(item):
            item[key] = TOOL_PAYLOAD_REDACTION
    elif (event_type in TOOL_ITEM_TYPES or "tool" in event_type or "command" in event_type) \
            and not event_type.endswith(".started"):
        for key in TOOL_PAYLOAD_FIELDS & set(clean):
            clean[key] = TOOL_PAYLOAD_REDACTION
    for key, nested in list(clean.items()):
        clean[key] = _redact_tool_payloads(nested)
    return clean


def sanitize_file(source: Path, destination: Path, *, workspace: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.casefold() == ".json":
        value = json.loads(source.read_text(encoding="utf-8", errors="replace"))
        destination.write_text(
            json.dumps(
                sanitize_public_record(value, workspace=workspace), indent=2,
                sort_keys=True, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
    elif source.suffix.casefold() == ".jsonl":
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        rendered: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            value = json.loads(line)
            rendered.append(json.dumps(
                sanitize_public_record(value, workspace=workspace),
                sort_keys=True, ensure_ascii=False,
            ))
        destination.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    else:
        destination.write_text(
            sanitize_text(source.read_text(encoding="utf-8", errors="replace"), workspace=workspace),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workspace")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sanitize_file(args.source, args.destination, workspace=args.workspace)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
