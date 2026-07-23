#!/usr/bin/env python3
"""Fail-closed replay for one sharded Codex knowledge-review JSONL stream.

The module performs no process execution.  It reuses the single-review event
and command grammar, binds the projected cache to the fixed shard manifest,
requires every assigned source and citation chunk to be read exactly once,
and validates the sole final JSON object with the sharded suite contract.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping

from tools import knowledge_review_shards as shard_plan
from tools import run_knowledge_review as review
from tools import run_knowledge_review_suite as suite


SHARD_TRACE_SCHEMA_VERSION = "hlsgraph.knowledge-review.shard-trace.v2"
MAX_RECOVERABLE_TRANSPORT_RETRY_EVENTS = 128
_TRANSPORT_RETRY_RE = re.compile(
    r"Reconnecting\.\.\. (?P<attempt>[1-5])/5 "
    r"\(stream disconnected before completion: (?P<reason>.+)\)"
)
_TRANSPORT_RETRY_REASONS = {
    "Transport error: network error: error decoding response body":
        "http_response_decode",
    (
        "error sending request for url "
        "(https://chatgpt.com/backend-api/codex/responses)"
    ): "http_request_send",
}


class ShardReplayError(ValueError):
    """A raw event, projected cache, or shard result failed closed replay."""


@dataclass(frozen=True)
class ShardReviewReplay:
    protocol_id: str
    shard_id: str
    invocation_id: str
    thread_id: str
    raw_sha256: str
    reported_input_tokens: int
    reported_cached_input_tokens: int
    reported_output_tokens: int
    reported_reasoning_output_tokens: int
    derived_input_plus_output_tokens: int
    transport_retry_event_count: int
    result: dict[str, Any] = field(repr=False)
    result_bytes: bytes = field(repr=False)
    trace_bytes: bytes = field(repr=False)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or review._SHA256_RE.fullmatch(value) is None:
        raise ShardReplayError(f"{label} must be a lowercase SHA-256")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ShardReplayError(f"{label} must be an array")
    return value


def _reject_compaction_metadata(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "compact" in str(key).casefold():
                raise ShardReplayError(f"{label} contains compaction metadata")
            _reject_compaction_metadata(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _reject_compaction_metadata(item, label=label)


def _validate_closed_event(event: Mapping[str, Any], *, index: int) -> None:
    event_type = event.get("type")
    optional_identity = {"thread_id", "invocation_id"}
    allowed = {
        "thread.started": {"type", "thread_id", "invocation_id"},
        "turn.started": {"type", *optional_identity},
        "turn.completed": {"type", "usage", *optional_identity},
        "item.started": {"type", "item", *optional_identity},
        "item.completed": {"type", "item", *optional_identity},
    }.get(str(event_type))
    if allowed is None or not set(event).issubset(allowed):
        raise ShardReplayError(f"raw event {index} is not a closed event object")
    required = {"type"}
    if event_type == "thread.started":
        required.add("thread_id")
    elif event_type == "turn.completed":
        required.add("usage")
    elif event_type in {"item.started", "item.completed"}:
        required.add("item")
    if not required <= set(event):
        raise ShardReplayError(f"raw event {index} omits a required field")
    _reject_compaction_metadata(event, label=f"raw event {index}")


def _validate_closed_item(item: Mapping[str, Any], *, index: int) -> None:
    item_type = item.get("type")
    allowed = {
        "command_execution": {
            "id", "call_id", "type", "command", "status", "exit_code",
            "aggregated_output", "output",
        },
        "reasoning": {"id", "type", "text", "summary"},
        "agent_message": {"id", "type", "text", "content"},
    }.get(str(item_type))
    if allowed is None:
        raise ShardReplayError(
            f"raw item event {index} uses forbidden or unknown tool {item_type!r}"
        )
    if not set(item).issubset(allowed):
        raise ShardReplayError(f"raw item event {index} is not a closed item object")
    if "type" not in item:
        raise ShardReplayError(f"raw item event {index} omits its type")


def _recoverable_transport_retry(
    event: Mapping[str, Any], *, index: int,
) -> dict[str, Any]:
    """Normalize one exact Codex HTTP reconnect notice.

    A reconnect notice is telemetry, not a successful review event.  It is
    accepted only when the same raw stream later closes its full turn,
    command inventory, evidence reads, and final result.  Terminal errors,
    unknown messages, alternate endpoints, and open event objects still fail.
    """

    if set(event) != {"type", "message"} or event.get("type") != "error":
        raise ShardReplayError(
            f"raw transport event {index} is not a closed event object"
        )
    message = event.get("message")
    if not isinstance(message, str) or len(message) > 512:
        raise ShardReplayError(
            f"raw transport event {index} has an invalid message"
        )
    match = _TRANSPORT_RETRY_RE.fullmatch(message)
    if match is None:
        raise ShardReplayError(
            f"raw transport event {index} is not a recognized recoverable retry"
        )
    reason_code = _TRANSPORT_RETRY_REASONS.get(match.group("reason"))
    if reason_code is None:
        raise ShardReplayError(
            f"raw transport event {index} has an unapproved retry reason"
        )
    return {
        "kind": "transport_retry",
        "raw_event_index": index,
        "attempt": int(match.group("attempt")),
        "retry_limit": 5,
        "reason_code": reason_code,
    }


def _validate_turn_usage(value: Any) -> dict[str, int]:
    required = {
        "input_tokens", "cached_input_tokens", "output_tokens",
        "reasoning_output_tokens",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ShardReplayError("turn.completed usage is not the pinned closed schema")
    usage: dict[str, int] = {}
    for field in sorted(required):
        item = value.get(field)
        if type(item) is not int or item < 0:
            raise ShardReplayError(f"turn.completed usage has invalid {field}")
        usage[field] = item
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise ShardReplayError("cached input tokens exceed reported input tokens")
    if usage["reasoning_output_tokens"] > usage["output_tokens"]:
        raise ShardReplayError(
            "reasoning output tokens exceed reported output tokens"
        )
    return usage


def _fixed_shard(shard_id: str) -> shard_plan.ShardDefinition:
    matches = [
        item for item in shard_plan.SHARD_DEFINITIONS
        if item.shard_id == shard_id
    ]
    if len(matches) != 1:
        raise ShardReplayError(f"unknown or duplicate shard ID: {shard_id!r}")
    return matches[0]


def _expected_assertions(protocol_id: str, shard_id: str) -> list[str]:
    owners = shard_plan.assertion_owners(protocol_id)
    return sorted(
        assertion_id for assertion_id, owner in owners.items()
        if owner == shard_id
    )


def _validate_rule_assignment(
    shard_manifest: Mapping[str, Any], *, shard_id: str,
) -> None:
    expected = {
        item.rule_id: item for item in shard_plan.RULE_REFERENCE_SPECS
        if item.shard_id == shard_id
    }
    rows = _require_list(
        shard_manifest.get("rule_references"), label="shard rule references",
    )
    seen_rules: set[str] = set()
    seen_references: set[str] = set()
    assigned_by_url: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "reference_id", "reference_surface_sha256", "rule_id",
            "citation_url", "section",
        }:
            raise ShardReplayError(
                f"shard rule reference {index} is not a closed object"
            )
        rule_id = row.get("rule_id")
        if not isinstance(rule_id, str) or rule_id in seen_rules:
            raise ShardReplayError("shard has a duplicate or malformed rule ID")
        spec = expected.get(rule_id)
        if spec is None:
            raise ShardReplayError(f"rule belongs to another shard: {rule_id}")
        reference_id = _require_sha256(
            row.get("reference_id"), label=f"reference ID for {rule_id}",
        )
        _require_sha256(
            row.get("reference_surface_sha256"),
            label=f"reference surface for {rule_id}",
        )
        if reference_id in seen_references:
            raise ShardReplayError("shard duplicates a reference ID")
        if (
            row.get("citation_url") != spec.citation_url
            or row.get("section") != spec.section
        ):
            raise ShardReplayError(f"rule URL or section is stale: {rule_id}")
        seen_rules.add(rule_id)
        seen_references.add(reference_id)
        assigned_by_url.setdefault(spec.citation_url, set()).add(reference_id)
    if seen_rules != set(expected):
        missing = sorted(set(expected) - seen_rules)
        raise ShardReplayError(
            "shard rule inventory is incomplete: " + ", ".join(missing)
        )

    citations = _require_list(
        shard_manifest.get("citations"), label="shard citations",
    )
    seen_urls: set[str] = set()
    for index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            raise ShardReplayError(f"shard citation {index} is not an object")
        url = citation.get("requested_url")
        if not isinstance(url, str) or url in seen_urls:
            raise ShardReplayError("shard has a duplicate or malformed citation URL")
        expected_ids = assigned_by_url.get(url)
        ids = citation.get("reference_ids")
        if (
            expected_ids is None
            or not isinstance(ids, list)
            or set(ids) != expected_ids
            or len(ids) != len(expected_ids)
        ):
            raise ShardReplayError(
                f"citation reference assignment is stale or cross-shard: {url}"
            )
        if citation.get("inspection_required") is not True:
            raise ShardReplayError(f"assigned rule citation is not inspectable: {url}")
        seen_urls.add(url)
    if seen_urls != set(assigned_by_url):
        missing = sorted(set(assigned_by_url) - seen_urls)
        raise ShardReplayError(
            "shard citation inventory is incomplete: " + ", ".join(missing)
        )


def _chunk_inventory(
    shard_manifest: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    source_chunks: set[str] = set()
    citation_chunks: set[str] = set()
    for file_row in _require_list(
        shard_manifest.get("files"), label="shard files",
    ):
        if not isinstance(file_row, dict):
            raise ShardReplayError("shard file row is not an object")
        chunks = _require_list(file_row.get("chunks"), label="source chunks")
        if not chunks:
            raise ShardReplayError("assigned source has no chunks")
        for chunk in chunks:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("path"), str):
                raise ShardReplayError("source chunk is malformed")
            path = str(chunk["path"])
            if path in source_chunks or path in citation_chunks:
                raise ShardReplayError("shard manifest duplicates a chunk path")
            source_chunks.add(path)
    for citation in _require_list(
        shard_manifest.get("citations"), label="shard citations",
    ):
        if not isinstance(citation, dict):
            raise ShardReplayError("shard citation row is not an object")
        for chunk in _require_list(
            citation.get("inspection_chunks"), label="citation chunks",
        ):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("path"), str):
                raise ShardReplayError("citation chunk is malformed")
            path = str(chunk["path"])
            if path in source_chunks or path in citation_chunks:
                raise ShardReplayError("shard manifest duplicates a chunk path")
            citation_chunks.add(path)
    return source_chunks, citation_chunks


def _validate_projected_cache(
    cache: review.ReviewCache, shard_manifest: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    if not isinstance(cache, review.ReviewCache):
        raise ShardReplayError("cache must be a projected ReviewCache")
    if not isinstance(cache.manifest, dict):
        raise ShardReplayError("projected cache manifest must be an object")
    parsed = review._strict_json_bytes(
        cache.manifest_bytes, label="projected cache manifest",
    )
    if parsed != cache.manifest:
        raise ShardReplayError("projected cache bytes and object disagree")

    if shard_manifest.get("schema_version") != suite.SHARD_MANIFEST_SCHEMA_VERSION:
        raise ShardReplayError("shard manifest has the wrong schema version")
    protocol_id = shard_manifest.get("protocol_id")
    if protocol_id not in {
        shard_plan.SEMANTIC_PROTOCOL_ID,
        shard_plan.ADVERSARIAL_PROTOCOL_ID,
    }:
        raise ShardReplayError("shard manifest has an unknown protocol")
    shard_id = shard_manifest.get("shard_id")
    if not isinstance(shard_id, str):
        raise ShardReplayError("shard manifest has no shard ID")
    fixed = _fixed_shard(shard_id)
    _require_sha256(
        shard_manifest.get("review_snapshot_sha256"),
        label="review snapshot hash",
    )
    _require_sha256(
        shard_manifest.get("shard_plan_sha256"), label="shard plan hash",
    )
    evidence_hash = _require_sha256(
        shard_manifest.get("citation_evidence_surface_sha256"),
        label="citation evidence surface hash",
    )
    _require_sha256(
        shard_manifest.get("full_citation_evidence_surface_sha256"),
        label="full citation evidence surface hash",
    )
    if shard_manifest.get("source_paths") != sorted(fixed.source_paths):
        raise ShardReplayError("shard source path inventory is not the fixed plan")
    if shard_manifest.get("assertion_ids") != _expected_assertions(
        str(protocol_id), shard_id,
    ):
        raise ShardReplayError("shard assertion inventory is not the fixed plan")
    if shard_manifest.get("token_budget_contract") != (
        shard_plan.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict()
    ):
        raise ShardReplayError("shard uses a stale token-budget contract")
    _validate_rule_assignment(shard_manifest, shard_id=shard_id)

    files = _require_list(shard_manifest.get("files"), label="shard files")
    file_paths = [
        row.get("path") if isinstance(row, dict) else None for row in files
    ]
    if (
        any(not isinstance(path, str) for path in file_paths)
        or len(file_paths) != len(set(file_paths))
        or set(file_paths) != set(fixed.source_paths)
    ):
        raise ShardReplayError("shard files differ from the fixed source paths")
    if cache.manifest.get("files") != files:
        raise ShardReplayError("projected cache contains different source files")
    if cache.manifest.get("citations") != shard_manifest.get("citations"):
        raise ShardReplayError("projected cache contains different citations")
    if cache.manifest.get("chunk_contract") != shard_manifest.get("chunk_contract"):
        raise ShardReplayError("projected cache uses another chunk contract")
    for key in (
        "protocol_id", "review_snapshot_sha256", "shard_plan_sha256",
        "shard_id", "citation_evidence_surface_sha256",
        "full_citation_evidence_surface_sha256",
    ):
        if cache.manifest.get(key) != shard_manifest.get(key):
            raise ShardReplayError(f"projected cache has a stale {key}")
    citations = _require_list(
        shard_manifest.get("citations"), label="shard citations",
    )
    if suite.citation_evidence_surface_sha256(citations) != evidence_hash:
        raise ShardReplayError("shard citation evidence surface is stale")

    source_chunks, citation_chunks = _chunk_inventory(shard_manifest)
    targets = review._cache_targets(cache)
    if set(targets) != source_chunks | citation_chunks:
        raise ShardReplayError("projected cache exposes an unassigned chunk")
    for path, target in targets.items():
        chunk = target.get("chunk")
        if not isinstance(chunk, dict):
            raise ShardReplayError(f"projected cache chunk is malformed: {path}")
        payload = review._read_private_cache_file(cache.root, path)
        if (
            hashlib.sha256(payload).hexdigest() != chunk.get("sha256")
            or len(payload) != chunk.get("size")
        ):
            raise ShardReplayError(f"projected cache chunk is stale: {path}")
    return source_chunks, citation_chunks


def _command_operations(
    cache: review.ReviewCache, command: str, output: str,
) -> list[dict[str, Any]]:
    """Accept exact raw output or the existing citation-only redaction marker."""

    inner_command = review._unwrap_codex_shell_event_command(command)
    expected, _rows, citation_content = review._expected_command(
        cache, inner_command,
    )
    normalized = output
    if citation_content and output == expected:
        normalized = review._citation_marker(expected)
    return review._command_operations(cache, inner_command, normalized)


def _replay(
    raw_bytes: bytes, *, cache: review.ReviewCache,
    shard_manifest: Mapping[str, Any],
) -> ShardReviewReplay:
    if not isinstance(raw_bytes, bytes):
        raise ShardReplayError("raw Codex stream must be bytes")
    if len(raw_bytes) > review.MAX_RAW_REVIEW_BYTES:
        raise ShardReplayError("raw Codex stream exceeds the fixed byte limit")
    source_chunks, citation_chunks = _validate_projected_cache(
        cache, shard_manifest,
    )
    protocol_id = str(shard_manifest["protocol_id"])
    shard_id = str(shard_manifest["shard_id"])

    events = review._strict_jsonl(
        raw_bytes, label="raw sharded Codex review stream",
    )
    event_types = [event.get("type") for event in events]
    if (
        len(events) < 4
        or event_types[0] != "thread.started"
        or event_types[1] != "turn.started"
        or event_types[-1] != "turn.completed"
    ):
        raise ShardReplayError(
            "raw shard stream must be one ordered thread/turn lifecycle"
        )

    thread_ids: list[str] = []
    referenced_thread_ids: set[str] = set()
    supplied_invocation_ids: set[str] = set()
    started_commands: dict[str, str] = {}
    completed_commands: set[str] = set()
    operations: list[dict[str, Any]] = []
    messages: list[str] = []
    turn_started = 0
    turn_completed = 0
    turn_usage: dict[str, int] | None = None
    final_message_seen = False
    transport_retries: list[dict[str, Any]] = []

    for index, event in enumerate(events, 1):
        event_type = event.get("type")
        if event_type == "error":
            if final_message_seen:
                raise ShardReplayError(
                    "raw shard stream has a transport retry after final JSON"
                )
            transport_retries.append(
                _recoverable_transport_retry(event, index=index)
            )
            if (
                len(transport_retries)
                > MAX_RECOVERABLE_TRANSPORT_RETRY_EVENTS
            ):
                raise ShardReplayError(
                    "raw shard stream exceeds the recoverable retry limit"
                )
            continue
        if event_type not in review._ALLOWED_EVENT_TYPES:
            raise ShardReplayError(
                f"raw event {index} has forbidden or unknown type {event_type!r}"
            )
        _validate_closed_event(event, index=index)
        if isinstance(event.get("thread_id"), str) and event["thread_id"]:
            referenced_thread_ids.add(str(event["thread_id"]))
        if "invocation_id" in event:
            invocation = event.get("invocation_id")
            if (
                not isinstance(invocation, str)
                or review._TOKEN_RE.fullmatch(invocation) is None
            ):
                raise ShardReplayError("raw shard stream has an invalid invocation ID")
            supplied_invocation_ids.add(invocation)
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if (
                not isinstance(thread_id, str)
                or review._TOKEN_RE.fullmatch(thread_id) is None
            ):
                raise ShardReplayError("raw shard stream has an invalid thread ID")
            thread_ids.append(thread_id)
            continue
        if event_type == "turn.started":
            turn_started += 1
            continue
        if event_type == "turn.completed":
            turn_completed += 1
            if turn_usage is not None:
                raise ShardReplayError("raw shard stream repeats turn usage")
            turn_usage = _validate_turn_usage(event.get("usage"))
            continue

        item = event.get("item")
        if not isinstance(item, dict):
            raise ShardReplayError(f"raw item event {index} has no object item")
        _validate_closed_item(item, index=index)
        item_type = item.get("type")
        if item_type in review._ALLOWED_NONCOMMAND_ITEMS:
            if event_type == "item.completed" and item_type == "agent_message":
                if final_message_seen:
                    raise ShardReplayError("raw shard stream has multiple final messages")
                text = review._content_text(item)
                if not isinstance(text, str) or not text.strip():
                    raise ShardReplayError("completed agent message has no text")
                messages.append(text.strip())
                final_message_seen = True
            elif item_type == "agent_message":
                raise ShardReplayError("agent message must be one completed final item")
            elif final_message_seen:
                raise ShardReplayError("raw shard stream has an item after final JSON")
            continue
        if item_type != "command_execution":
            raise ShardReplayError(
                f"raw item event {index} uses forbidden or unknown tool {item_type!r}"
            )
        if final_message_seen:
            raise ShardReplayError("raw shard stream executes after final JSON")
        call_id = item.get("id") or item.get("call_id")
        if (
            not isinstance(call_id, str)
            or review._CALL_ID_RE.fullmatch(call_id) is None
        ):
            raise ShardReplayError("command event has an invalid call ID")
        command = item.get("command")
        if not isinstance(command, str):
            raise ShardReplayError("command event has no literal command")
        if event_type == "item.started":
            if call_id in started_commands or call_id in completed_commands:
                raise ShardReplayError("command call ID is reused")
            review._split_command(command)
            started_commands[call_id] = command
            continue
        if call_id in completed_commands:
            raise ShardReplayError("completed command call ID is reused")
        if call_id not in started_commands:
            raise ShardReplayError("completed command has no matching start")
        if started_commands[call_id] != command:
            raise ShardReplayError("completed command differs from its start")
        if (
            item.get("status") != "completed"
            or type(item.get("exit_code")) is not int
            or item["exit_code"] != 0
        ):
            raise ShardReplayError("review command did not complete successfully")
        _output_key, output = review._command_output(item)
        operations.extend(_command_operations(cache, command, output))
        completed_commands.add(call_id)

    if len(thread_ids) != 1 or len(set(thread_ids)) != 1:
        raise ShardReplayError("raw shard stream must contain one unique thread")
    if referenced_thread_ids and referenced_thread_ids != {thread_ids[0]}:
        raise ShardReplayError("raw shard stream mixes thread identities")
    if len(supplied_invocation_ids) > 1:
        raise ShardReplayError("raw shard stream mixes invocation identities")
    if turn_started != 1 or turn_completed != 1 or turn_usage is None:
        raise ShardReplayError("raw shard stream must contain one completed turn")
    if set(started_commands) != completed_commands:
        raise ShardReplayError("raw shard stream has an incomplete command event")
    if len(messages) != 1:
        raise ShardReplayError("raw shard stream must contain one final agent JSON")

    source_reads = Counter(
        str(row["chunk_path"]) for row in operations
        if row.get("kind") == "file_chunk_read"
    )
    citation_reads = Counter(
        str(row["chunk_path"]) for row in operations
        if row.get("kind") == "citation_chunk_read"
    )
    if source_reads != Counter({path: 1 for path in source_chunks}):
        raise ShardReplayError(
            "assigned source chunks were not each read exactly once"
        )
    if citation_reads != Counter({path: 1 for path in citation_chunks}):
        raise ShardReplayError(
            "assigned citation chunks were not each read exactly once"
        )

    result = review._strict_json_bytes(
        messages[0].encode("utf-8"), label="final shard review result",
    )
    if not isinstance(result, dict):
        raise ShardReplayError("final shard review result is not an object")
    normalized_result = suite.validate_shard_result(result, shard_manifest)
    result_bytes = review._canonical_json(normalized_result)
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    protocol_label = (
        "semantic"
        if protocol_id == shard_plan.SEMANTIC_PROTOCOL_ID
        else "adversarial"
    )
    invocation_id = (
        f"review-{protocol_label}-{shard_id.replace('_', '-')}-"
        f"{raw_sha256[:32]}"
    )
    trace_rows: list[dict[str, Any]] = []
    for retry in transport_retries:
        trace_rows.append({
            "schema_version": SHARD_TRACE_SCHEMA_VERSION,
            "shard_id": shard_id,
            "shard_sequence": len(trace_rows) + 1,
            "invocation_id": invocation_id,
            **retry,
        })
    for operation in operations:
        trace_rows.append({
            "schema_version": SHARD_TRACE_SCHEMA_VERSION,
            "shard_id": shard_id,
            "shard_sequence": len(trace_rows) + 1,
            "invocation_id": invocation_id,
            **operation,
        })
    trace_rows.append({
        "schema_version": SHARD_TRACE_SCHEMA_VERSION,
        "shard_id": shard_id,
        "shard_sequence": len(trace_rows) + 1,
        "invocation_id": invocation_id,
        "kind": "shard_result_emit",
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
    })
    return ShardReviewReplay(
        protocol_id=protocol_id,
        shard_id=shard_id,
        invocation_id=invocation_id,
        thread_id=thread_ids[0],
        raw_sha256=raw_sha256,
        reported_input_tokens=turn_usage["input_tokens"],
        reported_cached_input_tokens=turn_usage["cached_input_tokens"],
        reported_output_tokens=turn_usage["output_tokens"],
        reported_reasoning_output_tokens=turn_usage["reasoning_output_tokens"],
        derived_input_plus_output_tokens=(
            turn_usage["input_tokens"] + turn_usage["output_tokens"]
        ),
        transport_retry_event_count=len(transport_retries),
        result=normalized_result,
        result_bytes=result_bytes,
        trace_bytes=review._canonical_jsonl(trace_rows),
    )


def replay_shard_raw_review(
    raw_bytes: bytes, *, cache: review.ReviewCache,
    shard_manifest: Mapping[str, Any],
) -> ShardReviewReplay:
    """Replay one projected shard without executing any external process."""

    try:
        return _replay(
            raw_bytes, cache=cache, shard_manifest=shard_manifest,
        )
    except ShardReplayError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ShardReplayError(str(exc)) from exc


__all__ = [
    "SHARD_TRACE_SCHEMA_VERSION",
    "MAX_RECOVERABLE_TRANSPORT_RETRY_EVENTS",
    "ShardReplayError",
    "ShardReviewReplay",
    "replay_shard_raw_review",
]
