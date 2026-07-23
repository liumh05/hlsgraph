#!/usr/bin/env python3
"""Build and verify the closed attestations for the three-shard review suite.

This module is deliberately execution-agnostic.  The restricted executor and
raw-stream replayer remain responsible for producing each ``invocation``
input.  This module accepts only those replay products, verifies their exact
shard/result/budget inventories, creates the public protocol trace and v6
receipt, and finally verifies the semantic/adversarial pair.

The old monolithic v4 receipt and v3 trace remain valid compatibility
contracts.  Nothing in this module weakens or silently upgrades them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from tools import knowledge_review_shards as shard_plan
from tools import run_knowledge_review_suite as suite


SUITE_TRACE_SCHEMA_VERSION = "hlsgraph.knowledge-review.suite-tool-trace.v2"
SUITE_RECEIPT_SCHEMA_VERSION = "hlsgraph.knowledge-review.cli-receipt.v6"
SUITE_PAIR_SEAL_SCHEMA_VERSION = "hlsgraph.knowledge-review.suite-pair-seal.v2"

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
CODEX_CLI_VERSION = "codex-cli 0.144.0"
OFFICIAL_CODEX_ELF_SHA256 = (
    "901923c1808a151f6926d41d703c17ad48815662cefb1c8d832a052c44271429"
)

PROTOCOL_TRACE_PATHS = {
    shard_plan.SEMANTIC_PROTOCOL_ID:
        "docs/knowledge-review-v0.3.semantic.trace.jsonl",
    shard_plan.ADVERSARIAL_PROTOCOL_ID:
        "docs/knowledge-review-v0.3.adversarial.trace.jsonl",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")

_INVOCATION_INPUT_FIELDS = frozenset({
    "shard_manifest", "shard_result", "invocation_id", "thread_id",
    "raw_output_sha256", "sanitized_output_sha256",
    "reported_input_tokens", "reported_cached_input_tokens",
    "reported_output_tokens",
    "reported_reasoning_output_tokens", "derived_input_plus_output_tokens",
    "normalized_shard_trace_sha256",
    "cache_manifest_sha256", "prompt_sha256", "command_sha256",
    "boundary_contract_sha256", "runtime_manifest_sha256",
    "replay_contract_sha256", "assigned_chunk_inventory_sha256",
    "token_budget", "completed", "exit_code", "compaction_event_count",
    "unknown_event_count", "assigned_chunks_complete",
})

_INVOCATION_PUBLIC_FIELDS = (
    "shard_id", "invocation_id", "thread_id", "raw_output_sha256",
    "sanitized_output_sha256",
    "reported_input_tokens", "reported_cached_input_tokens",
    "reported_output_tokens",
    "reported_reasoning_output_tokens", "derived_input_plus_output_tokens",
    "normalized_shard_trace_sha256", "shard_manifest_sha256",
    "shard_result_sha256", "cache_manifest_sha256", "prompt_sha256",
    "command_sha256", "boundary_contract_sha256",
    "runtime_manifest_sha256", "replay_contract_sha256",
    "assigned_chunk_inventory_sha256", "shard_evidence_surface_sha256",
    "assertion_ids", "reference_ids", "token_budget", "completed",
    "exit_code", "compaction_event_count", "unknown_event_count",
    "assigned_chunks_complete",
)

_TOKEN_BUDGET_FIELDS = frozenset({
    "contract", "prompt_tokens", "chunk_tokens", "command_tokens",
    "tool_event_count", "tool_event_overhead_tokens",
    "runtime_envelope_allowance_tokens",
    "visible_input_tokens", "context_reserve_tokens", "within_budget",
})

_RECEIPT_FIELDS = frozenset({
    "schema_version", "protocol_id", "suite_id", "model",
    "reasoning_effort", "codex_cli_version", "official_codex_elf_sha256",
    "review_snapshot_sha256", "citation_evidence_sha256",
    "full_evidence_surface_sha256", "shard_plan_sha256",
    "output_schema_sha256", "shard_output_schema_sha256",
    "suite_receipt_schema_sha256", "suite_trace_schema_sha256",
    "runtime_manifest_sha256", "result_sha256", "event_stream_path",
    "event_stream_sha256", "assertion_ids", "assertion_union_sha256",
    "rule_reference_ids", "rule_reference_union_sha256",
    "result_reference_ids", "result_reference_union_sha256",
    "shard_invocations", "invocation_count", "no_compaction",
    "no_unknown_events", "within_budget", "approved", "completed",
    "exit_code",
})


class SuiteSealError(ValueError):
    """The suite evidence cannot be sealed without weakening the contract."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(dict(row)) for row in rows)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_object(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def artifact_json_bytes(value: Any) -> bytes:
    """Return the exact canonical bytes used for public JSON artifacts.

    The existing review runner publishes indented, sorted JSON.  Artifact
    digests must hash those bytes—not a second compact object encoding—so a
    receipt can be checked directly against the file it names.
    """

    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")


def artifact_sha256(value: Any) -> str:
    return _sha256_bytes(artifact_json_bytes(value))


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SuiteSealError(f"{label} must be a lowercase SHA-256")
    return value


def _require_token(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise SuiteSealError(f"{label} has an invalid identifier")
    return value


def _require_protocol(value: Any) -> str:
    if value not in PROTOCOL_TRACE_PATHS:
        raise SuiteSealError("suite uses an unknown protocol")
    return str(value)


def _require_string_array(value: Any, *, label: str) -> list[str]:
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)):
        raise SuiteSealError(f"{label} must be an array of non-empty strings")
    if value != sorted(set(value)):
        raise SuiteSealError(f"{label} must be uniquely sorted")
    return list(value)


def _plan_rows(plan: Mapping[str, Any]) -> tuple[str, dict[str, Mapping[str, Any]]]:
    try:
        plan_hash = shard_plan.shard_plan_sha256(plan)
    except (TypeError, ValueError, shard_plan.ShardPlanError) as exc:
        raise SuiteSealError(f"invalid shard plan: {exc}") from exc
    rows = plan.get("shards")
    if not isinstance(rows, list):
        raise SuiteSealError("shard plan has no shard rows")
    by_id = {
        str(row.get("shard_id")): row for row in rows
        if isinstance(row, Mapping)
    }
    if tuple(by_id) != shard_plan.SHARD_ORDER or len(by_id) != 3:
        raise SuiteSealError("suite requires the fixed three-shard order")
    return plan_hash, by_id


def _expected_assertions(protocol_id: str, row: Mapping[str, Any]) -> list[str]:
    key = (
        "semantic_assertion_ids"
        if protocol_id == shard_plan.SEMANTIC_PROTOCOL_ID
        else "adversarial_assertion_ids"
    )
    values = row.get(key)
    if not isinstance(values, list):
        raise SuiteSealError(f"shard plan omits {key}")
    return sorted(str(value) for value in values)


def _expected_rule_references(row: Mapping[str, Any]) -> list[str]:
    values = row.get("rule_references")
    if not isinstance(values, list):
        raise SuiteSealError("shard plan omits rule references")
    result = sorted(
        str(value.get("reference_id")) for value in values
        if isinstance(value, Mapping)
    )
    if len(result) != len(values):
        raise SuiteSealError("shard plan has a malformed rule reference")
    for reference_id in result:
        _require_sha256(reference_id, label="rule reference ID")
    if len(result) != len(set(result)):
        raise SuiteSealError("shard plan duplicates a rule reference")
    return result


def _expected_result_references(citation_audit: Mapping[str, Any]) -> list[str]:
    rows = citation_audit.get("references")
    if not isinstance(rows, list):
        raise SuiteSealError("citation audit has no reference inventory")
    result: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SuiteSealError("citation audit has a malformed reference")
        reference_id = _require_sha256(
            row.get("reference_id"), label="citation-audit reference ID",
        )
        result.append(reference_id)
    if len(result) != 53 or len(result) != len(set(result)):
        raise SuiteSealError("suite requires exactly 53 unique result references")
    return sorted(result)


def _validate_token_budget(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TOKEN_BUDGET_FIELDS:
        raise SuiteSealError("shard invocation has a non-closed token budget")
    budget = copy.deepcopy(dict(value))
    contract = budget.get("contract")
    expected_contract = shard_plan.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict()
    if contract != expected_contract:
        raise SuiteSealError("shard invocation uses a stale token contract")
    integer_fields = (
        "prompt_tokens", "chunk_tokens", "command_tokens",
        "tool_event_count", "tool_event_overhead_tokens",
        "runtime_envelope_allowance_tokens",
        "visible_input_tokens", "context_reserve_tokens",
    )
    for field in integer_fields:
        item = budget.get(field)
        if type(item) is not int or item < 0:
            raise SuiteSealError(f"token budget has an invalid {field}")
    expected_overhead = (
        budget["tool_event_count"] * contract["tool_event_overhead_tokens"]
    )
    expected_visible = (
        budget["prompt_tokens"] + budget["chunk_tokens"]
        + budget["command_tokens"] + expected_overhead
        + contract["runtime_envelope_allowance_tokens"]
    )
    expected_reserve = contract["context_window_tokens"] - expected_visible
    if budget["tool_event_overhead_tokens"] != expected_overhead:
        raise SuiteSealError("token budget event overhead is inconsistent")
    if (budget["runtime_envelope_allowance_tokens"]
            != contract["runtime_envelope_allowance_tokens"]):
        raise SuiteSealError("token budget runtime allowance is inconsistent")
    if budget["visible_input_tokens"] != expected_visible:
        raise SuiteSealError("token budget visible-input total is inconsistent")
    if budget["context_reserve_tokens"] != expected_reserve:
        raise SuiteSealError("token budget context reserve is inconsistent")
    if (budget.get("within_budget") is not True
            or expected_visible > contract["max_visible_input_tokens"]
            or expected_reserve < contract["min_context_reserve_tokens"]):
        raise SuiteSealError("shard invocation breaches the fixed token budget")
    return budget


def _validate_invocation(
    value: Any, *, protocol_id: str, expected_shard: Mapping[str, Any],
    expected_plan_sha256: str, expected_snapshot_sha256: str,
    expected_full_evidence_surface_sha256: str,
    expected_runtime_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != _INVOCATION_INPUT_FIELDS:
        raise SuiteSealError("shard invocation does not match the closed input contract")
    source = dict(value)
    manifest = source.get("shard_manifest")
    result = source.get("shard_result")
    if not isinstance(manifest, Mapping) or not isinstance(result, Mapping):
        raise SuiteSealError("shard invocation lacks its manifest or result")
    shard_id = str(expected_shard.get("shard_id"))
    if (manifest.get("protocol_id") != protocol_id
            or manifest.get("shard_id") != shard_id
            or manifest.get("shard_plan_sha256") != expected_plan_sha256
            or manifest.get("review_snapshot_sha256")
            != expected_snapshot_sha256):
        raise SuiteSealError(f"{shard_id} manifest is bound to another review")
    try:
        validated_result = suite.validate_shard_result(result, manifest)
    except (TypeError, ValueError, suite.SuiteContractError) as exc:
        raise SuiteSealError(f"{shard_id} result is invalid: {exc}") from exc
    if (validated_result.get("approved") is not True
            or validated_result.get("issues") != []
            or validated_result.get("summary") != "approved_no_issues"):
        raise SuiteSealError(f"{shard_id} is not an approved shard result")

    assertion_ids = _expected_assertions(protocol_id, expected_shard)
    reference_ids = _expected_rule_references(expected_shard)
    if sorted(str(item) for item in manifest.get("assertion_ids", [])) != assertion_ids:
        raise SuiteSealError(f"{shard_id} manifest assertion inventory differs")
    manifest_references = manifest.get("rule_references")
    if not isinstance(manifest_references, list) or sorted(
        str(item.get("reference_id")) for item in manifest_references
        if isinstance(item, Mapping)
    ) != reference_ids:
        raise SuiteSealError(f"{shard_id} manifest reference inventory differs")

    token_budget = _validate_token_budget(source.get("token_budget"))
    reported_input = source.get("reported_input_tokens")
    reported_cached = source.get("reported_cached_input_tokens")
    reported_output = source.get("reported_output_tokens")
    reported_reasoning = source.get("reported_reasoning_output_tokens")
    derived_total = source.get("derived_input_plus_output_tokens")
    if (any(type(item) is not int or item < 0 for item in (
            reported_input, reported_cached, reported_output,
            reported_reasoning, derived_total,
        ))
            or reported_cached > reported_input
            or reported_reasoning > reported_output
            or derived_total != reported_input + reported_output):
        raise SuiteSealError(
            f"{shard_id} has inconsistent reported or derived token usage"
        )
    for field in (
        "raw_output_sha256", "sanitized_output_sha256",
        "normalized_shard_trace_sha256",
        "cache_manifest_sha256", "prompt_sha256", "command_sha256",
        "boundary_contract_sha256", "runtime_manifest_sha256",
        "replay_contract_sha256", "assigned_chunk_inventory_sha256",
    ):
        _require_sha256(source.get(field), label=f"{shard_id} {field}")
    _require_token(source.get("invocation_id"), label=f"{shard_id} invocation_id")
    _require_token(source.get("thread_id"), label=f"{shard_id} thread_id")
    if source.get("runtime_manifest_sha256") != expected_runtime_manifest_sha256:
        raise SuiteSealError(f"{shard_id} used another Codex runtime")
    if (source.get("completed") is not True or source.get("exit_code") != 0):
        raise SuiteSealError(f"{shard_id} invocation did not complete successfully")
    if type(source.get("compaction_event_count")) is not int or source.get(
        "compaction_event_count"
    ) != 0:
        raise SuiteSealError(f"{shard_id} contains a compaction event")
    if type(source.get("unknown_event_count")) is not int or source.get(
        "unknown_event_count"
    ) != 0:
        raise SuiteSealError(f"{shard_id} contains an unknown event")
    if source.get("assigned_chunks_complete") is not True:
        raise SuiteSealError(f"{shard_id} did not read its complete assigned inventory")

    shard_evidence = _require_sha256(
        manifest.get("citation_evidence_surface_sha256"),
        label=f"{shard_id} evidence surface",
    )
    _require_sha256(
        expected_full_evidence_surface_sha256,
        label="full evidence surface",
    )
    if manifest.get(
        "full_citation_evidence_surface_sha256"
    ) != expected_full_evidence_surface_sha256:
        raise SuiteSealError(f"{shard_id} does not bind the full evidence surface")
    public = {
        "shard_id": shard_id,
        "invocation_id": source["invocation_id"],
        "thread_id": source["thread_id"],
        "raw_output_sha256": source["raw_output_sha256"],
        "sanitized_output_sha256": source["sanitized_output_sha256"],
        "reported_input_tokens": reported_input,
        "reported_cached_input_tokens": reported_cached,
        "reported_output_tokens": reported_output,
        "reported_reasoning_output_tokens": reported_reasoning,
        "derived_input_plus_output_tokens": derived_total,
        "normalized_shard_trace_sha256": source[
            "normalized_shard_trace_sha256"
        ],
        "shard_manifest_sha256": artifact_sha256(dict(manifest)),
        "shard_result_sha256": artifact_sha256(validated_result),
        "cache_manifest_sha256": source["cache_manifest_sha256"],
        "prompt_sha256": source["prompt_sha256"],
        "command_sha256": source["command_sha256"],
        "boundary_contract_sha256": source["boundary_contract_sha256"],
        "runtime_manifest_sha256": source["runtime_manifest_sha256"],
        "replay_contract_sha256": source["replay_contract_sha256"],
        "assigned_chunk_inventory_sha256": source[
            "assigned_chunk_inventory_sha256"
        ],
        "shard_evidence_surface_sha256": shard_evidence,
        "assertion_ids": assertion_ids,
        "reference_ids": reference_ids,
        "token_budget": token_budget,
        "completed": True,
        "exit_code": 0,
        "compaction_event_count": 0,
        "unknown_event_count": 0,
        "assigned_chunks_complete": True,
    }
    if set(public) != set(_INVOCATION_PUBLIC_FIELDS):  # pragma: no cover
        raise AssertionError("internal invocation projection is not closed")
    return public, validated_result


def _result_reference_ids(
    result: Mapping[str, Any], expected_reference_ids: Sequence[str],
    *, protocol_id: str,
) -> list[str]:
    if (result.get("protocol_id") != protocol_id
            or result.get("approved") is not True
            or result.get("issues") != []
            or result.get("summary") != "approved_no_issues"):
        raise SuiteSealError("aggregate result is not an approved protocol result")
    citations = result.get("citation_results")
    if not isinstance(citations, list):
        raise SuiteSealError("aggregate result has no citation inventory")
    reference_ids: list[str] = []
    for row in citations:
        if not isinstance(row, Mapping):
            raise SuiteSealError("aggregate result has a malformed citation row")
        reference_id = _require_sha256(
            row.get("reference_id"), label="aggregate reference ID",
        )
        if (row.get("verdict") != "verified" or row.get("issues") != []
                or row.get("declared_version_matched") is not True):
            raise SuiteSealError("aggregate result contains an unverified citation")
        reference_ids.append(reference_id)
    if sorted(reference_ids) != list(expected_reference_ids):
        raise SuiteSealError("aggregate result reference inventory differs")
    if len(reference_ids) != len(set(reference_ids)):
        raise SuiteSealError("aggregate result duplicates a reference")
    return sorted(reference_ids)


def build_protocol_trace(
    *, protocol_id: str, plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any], review_snapshot_sha256: str,
    citation_evidence_sha256: str, full_evidence_surface_sha256: str,
    runtime_manifest_sha256: str, invocations: Sequence[Mapping[str, Any]],
    aggregate_result: Mapping[str, Any],
) -> bytes:
    """Create one canonical public JSONL trace from three replay products."""

    protocol = _require_protocol(protocol_id)
    plan_hash, plan_by_id = _plan_rows(plan)
    snapshot_hash = _require_sha256(
        review_snapshot_sha256, label="review snapshot",
    )
    evidence_hash = _require_sha256(
        citation_evidence_sha256, label="citation evidence mapping",
    )
    full_evidence_hash = _require_sha256(
        full_evidence_surface_sha256, label="full evidence surface",
    )
    runtime_hash = _require_sha256(
        runtime_manifest_sha256, label="runtime manifest",
    )
    if not isinstance(invocations, Sequence) or len(invocations) != 3:
        raise SuiteSealError("protocol trace requires exactly three invocations")
    by_shard: dict[str, Mapping[str, Any]] = {}
    for value in invocations:
        if not isinstance(value, Mapping):
            raise SuiteSealError("protocol trace contains a malformed invocation")
        manifest = value.get("shard_manifest")
        shard_id = manifest.get("shard_id") if isinstance(manifest, Mapping) else None
        if not isinstance(shard_id, str) or shard_id in by_shard:
            raise SuiteSealError("protocol trace duplicates or omits a shard")
        by_shard[shard_id] = value
    if set(by_shard) != set(shard_plan.SHARD_ORDER):
        raise SuiteSealError("protocol trace does not cover the fixed three shards")

    public_invocations: list[dict[str, Any]] = []
    for shard_id in shard_plan.SHARD_ORDER:
        public, _result = _validate_invocation(
            by_shard[shard_id], protocol_id=protocol,
            expected_shard=plan_by_id[shard_id],
            expected_plan_sha256=plan_hash,
            expected_snapshot_sha256=snapshot_hash,
            expected_full_evidence_surface_sha256=full_evidence_hash,
            expected_runtime_manifest_sha256=runtime_hash,
        )
        public_invocations.append(public)
    for field in (
        "invocation_id", "thread_id", "raw_output_sha256",
        "sanitized_output_sha256",
    ):
        values = [str(row[field]) for row in public_invocations]
        if len(values) != len(set(values)):
            raise SuiteSealError(f"protocol reuses one {field}")

    assertion_ids = sorted({
        item for row in public_invocations for item in row["assertion_ids"]
    })
    rule_reference_ids = sorted({
        item for row in public_invocations for item in row["reference_ids"]
    })
    expected_assertions = sorted({
        item for shard_id in shard_plan.SHARD_ORDER
        for item in _expected_assertions(protocol, plan_by_id[shard_id])
    })
    expected_rule_references = sorted({
        item for shard_id in shard_plan.SHARD_ORDER
        for item in _expected_rule_references(plan_by_id[shard_id])
    })
    if assertion_ids != expected_assertions:
        raise SuiteSealError("protocol assertion union is incomplete")
    if rule_reference_ids != expected_rule_references:
        raise SuiteSealError("protocol rule-reference union is incomplete")
    result_reference_ids = _expected_result_references(citation_audit)
    _result_reference_ids(
        aggregate_result, result_reference_ids, protocol_id=protocol,
    )

    rows: list[dict[str, Any]] = [{
        "schema_version": SUITE_TRACE_SCHEMA_VERSION,
        "sequence": 1,
        "kind": "suite_start",
        "protocol_id": protocol,
        "review_snapshot_sha256": snapshot_hash,
        "citation_evidence_sha256": evidence_hash,
        "full_evidence_surface_sha256": full_evidence_hash,
        "shard_plan_sha256": plan_hash,
        "runtime_manifest_sha256": runtime_hash,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "codex_cli_version": CODEX_CLI_VERSION,
        "official_codex_elf_sha256": OFFICIAL_CODEX_ELF_SHA256,
        "shard_count": 3,
    }]
    for sequence, public in enumerate(public_invocations, 2):
        rows.append({
            "schema_version": SUITE_TRACE_SCHEMA_VERSION,
            "sequence": sequence,
            "kind": "shard_invocation",
            "protocol_id": protocol,
            **copy.deepcopy(public),
        })
    rows.append({
        "schema_version": SUITE_TRACE_SCHEMA_VERSION,
        "sequence": 5,
        "kind": "aggregate_emit",
        "protocol_id": protocol,
        "result_sha256": artifact_sha256(dict(aggregate_result)),
        "assertion_ids": assertion_ids,
        "assertion_union_sha256": _sha256_object(assertion_ids),
        "rule_reference_ids": rule_reference_ids,
        "rule_reference_union_sha256": _sha256_object(rule_reference_ids),
        "result_reference_ids": result_reference_ids,
        "result_reference_union_sha256": _sha256_object(result_reference_ids),
        "approved": True,
    })
    return _canonical_jsonl(rows)


def parse_protocol_trace(trace_bytes: bytes) -> list[dict[str, Any]]:
    """Parse only canonical, closed five-row suite traces."""

    if not isinstance(trace_bytes, bytes) or not trace_bytes:
        raise SuiteSealError("suite trace is empty")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(trace_bytes.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise SuiteSealError("suite trace must end every row with LF")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SuiteSealError(f"suite trace row {index} is not strict JSON") from exc
        if not isinstance(value, dict) or _canonical_json(value) != line:
            raise SuiteSealError(f"suite trace row {index} is not canonical")
        rows.append(value)
    if (len(rows) != 5 or [row.get("sequence") for row in rows] != list(range(1, 6))
            or [row.get("kind") for row in rows]
            != ["suite_start", "shard_invocation", "shard_invocation",
                "shard_invocation", "aggregate_emit"]
            or any(row.get("schema_version") != SUITE_TRACE_SCHEMA_VERSION
                   for row in rows)):
        raise SuiteSealError("suite trace is not the closed five-row contract")
    protocol = _require_protocol(rows[0].get("protocol_id"))
    if any(row.get("protocol_id") != protocol for row in rows):
        raise SuiteSealError("suite trace mixes protocols")
    return rows


def validate_protocol_trace(
    trace_bytes: bytes, *, protocol_id: str, plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any], review_snapshot_sha256: str,
    citation_evidence_sha256: str, full_evidence_surface_sha256: str,
    runtime_manifest_sha256: str, invocations: Sequence[Mapping[str, Any]],
    aggregate_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replay the pure trace construction and require byte equality."""

    expected = build_protocol_trace(
        protocol_id=protocol_id, plan=plan, citation_audit=citation_audit,
        review_snapshot_sha256=review_snapshot_sha256,
        citation_evidence_sha256=citation_evidence_sha256,
        full_evidence_surface_sha256=full_evidence_surface_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        invocations=invocations, aggregate_result=aggregate_result,
    )
    if trace_bytes != expected:
        raise SuiteSealError("stored suite trace differs from deterministic replay")
    return parse_protocol_trace(trace_bytes)


def build_protocol_receipt(
    *, trace_bytes: bytes, protocol_id: str, plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any], review_snapshot_sha256: str,
    citation_evidence_sha256: str, full_evidence_surface_sha256: str,
    runtime_manifest_sha256: str, invocations: Sequence[Mapping[str, Any]],
    aggregate_result: Mapping[str, Any], output_schema_sha256: str,
    shard_output_schema_sha256: str, suite_receipt_schema_sha256: str,
    suite_trace_schema_sha256: str, event_stream_path: str | None = None,
) -> dict[str, Any]:
    """Build one closed v6 receipt after deterministic trace verification."""

    protocol = _require_protocol(protocol_id)
    rows = validate_protocol_trace(
        trace_bytes, protocol_id=protocol, plan=plan,
        citation_audit=citation_audit,
        review_snapshot_sha256=review_snapshot_sha256,
        citation_evidence_sha256=citation_evidence_sha256,
        full_evidence_surface_sha256=full_evidence_surface_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        invocations=invocations, aggregate_result=aggregate_result,
    )
    plan_hash, _plan_by_id = _plan_rows(plan)
    aggregate = rows[-1]
    public_invocations = [
        {key: copy.deepcopy(row[key]) for key in _INVOCATION_PUBLIC_FIELDS}
        for row in rows[1:4]
    ]
    path = event_stream_path or PROTOCOL_TRACE_PATHS[protocol]
    if path != PROTOCOL_TRACE_PATHS[protocol]:
        raise SuiteSealError("suite trace is not the fixed public artifact path")
    for digest, label in (
        (output_schema_sha256, "aggregate output schema"),
        (shard_output_schema_sha256, "shard output schema"),
        (suite_receipt_schema_sha256, "suite receipt schema"),
        (suite_trace_schema_sha256, "suite trace schema"),
    ):
        _require_sha256(digest, label=label)
    trace_hash = _sha256_bytes(trace_bytes)
    result_hash = artifact_sha256(dict(aggregate_result))
    receipt = {
        "schema_version": SUITE_RECEIPT_SCHEMA_VERSION,
        "protocol_id": protocol,
        "suite_id": f"review-suite-{protocol.rsplit('.', 2)[-2]}-{trace_hash[:32]}",
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "codex_cli_version": CODEX_CLI_VERSION,
        "official_codex_elf_sha256": OFFICIAL_CODEX_ELF_SHA256,
        "review_snapshot_sha256": review_snapshot_sha256,
        "citation_evidence_sha256": citation_evidence_sha256,
        "full_evidence_surface_sha256": full_evidence_surface_sha256,
        "shard_plan_sha256": plan_hash,
        "output_schema_sha256": output_schema_sha256,
        "shard_output_schema_sha256": shard_output_schema_sha256,
        "suite_receipt_schema_sha256": suite_receipt_schema_sha256,
        "suite_trace_schema_sha256": suite_trace_schema_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "result_sha256": result_hash,
        "event_stream_path": path,
        "event_stream_sha256": trace_hash,
        "assertion_ids": aggregate["assertion_ids"],
        "assertion_union_sha256": aggregate["assertion_union_sha256"],
        "rule_reference_ids": aggregate["rule_reference_ids"],
        "rule_reference_union_sha256": aggregate[
            "rule_reference_union_sha256"
        ],
        "result_reference_ids": aggregate["result_reference_ids"],
        "result_reference_union_sha256": aggregate[
            "result_reference_union_sha256"
        ],
        "shard_invocations": public_invocations,
        "invocation_count": 3,
        "no_compaction": True,
        "no_unknown_events": True,
        "within_budget": True,
        "approved": True,
        "completed": True,
        "exit_code": 0,
    }
    if set(receipt) != _RECEIPT_FIELDS:  # pragma: no cover
        raise AssertionError("internal suite receipt is not closed")
    return receipt


def validate_protocol_receipt(
    receipt: Mapping[str, Any], *, trace_bytes: bytes, protocol_id: str,
    plan: Mapping[str, Any], citation_audit: Mapping[str, Any],
    review_snapshot_sha256: str, citation_evidence_sha256: str,
    full_evidence_surface_sha256: str, runtime_manifest_sha256: str,
    invocations: Sequence[Mapping[str, Any]],
    aggregate_result: Mapping[str, Any], output_schema_sha256: str,
    shard_output_schema_sha256: str, suite_receipt_schema_sha256: str,
    suite_trace_schema_sha256: str,
) -> dict[str, Any]:
    """Require byte-semantic equality with a freshly rebuilt v6 receipt."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise SuiteSealError("suite receipt does not match the closed v6 contract")
    expected = build_protocol_receipt(
        trace_bytes=trace_bytes, protocol_id=protocol_id, plan=plan,
        citation_audit=citation_audit,
        review_snapshot_sha256=review_snapshot_sha256,
        citation_evidence_sha256=citation_evidence_sha256,
        full_evidence_surface_sha256=full_evidence_surface_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        invocations=invocations, aggregate_result=aggregate_result,
        output_schema_sha256=output_schema_sha256,
        shard_output_schema_sha256=shard_output_schema_sha256,
        suite_receipt_schema_sha256=suite_receipt_schema_sha256,
        suite_trace_schema_sha256=suite_trace_schema_sha256,
        event_stream_path=PROTOCOL_TRACE_PATHS[_require_protocol(protocol_id)],
    )
    normalized = copy.deepcopy(dict(receipt))
    if normalized != expected:
        raise SuiteSealError("stored suite receipt differs from deterministic replay")
    return normalized


def validate_suite_pair(
    *, semantic_receipt: Mapping[str, Any],
    adversarial_receipt: Mapping[str, Any],
    semantic_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
    plan: Mapping[str, Any], citation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the already protocol-validated pair and enforce six-way isolation.

    Protocol receipts are normally reconstructed first with
    :func:`validate_protocol_receipt`.  This final check nevertheless repeats
    every public inventory, constant, and arithmetic invariant; a caller
    cannot weaken the pair seal merely by skipping that earlier call.
    """

    receipts = [copy.deepcopy(dict(semantic_receipt)), copy.deepcopy(dict(adversarial_receipt))]
    if any(set(receipt) != _RECEIPT_FIELDS for receipt in receipts):
        raise SuiteSealError("suite pair contains a non-v6 receipt")
    by_protocol = {receipt.get("protocol_id"): receipt for receipt in receipts}
    if set(by_protocol) != {
        shard_plan.SEMANTIC_PROTOCOL_ID, shard_plan.ADVERSARIAL_PROTOCOL_ID,
    }:
        raise SuiteSealError("suite pair requires semantic and adversarial receipts")
    semantic = by_protocol[shard_plan.SEMANTIC_PROTOCOL_ID]
    adversarial = by_protocol[shard_plan.ADVERSARIAL_PROTOCOL_ID]
    plan_hash, plan_by_id = _plan_rows(plan)
    expected_rule_references = sorted({
        item for shard_id in shard_plan.SHARD_ORDER
        for item in _expected_rule_references(plan_by_id[shard_id])
    })
    expected_result_references = _expected_result_references(citation_audit)
    expected_assertions = {
        protocol: sorted({
            item for shard_id in shard_plan.SHARD_ORDER
            for item in _expected_assertions(protocol, plan_by_id[shard_id])
        })
        for protocol in (
            shard_plan.SEMANTIC_PROTOCOL_ID,
            shard_plan.ADVERSARIAL_PROTOCOL_ID,
        )
    }
    for protocol, receipt in by_protocol.items():
        fixed = {
            "schema_version": SUITE_RECEIPT_SCHEMA_VERSION,
            "protocol_id": protocol,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "codex_cli_version": CODEX_CLI_VERSION,
            "official_codex_elf_sha256": OFFICIAL_CODEX_ELF_SHA256,
            "shard_plan_sha256": plan_hash,
            "event_stream_path": PROTOCOL_TRACE_PATHS[str(protocol)],
            "invocation_count": 3,
            "no_compaction": True,
            "no_unknown_events": True,
            "within_budget": True,
            "approved": True,
            "completed": True,
            "exit_code": 0,
        }
        for field, expected in fixed.items():
            if receipt.get(field) != expected:
                raise SuiteSealError(f"suite receipt has invalid {field}")
        _require_token(receipt.get("suite_id"), label="suite_id")
        for field in (
            "review_snapshot_sha256", "citation_evidence_sha256",
            "full_evidence_surface_sha256", "output_schema_sha256",
            "shard_output_schema_sha256", "suite_receipt_schema_sha256",
            "suite_trace_schema_sha256", "runtime_manifest_sha256",
            "result_sha256", "event_stream_sha256",
        ):
            _require_sha256(receipt.get(field), label=f"receipt {field}")
        if receipt.get("assertion_ids") != expected_assertions[protocol]:
            raise SuiteSealError("suite receipt assertion union differs from the plan")
        if receipt.get("assertion_union_sha256") != _sha256_object(
            expected_assertions[protocol]
        ):
            raise SuiteSealError("suite receipt assertion-union hash is invalid")
        if receipt.get("rule_reference_ids") != expected_rule_references:
            raise SuiteSealError("suite receipt rule-reference union differs from the plan")
        if receipt.get("rule_reference_union_sha256") != _sha256_object(
            expected_rule_references
        ):
            raise SuiteSealError("suite receipt rule-reference hash is invalid")
        if receipt.get("result_reference_ids") != expected_result_references:
            raise SuiteSealError("suite receipt result-reference union differs from the audit")
        if receipt.get("result_reference_union_sha256") != _sha256_object(
            expected_result_references
        ):
            raise SuiteSealError("suite receipt result-reference hash is invalid")
        rows = receipt.get("shard_invocations")
        if not isinstance(rows, list) or len(rows) != 3:
            raise SuiteSealError("suite receipt does not have three shard invocations")
        if [row.get("shard_id") if isinstance(row, Mapping) else None for row in rows] != list(
            shard_plan.SHARD_ORDER
        ):
            raise SuiteSealError("suite receipt shard order is not canonical")
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != set(_INVOCATION_PUBLIC_FIELDS):
                raise SuiteSealError("suite receipt has a non-closed shard invocation")
            shard_id = str(row["shard_id"])
            if row.get("assertion_ids") != _expected_assertions(
                str(protocol), plan_by_id[shard_id]
            ):
                raise SuiteSealError("shard assertion inventory differs from the plan")
            if row.get("reference_ids") != _expected_rule_references(
                plan_by_id[shard_id]
            ):
                raise SuiteSealError("shard reference inventory differs from the plan")
            _validate_token_budget(row.get("token_budget"))
    common_fields = (
        "model", "reasoning_effort", "codex_cli_version",
        "official_codex_elf_sha256", "citation_evidence_sha256",
        "full_evidence_surface_sha256", "shard_plan_sha256",
        "output_schema_sha256", "shard_output_schema_sha256",
        "suite_receipt_schema_sha256", "suite_trace_schema_sha256",
        "runtime_manifest_sha256", "rule_reference_ids",
        "rule_reference_union_sha256", "result_reference_ids",
        "result_reference_union_sha256",
    )
    for field in common_fields:
        if semantic.get(field) != adversarial.get(field):
            raise SuiteSealError(f"suite protocols disagree on {field}")
    if semantic.get("suite_id") == adversarial.get("suite_id"):
        raise SuiteSealError("suite protocols reuse one suite identity")
    if semantic.get("event_stream_sha256") == adversarial.get(
        "event_stream_sha256"
    ):
        raise SuiteSealError("suite protocols reuse one normalized trace")

    invocations = [
        row for receipt in (semantic, adversarial)
        for row in receipt.get("shard_invocations", [])
        if isinstance(row, Mapping)
    ]
    if len(invocations) != 6:
        raise SuiteSealError("suite pair does not contain exactly six invocations")
    unique_fields = (
        "invocation_id", "thread_id", "raw_output_sha256",
        "sanitized_output_sha256",
    )
    for field in unique_fields:
        values = [row.get(field) for row in invocations]
        if (any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))):
            raise SuiteSealError(f"six review shards do not have unique {field}")
    if any(
        row.get("runtime_manifest_sha256")
        != semantic.get("runtime_manifest_sha256")
        for row in invocations
    ):
        raise SuiteSealError("six review shards did not use one runtime")
    if any(
        row.get("compaction_event_count") != 0
        or row.get("unknown_event_count") != 0
        or row.get("assigned_chunks_complete") is not True
        or row.get("token_budget", {}).get("within_budget") is not True
        for row in invocations
    ):
        raise SuiteSealError("suite pair contains compaction, unknown, incomplete, or over-budget evidence")

    if (semantic_result.get("citation_results")
            != adversarial_result.get("citation_results")):
        raise SuiteSealError("semantic and adversarial citation verdicts disagree")
    for receipt, result in (
        (semantic, semantic_result), (adversarial, adversarial_result),
    ):
        if (receipt.get("result_sha256") != artifact_sha256(dict(result))
                or receipt.get("approved") is not True
                or receipt.get("completed") is not True
                or receipt.get("exit_code") != 0
                or receipt.get("no_compaction") is not True
                or receipt.get("no_unknown_events") is not True
                or receipt.get("within_budget") is not True):
            raise SuiteSealError("suite pair contains an unapproved protocol receipt")

    assertion_ids = sorted(
        list(semantic["assertion_ids"]) + list(adversarial["assertion_ids"])
    )
    if len(assertion_ids) != len(set(assertion_ids)):
        raise SuiteSealError("semantic and adversarial assertion inventories overlap")
    raw_hashes = sorted(str(row["raw_output_sha256"]) for row in invocations)
    sanitized_hashes = sorted(
        str(row["sanitized_output_sha256"]) for row in invocations
    )
    invocation_ids = sorted(str(row["invocation_id"]) for row in invocations)
    thread_ids = sorted(str(row["thread_id"]) for row in invocations)
    receipt_hashes = sorted(artifact_sha256(receipt) for receipt in receipts)
    trace_hashes = sorted(str(receipt["event_stream_sha256"]) for receipt in receipts)
    result_hashes = sorted(str(receipt["result_sha256"]) for receipt in receipts)
    return {
        "schema_version": SUITE_PAIR_SEAL_SCHEMA_VERSION,
        "protocols": sorted(by_protocol),
        "receipt_sha256s": receipt_hashes,
        "trace_sha256s": trace_hashes,
        "result_sha256s": result_hashes,
        "runtime_manifest_sha256": semantic["runtime_manifest_sha256"],
        "full_evidence_surface_sha256": semantic[
            "full_evidence_surface_sha256"
        ],
        "shard_plan_sha256": semantic["shard_plan_sha256"],
        "invocation_ids": invocation_ids,
        "thread_ids": thread_ids,
        "raw_output_sha256s": raw_hashes,
        "sanitized_output_sha256s": sanitized_hashes,
        "assertion_ids": assertion_ids,
        "rule_reference_ids": list(semantic["rule_reference_ids"]),
        "result_reference_ids": list(semantic["result_reference_ids"]),
        "invocation_count": 6,
        "no_compaction": True,
        "no_unknown_events": True,
        "within_budget": True,
        "approved": True,
    }


__all__ = [
    "CODEX_CLI_VERSION", "MODEL", "OFFICIAL_CODEX_ELF_SHA256",
    "PROTOCOL_TRACE_PATHS", "REASONING_EFFORT",
    "SUITE_PAIR_SEAL_SCHEMA_VERSION", "SUITE_RECEIPT_SCHEMA_VERSION",
    "SUITE_TRACE_SCHEMA_VERSION", "SuiteSealError", "build_protocol_receipt",
    "build_protocol_trace", "parse_protocol_trace", "validate_protocol_receipt",
    "validate_protocol_trace", "validate_suite_pair", "artifact_json_bytes",
    "artifact_sha256",
]
