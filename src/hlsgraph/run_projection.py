"""Fail-closed public projections for immutable tool-run provenance.

Run metadata is intentionally an extensible internal object.  Public REST, MCP,
status/health, and ML surfaces must therefore share a positive schema rather
than recursively copying that object.  Unknown keys and malformed legacy values
are omitted; they are never stringified into a response.
"""
from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping

from .model import AuthorityClass, FailureClass, GateKind, GateStatus, RunStatus


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")

PUBLIC_RUN_AUTHORITIES = frozenset({
    *(item.value for item in AuthorityClass),
    # Runner event classifications are provenance states, not additional fact
    # authorities.  They are nevertheless bounded public values.
    "infrastructure", "replay", "deterministic_derivation",
})
PUBLIC_RUN_STATUSES = frozenset(item.value for item in RunStatus)
PUBLIC_FAILURE_CLASSES = frozenset(item.value for item in FailureClass)
PUBLIC_GATE_KINDS = frozenset(item.value for item in GateKind)
PUBLIC_GATE_STATUSES = frozenset(item.value for item in GateStatus)

_BOOLEAN_FIELDS = frozenset({
    "execution_enabled", "fresh_execution", "fresh_tool_truth",
    "input_validation_failed", "output_embedded", "partial_graph_persisted",
    "remote_environment_verified", "remote_inputs_verified", "snapshot_stale",
    "resource_guard_checked", "resource_guard_configured", "resource_guard_passed",
    "runtime_guard_checked", "runtime_guard_configured", "runtime_guard_passed",
    "runtime_guard_triggered",
    "tool_truth",
})
_BYTE_COUNT_FIELDS = frozenset({"stderr_bytes", "stdout_bytes"})
_DIGEST_FIELDS = frozenset({
    "bootstrap_environment_hash", "expected_remote_environment_hash",
    "inherited_environment_hash",
    "replayed_request_hash", "runner_fingerprint", "stderr_sha256",
    "stdout_sha256",
})
_IDENTIFIER_FIELDS = frozenset({
    "campaign_id", "replayed_from_run_id", "workload_id",
})


def public_identifier(value: Any) -> str | None:
    """Return a bounded public identifier, never a coerced arbitrary value."""
    if isinstance(value, str) and _SAFE_ID.fullmatch(value):
        return value
    return None


def public_sha256(value: Any) -> str | None:
    """Return a normalized SHA-256 digest when the value has the exact shape."""
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return value.lower()
    return None


def public_enum(value: Any, allowed: frozenset[str]) -> str | None:
    """Project one string through an explicit finite vocabulary."""
    return value if isinstance(value, str) and value in allowed else None


def public_timestamp(value: Any) -> str | None:
    """Return one bounded timezone-aware ISO-8601 timestamp."""
    if not isinstance(value, str) or len(value) > 64 or "\x00" in value:
        return None
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError:
        return None
    return value if parsed.tzinfo is not None else None


def public_identifier_list(value: Any) -> list[str]:
    """Return only safe IDs from a list; other container types fail closed."""
    if not isinstance(value, list):
        return []
    return [safe for item in value if (safe := public_identifier(item)) is not None]


def sanitize_run_metadata(value: Any) -> dict[str, Any]:
    """Apply the shared positive public schema to one run metadata object."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in sorted(_BOOLEAN_FIELDS):
        item = value.get(key)
        if isinstance(item, bool):
            result[key] = item
    for key in sorted(_BYTE_COUNT_FIELDS):
        item = value.get(key)
        if (isinstance(item, int) and not isinstance(item, bool)
                and 0 <= item <= 9_223_372_036_854_775_807):
            result[key] = item
    for key in sorted(_DIGEST_FIELDS):
        item = public_sha256(value.get(key))
        if item is not None:
            result[key] = item
    for key in sorted(_IDENTIFIER_FIELDS):
        item = public_identifier(value.get(key))
        if item is not None:
            result[key] = item
    authority = public_enum(value.get("authority"), PUBLIC_RUN_AUTHORITIES)
    if authority is not None:
        result["authority"] = authority
    mismatch_ids = public_identifier_list(value.get("input_mismatch_ids"))
    if mismatch_ids:
        result["input_mismatch_ids"] = mismatch_ids
    # A future backend may report a failure subtype, but it is public only when
    # it uses the already versioned FailureClass vocabulary.
    # ``infra_resource_guard`` is privileged structured runner provenance and
    # is public only as ToolRun.failure_class.  Extensible metadata must never
    # synthesize that classification.
    metadata_failure_classes = PUBLIC_FAILURE_CLASSES - {
        FailureClass.INFRA_RESOURCE_GUARD.value,
    }
    failure_type = public_enum(value.get("failure_type"), metadata_failure_classes)
    if failure_type is not None:
        result["failure_type"] = failure_type
    return result


__all__ = [
    "PUBLIC_FAILURE_CLASSES", "PUBLIC_GATE_KINDS", "PUBLIC_GATE_STATUSES",
    "PUBLIC_RUN_AUTHORITIES", "PUBLIC_RUN_STATUSES", "public_enum",
    "public_identifier", "public_identifier_list", "public_sha256",
    "public_timestamp", "sanitize_run_metadata",
]
