"""Fail-closed public projection for local diagnostic records.

Diagnostic messages, guidance, and extension metadata are useful to a trusted
local operator, but vendor tools and plugins may place paths, source excerpts,
commands, or other project-private values in them.  Public SDK query results,
CLI status, REST, and MCP therefore share this positive schema instead of
recursively serializing the local record.
"""
from __future__ import annotations

from typing import Any, Mapping

from .model import Diagnostic, DiagnosticSeverity, SourceAnchor, json_ready, stable_hash
from .run_projection import public_enum, public_identifier, public_sha256


PUBLIC_DIAGNOSTIC_SEVERITIES = frozenset(
    item.value for item in DiagnosticSeverity
)

_ANCHOR_INTEGER_FIELDS = (
    "start_line", "start_column", "end_line", "end_column",
)
_ANCHOR_TEXT_FIELDS = (
    "symbol", "ir_location", "mapping_kind", "ambiguity",
)
_PUBLIC_MESSAGE = (
    "Diagnostic details are redacted; inspect the trusted local bundle by ID."
)


def _public_anchor(value: Any) -> dict[str, Any] | None:
    """Revalidate and positively project one source/IR evidence anchor."""
    ready = json_ready(value)
    if not isinstance(ready, Mapping):
        return None
    artifact_id = public_identifier(ready.get("artifact_id"))
    if artifact_id is None:
        return None
    candidate: dict[str, Any] = {"artifact_id": artifact_id}
    for field in _ANCHOR_INTEGER_FIELDS:
        item = ready.get(field)
        if item is not None:
            if (not isinstance(item, int) or isinstance(item, bool)
                    or not 1 <= item <= 2_147_483_647):
                return None
            candidate[field] = item
    for field in _ANCHOR_TEXT_FIELDS:
        item = ready.get(field)
        if item is not None:
            if not isinstance(item, str):
                return None
            candidate[field] = item
    try:
        # Construction reapplies length/path validation even for a mutated or
        # legacy object.  Absolute locations become non-reversible digests.
        anchor = SourceAnchor.from_dict(candidate)
    except (TypeError, ValueError):
        return None
    return {
        key: item for key, item in json_ready(anchor).items()
        if item is not None
    }


def _detail_hash(raw: Mapping[str, Any]) -> str:
    details = {
        "message": raw.get("message"),
        "guidance": raw.get("guidance"),
        "metadata": raw.get("metadata"),
    }
    try:
        return stable_hash(details)
    except (TypeError, ValueError, UnicodeError, OverflowError):
        # Malformed legacy/plugin detail still receives a deterministic public
        # marker without stringifying or disclosing the offending object.
        return stable_hash({"diagnostic_detail": "unavailable"})


def public_diagnostic(value: Any) -> dict[str, Any]:
    """Return the bounded public view of one local :class:`Diagnostic`.

    The raw message, guidance, and metadata never cross this boundary.  Their
    digest allows a local operator to correlate a public report with the exact
    trusted ledger row without publishing the details themselves.
    """
    ready = json_ready(value)
    raw = dict(ready) if isinstance(ready, Mapping) else {}
    return {
        "id": public_identifier(raw.get("id")),
        "snapshot_id": public_identifier(raw.get("snapshot_id")),
        "code": public_identifier(raw.get("code")),
        "severity": public_enum(
            raw.get("severity"), PUBLIC_DIAGNOSTIC_SEVERITIES,
        ),
        "stage": public_identifier(raw.get("stage")),
        "run_id": public_identifier(raw.get("run_id")),
        "subject_id": public_identifier(raw.get("subject_id")),
        "artifact_id": public_identifier(raw.get("artifact_id")),
        "anchor": _public_anchor(raw.get("anchor")),
        "detail_sha256": public_sha256(_detail_hash(raw)),
        "detail_redacted": True,
        "message": _PUBLIC_MESSAGE,
    }


def redacted_diagnostic_record(value: Any) -> Diagnostic | None:
    """Return a detail-free typed record for an existing typed-only consumer.

    The renderer currently consumes ``Diagnostic`` objects.  MCP uses this
    adapter so its in-memory render output cannot bypass the public projection
    merely because the final payload is HTML/JSON text.
    """
    projected = public_diagnostic(value)
    required = {
        key: projected.get(key) for key in (
            "snapshot_id", "code", "severity", "stage",
        )
    }
    if not all(isinstance(item, str) for item in required.values()):
        return None
    anchor_value = projected.get("anchor")
    try:
        anchor = (SourceAnchor.from_dict(anchor_value)
                  if isinstance(anchor_value, Mapping) else None)
        return Diagnostic(
            snapshot_id=required["snapshot_id"],
            code=required["code"],
            severity=required["severity"],
            message=_PUBLIC_MESSAGE,
            id=projected.get("id") or "",
            stage=required["stage"],
            run_id=projected.get("run_id"),
            subject_id=projected.get("subject_id"),
            artifact_id=projected.get("artifact_id"),
            anchor=anchor,
        )
    except (TypeError, ValueError):
        return None


__all__ = [
    "PUBLIC_DIAGNOSTIC_SEVERITIES", "public_diagnostic",
    "redacted_diagnostic_record",
]
