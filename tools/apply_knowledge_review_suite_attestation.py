#!/usr/bin/env python3
"""Apply a verified v6 review-suite attestation to the public knowledge packs.

This module is intentionally separate from review execution.  It neither
starts Codex nor trusts a caller-provided pair seal.  The semantic and
adversarial receipts are re-closed with :func:`validate_suite_pair`, the two
frozen :class:`ReviewSnapshot` values are checked against the checkout, and
all three candidate packs are loaded as ``review_ready`` before any write.

``build_updates`` is side-effect free with respect to the checkout.  The
``apply_attestation`` entry point rebuilds the same updates and delegates the
three-file transaction to the established rollback-capable atomic replacer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from tools import knowledge_review_shards as shard_plan
from tools import run_knowledge_review as legacy_review
from tools import seal_knowledge_review_suite as suite_seal


ATTESTATION_STATUS = "machine_repeated_reviewed"
EXPECTED_PACK_COUNT = 3
REVIEW_INVOCATIONS_KEY = "review_invocations"

# The snapshot builder owns the single suite-TCB inventory.  The external
# private evidence tree is deliberately absent: it is an execution input,
# never a public source artifact.
SUITE_REVIEW_SOURCE_PATHS = legacy_review.SUITE_REVIEW_SOURCE_PATHS

_AGGREGATE_RESULT_FIELDS = frozenset({
    "protocol_id", "review_surface_sha256",
    "implementation_surface_sha256", "citation_audit_sha256",
    "citation_results", "approved", "issues", "summary",
})
_CITATION_RESULT_FIELDS = frozenset({
    "reference_id", "reference_surface_sha256", "verdict",
    "exact_locator_inspected", "declared_version_matched",
    "declared_section_matched", "paraphrase_supported",
    "applicability_not_broader", "issues",
})
_PUBLIC_INVOCATION_FIELDS = frozenset({
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
})
_ENVELOPE_CONTEXT_FIELDS = frozenset({
    "protocol_id", "suite_id", "protocol_receipt_sha256",
})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class SuiteAttestationError(ValueError):
    """The supplied review evidence cannot safely activate the packs."""


def _pack_json(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SuiteAttestationError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SuiteAttestationError(f"{label} must be an object")
    try:
        return copy.deepcopy(dict(value))
    except (TypeError, ValueError) as exc:
        raise SuiteAttestationError(f"{label} cannot be copied safely") from exc


def _assert_no_private_paths_or_text(value: Any, *, label: str) -> None:
    """Reject path-like strings in the closed public evidence envelopes.

    Arbitrary prose cannot enter these envelopes because their keys are
    separately allow-listed.  This recursive check closes the remaining
    absolute-path and URI-shaped value channel.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or "\x00" in key:
                raise SuiteAttestationError(f"{label} has an invalid key")
            _assert_no_private_paths_or_text(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_private_paths_or_text(item, label=f"{label}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if ("\x00" in value or value.startswith(("/", "\\\\"))
                or _WINDOWS_ABSOLUTE_RE.match(value)
                or lowered.startswith(("file://", "file:\\"))):
            raise SuiteAttestationError(
                f"{label} contains a filesystem path instead of public evidence"
            )


def _validate_aggregate_result(
    result: Mapping[str, Any], *, protocol_id: str,
    snapshot: legacy_review.ReviewSnapshot,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = _strict_mapping(result, label=f"{protocol_id} aggregate result")
    if set(value) != _AGGREGATE_RESULT_FIELDS:
        raise SuiteAttestationError(
            f"{protocol_id} aggregate result is not the closed result contract"
        )
    if (value.get("protocol_id") != protocol_id
            or value.get("approved") is not True
            or value.get("issues") != []
            or value.get("summary") != "approved_no_issues"):
        raise SuiteAttestationError(
            f"{protocol_id} aggregate result is not approved and issue-free"
        )
    if value.get("review_surface_sha256") != snapshot.surfaces:
        raise SuiteAttestationError(
            f"{protocol_id} aggregate result binds another pack surface"
        )
    if (value.get("implementation_surface_sha256")
            != snapshot.implementation_surface_sha256
            or value.get("citation_audit_sha256")
            != snapshot.citation_audit_sha256):
        raise SuiteAttestationError(
            f"{protocol_id} aggregate result binds another frozen snapshot"
        )
    rows = value.get("citation_results")
    if not isinstance(rows, list) or len(rows) != 53:
        raise SuiteAttestationError(
            f"{protocol_id} aggregate result must contain 53 citations"
        )
    reference_ids: list[str] = []
    rule_reference_ids = receipt.get("rule_reference_ids")
    if (not isinstance(rule_reference_ids, list)
            or len(rule_reference_ids) != 38
            or len(rule_reference_ids) != len(set(rule_reference_ids))):
        raise SuiteAttestationError(
            f"{protocol_id} receipt has an invalid rule-reference inventory"
        )
    rule_references = set(rule_reference_ids)
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _CITATION_RESULT_FIELDS:
            raise SuiteAttestationError(
                f"{protocol_id} aggregate result has a non-closed citation"
            )
        reference_id = _require_sha256(
            row.get("reference_id"), label="aggregate citation reference ID",
        )
        _require_sha256(
            row.get("reference_surface_sha256"),
            label="aggregate citation surface",
        )
        rule_claims = (
            row.get("exact_locator_inspected") is True
            and row.get("declared_section_matched") is True
            and row.get("paraphrase_supported") is True
            and row.get("applicability_not_broader") is True
        )
        document_identity = (
            row.get("exact_locator_inspected") is False
            and row.get("declared_section_matched") is None
            and row.get("paraphrase_supported") is None
            and row.get("applicability_not_broader") is None
        )
        if (row.get("verdict") != "verified" or row.get("issues") != []
                or row.get("declared_version_matched") is not True
                or (reference_id in rule_references and not rule_claims)
                or (reference_id not in rule_references and not document_identity)):
            raise SuiteAttestationError(
                f"{protocol_id} aggregate result contains an unverified citation"
            )
        reference_ids.append(reference_id)
    expected_references = receipt.get("result_reference_ids")
    if (not isinstance(expected_references, list)
            or reference_ids != sorted(reference_ids)
            or len(reference_ids) != len(set(reference_ids))
            or reference_ids != expected_references):
        raise SuiteAttestationError(
            f"{protocol_id} aggregate citation inventory differs from its receipt"
        )
    if receipt.get("result_sha256") != suite_seal.artifact_sha256(value):
        raise SuiteAttestationError(
            f"{protocol_id} aggregate result hash differs from its receipt"
        )
    return value


def _validate_snapshot(
    snapshot: legacy_review.ReviewSnapshot, *, protocol_id: str,
    receipt: Mapping[str, Any], aggregate_result: Mapping[str, Any],
) -> None:
    if not isinstance(snapshot, legacy_review.ReviewSnapshot):
        raise SuiteAttestationError(
            f"{protocol_id} requires a frozen ReviewSnapshot"
        )
    if snapshot.protocol_id != protocol_id:
        raise SuiteAttestationError(
            f"{protocol_id} received another protocol's ReviewSnapshot"
        )
    expected = {
        "review_snapshot_sha256": snapshot.sha256,
        "citation_evidence_sha256": snapshot.citation_evidence_sha256,
        "output_schema_sha256": snapshot.output_schema_sha256,
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise SuiteAttestationError(
                f"{protocol_id} receipt does not bind the frozen {field}"
            )
    _validate_aggregate_result(
        aggregate_result, protocol_id=protocol_id, snapshot=snapshot,
        receipt=receipt,
    )


def _assert_common_snapshots(
    semantic: legacy_review.ReviewSnapshot,
    adversarial: legacy_review.ReviewSnapshot,
) -> None:
    fields = (
        "review_surface_sha256", "implementation_surface_sha256",
        "citation_audit_sha256", "citation_evidence_sha256",
        "output_schema_sha256", "receipt_schema_sha256",
        "exact_citation_urls", "files",
    )
    for field in fields:
        if getattr(semantic, field) != getattr(adversarial, field):
            raise SuiteAttestationError(
                f"semantic and adversarial snapshots disagree on {field}"
            )


def _assert_snapshots_current(
    root: Path, semantic: legacy_review.ReviewSnapshot,
    adversarial: legacy_review.ReviewSnapshot,
) -> None:
    for expected in (semantic, adversarial):
        try:
            current = legacy_review.freeze_review_snapshot(
                root, expected.protocol_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SuiteAttestationError(
                f"cannot re-freeze {expected.protocol_id} snapshot"
            ) from exc
        if current != expected:
            raise SuiteAttestationError(
                f"{expected.protocol_id} ReviewSnapshot changed before attestation"
            )


def suite_review_source_hashes(
    root: Path, surfaces: Mapping[str, str], implementation_sha256: str,
) -> dict[str, str]:
    """Return the legacy closure plus every v6 suite implementation input."""

    resolved = Path(root).resolve(strict=True)
    try:
        hashes = legacy_review.review_source_hashes(
            resolved, dict(surfaces), implementation_sha256,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SuiteAttestationError(
            "cannot compute the deterministic review source closure"
        ) from exc
    for relative in sorted(SUITE_REVIEW_SOURCE_PATHS):
        path = resolved / PurePosixPath(relative)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise SuiteAttestationError(
                f"v6 review source closure is missing {relative}"
            ) from exc
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    return dict(sorted(hashes.items()))


def _receipt_hashes(
    semantic_receipt: Mapping[str, Any], adversarial_receipt: Mapping[str, Any],
) -> dict[str, str]:
    values = {
        shard_plan.SEMANTIC_PROTOCOL_ID: suite_seal.artifact_sha256(
            semantic_receipt,
        ),
        shard_plan.ADVERSARIAL_PROTOCOL_ID: suite_seal.artifact_sha256(
            adversarial_receipt,
        ),
    }
    return dict(sorted(values.items()))


def public_invocation_envelopes(
    semantic_receipt: Mapping[str, Any], adversarial_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project exactly six path-free public invocation envelopes."""

    rows: list[dict[str, Any]] = []
    receipts = (
        (shard_plan.SEMANTIC_PROTOCOL_ID, semantic_receipt),
        (shard_plan.ADVERSARIAL_PROTOCOL_ID, adversarial_receipt),
    )
    for protocol_id, receipt in receipts:
        if receipt.get("protocol_id") != protocol_id:
            raise SuiteAttestationError("receipt protocol identity changed")
        receipt_hash = suite_seal.artifact_sha256(receipt)
        invocations = receipt.get("shard_invocations")
        if not isinstance(invocations, list) or len(invocations) != 3:
            raise SuiteAttestationError(
                f"{protocol_id} receipt lacks three public invocations"
            )
        for invocation in invocations:
            if (not isinstance(invocation, Mapping)
                    or set(invocation) != _PUBLIC_INVOCATION_FIELDS):
                raise SuiteAttestationError(
                    f"{protocol_id} receipt has a non-closed public invocation"
                )
            envelope = copy.deepcopy(dict(invocation))
            envelope.update({
                "protocol_id": protocol_id,
                "suite_id": receipt.get("suite_id"),
                "protocol_receipt_sha256": receipt_hash,
            })
            if set(envelope) != _PUBLIC_INVOCATION_FIELDS | _ENVELOPE_CONTEXT_FIELDS:
                raise AssertionError("public invocation projection is not closed")
            _assert_no_private_paths_or_text(
                envelope, label="public review invocation",
            )
            rows.append(envelope)
    rows.sort(key=lambda item: (
        str(item["protocol_id"]), str(item["shard_id"]),
        str(item["invocation_id"]),
    ))
    for field in (
        "invocation_id", "thread_id", "raw_output_sha256",
        "sanitized_output_sha256",
    ):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise SuiteAttestationError(
                f"public review invocations reuse one {field}"
            )
    if len(rows) != 6:
        raise SuiteAttestationError("review attestation requires six invocations")
    return rows


def build_attestation_material(
    root: Path, *, semantic_receipt: Mapping[str, Any],
    adversarial_receipt: Mapping[str, Any],
    semantic_result: Mapping[str, Any], adversarial_result: Mapping[str, Any],
    suite_pair_seal: Mapping[str, Any], plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any],
    semantic_snapshot: legacy_review.ReviewSnapshot,
    adversarial_snapshot: legacy_review.ReviewSnapshot,
) -> dict[str, Any]:
    """Validate all v6 inputs and construct deterministic public evidence."""

    resolved = Path(root).resolve(strict=True)
    semantic = _strict_mapping(semantic_receipt, label="semantic receipt")
    adversarial = _strict_mapping(
        adversarial_receipt, label="adversarial receipt",
    )
    semantic_aggregate = _strict_mapping(
        semantic_result, label="semantic aggregate result",
    )
    adversarial_aggregate = _strict_mapping(
        adversarial_result, label="adversarial aggregate result",
    )
    supplied_pair = _strict_mapping(suite_pair_seal, label="suite pair seal")
    try:
        validated_pair = suite_seal.validate_suite_pair(
            semantic_receipt=semantic,
            adversarial_receipt=adversarial,
            semantic_result=semantic_aggregate,
            adversarial_result=adversarial_aggregate,
            plan=plan, citation_audit=citation_audit,
        )
    except (TypeError, ValueError, suite_seal.SuiteSealError) as exc:
        raise SuiteAttestationError("v6 suite pair validation failed") from exc
    if supplied_pair != validated_pair:
        raise SuiteAttestationError(
            "supplied suite pair seal differs from deterministic validation"
        )

    _validate_snapshot(
        semantic_snapshot, protocol_id=shard_plan.SEMANTIC_PROTOCOL_ID,
        receipt=semantic, aggregate_result=semantic_aggregate,
    )
    _validate_snapshot(
        adversarial_snapshot, protocol_id=shard_plan.ADVERSARIAL_PROTOCOL_ID,
        receipt=adversarial, aggregate_result=adversarial_aggregate,
    )
    _assert_common_snapshots(semantic_snapshot, adversarial_snapshot)
    _assert_snapshots_current(resolved, semantic_snapshot, adversarial_snapshot)

    receipt_hashes = _receipt_hashes(semantic, adversarial)
    if sorted(receipt_hashes.values()) != validated_pair.get("receipt_sha256s"):
        raise SuiteAttestationError("pair seal does not bind both receipt hashes")
    invocations = public_invocation_envelopes(semantic, adversarial)
    if ([row["invocation_id"] for row in invocations]
            and sorted(row["invocation_id"] for row in invocations)
            != validated_pair.get("invocation_ids")):
        raise SuiteAttestationError("pair seal does not bind all six invocations")
    reviewers = sorted(
        f"{suite_seal.MODEL}@{suite_seal.REASONING_EFFORT}#{row['invocation_id']}"
        for row in invocations
    )
    if len(reviewers) != 6 or len(reviewers) != len(set(reviewers)):
        raise SuiteAttestationError("reviewers do not identify six invocations")
    source_hashes = suite_review_source_hashes(
        resolved, semantic_snapshot.surfaces,
        semantic_snapshot.implementation_surface_sha256,
    )
    evidence = {
        "independent_invocations": True,
        "same_model_repeated_review": True,
        "distinct_model_families": False,
        "citation_verified": True,
        "review_agreement": True,
        "unresolved_conflicts": False,
        "suite_pair_seal_sha256": suite_seal.artifact_sha256(validated_pair),
        "suite_pair_seal": copy.deepcopy(validated_pair),
        "protocol_receipt_sha256s": receipt_hashes,
        REVIEW_INVOCATIONS_KEY: invocations,
    }
    _assert_no_private_paths_or_text(
        validated_pair, label="suite pair seal",
    )
    # source_hash keys are intentionally public repository-relative paths;
    # only invocation envelopes and the suite seal carry the no-path rule.
    return {
        "reviewers": reviewers,
        "source_hashes": source_hashes,
        "review_evidence": evidence,
    }


def _pack_paths(root: Path) -> list[Path]:
    pack_root = root / "src" / "hlsgraph" / "knowledge" / "packs"
    try:
        resolved_pack_root = pack_root.resolve(strict=True)
    except OSError as exc:
        raise SuiteAttestationError("knowledge pack directory is missing") from exc
    paths = sorted(resolved_pack_root.glob("*.json"), key=lambda path: path.name)
    if len(paths) != EXPECTED_PACK_COUNT:
        raise SuiteAttestationError(
            f"suite attestation requires exactly {EXPECTED_PACK_COUNT} packs"
        )
    for path in paths:
        try:
            if path.is_symlink() or path.resolve(strict=True).parent != resolved_pack_root:
                raise SuiteAttestationError(
                    f"knowledge pack escaped its directory: {path.name}"
                )
        except OSError as exc:
            raise SuiteAttestationError(
                f"cannot resolve knowledge pack {path.name}"
            ) from exc
    return paths


def _build_updates_and_originals(
    root: Path, *, semantic_receipt: Mapping[str, Any],
    adversarial_receipt: Mapping[str, Any],
    semantic_result: Mapping[str, Any], adversarial_result: Mapping[str, Any],
    suite_pair_seal: Mapping[str, Any], plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any],
    semantic_snapshot: legacy_review.ReviewSnapshot,
    adversarial_snapshot: legacy_review.ReviewSnapshot,
) -> tuple[dict[Path, bytes], dict[Path, bytes]]:
    resolved = Path(root).resolve(strict=True)
    material = build_attestation_material(
        resolved,
        semantic_receipt=semantic_receipt,
        adversarial_receipt=adversarial_receipt,
        semantic_result=semantic_result,
        adversarial_result=adversarial_result,
        suite_pair_seal=suite_pair_seal,
        plan=plan, citation_audit=citation_audit,
        semantic_snapshot=semantic_snapshot,
        adversarial_snapshot=adversarial_snapshot,
    )
    updates: dict[Path, bytes] = {}
    originals: dict[Path, bytes] = {}
    pack_ids: set[str] = set()
    try:
        from hlsgraph.knowledge import load_pack
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise SuiteAttestationError("cannot load the knowledge-pack validator") from exc
    for path in _pack_paths(resolved):
        try:
            original = path.read_bytes()
            value = legacy_review._strict_json_bytes(
                original, label=f"knowledge pack {path.name}",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SuiteAttestationError(
                f"cannot parse knowledge pack {path.name}"
            ) from exc
        if not isinstance(value, dict):
            raise SuiteAttestationError(
                f"knowledge pack {path.name} is not an object"
            )
        try:
            pack_id, before_surface, _payload = (
                legacy_review._semantic_pack_projection(
                    original, label=f"knowledge pack {path.name}",
                )
            )
        except (TypeError, ValueError) as exc:
            raise SuiteAttestationError(
                f"cannot project knowledge pack {path.name}"
            ) from exc
        if (pack_id in pack_ids
                or semantic_snapshot.surfaces.get(pack_id) != before_surface):
            raise SuiteAttestationError(
                f"knowledge pack surface differs from the review: {pack_id}"
            )
        pack_ids.add(pack_id)
        metadata = value.get("metadata")
        coverage = value.get("coverage")
        if not isinstance(metadata, dict) or not isinstance(coverage, dict):
            raise SuiteAttestationError(
                f"knowledge pack lacks review fields: {pack_id}"
            )
        metadata["review_status"] = ATTESTATION_STATUS
        coverage.update(copy.deepcopy(material))
        coverage["review_status"] = ATTESTATION_STATUS
        # Coverage identity includes review evidence and must be recomputed by
        # the public loader.  Retaining an earlier identity would be a forgery.
        coverage.pop("id", None)
        encoded = _pack_json(value)
        try:
            after_id, after_surface, _payload = (
                legacy_review._semantic_pack_projection(
                    encoded, label=f"attested knowledge pack {path.name}",
                )
            )
            loaded = load_pack(json.loads(encoded))
        except (TypeError, ValueError) as exc:
            raise SuiteAttestationError(
                f"attested knowledge pack does not load: {pack_id}"
            ) from exc
        if (after_id, after_surface) != (pack_id, before_surface):
            raise SuiteAttestationError(
                f"attestation changed semantic pack surface: {pack_id}"
            )
        if not loaded.review_ready:
            raise SuiteAttestationError(
                f"attested knowledge pack is not review_ready: {pack_id}"
            )
        updates[path] = encoded
        originals[path] = original
    if pack_ids != set(semantic_snapshot.surfaces):
        raise SuiteAttestationError(
            "reviewed pack inventory differs from the three public packs"
        )
    return updates, originals


def build_updates(
    root: Path, *, semantic_receipt: Mapping[str, Any],
    adversarial_receipt: Mapping[str, Any],
    semantic_result: Mapping[str, Any], adversarial_result: Mapping[str, Any],
    suite_pair_seal: Mapping[str, Any], plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any],
    semantic_snapshot: legacy_review.ReviewSnapshot,
    adversarial_snapshot: legacy_review.ReviewSnapshot,
) -> dict[Path, bytes]:
    """Build all three attested pack bytes without modifying the checkout."""

    updates, _originals = _build_updates_and_originals(
        root,
        semantic_receipt=semantic_receipt,
        adversarial_receipt=adversarial_receipt,
        semantic_result=semantic_result,
        adversarial_result=adversarial_result,
        suite_pair_seal=suite_pair_seal,
        plan=plan, citation_audit=citation_audit,
        semantic_snapshot=semantic_snapshot,
        adversarial_snapshot=adversarial_snapshot,
    )
    return updates


def apply_attestation(
    root: Path, *, semantic_receipt: Mapping[str, Any],
    adversarial_receipt: Mapping[str, Any],
    semantic_result: Mapping[str, Any], adversarial_result: Mapping[str, Any],
    suite_pair_seal: Mapping[str, Any], plan: Mapping[str, Any],
    citation_audit: Mapping[str, Any],
    semantic_snapshot: legacy_review.ReviewSnapshot,
    adversarial_snapshot: legacy_review.ReviewSnapshot,
) -> dict[Path, bytes]:
    """Atomically activate all three packs, rolling back on any late failure."""

    resolved = Path(root).resolve(strict=True)
    updates, originals = _build_updates_and_originals(
        resolved,
        semantic_receipt=semantic_receipt,
        adversarial_receipt=adversarial_receipt,
        semantic_result=semantic_result,
        adversarial_result=adversarial_result,
        suite_pair_seal=suite_pair_seal,
        plan=plan, citation_audit=citation_audit,
        semantic_snapshot=semantic_snapshot,
        adversarial_snapshot=adversarial_snapshot,
    )
    # Close the build/apply gap before delegating to the established atomic
    # replacement helper.  A changed byte aborts before the first replace.
    for path, original in originals.items():
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise SuiteAttestationError(
                f"knowledge pack disappeared before apply: {path.name}"
            ) from exc
        if current != original:
            raise SuiteAttestationError(
                f"knowledge pack changed before apply: {path.name}"
            )
    _assert_snapshots_current(resolved, semantic_snapshot, adversarial_snapshot)
    try:
        legacy_review._atomic_replace_pack_bytes(resolved, updates)
        if any(path.read_bytes() != payload for path, payload in updates.items()):
            raise SuiteAttestationError("one attested pack was not replaced exactly")
        _assert_snapshots_current(
            resolved, semantic_snapshot, adversarial_snapshot,
        )
    except BaseException:
        # The delegated helper already rolls back partial os.replace failures.
        # This outer rollback covers a successful replace followed by a failed
        # snapshot/load verification.
        try:
            if any(path.read_bytes() != payload for path, payload in originals.items()):
                legacy_review._atomic_replace_pack_bytes(resolved, originals)
        except BaseException:
            # Preserve the initiating failure while making recovery failure
            # explicit through exception chaining at the call site.
            pass
        raise
    return updates


def finalize_attestation(
    root: Path, *, suite_evidence: Path,
) -> dict[Path, bytes]:
    """Audit every private v6 input before atomically activating the packs.

    This is the only operational finalizer.  It deliberately consumes the
    release auditor's pure replay result instead of accepting receipts,
    results, snapshots, or a pair seal from command-line arguments.  A failed
    raw/cache/process replay therefore cannot be converted into public review
    metadata by calling the lower-level transaction directly.
    """

    from tools import audit_release

    resolved = Path(root).resolve(strict=True)
    semantic_result_path = resolved / PurePosixPath(
        legacy_review.PROTOCOL_FILES[shard_plan.SEMANTIC_PROTOCOL_ID]["result"]
    )
    adversarial_result_path = resolved / PurePosixPath(
        legacy_review.PROTOCOL_FILES[shard_plan.ADVERSARIAL_PROTOCOL_ID]["result"]
    )
    replay = audit_release.replay_knowledge_review_suite_evidence(
        resolved,
        semantic_review=semantic_result_path,
        adversarial_review=adversarial_result_path,
        suite_evidence=Path(suite_evidence),
    )
    if not replay.verified:
        detail = "; ".join(replay.issues) or "incomplete replay products"
        raise SuiteAttestationError(
            f"knowledge-review suite evidence audit failed: {detail}"
        )
    required = {
        "semantic_receipt": replay.semantic_receipt,
        "adversarial_receipt": replay.adversarial_receipt,
        "semantic_result": replay.semantic_result,
        "adversarial_result": replay.adversarial_result,
        "suite_pair_seal": replay.pair_seal,
        "plan": replay.plan,
        "citation_audit": replay.citation_audit,
        "semantic_snapshot": replay.semantic_snapshot,
        "adversarial_snapshot": replay.adversarial_snapshot,
    }
    if any(value is None for value in required.values()):
        raise SuiteAttestationError(
            "knowledge-review suite evidence audit omitted a required product"
        )
    return apply_attestation(resolved, **required)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "fully replay v6 private evidence, then atomically attest all "
            "public knowledge packs"
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--suite-evidence", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        updates = finalize_attestation(
            args.root, suite_evidence=args.suite_evidence,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"knowledge-review attestation refused: {exc}", file=sys.stderr)
        return 1
    summary = {
        path.name: hashlib.sha256(payload).hexdigest()
        for path, payload in sorted(updates.items(), key=lambda item: item[0].name)
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - formal CLI only
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_STATUS", "EXPECTED_PACK_COUNT", "REVIEW_INVOCATIONS_KEY",
    "SUITE_REVIEW_SOURCE_PATHS", "SuiteAttestationError",
    "apply_attestation", "build_attestation_material", "build_updates",
    "finalize_attestation", "main",
    "public_invocation_envelopes", "suite_review_source_hashes",
]
