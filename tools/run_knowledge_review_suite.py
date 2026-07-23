#!/usr/bin/env python3
"""Pure deterministic core for sharded public knowledge reviews.

This module deliberately does not execute Codex, read cache files, write
artifacts, replay a tool trace, or seal an attestation.  It only projects an
already validated full review-cache manifest, constructs a bounded shard
prompt, validates closed shard results, and aggregates the three fixed shards
back into the existing full-result shape.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from tools import knowledge_review_shards as shard_plan


SHARD_MANIFEST_SCHEMA_VERSION = "hlsgraph.knowledge-review.shard-manifest.v1"
SHARD_RESULT_SCHEMA_VERSION = "hlsgraph.knowledge-review.shard-result.v1"
CITATION_EVIDENCE_SURFACE_VERSION = (
    "hlsgraph.knowledge-review.citation-evidence-surface.v1"
)
PROMPT_CONTRACT_VERSION = "hlsgraph.knowledge-review.shard-prompt.v2"

CONTROLLED_ASSERTION_ISSUES = frozenset({
    "assertion_rejected", "evidence_incomplete", "contract_violation",
})
CONTROLLED_CITATION_ISSUES = frozenset({
    "locator_unavailable", "resolver_mismatch", "version_mismatch",
    "section_mismatch", "paraphrase_unsupported",
    "applicability_too_broad", "inspection_incomplete",
})
CONTROLLED_REVIEW_ISSUES = frozenset({
    "semantic_gap", "activation_bypass", "citation_unavailable",
    "citation_rejected", "contract_violation",
})
_SEVERITIES = frozenset({"critical", "high", "medium", "low"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ASSERTION_ID_RE = re.compile(r"[^\x00-\x1f\x7f]+")

_SHARD_RESULT_KEYS = frozenset({
    "schema_version", "protocol_id", "review_snapshot_sha256",
    "shard_plan_sha256", "shard_id", "citation_evidence_surface_sha256",
    "assertion_results", "citation_results", "approved", "issues", "summary",
})
_ASSERTION_RESULT_KEYS = frozenset({"assertion_id", "verdict", "issues"})
_CITATION_RESULT_KEYS = frozenset({
    "reference_id", "reference_surface_sha256", "verdict",
    "exact_locator_inspected", "declared_version_matched",
    "declared_section_matched", "paraphrase_supported",
    "applicability_not_broader", "issues",
})
_REVIEW_ISSUE_KEYS = frozenset({"severity", "code"})

_CITATION_SURFACE_FIELDS = (
    "requested_url", "evidence_url", "final_url", "redirect_chain",
    "resolver_id", "status", "content_type", "body_sha256", "body_size",
    "inspection_required", "identity_verified", "available",
    "inspection_sha256", "inspection_size", "parser_id", "parser_version",
    "parser_command_sha256", "parser_executable_sha256",
    "parser_version_output_sha256", "resolver_artifacts", "inspection_chunks",
    "error_code", "reference_ids",
)
_RESOLVER_ARTIFACT_FIELDS = (
    "kind", "requested_url", "status", "final_url", "redirect_chain",
    "content_type", "body_sha256", "body_size",
)
_CHUNK_FIELDS = (
    "index", "path", "sha256", "size", "byte_start", "byte_end",
    "original_sha256", "original_size",
)
_FILE_FIELDS = (
    "path", "hash_kind", "sha256", "cache_sha256", "cache_size",
    "model_inspection_required", "chunks",
)


class SuiteContractError(ValueError):
    """One suite input or result violates the closed deterministic contract."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _pretty_json(value: Any) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def review_snapshot_sha256(snapshot_inventory: Mapping[str, Any]) -> str:
    """Reproduce ``ReviewSnapshot.sha256`` without importing the full runner."""

    if not isinstance(snapshot_inventory, Mapping):
        raise SuiteContractError("snapshot inventory must be an object")
    return hashlib.sha256(_pretty_json(dict(snapshot_inventory))).hexdigest()


def _protocol_id(value: str) -> str:
    if value in {"semantic", shard_plan.SEMANTIC_PROTOCOL_ID}:
        return shard_plan.SEMANTIC_PROTOCOL_ID
    if value in {"adversarial", shard_plan.ADVERSARIAL_PROTOCOL_ID}:
        return shard_plan.ADVERSARIAL_PROTOCOL_ID
    raise SuiteContractError(f"unknown knowledge-review protocol: {value!r}")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SuiteContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_array(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SuiteContractError(f"{label} must be an array")
    return value


def _find_shard(plan: Mapping[str, Any], shard_id: str) -> dict[str, Any]:
    if plan.get("schema_version") != shard_plan.PLAN_SCHEMA_VERSION:
        raise SuiteContractError("shard plan has the wrong schema version")
    if plan.get("shard_order") != list(shard_plan.SHARD_ORDER):
        raise SuiteContractError("shard plan has a non-canonical shard order")
    rows = _require_array(plan.get("shards"), label="shard plan shards")
    matches = [row for row in rows if isinstance(row, dict) and row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise SuiteContractError(f"shard plan does not contain exactly one {shard_id!r}")
    return matches[0]


def _assertion_ids(protocol_id: str, shard: Mapping[str, Any]) -> list[str]:
    key = (
        "semantic_assertion_ids"
        if protocol_id == shard_plan.SEMANTIC_PROTOCOL_ID
        else "adversarial_assertion_ids"
    )
    values = _require_array(shard.get(key), label=f"{key} for shard")
    if any(not isinstance(value, str) or not value for value in values):
        raise SuiteContractError(f"{key} contains a malformed assertion ID")
    if len(values) != len(set(values)):
        raise SuiteContractError(f"{key} contains duplicate assertion IDs")
    return sorted(values)


def _safe_chunks(value: Any, *, label: str) -> list[dict[str, Any]]:
    rows = _require_array(value, label=label)
    projected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SuiteContractError(f"{label} row {index} is not an object")
        projected.append({key: copy.deepcopy(row.get(key)) for key in _CHUNK_FIELDS})
    projected.sort(key=lambda row: (row.get("index"), str(row.get("path"))))
    paths = [row.get("path") for row in projected]
    if any(not isinstance(path, str) or not path for path in paths):
        raise SuiteContractError(f"{label} has a chunk without a path")
    if len(paths) != len(set(paths)):
        raise SuiteContractError(f"{label} contains duplicate chunk paths")
    return projected


def _safe_resolver_artifacts(value: Any) -> list[dict[str, Any]]:
    rows = _require_array(value, label="resolver artifacts")
    projected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SuiteContractError(f"resolver artifact {index} is not an object")
        projected.append({
            key: copy.deepcopy(row.get(key)) for key in _RESOLVER_ARTIFACT_FIELDS
        })
    projected.sort(key=lambda row: (
        str(row.get("kind")), str(row.get("requested_url")),
        str(row.get("body_sha256")),
    ))
    return projected


def _citation_surface_rows(
    citations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for index, source in enumerate(citations):
        if not isinstance(source, Mapping):
            raise SuiteContractError(f"citation evidence row {index} is not an object")
        requested = source.get("requested_url")
        if not isinstance(requested, str) or not requested or requested in seen_urls:
            raise SuiteContractError("citation evidence has a duplicate or invalid URL")
        seen_urls.add(requested)
        row = {
            key: copy.deepcopy(source.get(key)) for key in _CITATION_SURFACE_FIELDS
        }
        ids = row["reference_ids"]
        if not isinstance(ids, list) or any(
            not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None
            for item in ids
        ):
            raise SuiteContractError(f"citation {requested} has invalid reference IDs")
        if len(ids) != len(set(ids)) or not ids:
            raise SuiteContractError(f"citation {requested} has duplicate or no references")
        row["reference_ids"] = sorted(ids)
        if type(row["identity_verified"]) is not bool:
            raise SuiteContractError(f"citation {requested} lacks identity verification")
        if type(row["available"]) is not bool:
            raise SuiteContractError(f"citation {requested} lacks availability")
        row["resolver_artifacts"] = _safe_resolver_artifacts(
            row["resolver_artifacts"],
        )
        row["inspection_chunks"] = _safe_chunks(
            row["inspection_chunks"], label=f"citation {requested} chunks",
        )
        projected.append(row)
    return sorted(projected, key=lambda row: row["requested_url"])


def citation_evidence_surface_sha256(
    citations: Sequence[Mapping[str, Any]],
) -> str:
    """Hash protocol-neutral, body-free citation evidence metadata."""

    payload = {
        "schema_version": CITATION_EVIDENCE_SURFACE_VERSION,
        "citations": _citation_surface_rows(citations),
    }
    return _sha256(payload)


def _file_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: copy.deepcopy(source.get(key)) for key in _FILE_FIELDS}
    path = row.get("path")
    if not isinstance(path, str) or not path:
        raise SuiteContractError("cache file row has no path")
    if row.get("model_inspection_required") is not True:
        raise SuiteContractError(f"assigned source is not model-readable: {path}")
    row["chunks"] = _safe_chunks(row.get("chunks"), label=f"source {path} chunks")
    if not row["chunks"]:
        raise SuiteContractError(f"assigned source has no inspection chunks: {path}")
    return row


def project_shard_manifest(
    *, protocol_id: str, snapshot_inventory: Mapping[str, Any],
    cache_manifest: Mapping[str, Any], plan: Mapping[str, Any], shard_id: str,
) -> dict[str, Any]:
    """Project one body-free shard from a validated full cache manifest."""

    protocol = _protocol_id(protocol_id)
    shard = _find_shard(plan, shard_id)
    snapshot = dict(snapshot_inventory)
    snapshot_hash = review_snapshot_sha256(snapshot)
    if snapshot.get("protocol_id") != protocol:
        raise SuiteContractError("snapshot protocol differs from shard protocol")
    if (cache_manifest.get("review_snapshot") != snapshot
            or cache_manifest.get("review_snapshot_sha256") != snapshot_hash
            or cache_manifest.get("protocol_id") != protocol):
        raise SuiteContractError("full cache does not bind the supplied snapshot")

    source_paths = _require_array(shard.get("source_paths"), label="shard source paths")
    if (any(not isinstance(path, str) or not path for path in source_paths)
            or len(source_paths) != len(set(source_paths))):
        raise SuiteContractError("shard source paths are malformed or duplicated")
    full_files = _require_array(cache_manifest.get("files"), label="cache files")
    files_by_path: dict[str, Mapping[str, Any]] = {}
    for row in full_files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise SuiteContractError("full cache has a malformed file row")
        path = str(row["path"])
        if path in files_by_path:
            raise SuiteContractError(f"full cache duplicates source path {path}")
        files_by_path[path] = row
    missing_sources = sorted(set(source_paths) - set(files_by_path))
    if missing_sources:
        raise SuiteContractError(
            "full cache lacks assigned source paths: " + ", ".join(missing_sources)
        )
    files = [_file_projection(files_by_path[path]) for path in sorted(source_paths)]

    rule_references = _require_array(
        shard.get("rule_references"), label="shard rule references",
    )
    expected_by_url: dict[str, list[str]] = {}
    normalized_references: list[dict[str, Any]] = []
    seen_reference_ids: set[str] = set()
    for row in rule_references:
        if not isinstance(row, dict) or set(row) != {
            "reference_id", "reference_surface_sha256", "rule_id",
            "citation_url", "section",
        }:
            raise SuiteContractError("shard plan has a malformed rule reference")
        reference_id = _require_sha256(
            row.get("reference_id"), label="rule reference ID",
        )
        _require_sha256(
            row.get("reference_surface_sha256"), label="rule reference surface",
        )
        if reference_id in seen_reference_ids:
            raise SuiteContractError("shard plan duplicates a rule reference")
        seen_reference_ids.add(reference_id)
        url = row.get("citation_url")
        if not isinstance(url, str) or not url:
            raise SuiteContractError("shard rule reference has no citation URL")
        expected_by_url.setdefault(url, []).append(reference_id)
        normalized_references.append(copy.deepcopy(row))
    normalized_references.sort(key=lambda row: row["reference_id"])

    full_citations = _require_array(
        cache_manifest.get("citations"), label="cache citations",
    )
    citations_by_url: dict[str, Mapping[str, Any]] = {}
    for row in full_citations:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("requested_url"), str,
        ):
            raise SuiteContractError("full cache has a malformed citation row")
        url = str(row["requested_url"])
        if url in citations_by_url:
            raise SuiteContractError(f"full cache duplicates citation URL {url}")
        citations_by_url[url] = row
    missing_urls = sorted(set(expected_by_url) - set(citations_by_url))
    if missing_urls:
        raise SuiteContractError(
            "full cache lacks assigned citation URLs: " + ", ".join(missing_urls)
        )
    selected: list[dict[str, Any]] = []
    for url in sorted(expected_by_url):
        source = copy.deepcopy(dict(citations_by_url[url]))
        assigned = sorted(expected_by_url[url])
        cached_ids = source.get("reference_ids")
        if not isinstance(cached_ids, list) or not set(assigned).issubset(cached_ids):
            raise SuiteContractError(f"citation {url} lacks an assigned rule reference")
        source["reference_ids"] = assigned
        if source.get("inspection_required") is not True:
            raise SuiteContractError(f"rule citation is not inspection-required: {url}")
        selected.append(source)
    safe_citations = _citation_surface_rows(selected)
    evidence_hash = citation_evidence_surface_sha256(safe_citations)
    full_evidence_hash = citation_evidence_surface_sha256(full_citations)

    plan_hash = shard_plan.shard_plan_sha256(plan)
    budget = copy.deepcopy(plan.get("token_budget_contract"))
    if budget != shard_plan.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict():
        raise SuiteContractError("shard plan uses a stale token-budget contract")
    return {
        "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol,
        "review_snapshot_sha256": snapshot_hash,
        "shard_plan_sha256": plan_hash,
        "shard_id": shard_id,
        "citation_evidence_surface_sha256": evidence_hash,
        "full_citation_evidence_surface_sha256": full_evidence_hash,
        "source_paths": sorted(source_paths),
        "assertion_ids": _assertion_ids(protocol, shard),
        "rule_references": normalized_references,
        "files": files,
        "citations": safe_citations,
        "chunk_contract": copy.deepcopy(cache_manifest.get("chunk_contract")),
        "token_budget_contract": budget,
    }


def _budget_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        method = getattr(value, "to_dict", None)
        if not callable(method):
            raise SuiteContractError("budget contract must be a mapping or expose to_dict")
        result = method()
    if result != shard_plan.DEFAULT_TOKEN_BUDGET_CONTRACT.to_dict():
        raise SuiteContractError("prompt uses a stale token-budget contract")
    return result


def build_shard_prompt(
    *, base_protocol_text: str | bytes, snapshot_inventory: Mapping[str, Any],
    plan_sha256: str, shard_projection: Mapping[str, Any],
    budget_contract: Any = shard_plan.DEFAULT_TOKEN_BUDGET_CONTRACT,
) -> bytes:
    """Build a deterministic prompt that exposes only assigned shard work."""

    if isinstance(base_protocol_text, bytes):
        try:
            base = base_protocol_text.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SuiteContractError("base protocol text is not strict UTF-8") from exc
    elif isinstance(base_protocol_text, str):
        base = base_protocol_text
    else:
        raise SuiteContractError("base protocol text must be str or bytes")
    _require_sha256(plan_sha256, label="shard plan hash")
    projection = copy.deepcopy(dict(shard_projection))
    if (projection.get("schema_version") != SHARD_MANIFEST_SCHEMA_VERSION
            or projection.get("review_snapshot_sha256")
            != review_snapshot_sha256(snapshot_inventory)
            or projection.get("shard_plan_sha256") != plan_sha256):
        raise SuiteContractError("prompt inputs do not bind one shard projection")
    budget = _budget_dict(budget_contract)
    if projection.get("token_budget_contract") != budget:
        raise SuiteContractError("shard projection and prompt budget differ")
    snapshot_identity = {
        key: copy.deepcopy(snapshot_inventory.get(key)) for key in (
            "protocol_id", "review_surface_sha256",
            "implementation_surface_sha256", "citation_audit_sha256",
            "citation_evidence_sha256", "output_schema_sha256",
            "receipt_schema_sha256",
        )
    }
    contract = {
        "schema_version": PROMPT_CONTRACT_VERSION,
        "review_snapshot_sha256": projection["review_snapshot_sha256"],
        "shard_plan_sha256": plan_sha256,
        "shard_id": projection["shard_id"],
        "protocol_id": projection["protocol_id"],
        "citation_evidence_surface_sha256": projection[
            "citation_evidence_surface_sha256"
        ],
        "full_citation_evidence_surface_sha256": projection[
            "full_citation_evidence_surface_sha256"
        ],
        "snapshot_identity": snapshot_identity,
        "budget_contract": budget,
        "assertion_contract": shard_plan.assertion_contract(
            str(projection["protocol_id"]), projection.get("assertion_ids", []),
        ),
        "shard_manifest": projection,
        "instructions": [
            "Before any agent message, use one separate shell call for every manifested chunk path.",
            "Each call must be exactly: head -n 100000000 PATH, with PATH copied verbatim from the manifest.",
            "A rejecting result still requires every assigned chunk read exactly once.",
            "Review exactly shard_manifest.assertion_ids and no other assertions.",
            "Emit citation results only for shard_manifest.rule_references.",
            "Document references are suite-generated and must not be reviewed or emitted.",
            "Do not infer evidence from another shard or an unlisted source path.",
        ],
    }
    separator = "\n\n--- deterministic shard contract ---\n"
    return (base.rstrip() + separator).encode("utf-8") + _pretty_json(contract)


def _review_issues(value: Any) -> list[dict[str, str]]:
    rows = _require_array(value, label="review issues")
    result: list[dict[str, str]] = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != _REVIEW_ISSUE_KEYS
                or row.get("severity") not in _SEVERITIES
                or row.get("code") not in CONTROLLED_REVIEW_ISSUES):
            raise SuiteContractError("shard result contains an uncontrolled review issue")
        result.append(dict(row))
    canonical = sorted(result, key=lambda row: (row["severity"], row["code"]))
    if result != canonical or len({_canonical_json(row) for row in result}) != len(result):
        raise SuiteContractError("shard review issues are duplicated or not sorted")
    return result


def _validate_assertions(
    value: Any, expected_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], bool]:
    rows = _require_array(value, label="assertion_results")
    if (not isinstance(expected_ids, list)
            or any(
                not isinstance(item, str)
                or _ASSERTION_ID_RE.fullmatch(item) is None
                for item in expected_ids
            )
            or len(expected_ids) != len(set(expected_ids))):
        raise SuiteContractError("shard manifest has malformed assertion IDs")
    expected = set(expected_ids)
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ASSERTION_RESULT_KEYS:
            raise SuiteContractError("assertion result is not a closed object")
        assertion_id = row.get("assertion_id")
        if (not isinstance(assertion_id, str)
                or _ASSERTION_ID_RE.fullmatch(assertion_id) is None
                or assertion_id in seen):
            raise SuiteContractError("assertion results contain a duplicate or malformed ID")
        if assertion_id not in expected:
            raise SuiteContractError(f"assertion belongs to another shard: {assertion_id}")
        issues = row.get("issues")
        if (not isinstance(issues, list)
                or any(not isinstance(item, str) for item in issues)
                or len(issues) != len(set(issues))
                or any(item not in CONTROLLED_ASSERTION_ISSUES for item in issues)):
            raise SuiteContractError(f"assertion {assertion_id} has uncontrolled issues")
        verdict = row.get("verdict")
        if verdict not in {"verified", "rejected"}:
            raise SuiteContractError(f"assertion {assertion_id} has an invalid verdict")
        if (verdict == "verified") != (issues == []):
            raise SuiteContractError(f"assertion {assertion_id} verdict and issues disagree")
        seen[assertion_id] = copy.deepcopy(row)
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        raise SuiteContractError("assertion inventory is incomplete: " + ", ".join(missing))
    ordered = [seen[key] for key in sorted(seen)]
    if rows != ordered:
        raise SuiteContractError("assertion results are not canonically sorted")
    return ordered, all(row["verdict"] == "verified" for row in ordered)


def _validate_rule_citations(
    value: Any, shard_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    rows = _require_array(value, label="citation_results")
    expected_rows = shard_manifest.get("rule_references")
    if not isinstance(expected_rows, list):
        raise SuiteContractError("shard manifest has no rule references")
    expected = {row["reference_id"]: row for row in expected_rows}
    cache_by_reference: dict[str, Mapping[str, Any]] = {}
    for citation in shard_manifest.get("citations", []):
        for reference_id in citation.get("reference_ids", []):
            if reference_id in cache_by_reference:
                raise SuiteContractError("shard manifest duplicates a citation reference")
            cache_by_reference[reference_id] = citation
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _CITATION_RESULT_KEYS:
            raise SuiteContractError("rule citation result is not a closed object")
        reference_id = row.get("reference_id")
        if not isinstance(reference_id, str) or reference_id in seen:
            raise SuiteContractError("citation results contain a duplicate or malformed ID")
        expected_row = expected.get(reference_id)
        if expected_row is None:
            raise SuiteContractError(f"citation belongs to another shard: {reference_id}")
        if row.get("reference_surface_sha256") != expected_row.get(
            "reference_surface_sha256"
        ):
            raise SuiteContractError(f"citation {reference_id} has a stale surface")
        issues = row.get("issues")
        if (not isinstance(issues, list)
                or any(not isinstance(item, str) for item in issues)
                or len(issues) != len(set(issues))
                or any(item not in CONTROLLED_CITATION_ISSUES for item in issues)):
            raise SuiteContractError(f"citation {reference_id} has uncontrolled issues")
        verdict = row.get("verdict")
        if verdict not in {"verified", "unavailable", "rejected"}:
            raise SuiteContractError(f"citation {reference_id} has an invalid verdict")
        verified_fields = (
            row.get("exact_locator_inspected") is True
            and row.get("declared_version_matched") is True
            and row.get("declared_section_matched") is True
            and row.get("paraphrase_supported") is True
            and row.get("applicability_not_broader") is True
            and issues == []
        )
        evidence = cache_by_reference.get(reference_id)
        evidence_ready = bool(
            evidence is not None
            and evidence.get("available") is True
            and evidence.get("identity_verified") is True
            and evidence.get("inspection_required") is True
            and evidence.get("inspection_chunks")
        )
        if verdict == "verified" and not (verified_fields and evidence_ready):
            raise SuiteContractError(f"verified citation lacks shard evidence: {reference_id}")
        if verdict != "verified" and not issues:
            raise SuiteContractError(f"non-verified citation has no controlled issue: {reference_id}")
        seen[reference_id] = copy.deepcopy(row)
    if set(seen) != set(expected):
        missing = sorted(set(expected) - set(seen))
        raise SuiteContractError("rule citation inventory is incomplete: " + ", ".join(missing))
    ordered = [seen[key] for key in sorted(seen)]
    if rows != ordered:
        raise SuiteContractError("citation results are not canonically sorted")
    return ordered, all(row["verdict"] == "verified" for row in ordered)


def validate_shard_result(
    result: Mapping[str, Any], shard_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one result against the exact assigned assertion/reference set."""

    if not isinstance(result, Mapping) or set(result) != _SHARD_RESULT_KEYS:
        raise SuiteContractError("shard result does not match the closed contract")
    expected_header = {
        "schema_version": SHARD_RESULT_SCHEMA_VERSION,
        "protocol_id": shard_manifest.get("protocol_id"),
        "review_snapshot_sha256": shard_manifest.get("review_snapshot_sha256"),
        "shard_plan_sha256": shard_manifest.get("shard_plan_sha256"),
        "shard_id": shard_manifest.get("shard_id"),
        "citation_evidence_surface_sha256": shard_manifest.get(
            "citation_evidence_surface_sha256"
        ),
    }
    if any(result.get(key) != value for key, value in expected_header.items()):
        raise SuiteContractError("shard result header is stale or cross-shard")
    assertions, assertions_ok = _validate_assertions(
        result.get("assertion_results"), shard_manifest.get("assertion_ids", []),
    )
    citations, citations_ok = _validate_rule_citations(
        result.get("citation_results"), shard_manifest,
    )
    issues = _review_issues(result.get("issues"))
    approved = assertions_ok and citations_ok and issues == []
    if result.get("approved") is not approved:
        raise SuiteContractError("shard approved is not the pure closed-set conjunction")
    if not approved and not issues:
        raise SuiteContractError("rejected shard has no controlled top-level issue")
    summary = "approved_no_issues" if approved else "rejected_with_controlled_issues"
    if result.get("summary") != summary:
        raise SuiteContractError("shard summary disagrees with approved")
    normalized = copy.deepcopy(dict(result))
    normalized["assertion_results"] = assertions
    normalized["citation_results"] = citations
    normalized["issues"] = issues
    return normalized


def _document_result(
    reference: Mapping[str, Any], evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, str] | None]:
    available = evidence.get("available") is True
    identity = evidence.get("identity_verified") is True
    verified = available and identity
    if verified:
        verdict = "verified"
        citation_issues: list[str] = []
        review_issue = None
    elif not available:
        verdict = "unavailable"
        citation_issues = ["locator_unavailable"]
        review_issue = {"severity": "high", "code": "citation_unavailable"}
    else:
        verdict = "rejected"
        citation_issues = ["resolver_mismatch"]
        review_issue = {"severity": "high", "code": "citation_rejected"}
    return ({
        "reference_id": reference["reference_id"],
        "reference_surface_sha256": reference["reference_surface_sha256"],
        "verdict": verdict,
        "exact_locator_inspected": False,
        "declared_version_matched": verified,
        "declared_section_matched": None,
        "paraphrase_supported": None,
        "applicability_not_broader": None,
        "issues": citation_issues,
    }, verified, review_issue)


def aggregate_shard_results(
    *, protocol_id: str, snapshot_inventory: Mapping[str, Any],
    cache_manifest: Mapping[str, Any], citation_audit: Mapping[str, Any],
    plan: Mapping[str, Any], shard_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly three validated shards into the legacy full result."""

    protocol = _protocol_id(protocol_id)
    rebuilt = shard_plan.build_shard_plan(citation_audit)
    if rebuilt != dict(plan):
        raise SuiteContractError("aggregate shard plan differs from the citation audit")
    by_shard: dict[str, Mapping[str, Any]] = {}
    for result in shard_results:
        if not isinstance(result, Mapping):
            raise SuiteContractError("aggregate received a non-object shard result")
        shard_id = result.get("shard_id")
        if not isinstance(shard_id, str) or shard_id in by_shard:
            raise SuiteContractError("aggregate has a duplicate or malformed shard result")
        by_shard[shard_id] = result
    if set(by_shard) != set(shard_plan.SHARD_ORDER):
        raise SuiteContractError("aggregate must contain exactly the fixed three shards")

    validated: list[dict[str, Any]] = []
    rule_rows: dict[str, dict[str, Any]] = {}
    for shard_id in shard_plan.SHARD_ORDER:
        projection = project_shard_manifest(
            protocol_id=protocol, snapshot_inventory=snapshot_inventory,
            cache_manifest=cache_manifest, plan=plan, shard_id=shard_id,
        )
        result = validate_shard_result(by_shard[shard_id], projection)
        validated.append(result)
        for row in result["citation_results"]:
            reference_id = row["reference_id"]
            if reference_id in rule_rows:
                raise SuiteContractError("one rule reference appears in multiple shards")
            rule_rows[reference_id] = copy.deepcopy(row)

    references = _require_array(citation_audit.get("references"), label="citation references")
    reference_ids: set[str] = set()
    documents: list[Mapping[str, Any]] = []
    rules: list[Mapping[str, Any]] = []
    for row in references:
        if not isinstance(row, Mapping):
            raise SuiteContractError("citation audit contains a non-object reference")
        reference_id = row.get("reference_id")
        if not isinstance(reference_id, str) or reference_id in reference_ids:
            raise SuiteContractError("citation audit has a duplicate reference ID")
        reference_ids.add(reference_id)
        if row.get("reference_kind") == "document":
            documents.append(row)
        elif row.get("reference_kind") == "rule":
            rules.append(row)
        else:
            raise SuiteContractError("citation audit has an unknown reference kind")
    if len(rules) != 38 or len(documents) != 15:
        raise SuiteContractError("suite requires exactly 38 rule and 15 document rows")
    if set(rule_rows) != {str(row["reference_id"]) for row in rules}:
        raise SuiteContractError("aggregate rule inventory is incomplete or cross-shard")

    cache_by_url: dict[str, Mapping[str, Any]] = {}
    for row in _require_array(cache_manifest.get("citations"), label="cache citations"):
        if not isinstance(row, Mapping) or not isinstance(row.get("requested_url"), str):
            raise SuiteContractError("cache contains a malformed citation row")
        url = str(row["requested_url"])
        if url in cache_by_url:
            raise SuiteContractError("cache duplicates a citation URL")
        cache_by_url[url] = row

    all_rows = list(rule_rows.values())
    document_ok = True
    aggregate_issues: list[dict[str, str]] = [
        copy.deepcopy(issue)
        for result in validated for issue in result["issues"]
    ]
    for reference in documents:
        url = reference.get("citation_url")
        evidence = cache_by_url.get(str(url))
        if evidence is None:
            raise SuiteContractError(f"cache lacks document citation {url}")
        row, verified, issue = _document_result(reference, evidence)
        all_rows.append(row)
        document_ok = document_ok and verified
        if issue is not None:
            aggregate_issues.append(issue)
    unique_issues = {
        (row["severity"], row["code"]): row for row in aggregate_issues
    }
    aggregate_issues = [
        unique_issues[key] for key in sorted(unique_issues)
    ]
    approved = all(result["approved"] for result in validated) and document_ok
    snapshot = dict(snapshot_inventory)
    if snapshot.get("protocol_id") != protocol:
        raise SuiteContractError("aggregate snapshot has the wrong protocol")
    result = {
        "protocol_id": protocol,
        "review_surface_sha256": copy.deepcopy(snapshot.get("review_surface_sha256")),
        "implementation_surface_sha256": snapshot.get(
            "implementation_surface_sha256"
        ),
        "citation_audit_sha256": snapshot.get("citation_audit_sha256"),
        "citation_results": sorted(all_rows, key=lambda row: row["reference_id"]),
        "approved": approved,
        "issues": aggregate_issues,
        "summary": (
            "approved_no_issues" if approved
            else "rejected_with_controlled_issues"
        ),
    }
    if set(row["reference_id"] for row in result["citation_results"]) != reference_ids:
        raise SuiteContractError("aggregate citation inventory is not closed")
    if approved != (aggregate_issues == []):
        raise SuiteContractError("aggregate issues disagree with pure conjunction")
    return result


__all__ = [
    "CITATION_EVIDENCE_SURFACE_VERSION",
    "PROMPT_CONTRACT_VERSION",
    "SHARD_MANIFEST_SCHEMA_VERSION",
    "SHARD_RESULT_SCHEMA_VERSION",
    "SuiteContractError",
    "aggregate_shard_results",
    "build_shard_prompt",
    "citation_evidence_surface_sha256",
    "project_shard_manifest",
    "review_snapshot_sha256",
    "validate_shard_result",
]
