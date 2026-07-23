"""Content-bound validation for protected canonical static features.

The legacy :class:`~hlsgraph.model.Derivation` identifier intentionally does
not include value or metadata, because changing that identity would rewrite
v0.1/v0.2 evidence lineages.  A complete v0.3 aggregate therefore carries a
separate receipt.  The receipt is useful only when an index commit also proves
that its producer/domain proof came from the extraction pipeline; serialized
graph attributes alone never authorize a producer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from .graph import CanonicalGraph
from .model import (
    ArtifactRef,
    Completeness,
    Derivation,
    EvidenceKind,
    json_ready,
    stable_hash,
    stable_id,
)


STANDARD_STATIC_AGGREGATE_PREDICATES = frozenset({
    "feature.operation_histogram",
    "feature.index_histogram",
    "feature.bitwidth",
    "feature.memory_access",
    "feature.trip_count",
    "feature.loop_bounds",
    "feature.dependence_distance",
})
STATIC_AGGREGATE_RECEIPT_CONTRACT = "hlsgraph.static_aggregate_receipt.v1"
_RECEIPT_METADATA_KEY = "static_aggregate_receipt"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PARSER_CONTRACTS = {
    ("mlir", "mlir", "ir.mlir_text", "3"):
        "hlsgraph.ir.mlir_text.static_feature_domain.v1",
    ("llvm", "llvm", "ir.llvm_text", "2"):
        "hlsgraph.ir.llvm_text.static_feature_domain.v1",
}


def static_aggregate_receipt_required(
    value: Derivation | Mapping[str, Any],
) -> bool:
    """Return whether a row makes the complete-value claim receipts protect."""

    if isinstance(value, Derivation):
        predicate = value.predicate
        completeness = str(value.completeness)
        payload = value.value
    else:
        predicate = value.get("predicate")
        completeness = str(value.get("completeness"))
        payload = value.get("value")
    return bool(
        predicate in STANDARD_STATIC_AGGREGATE_PREDICATES
        and completeness == Completeness.COMPLETE.value
        and payload is not None
    )


class StaticAggregateError(ValueError):
    """Raised when an aggregate cannot close to its exact evidence domain."""


def _sha256(value: str, label: str) -> str:
    digest = str(value).casefold()
    if _SHA256.fullmatch(digest) is None:
        raise StaticAggregateError(f"{label} must be a lowercase SHA-256")
    return digest


def _sorted_unique(values: tuple[str, ...] | list[str], label: str) -> tuple[str, ...]:
    result = tuple(sorted(str(item) for item in values))
    if not result or any(not item for item in result):
        raise StaticAggregateError(f"{label} must contain non-empty identifiers")
    if len(result) != len(set(result)):
        raise StaticAggregateError(f"{label} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class StaticFeatureDomainProof:
    """Serializable domain identity prepared from pipeline-private origins."""

    snapshot_id: str
    scope_id: str
    layer: str
    stage: str
    parser_name: str
    parser_version: str
    parser_contract: str
    extractor_identity_sha256: str
    origin_manifest_sha256: str
    artifact_hashes: tuple[tuple[str, str], ...]
    entity_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    contract: str = "hlsgraph.static_feature_domain_proof.v1"
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.scope_id:
            raise StaticAggregateError("domain proof requires snapshot and scope IDs")
        expected = _PARSER_CONTRACTS.get((
            self.layer, self.stage, self.parser_name, self.parser_version,
        ))
        if expected is None or self.parser_contract != expected:
            raise StaticAggregateError("domain proof uses an untrusted parser contract")
        object.__setattr__(
            self, "extractor_identity_sha256",
            _sha256(self.extractor_identity_sha256, "extractor identity"),
        )
        object.__setattr__(
            self, "origin_manifest_sha256",
            _sha256(self.origin_manifest_sha256, "origin manifest"),
        )
        artifacts = tuple(sorted(
            (str(identifier), _sha256(digest, "artifact hash"))
            for identifier, digest in self.artifact_hashes
        ))
        if not artifacts or any(not identifier for identifier, _digest in artifacts):
            raise StaticAggregateError("domain proof requires artifact identities")
        if len({identifier for identifier, _digest in artifacts}) != len(artifacts):
            raise StaticAggregateError("domain proof artifact identities must be unique")
        object.__setattr__(self, "artifact_hashes", artifacts)
        object.__setattr__(
            self, "entity_ids", _sorted_unique(self.entity_ids, "domain entities"),
        )
        relations = tuple(sorted(str(item) for item in self.relation_ids))
        if any(not item for item in relations) or len(relations) != len(set(relations)):
            raise StaticAggregateError("domain relation identifiers must be unique")
        object.__setattr__(self, "relation_ids", relations)
        if self.scope_id not in self.entity_ids:
            raise StaticAggregateError("domain proof must include its scope entity")
        if self.contract != "hlsgraph.static_feature_domain_proof.v1":
            raise StaticAggregateError("unsupported static feature domain proof")
        identity = json_ready(self)
        identity["id"] = ""
        expected_id = stable_id("static_feature_domain_proof", identity)
        if self.id and self.id != expected_id:
            raise StaticAggregateError("static feature domain proof ID is invalid")
        object.__setattr__(self, "id", expected_id)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StaticFeatureDomainProof":
        data = dict(value)
        data["artifact_hashes"] = tuple(
            (str(item[0]), str(item[1])) for item in data.get("artifact_hashes", [])
        )
        data["entity_ids"] = tuple(str(item) for item in data.get("entity_ids", []))
        data["relation_ids"] = tuple(str(item) for item in data.get("relation_ids", []))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class StaticAggregateReceipt:
    """Content identity of one exactly recomputed complete aggregate."""

    snapshot_id: str
    derivation_id: str
    subject_id: str
    predicate: str
    graph_hash: str
    semantic_payload_sha256: str
    evidence_ref_sha256: str
    domain_proof: StaticFeatureDomainProof
    contract: str = STATIC_AGGREGATE_RECEIPT_CONTRACT
    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.predicate not in STANDARD_STATIC_AGGREGATE_PREDICATES:
            raise StaticAggregateError("receipt predicate is not a standard aggregate")
        if (self.snapshot_id != self.domain_proof.snapshot_id
                or self.subject_id != self.domain_proof.scope_id):
            raise StaticAggregateError("receipt and domain proof identities differ")
        for name in ("graph_hash", "semantic_payload_sha256", "evidence_ref_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.contract != STATIC_AGGREGATE_RECEIPT_CONTRACT:
            raise StaticAggregateError("unsupported static aggregate receipt")
        identity = json_ready(self)
        identity["id"] = ""
        expected_id = stable_id("static_aggregate_receipt", identity)
        if self.id and self.id != expected_id:
            raise StaticAggregateError("static aggregate receipt ID is invalid")
        object.__setattr__(self, "id", expected_id)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StaticAggregateReceipt":
        data = dict(value)
        data["domain_proof"] = StaticFeatureDomainProof.from_dict(
            data["domain_proof"]
        )
        return cls(**data)


def _without_receipt(value: Derivation | Mapping[str, Any]) -> dict[str, Any]:
    payload = json_ready(value)
    metadata = dict(payload.get("metadata", {}))
    metadata.pop(_RECEIPT_METADATA_KEY, None)
    payload["metadata"] = metadata
    return payload


def _expected_derivation(
    graph: CanonicalGraph, subject_id: str, predicate: str,
) -> Derivation:
    # Import lazily to keep the extraction pass independent of the read-side
    # verifier and to ensure there is only one aggregation implementation.
    from .extract.base import ExtractionResult
    from .extract.static_features import derive_static_features

    recomputed = ExtractionResult(graph=graph)
    derive_static_features(recomputed)
    matches = [
        item for item in recomputed.derivations
        if item.subject_id == subject_id and item.predicate == predicate
    ]
    if len(matches) != 1:
        raise StaticAggregateError(
            "standard aggregate does not have one deterministic recomputation"
        )
    return matches[0]


def _artifact_map(
    artifacts: Mapping[str, ArtifactRef] | list[ArtifactRef],
) -> dict[str, ArtifactRef]:
    values = (
        dict(artifacts) if isinstance(artifacts, Mapping)
        else {item.id: item for item in artifacts}
    )
    if any(not isinstance(item, ArtifactRef) for item in values.values()):
        raise StaticAggregateError("aggregate artifacts must be ArtifactRef values")
    return values


def validate_static_aggregate(
    graph: CanonicalGraph,
    derivation: Derivation | Mapping[str, Any],
    artifacts: Mapping[str, ArtifactRef] | list[ArtifactRef],
    domain_proof: StaticFeatureDomainProof | Mapping[str, Any],
    *,
    _expected: Derivation | None = None,
) -> StaticAggregateReceipt:
    """Recompute and content-bind one complete standard aggregate.

    This validates semantic content and exact evidence closure.  The caller
    must separately prove that ``domain_proof`` came from a pipeline-private
    extractor origin; a JSON round-trip of the proof is not authorization.
    """

    item = (
        derivation if isinstance(derivation, Derivation)
        else Derivation.from_dict(derivation)
    )
    proof = (
        domain_proof if isinstance(domain_proof, StaticFeatureDomainProof)
        else StaticFeatureDomainProof.from_dict(domain_proof)
    )
    if item.predicate not in STANDARD_STATIC_AGGREGATE_PREDICATES:
        raise StaticAggregateError("derivation is not a standard aggregate")
    if (item.snapshot_id != graph.snapshot_id
            or item.snapshot_id != proof.snapshot_id
            or item.subject_id != proof.scope_id):
        raise StaticAggregateError("aggregate snapshot/scope identity mismatch")
    if item.completeness != Completeness.COMPLETE or item.value is None:
        raise StaticAggregateError("only complete, known aggregates receive receipts")

    expected = _expected or _expected_derivation(
        graph, item.subject_id, item.predicate,
    )
    if (
        expected.snapshot_id != item.snapshot_id
        or expected.subject_id != item.subject_id
        or expected.predicate != item.predicate
    ):
        raise StaticAggregateError(
            "precomputed aggregate belongs to another snapshot/scope/predicate"
        )
    actual_payload = _without_receipt(item)
    expected_payload = _without_receipt(expected)
    if actual_payload != expected_payload:
        raise StaticAggregateError(
            "aggregate payload differs from deterministic recomputation"
        )

    refs = item.evidence_refs
    entity_ids = tuple(sorted(
        ref.target_id for ref in refs if ref.kind == EvidenceKind.ENTITY_ANCHOR
    ))
    relation_ids = tuple(sorted(
        ref.target_id for ref in refs if ref.kind == EvidenceKind.RELATION
    ))
    artifact_ids = tuple(sorted(
        ref.target_id for ref in refs if ref.kind == EvidenceKind.ARTIFACT
    ))
    if entity_ids != proof.entity_ids or relation_ids != proof.relation_ids:
        raise StaticAggregateError("domain proof does not cover exact evidence refs")
    if any(identifier not in graph.entities for identifier in entity_ids):
        raise StaticAggregateError("aggregate cites an unavailable entity")
    if any(identifier not in graph.relations for identifier in relation_ids):
        raise StaticAggregateError("aggregate cites an unavailable relation")

    by_artifact = _artifact_map(artifacts)
    expected_artifacts = tuple(
        (identifier, by_artifact[identifier].sha256)
        for identifier in artifact_ids if identifier in by_artifact
    )
    if len(expected_artifacts) != len(artifact_ids):
        raise StaticAggregateError("aggregate cites an unavailable artifact")
    if expected_artifacts != proof.artifact_hashes:
        raise StaticAggregateError("domain proof artifact bytes do not match evidence")

    for identifier in entity_ids:
        entity = graph.entities[identifier]
        if (entity.attrs.get("static_feature_parser") != proof.parser_name
                or entity.attrs.get("static_feature_parser_version")
                != proof.parser_version
                or entity.attrs.get("static_feature_domain_contract")
                != proof.parser_contract
                or entity.attrs.get("static_feature_domain_complete") is not True
                or entity.attrs.get("static_feature_unparsed_construct_count") != 0
                or entity.attrs.get("static_feature_artifact_id")
                not in artifact_ids):
            raise StaticAggregateError(
                "aggregate entity lacks the exact complete parser-domain contract"
            )
    evidence_payload = [json_ready(ref) for ref in refs]
    return StaticAggregateReceipt(
        snapshot_id=item.snapshot_id,
        derivation_id=item.id,
        subject_id=item.subject_id,
        predicate=item.predicate,
        graph_hash=graph.graph_hash,
        semantic_payload_sha256=stable_hash(actual_payload),
        evidence_ref_sha256=stable_hash(evidence_payload),
        domain_proof=proof,
    )


def attach_static_aggregate_receipt(
    derivation: Derivation, receipt: StaticAggregateReceipt,
) -> Derivation:
    """Return the legacy-ID-compatible derivation with its receipt embedded."""

    if (derivation.id != receipt.derivation_id
            or derivation.snapshot_id != receipt.snapshot_id
            or derivation.subject_id != receipt.subject_id
            or derivation.predicate != receipt.predicate):
        raise StaticAggregateError("receipt belongs to another derivation")
    payload = json_ready(derivation)
    metadata = dict(payload.get("metadata", {}))
    metadata[_RECEIPT_METADATA_KEY] = json_ready(receipt)
    payload["metadata"] = metadata
    rebuilt = Derivation.from_dict(payload)
    if rebuilt.id != derivation.id:
        raise StaticAggregateError("receipt changed the legacy derivation identity")
    return rebuilt


def verify_static_aggregate_receipt(
    graph: CanonicalGraph,
    derivation: Derivation | Mapping[str, Any],
    artifacts: Mapping[str, ArtifactRef] | list[ArtifactRef],
) -> StaticAggregateReceipt:
    """Revalidate an embedded receipt against current graph and artifact state."""

    item = (
        derivation if isinstance(derivation, Derivation)
        else Derivation.from_dict(derivation)
    )
    raw = item.metadata.get(_RECEIPT_METADATA_KEY)
    if not isinstance(raw, Mapping):
        raise StaticAggregateError("complete standard aggregate has no receipt")
    claimed = StaticAggregateReceipt.from_dict(raw)
    actual = validate_static_aggregate(
        graph, item, artifacts, claimed.domain_proof,
    )
    if json_ready(actual) != json_ready(claimed):
        raise StaticAggregateError("static aggregate receipt does not revalidate")
    return actual


def verify_static_aggregate_receipts(
    graph: CanonicalGraph,
    derivations: list[Derivation] | tuple[Derivation, ...],
    artifacts: Mapping[str, ArtifactRef] | list[ArtifactRef],
) -> dict[str, StaticAggregateReceipt]:
    """Batch-recompute and verify complete standard aggregates once per graph.

    Calling :func:`verify_static_aggregate_receipt` independently for every
    scope would rerun the complete static-feature pass for every row.  This
    helper preserves identical validation semantics while sharing one
    deterministic recomputation across the snapshot.
    """

    from .extract.base import ExtractionResult
    from .extract.static_features import derive_static_features

    recomputed = ExtractionResult(graph=graph)
    derive_static_features(recomputed)
    expected: dict[tuple[str, str], Derivation] = {}
    for item in recomputed.derivations:
        key = (item.subject_id, item.predicate)
        if key in expected:
            raise StaticAggregateError(
                "deterministic recomputation produced duplicate aggregate keys"
            )
        expected[key] = item

    verified: dict[str, StaticAggregateReceipt] = {}
    for item in derivations:
        if (
            item.predicate not in STANDARD_STATIC_AGGREGATE_PREDICATES
            or item.completeness != Completeness.COMPLETE
            or item.value is None
        ):
            continue
        raw = item.metadata.get(_RECEIPT_METADATA_KEY)
        if not isinstance(raw, Mapping):
            raise StaticAggregateError(
                "complete standard aggregate has no receipt"
            )
        claimed = StaticAggregateReceipt.from_dict(raw)
        selected = expected.get((item.subject_id, item.predicate))
        if selected is None:
            raise StaticAggregateError(
                "standard aggregate has no deterministic recomputation"
            )
        actual = validate_static_aggregate(
            graph,
            item,
            artifacts,
            claimed.domain_proof,
            _expected=selected,
        )
        if json_ready(actual) != json_ready(claimed):
            raise StaticAggregateError(
                "static aggregate receipt does not revalidate"
            )
        if item.id in verified:
            raise StaticAggregateError(
                "duplicate complete standard aggregate identifier"
            )
        verified[item.id] = actual
    return verified


__all__ = [
    "STANDARD_STATIC_AGGREGATE_PREDICATES",
    "STATIC_AGGREGATE_RECEIPT_CONTRACT",
    "StaticAggregateError",
    "StaticAggregateReceipt",
    "StaticFeatureDomainProof",
    "attach_static_aggregate_receipt",
    "static_aggregate_receipt_required",
    "validate_static_aggregate",
    "verify_static_aggregate_receipt",
    "verify_static_aggregate_receipts",
]
