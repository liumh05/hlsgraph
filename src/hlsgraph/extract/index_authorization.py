"""Pipeline-private authorization for content-bound index commits.

Serialized parser attributes are evidence, not capabilities.  This module
records which live extractor object produced each graph record, then converts
that process-local origin map into one-shot authorization only after the SDK
has finalized the graph and recomputed every complete standard aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass
import secrets
import threading
from typing import Any, Mapping

from ..graph import CanonicalGraph
from ..model import (
    ArtifactRef,
    Completeness,
    Derivation,
    EvidenceKind,
    json_ready,
    stable_hash,
    stable_id,
)
from ..static_aggregate import (
    STANDARD_STATIC_AGGREGATE_PREDICATES,
    StaticAggregateError,
    StaticAggregateReceipt,
    StaticFeatureDomainProof,
    attach_static_aggregate_receipt,
    validate_static_aggregate,
    verify_static_aggregate_receipts,
)


INDEX_COMMIT_RECEIPT_CONTRACT = "hlsgraph.index_commit_receipt.v1"
_TRUSTED_STATIC_PARSERS = {
    (
        "ir.mlir_text", "3", "hlsgraph.extract.mlir", "MlirTextExtractor",
    ): ("mlir", "mlir", "hlsgraph.ir.mlir_text.static_feature_domain.v1"),
    (
        "ir.llvm_text", "2", "hlsgraph.extract.llvm", "LlvmIrExtractor",
    ): ("llvm", "llvm", "hlsgraph.ir.llvm_text.static_feature_domain.v1"),
}
_SENTINEL = object()
_LOCK = threading.Lock()
_ACTIVE_ORIGIN_CAPABILITIES: dict[str, object] = {}
_ACTIVE_INDEX_AUTHORIZATIONS: dict[str, object] = {}


class IndexAuthorizationError(ValueError):
    """Raised when a caller cannot prove a pipeline-issued index candidate."""


@dataclass(frozen=True, slots=True)
class _OriginRecord:
    target_kind: str
    target_id: str
    payload_sha256: str
    extractor_identity_sha256: str
    extractor_name: str
    extractor_version: str
    implementation_module: str
    implementation_qualname: str
    artifact_ids: tuple[str, ...]
    authorized_parser_key: tuple[str, str, str, str] | None

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


class _OriginAccumulator:
    __slots__ = ("snapshot_id", "records", "_seal")

    def __init__(self, snapshot_id: str, *, _sentinel: object) -> None:
        if _sentinel is not _SENTINEL:
            raise IndexAuthorizationError("origin accumulator is pipeline-internal")
        self.snapshot_id = snapshot_id
        self.records: dict[tuple[str, str], list[_OriginRecord]] = {}
        self._seal = object()


class _OriginCapability:
    __slots__ = ("snapshot_id", "records", "candidate_graph_content_sha256",
                 "nonce", "_seal")

    def __init__(
        self, *, snapshot_id: str,
        records: dict[tuple[str, str], tuple[_OriginRecord, ...]],
        candidate_graph_content_sha256: str, _sentinel: object,
    ) -> None:
        if _sentinel is not _SENTINEL:
            raise IndexAuthorizationError("origin capability is pipeline-internal")
        self.snapshot_id = snapshot_id
        self.records = records
        self.candidate_graph_content_sha256 = candidate_graph_content_sha256
        self.nonce = secrets.token_hex(32)
        self._seal = object()

    def __copy__(self) -> object:  # pragma: no cover - defensive guard
        raise TypeError("origin capability is non-copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("origin capability is non-copyable")

    def __reduce__(self) -> object:
        raise TypeError("origin capability is non-serializable")


class _IndexCommitAuthorization:
    __slots__ = (
        "snapshot_id", "graph_hash", "artifact_hashes",
        "standard_derivation_hashes", "receipt_hashes",
        "origin_manifest_sha256", "nonce", "_seal",
    )

    def __init__(
        self, *, snapshot_id: str, graph_hash: str,
        artifact_hashes: tuple[tuple[str, str], ...],
        standard_derivation_hashes: tuple[tuple[str, str], ...],
        receipt_hashes: tuple[tuple[str, str], ...],
        origin_manifest_sha256: str, _sentinel: object,
    ) -> None:
        if _sentinel is not _SENTINEL:
            raise IndexAuthorizationError("index authorization is pipeline-internal")
        self.snapshot_id = snapshot_id
        self.graph_hash = graph_hash
        self.artifact_hashes = artifact_hashes
        self.standard_derivation_hashes = standard_derivation_hashes
        self.receipt_hashes = receipt_hashes
        self.origin_manifest_sha256 = origin_manifest_sha256
        self.nonce = secrets.token_hex(32)
        self._seal = object()

    def __copy__(self) -> object:  # pragma: no cover - defensive guard
        raise TypeError("index authorization is non-copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("index authorization is non-copyable")

    def __reduce__(self) -> object:
        raise TypeError("index authorization is non-serializable")


def _graph_content_hash(graph: CanonicalGraph) -> str:
    """Hash graph evidence without mutable presentation metadata."""

    return stable_hash({
        "schema_version": graph.schema_version,
        "snapshot_id": graph.snapshot_id,
        "entities": [
            json_ready(graph.entities[key]) for key in sorted(graph.entities)
        ],
        "relations": [
            json_ready(graph.relations[key]) for key in sorted(graph.relations)
        ],
    })


def _new_origin_accumulator(snapshot_id: str) -> _OriginAccumulator:
    return _OriginAccumulator(snapshot_id, _sentinel=_SENTINEL)


def _record_extractor_result(
    accumulator: _OriginAccumulator,
    graph: CanonicalGraph,
    extractor_identity: Mapping[str, Any],
    extractor: object,
) -> None:
    """Record actual runtime output ownership before graphs are merged."""

    if graph.snapshot_id != accumulator.snapshot_id:
        raise IndexAuthorizationError("extractor result belongs to another snapshot")
    identity = {
        "name": str(extractor_identity.get("name", "")),
        "version": str(extractor_identity.get("version", "")),
        "implementation_module": str(
            extractor_identity.get("implementation_module", "")
        ),
        "implementation_qualname": str(
            extractor_identity.get("implementation_qualname", "")
        ),
        **({
            "runtime": extractor_identity["runtime"],
        } if "runtime" in extractor_identity else {}),
    }
    identity_sha256 = stable_hash(identity)
    # Strings supplied by a plugin are evidence, not authority.  Only the
    # exact built-in class object may close a complete parser domain.  This is
    # an integrity boundary against accidental/forged plugin output, not a
    # sandbox against arbitrary Python code already executing in-process.
    from .llvm import LlvmIrExtractor
    from .mlir import MlirTextExtractor

    authorized_parser_key: tuple[str, str, str, str] | None = None
    if type(extractor) is LlvmIrExtractor:
        authorized_parser_key = (
            "ir.llvm_text", "2", "hlsgraph.extract.llvm", "LlvmIrExtractor",
        )
    elif type(extractor) is MlirTextExtractor:
        authorized_parser_key = (
            "ir.mlir_text", "3", "hlsgraph.extract.mlir", "MlirTextExtractor",
        )
    if authorized_parser_key is not None and (
        identity["name"], identity["version"],
        identity["implementation_module"], identity["implementation_qualname"],
    ) != authorized_parser_key:
        authorized_parser_key = None

    def artifact_ids_for(target: Any) -> tuple[str, ...]:
        identifiers = {anchor.artifact_id for anchor in target.anchors}
        if hasattr(target, "src") and hasattr(target, "dst"):
            for endpoint in (graph.entities.get(target.src),
                             graph.entities.get(target.dst)):
                if endpoint is not None:
                    identifiers.update(
                        anchor.artifact_id for anchor in endpoint.anchors
                    )
        return tuple(sorted(identifiers))

    common = {
        "extractor_identity_sha256": identity_sha256,
        "extractor_name": identity["name"],
        "extractor_version": identity["version"],
        "implementation_module": identity["implementation_module"],
        "implementation_qualname": identity["implementation_qualname"],
        "authorized_parser_key": authorized_parser_key,
    }
    for kind, values in (("entity", graph.entities), ("relation", graph.relations)):
        for identifier, target in values.items():
            record = _OriginRecord(
                target_kind=kind,
                target_id=identifier,
                payload_sha256=stable_hash(json_ready(target)),
                artifact_ids=artifact_ids_for(target),
                **common,
            )
            accumulator.records.setdefault((kind, identifier), []).append(record)


def _issue_origin_capability(
    accumulator: _OriginAccumulator, graph: CanonicalGraph,
) -> _OriginCapability:
    if graph.snapshot_id != accumulator.snapshot_id:
        raise IndexAuthorizationError("merged graph belongs to another snapshot")
    records = {
        key: tuple(sorted(values, key=lambda item: stable_hash(item.to_dict())))
        for key, values in accumulator.records.items()
    }
    capability = _OriginCapability(
        snapshot_id=graph.snapshot_id,
        records=records,
        candidate_graph_content_sha256=_graph_content_hash(graph),
        _sentinel=_SENTINEL,
    )
    with _LOCK:
        _ACTIVE_ORIGIN_CAPABILITIES[capability.nonce] = capability
    return capability


def _consume_origin_capability(
    value: object, graph: CanonicalGraph,
) -> _OriginCapability:
    if type(value) is not _OriginCapability:
        raise IndexAuthorizationError(
            "index candidate requires an ExtractionPipeline-issued origin capability"
        )
    capability = value
    with _LOCK:
        expected = _ACTIVE_ORIGIN_CAPABILITIES.pop(capability.nonce, None)
    if expected is not capability:
        raise IndexAuthorizationError(
            "origin capability is unknown, copied, or already consumed"
        )
    if (capability.snapshot_id != graph.snapshot_id
            or capability.candidate_graph_content_sha256
            != _graph_content_hash(graph)):
        raise IndexAuthorizationError("graph evidence changed after extraction")
    return capability


def _origin_for(
    capability: _OriginCapability, kind: str, identifier: str,
    graph: CanonicalGraph,
) -> _OriginRecord:
    values = capability.records.get((kind, identifier), ())
    if len(values) != 1:
        raise IndexAuthorizationError(
            f"{kind} {identifier} has ambiguous or missing extractor origin"
        )
    origin = values[0]
    target = (
        graph.entities.get(identifier) if kind == "entity"
        else graph.relations.get(identifier)
    )
    if target is None or stable_hash(json_ready(target)) != origin.payload_sha256:
        raise IndexAuthorizationError(f"{kind} {identifier} changed after extraction")
    return origin


def _domain_proof(
    capability: _OriginCapability,
    graph: CanonicalGraph,
    derivation: Derivation,
    artifacts: Mapping[str, ArtifactRef],
) -> StaticFeatureDomainProof:
    entity_ids = tuple(sorted(
        item.target_id for item in derivation.evidence_refs
        if item.kind == EvidenceKind.ENTITY_ANCHOR
    ))
    relation_ids = tuple(sorted(
        item.target_id for item in derivation.evidence_refs
        if item.kind == EvidenceKind.RELATION
    ))
    artifact_ids = tuple(sorted(
        item.target_id for item in derivation.evidence_refs
        if item.kind == EvidenceKind.ARTIFACT
    ))
    origins = [
        *(_origin_for(capability, "entity", item, graph) for item in entity_ids),
        *(_origin_for(capability, "relation", item, graph) for item in relation_ids),
    ]
    producer_keys = {
        (
            item.extractor_name, item.extractor_version,
            item.implementation_module, item.implementation_qualname,
        )
        for item in origins
    }
    identity_hashes = {item.extractor_identity_sha256 for item in origins}
    authorized_parser_keys = {item.authorized_parser_key for item in origins}
    if len(producer_keys) != 1 or len(identity_hashes) != 1:
        raise IndexAuthorizationError(
            "aggregate evidence does not close to one extractor identity"
        )
    producer = next(iter(producer_keys))
    if authorized_parser_keys != {producer}:
        raise IndexAuthorizationError(
            "aggregate producer was not the exact authorized built-in parser"
        )
    parser = _TRUSTED_STATIC_PARSERS.get(producer)
    if parser is None:
        raise IndexAuthorizationError(
            "aggregate producer is not an authorized built-in parser"
        )
    layer, stage, parser_contract = parser
    if any(identifier not in artifacts for identifier in artifact_ids):
        raise IndexAuthorizationError("aggregate origin cites an unavailable artifact")
    if any(
        not set(origin.artifact_ids).issubset(set(artifact_ids))
        for origin in origins
    ):
        raise IndexAuthorizationError(
            "aggregate origin artifacts differ from derivation evidence"
        )
    origin_rows = [item.to_dict() for item in origins]
    return StaticFeatureDomainProof(
        snapshot_id=graph.snapshot_id,
        scope_id=derivation.subject_id,
        layer=layer,
        stage=stage,
        parser_name=producer[0],
        parser_version=producer[1],
        parser_contract=parser_contract,
        extractor_identity_sha256=next(iter(identity_hashes)),
        origin_manifest_sha256=stable_hash(origin_rows),
        artifact_hashes=tuple(
            (identifier, artifacts[identifier].sha256)
            for identifier in artifact_ids
        ),
        entity_ids=entity_ids,
        relation_ids=relation_ids,
    )


def _downgrade_untrusted_aggregate(
    derivation: Derivation, reason: str,
) -> Derivation:
    payload = json_ready(derivation)
    value = payload.get("value")
    known = value is not None
    payload["value"] = value if known else None
    payload["completeness"] = (
        Completeness.PARTIAL.value if known else Completeness.MISSING.value
    )
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("static_aggregate_receipt", None)
    for key in tuple(metadata):
        if key.endswith("_domain_complete"):
            metadata.pop(key)
    metadata["static_aggregate_receipt_status"] = "withheld"
    metadata["static_aggregate_receipt_reason"] = reason
    payload["metadata"] = metadata
    return Derivation.from_dict(payload)


def _finalize_index_authorization(
    origin_capability: object,
    graph: CanonicalGraph,
    derivations: list[Derivation],
    artifacts: Mapping[str, ArtifactRef] | list[ArtifactRef],
) -> tuple[list[Derivation], object | None, list[str]]:
    """Attach valid receipts and issue a one-use ledger authorization."""

    capability = _consume_origin_capability(origin_capability, graph)
    by_artifact = (
        dict(artifacts) if isinstance(artifacts, Mapping)
        else {item.id: item for item in artifacts}
    )
    finalized: list[Derivation] = []
    withheld: list[str] = []
    receipt_hashes: list[tuple[str, str]] = []
    origin_hashes: list[str] = []
    for item in derivations:
        if item.predicate not in STANDARD_STATIC_AGGREGATE_PREDICATES:
            finalized.append(item)
            continue
        if item.completeness == Completeness.COMPLETE and item.value is None:
            finalized.append(
                _downgrade_untrusted_aggregate(
                    item, "InvalidCompleteNullClaim",
                )
            )
            withheld.append(item.id)
            continue
        if item.completeness != Completeness.COMPLETE:
            finalized.append(item)
            continue
        try:
            proof = _domain_proof(
                capability, graph, item, by_artifact,
            )
            receipt = validate_static_aggregate(
                graph, item, by_artifact, proof,
            )
            attached = attach_static_aggregate_receipt(item, receipt)
        except (IndexAuthorizationError, StaticAggregateError, KeyError) as exc:
            finalized.append(
                _downgrade_untrusted_aggregate(item, type(exc).__name__)
            )
            withheld.append(item.id)
            continue
        finalized.append(attached)
        receipt_hashes.append((receipt.id, stable_hash(json_ready(receipt))))
        origin_hashes.append(receipt.domain_proof.origin_manifest_sha256)

    standard_derivation_hashes = tuple(sorted(
        (item.id, stable_hash(json_ready(item)))
        for item in finalized
        if item.predicate in STANDARD_STATIC_AGGREGATE_PREDICATES
        and item.completeness == Completeness.COMPLETE
        and item.value is not None
    ))
    if len(standard_derivation_hashes) != len(receipt_hashes):
        raise IndexAuthorizationError(
            "every complete standard aggregate must have exactly one receipt"
        )
    if not standard_derivation_hashes:
        return finalized, None, sorted(withheld)
    authorization = _IndexCommitAuthorization(
        snapshot_id=graph.snapshot_id,
        graph_hash=graph.graph_hash,
        artifact_hashes=tuple(sorted(
            (identifier, item.sha256) for identifier, item in by_artifact.items()
        )),
        standard_derivation_hashes=standard_derivation_hashes,
        receipt_hashes=tuple(sorted(receipt_hashes)),
        origin_manifest_sha256=stable_hash(sorted(origin_hashes)),
        _sentinel=_SENTINEL,
    )
    with _LOCK:
        _ACTIVE_INDEX_AUTHORIZATIONS[authorization.nonce] = authorization
    return finalized, authorization, sorted(withheld)


def _consume_index_authorization(
    value: object,
    graph: CanonicalGraph,
    derivations: list[Derivation],
    artifacts: Mapping[str, ArtifactRef] | list[ArtifactRef],
) -> dict[str, Any]:
    """Consume the exact authorization and return its public receipt payload."""

    if type(value) is not _IndexCommitAuthorization:
        raise IndexAuthorizationError(
            "complete static aggregates require pipeline index authorization"
        )
    authorization = value
    with _LOCK:
        expected = _ACTIVE_INDEX_AUTHORIZATIONS.pop(
            authorization.nonce, None,
        )
    if expected is not authorization:
        raise IndexAuthorizationError(
            "index authorization is unknown, copied, or already consumed"
        )
    by_artifact = (
        dict(artifacts) if isinstance(artifacts, Mapping)
        else {item.id: item for item in artifacts}
    )
    actual_artifacts = tuple(sorted(
        (identifier, item.sha256) for identifier, item in by_artifact.items()
    ))
    actual_derivations = tuple(sorted(
        (item.id, stable_hash(json_ready(item)))
        for item in derivations
        if item.predicate in STANDARD_STATIC_AGGREGATE_PREDICATES
        and item.completeness == Completeness.COMPLETE
        and item.value is not None
    ))
    complete_aggregates = [
        item for item in derivations
        if item.predicate in STANDARD_STATIC_AGGREGATE_PREDICATES
        and item.completeness == Completeness.COMPLETE
        and item.value is not None
    ]
    verified_receipts = verify_static_aggregate_receipts(
        graph, complete_aggregates, by_artifact,
    )
    if set(verified_receipts) != {item.id for item in complete_aggregates}:
        raise IndexAuthorizationError(
            "complete aggregate receipt verification is incomplete"
        )
    actual_receipts: list[tuple[str, str]] = []
    for item in complete_aggregates:
        receipt = verified_receipts[item.id]
        actual_receipts.append((receipt.id, stable_hash(json_ready(receipt))))
    if (
        authorization.snapshot_id != graph.snapshot_id
        or authorization.graph_hash != graph.graph_hash
        or authorization.artifact_hashes != actual_artifacts
        or authorization.standard_derivation_hashes != actual_derivations
        or authorization.receipt_hashes != tuple(sorted(actual_receipts))
    ):
        raise IndexAuthorizationError(
            "index candidate changed after pipeline authorization"
        )
    return {
        "protocol_version": INDEX_COMMIT_RECEIPT_CONTRACT,
        "snapshot_id": graph.snapshot_id,
        "graph_hash": graph.graph_hash,
        "artifact_hashes": [list(item) for item in actual_artifacts],
        "standard_derivation_hashes": [
            list(item) for item in actual_derivations
        ],
        "static_aggregate_receipt_hashes": [
            list(item) for item in sorted(actual_receipts)
        ],
        "origin_manifest_sha256": authorization.origin_manifest_sha256,
    }


def build_index_commit_receipt(
    authorization_payload: Mapping[str, Any],
    *,
    run_id: str,
    run_payload_sha256: str,
) -> dict[str, Any]:
    payload = {
        **dict(authorization_payload),
        "run_id": run_id,
        "run_payload_sha256": run_payload_sha256,
    }
    payload["id"] = stable_id("index_commit_receipt", payload)
    return payload


__all__ = [
    "INDEX_COMMIT_RECEIPT_CONTRACT",
    "IndexAuthorizationError",
    "_consume_index_authorization",
    "_finalize_index_authorization",
    "_issue_origin_capability",
    "_new_origin_accumulator",
    "_record_extractor_result",
    "build_index_commit_receipt",
]
