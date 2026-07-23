"""Extractor SPI and deterministic merge pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..graph import CanonicalGraph
from ..model import (
    ArtifactRef,
    ArtifactSemanticAttestation,
    ArtifactSemanticClaim,
    AuthorityClass,
    DesignSnapshot,
    Derivation,
    Diagnostic,
    DiagnosticSeverity,
    Observation,
    ProjectManifest,
    VerificationResult,
    json_ready,
    reject_embedded_body_fields,
    stable_hash,
)


class ExtractionError(RuntimeError):
    pass


@dataclass(slots=True)
class ExtractionContext:
    project_root: Path
    manifest: ProjectManifest
    snapshot: DesignSnapshot
    artifacts: dict[str, ArtifactRef]
    allow_degraded: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def artifact_for_uri(self, uri: str) -> ArtifactRef | None:
        normalized = uri.replace("\\", "/")
        return next((item for item in self.artifacts.values() if item.uri == normalized), None)

    @staticmethod
    def authority_for(artifact: ArtifactRef, default: AuthorityClass) -> AuthorityClass:
        """Prevent fake/synthetic CI artifacts from masquerading as tool truth."""
        marker = str(artifact.metadata.get("fixture_authority", "")).casefold()
        return AuthorityClass.SYNTHETIC if marker in {"synthetic", "fake"} else default


@dataclass(slots=True)
class ExtractionResult:
    graph: CanonicalGraph
    observations: list[Observation] = field(default_factory=list)
    derivations: list[Derivation] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    verifications: list[VerificationResult] = field(default_factory=list)
    produced_artifacts: list[ArtifactRef] = field(default_factory=list)
    artifact_semantic_claims: list[ArtifactSemanticClaim] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    # Process-local proof of which live extractor produced each graph record.
    # It is deliberately non-serializable and is consumed once by Project.index.
    _index_origin_capability: object | None = field(
        default=None, repr=False,
    )


@runtime_checkable
class Extractor(Protocol):
    name: str
    version: str

    def supports(self, context: ExtractionContext) -> bool:
        ...

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        ...


class ExtractionPipeline:
    def __init__(self, extractors: list[Extractor], *,
                 extractor_identities: list[dict[str, Any]] | None = None):
        self.extractors = extractors
        if extractor_identities is not None and len(extractor_identities) != len(extractors):
            raise ValueError("extractor identities must correspond one-to-one with extractors")
        self.extractor_identities = extractor_identities

    @staticmethod
    def _identity(extractor: Extractor) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "name": str(extractor.name), "version": str(extractor.version),
            "implementation_module": type(extractor).__module__,
            "implementation_qualname": type(extractor).__qualname__,
        }
        runtime_identity = getattr(extractor, "runtime_identity", None)
        if callable(runtime_identity):
            identity["runtime"] = runtime_identity()
        # Canonical serialization both validates the shape and gives the
        # immutable attestation a deterministic producer fingerprint.
        stable_hash(identity)
        return identity

    @staticmethod
    def _attest_claims(
        result: ExtractionResult, *, context: ExtractionContext,
        extractor: Extractor, extractor_identity: dict[str, Any],
    ) -> list[ArtifactSemanticAttestation]:
        # A parser implementation identity is not an authorization to assert
        # conformance with an external language specification.  The public
        # v0.3 pipeline has no persisted, capability-issued semantic adapter
        # authorization contract yet, so every claim is rejected.  Default
        # MLIR/LLVM text extractors still emit structural evidence normally.
        if result.artifact_semantic_claims:
            raise ValueError(
                "no public extractor is authorized to issue language-spec "
                "semantic attestations"
            )
        return []

    def run(self, context: ExtractionContext) -> ExtractionResult:
        from .index_authorization import (
            _issue_origin_capability,
            _new_origin_accumulator,
            _record_extractor_result,
        )

        merged = ExtractionResult(graph=CanonicalGraph(snapshot_id=context.snapshot.id))
        origin_accumulator = _new_origin_accumulator(context.snapshot.id)
        semantic_attestations: dict[str, ArtifactSemanticAttestation] = {}
        for position, extractor in enumerate(self.extractors):
            if not extractor.supports(context):
                continue
            try:
                runtime_extractor_identity = self._identity(extractor)
                extractor_identity = (
                    self.extractor_identities[position]
                    if self.extractor_identities is not None
                    else runtime_extractor_identity
                )
                context.options["existing_graph"] = merged.graph
                result = extractor.extract(context)
                reject_embedded_body_fields(
                    result.graph.metadata, f"extractor {extractor.name} graph metadata",
                )
                reject_embedded_body_fields(
                    result.coverage, f"extractor {extractor.name} coverage",
                )
                attestations = self._attest_claims(
                    result, context=context, extractor=extractor,
                    extractor_identity=extractor_identity,
                )
                _record_extractor_result(
                    origin_accumulator, result.graph,
                    runtime_extractor_identity,
                    extractor,
                )
                if result.graph.metadata:
                    merged.graph.metadata.setdefault("extractor_metadata", {})[extractor.name] = result.graph.metadata
                for entity in result.graph.entities.values():
                    merged.graph.add_entity(entity)
                for relation in result.graph.relations.values():
                    merged.graph.add_relation(relation)
                merged.observations.extend(result.observations)
                merged.derivations.extend(result.derivations)
                merged.diagnostics.extend(result.diagnostics)
                merged.verifications.extend(result.verifications)
                merged.produced_artifacts.extend(result.produced_artifacts)
                for attestation in attestations:
                    semantic_attestations[attestation.id] = attestation
                merged.coverage[extractor.name] = result.coverage
                merged.capabilities.extend(result.capabilities)
            except Exception as exc:
                error_type = type(exc).__name__
                error_fingerprint = stable_hash({
                    "extractor": extractor.name,
                    "version": extractor.version,
                    "error_type": error_type,
                    "message": str(exc),
                })
                merged.diagnostics.append(Diagnostic(
                    snapshot_id=context.snapshot.id,
                    code="extractor.failed",
                    severity=DiagnosticSeverity.ERROR,
                    message=(f"{extractor.name} failed with {error_type}; "
                             f"details withheld (fingerprint {error_fingerprint[:16]})"),
                    stage="unknown",
                    metadata={"extractor": extractor.name, "version": extractor.version,
                              "error_type": error_type,
                              "error_fingerprint": error_fingerprint},
                ))
        try:
            from .directives import resolve_directives
            from .static_features import derive_static_features
            resolve_directives(merged)
            derive_static_features(merged)
        finally:
            context.options.pop("existing_graph", None)
        merged.capabilities = sorted(set(merged.capabilities))
        merged.graph.metadata["coverage"] = merged.coverage
        merged.graph.metadata["capabilities"] = merged.capabilities
        if semantic_attestations:
            merged.graph.metadata["artifact_semantic_attestations"] = [
                json_ready(semantic_attestations[key])
                for key in sorted(semantic_attestations)
            ]
        from ..static_aggregate import (
            STANDARD_STATIC_AGGREGATE_PREDICATES,
        )
        if any(
            item.predicate in STANDARD_STATIC_AGGREGATE_PREDICATES
            and str(item.completeness) == "complete"
            for item in merged.derivations
        ):
            merged._index_origin_capability = _issue_origin_capability(
                origin_accumulator, merged.graph,
            )
        reject_embedded_body_fields(merged.graph.metadata, "merged extractor graph metadata")
        return merged
