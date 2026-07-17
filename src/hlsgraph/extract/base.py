"""Extractor SPI and deterministic merge pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..graph import CanonicalGraph
from ..model import (
    ArtifactRef,
    AuthorityClass,
    DesignSnapshot,
    Derivation,
    Diagnostic,
    DiagnosticSeverity,
    Observation,
    ProjectManifest,
    VerificationResult,
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
    coverage: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)


@runtime_checkable
class Extractor(Protocol):
    name: str
    version: str

    def supports(self, context: ExtractionContext) -> bool:
        ...

    def extract(self, context: ExtractionContext) -> ExtractionResult:
        ...


class ExtractionPipeline:
    def __init__(self, extractors: list[Extractor]):
        self.extractors = extractors

    def run(self, context: ExtractionContext) -> ExtractionResult:
        merged = ExtractionResult(graph=CanonicalGraph(snapshot_id=context.snapshot.id))
        for extractor in self.extractors:
            if not extractor.supports(context):
                continue
            try:
                context.options["existing_graph"] = merged.graph
                result = extractor.extract(context)
                reject_embedded_body_fields(
                    result.graph.metadata, f"extractor {extractor.name} graph metadata",
                )
                reject_embedded_body_fields(
                    result.coverage, f"extractor {extractor.name} coverage",
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
        merged.capabilities = sorted(set(merged.capabilities))
        try:
            from .directives import resolve_directives
            resolve_directives(merged)
        finally:
            context.options.pop("existing_graph", None)
        merged.graph.metadata["coverage"] = merged.coverage
        merged.graph.metadata["capabilities"] = merged.capabilities
        reject_embedded_body_fields(merged.graph.metadata, "merged extractor graph metadata")
        return merged
