"""Fail-closed replay proof for source and external HLS directives.

Directive entities are public data objects, not signatures.  A caller can
construct one (and an ``hls.annotates`` relation) without ever running the
standard source extractor.  This module therefore re-runs the two fixed,
public directive parsers over the immutable snapshot inputs before retrieval
may mint any directive-source capability.

The replay is deliberately local and ephemeral.  It stores hashes of the
matched records and source spelling, never source text, and it does not write
an attestation back into the caller-controlled graph or ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..graph import CanonicalGraph
from ..manifest import project_path
from ..model import (
    ArtifactRef,
    DesignSnapshot,
    DiagnosticSeverity,
    Entity,
    Observation,
    ProjectManifest,
    json_ready,
    stable_hash,
    stable_id,
)
from .base import ExtractionContext, ExtractionPipeline
from .directives import ExternalDirectiveExtractor
from .source import LibClangExtractor


DIRECTIVE_REPLAY_CONTRACT = "hlsgraph.directive_parser_replay.v1"


@dataclass(frozen=True, slots=True)
class DirectiveReplayProof:
    """One exact directive declaration reproduced by the fixed parsers."""

    directive_id: str
    directive_kind: str
    scope_id: str
    scope_kind: str
    scope_resolution: str
    operand_id: str | None
    operand_role: str | None
    source_artifact_id: str
    source_artifact_sha256: str
    source_artifact_size: int
    source_anchor_hash: str
    source_spelling_hash: str
    directive_record_hash: str
    scope_record_hash: str
    operand_record_hash: str | None
    annotates_relation_hash: str
    requested_observation_hash: str
    parser_identity: str
    replay_identity: str
    contract: str = DIRECTIVE_REPLAY_CONTRACT


@dataclass(frozen=True, slots=True)
class DirectiveReplayIndex:
    """Ephemeral replay result; an empty index is a fail-closed outcome."""

    proofs: Mapping[str, tuple[DirectiveReplayProof, ...]] = field(
        default_factory=dict,
    )
    replay_identity: str | None = None
    failure_reason: str | None = None

    @classmethod
    def failed(cls, reason: str) -> "DirectiveReplayIndex":
        return cls(failure_reason=reason)


def _record_hash(value: Any) -> str:
    return stable_hash(json_ready(value))


def _snapshot_identity_valid(
    manifest: ProjectManifest,
    snapshot: DesignSnapshot,
) -> bool:
    return bool(
        snapshot.id == stable_id("snapshot", snapshot.identity_payload(), 32)
        and stable_hash(manifest.identity_payload()) == snapshot.manifest_hash
        and stable_hash(manifest.build) == snapshot.build_hash
        and stable_hash(manifest.target) == snapshot.target_hash
        and stable_hash(manifest.constraints) == snapshot.constraint_hash
        and stable_hash({
            "toolchains": manifest.toolchains,
            "stage_toolchains": manifest.stage_toolchains,
        }) == snapshot.toolchain_hash
    )


def _source_line_hash(
    project_root: Path,
    artifact: ArtifactRef,
    line_number: int | None,
) -> str | None:
    if not isinstance(line_number, int) or line_number < 1:
        return None
    try:
        data = project_path(project_root, artifact.uri).read_bytes()
        if b"\x00" in data:
            return None
        lines = data.decode("utf-8-sig").splitlines()
    except (OSError, UnicodeError, ValueError):
        return None
    if line_number > len(lines):
        return None
    # The raw spelling is intentionally consumed only inside this hash.
    return stable_hash({
        "artifact_sha256": artifact.sha256,
        "line": line_number,
        "spelling": lines[line_number - 1],
    })


def _directive_operand(
    directive: Entity,
) -> tuple[str | None, str | None]:
    kind = str(
        directive.attrs.get("directive_kind") or directive.name
    ).upper()
    if kind == "INTERFACE":
        value = directive.attrs.get("port_id")
        return (value, "port_id") if isinstance(value, str) and value else (None, None)
    if kind in {"ARRAY_PARTITION", "STREAM", "DEPENDENCE"}:
        value = directive.attrs.get("variable_id")
        return (value, "variable_id") if isinstance(value, str) and value else (None, None)
    return None, None


def _proof_for_directive(
    *,
    graph: CanonicalGraph,
    directive: Entity,
    observations: Sequence[Observation],
    artifacts: Mapping[str, ArtifactRef],
    project_root: Path,
    parser_identity: str,
) -> DirectiveReplayProof | None:
    if (
        directive.kind != "hls.directive"
        or directive.stage != "source"
        or str(directive.authority) != "declared_constraint"
        or str(directive.completeness) != "complete"
        or directive.attrs.get("directive_instance_id") != directive.id
    ):
        return None
    kind = str(
        directive.attrs.get("directive_kind") or directive.name
    ).upper()
    scope_id = directive.attrs.get("scope_id")
    scope_kind = directive.attrs.get("scope_kind")
    scope_resolution = directive.attrs.get("scope_resolution")
    if (
        not isinstance(scope_id, str)
        or not isinstance(scope_kind, str)
        or scope_resolution not in {"source_ast", "external_exact"}
    ):
        return None
    scope = graph.entities.get(scope_id)
    if scope is None or scope.kind != scope_kind:
        return None
    annotations = [
        relation
        for relation in graph.relations.values()
        if relation.kind == "hls.annotates"
        and relation.src == directive.id
        and relation.dst == scope_id
        and relation.stage == "source"
        and str(relation.authority) == "declared_constraint"
        and str(relation.completeness) == "complete"
        and relation.attrs.get("scope_node_id") == scope_id
        and relation.attrs.get("scope_resolution") == scope_resolution
    ]
    requested = [
        observation
        for observation in observations
        if observation.snapshot_id == directive.snapshot_id
        and observation.subject_id == directive.id
        and observation.predicate == "directive.requested"
        and observation.stage == "source"
        and str(observation.authority) == "declared_constraint"
        and str(observation.completeness) == "complete"
        and observation.run_id is None
    ]
    if len(annotations) != 1 or len(requested) != 1:
        return None
    annotation = annotations[0]
    observation = requested[0]
    if (
        len(directive.anchors) != 1
        or len(annotation.anchors) != 1
        or observation.anchor is None
        or observation.artifact_id is None
        or observation.anchor.artifact_id != observation.artifact_id
        or _record_hash(directive.anchors[0]) != _record_hash(observation.anchor)
        or _record_hash(annotation.anchors[0]) != _record_hash(observation.anchor)
        or _record_hash(observation.value)
        != _record_hash(directive.attrs.get("options") or True)
    ):
        return None
    artifact = artifacts.get(observation.artifact_id)
    if artifact is None or artifact.producer_run_id is not None:
        return None
    spelling_hash = _source_line_hash(
        project_root, artifact, observation.anchor.start_line,
    )
    if spelling_hash is None:
        return None
    operand_id, operand_role = _directive_operand(directive)
    operand = graph.entities.get(operand_id) if operand_id else None
    if kind in {"ARRAY_PARTITION", "STREAM", "INTERFACE"}:
        if operand is None or operand_id != scope_id:
            return None
    elif kind == "DEPENDENCE":
        if operand is None or operand_id == scope_id:
            return None
    directive_hash = _record_hash(directive)
    scope_hash = _record_hash(scope)
    operand_hash = _record_hash(operand) if operand is not None else None
    relation_hash = _record_hash(annotation)
    observation_hash = _record_hash(observation)
    replay_identity = stable_hash({
        "contract": DIRECTIVE_REPLAY_CONTRACT,
        "parser_identity": parser_identity,
        "directive_record_hash": directive_hash,
        "scope_record_hash": scope_hash,
        "operand_record_hash": operand_hash,
        "annotates_relation_hash": relation_hash,
        "requested_observation_hash": observation_hash,
        "source_artifact_id": artifact.id,
        "source_artifact_sha256": artifact.sha256,
        "source_artifact_size": artifact.size,
        "source_spelling_hash": spelling_hash,
    })
    return DirectiveReplayProof(
        directive_id=directive.id,
        directive_kind=kind,
        scope_id=scope_id,
        scope_kind=scope_kind,
        scope_resolution=str(scope_resolution),
        operand_id=operand_id,
        operand_role=operand_role,
        source_artifact_id=artifact.id,
        source_artifact_sha256=artifact.sha256,
        source_artifact_size=artifact.size,
        source_anchor_hash=_record_hash(observation.anchor),
        source_spelling_hash=spelling_hash,
        directive_record_hash=directive_hash,
        scope_record_hash=scope_hash,
        operand_record_hash=operand_hash,
        annotates_relation_hash=relation_hash,
        requested_observation_hash=observation_hash,
        parser_identity=parser_identity,
        replay_identity=replay_identity,
    )


def replay_directive_declarations(
    *,
    project_root: Path,
    manifest: ProjectManifest,
    snapshot: DesignSnapshot,
    artifacts: Mapping[str, ArtifactRef],
    artifact_bytes_valid: Callable[[ArtifactRef], bool],
) -> DirectiveReplayIndex:
    """Reparse exact snapshot inputs with the fixed non-degraded adapters."""
    if not _snapshot_identity_valid(manifest, snapshot):
        return DirectiveReplayIndex.failed("snapshot_identity_invalid")
    base_artifacts = {
        artifact.id: artifact
        for artifact in artifacts.values()
        if artifact.producer_run_id is None
    }
    by_uri: dict[str, list[ArtifactRef]] = {}
    for artifact in base_artifacts.values():
        by_uri.setdefault(artifact.uri, []).append(artifact)
    if (
        set(by_uri) != set(snapshot.artifact_hashes)
        or any(len(values) != 1 for values in by_uri.values())
        or any(
            values[0].sha256 != snapshot.artifact_hashes[uri]
            for uri, values in by_uri.items()
        )
        or not base_artifacts
    ):
        return DirectiveReplayIndex.failed("snapshot_artifact_closure_invalid")
    # Validate every immutable input, not merely the directive's own file:
    # preprocessing and exact scope depend on headers, macros, and external
    # Tcl/config inputs as well.
    if not all(artifact_bytes_valid(item) for item in base_artifacts.values()):
        return DirectiveReplayIndex.failed("snapshot_input_bytes_invalid")

    source = LibClangExtractor()
    external = ExternalDirectiveExtractor()
    if (
        type(source) is not LibClangExtractor
        or source.name != "source.libclang"
        or source.version != "2"
        or type(external) is not ExternalDirectiveExtractor
        or external.name != "directive.external"
        or external.version != "1"
        or not source.available()
    ):
        return DirectiveReplayIndex.failed("fixed_parser_unavailable")
    runtime = source.runtime_identity()
    if runtime.get("available") is not True:
        return DirectiveReplayIndex.failed("fixed_parser_runtime_unavailable")
    parser_identity = stable_hash({
        "contract": DIRECTIVE_REPLAY_CONTRACT,
        "source": {
            "name": source.name,
            "version": source.version,
            "implementation_module": type(source).__module__,
            "implementation_qualname": type(source).__qualname__,
            "runtime": runtime,
        },
        "external": {
            "name": external.name,
            "version": external.version,
            "implementation_module": type(external).__module__,
            "implementation_qualname": type(external).__qualname__,
        },
    })
    context = ExtractionContext(
        project_root=project_root,
        manifest=manifest,
        snapshot=snapshot,
        artifacts=base_artifacts,
        allow_degraded=False,
        options={},
    )
    try:
        result = ExtractionPipeline([source, external]).run(context)
    except Exception:
        # Retrieval is read-only.  Parser/resolver failures are represented by
        # absence of the capability rather than escaping as query failures or
        # persisting attacker-influenced exception text.
        return DirectiveReplayIndex.failed("fixed_parser_replay_failed")
    source_metadata = result.graph.metadata.get("extractor_metadata", {}).get(
        source.name, {},
    )
    if (
        source_metadata.get("source_backend") != source.name
        or source_metadata.get("fidelity") != "ast"
        or "source.ast" not in result.capabilities
        or "directive.source_scope" not in result.capabilities
        or "source.degraded" in result.capabilities
        or any(
            item.severity
            in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}
            for item in result.diagnostics
        )
    ):
        return DirectiveReplayIndex.failed("fixed_parser_replay_incomplete")
    # Close the parse against the same live bytes a second time.  A changed,
    # replaced, linked, or missing input invalidates the entire ephemeral proof.
    if not all(artifact_bytes_valid(item) for item in base_artifacts.values()):
        return DirectiveReplayIndex.failed("snapshot_input_changed_during_replay")

    proofs: dict[str, list[DirectiveReplayProof]] = {}
    for directive in sorted(result.graph.entities.values(), key=lambda item: item.id):
        if directive.kind != "hls.directive":
            continue
        proof = _proof_for_directive(
            graph=result.graph,
            directive=directive,
            observations=result.observations,
            artifacts=base_artifacts,
            project_root=project_root,
            parser_identity=parser_identity,
        )
        if proof is not None:
            proofs.setdefault(proof.directive_id, []).append(proof)
    frozen = {
        key: tuple(sorted(values, key=lambda item: item.replay_identity))
        for key, values in sorted(proofs.items())
    }
    # ``_proof_for_directive`` reads the exact source line to bind the raw
    # spelling hash.  Revalidate once more after those reads so a replacement
    # between parser completion and proof construction cannot be accepted.
    if not all(artifact_bytes_valid(item) for item in base_artifacts.values()):
        return DirectiveReplayIndex.failed("snapshot_input_changed_during_proof")
    replay_identity = stable_hash({
        "contract": DIRECTIVE_REPLAY_CONTRACT,
        "snapshot_id": snapshot.id,
        "parser_identity": parser_identity,
        "proofs": [
            proof.replay_identity
            for key in sorted(frozen)
            for proof in frozen[key]
        ],
    })
    return DirectiveReplayIndex(
        proofs=frozen,
        replay_identity=replay_identity,
    )


def match_directive_replay(
    index: DirectiveReplayIndex,
    *,
    graph: CanonicalGraph,
    observations: Sequence[Observation],
    directive_id: str,
) -> DirectiveReplayProof | None:
    """Match a caller-visible record to exactly one independently replayed fact."""
    candidates = index.proofs.get(directive_id, ())
    if len(candidates) != 1:
        return None
    proof = candidates[0]
    directive = graph.entities.get(directive_id)
    scope = graph.entities.get(proof.scope_id)
    operand = graph.entities.get(proof.operand_id) if proof.operand_id else None
    if (
        directive is None
        or scope is None
        or _record_hash(directive) != proof.directive_record_hash
        or _record_hash(scope) != proof.scope_record_hash
        or (
            proof.operand_record_hash is not None
            and (operand is None or _record_hash(operand) != proof.operand_record_hash)
        )
    ):
        return None
    annotations = [
        relation
        for relation in graph.relations.values()
        if relation.kind == "hls.annotates" and relation.src == directive_id
    ]
    requested = [
        observation
        for observation in observations
        if observation.subject_id == directive_id
        and observation.predicate == "directive.requested"
    ]
    if (
        len(annotations) != 1
        or len(requested) != 1
        or _record_hash(annotations[0]) != proof.annotates_relation_hash
        or _record_hash(requested[0]) != proof.requested_observation_hash
    ):
        return None
    return proof


__all__ = [
    "DIRECTIVE_REPLAY_CONTRACT",
    "DirectiveReplayIndex",
    "DirectiveReplayProof",
    "match_directive_replay",
    "replay_directive_declarations",
]
