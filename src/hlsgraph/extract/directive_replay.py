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


DIRECTIVE_REPLAY_CONTRACT = "hlsgraph.directive_parser_replay.v4"


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
    scope_owner_id: str
    scope_ownership_hash: str
    operand_record_hash: str | None
    operand_owner_id: str | None
    operand_ownership_hash: str | None
    annotates_relation_hash: str
    requested_observation_hash: str
    port_owner_id: str | None
    port_owner_kind: str | None
    port_owner_record_hash: str | None
    port_owner_relation_hash: str | None
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


def _ownership_closure(
    graph: CanonicalGraph, entity: Entity,
) -> tuple[str, Entity, tuple[Any, ...]] | None:
    """Close one AST entity to one exact nearest function-like owner.

    Every incoming ``hls.contains`` relation participates in the ambiguity
    check.  A second partial, stale, non-AST, or non-function path is therefore
    a rejection, not evidence that can be ignored by filtering first.
    """

    def entity_valid(item: Entity) -> bool:
        return bool(
            item.snapshot_id == graph.snapshot_id
            and item.stage == "ast"
            and str(item.authority) == "static_fact"
            and str(item.completeness) == "complete"
        )

    if not entity_valid(entity):
        return None
    current = entity
    entities = [current]
    relations: list[Any] = []
    visited = {current.id}
    while current.kind not in {"hls.kernel", "hls.function"}:
        incoming = [
            relation for relation in graph.relations.values()
            if relation.kind == "hls.contains" and relation.dst == current.id
        ]
        if len(incoming) != 1:
            return None
        relation = incoming[0]
        parent = graph.entities.get(relation.src)
        if (
            parent is None
            or relation.snapshot_id != graph.snapshot_id
            or relation.stage != "ast"
            or str(relation.authority) != "static_fact"
            or str(relation.completeness) != "complete"
            or not entity_valid(parent)
            or parent.id in visited
        ):
            return None
        relations.append(relation)
        current = parent
        entities.append(current)
        visited.add(current.id)
    closure_hash = stable_hash({
        "entity_ids": [item.id for item in entities],
        "entity_hashes": [_record_hash(item) for item in entities],
        "relation_ids": [item.id for item in relations],
        "relation_hashes": [_record_hash(item) for item in relations],
        "owner_id": current.id,
    })
    return closure_hash, current, tuple(relations)


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


def _source_anchor_spelling_hash(
    project_root: Path,
    artifact: ArtifactRef,
    anchor: Any,
    *,
    require_pragma: bool,
) -> str | None:
    line_number = getattr(anchor, "start_line", None)
    end_line = getattr(anchor, "end_line", None)
    start_column = getattr(anchor, "start_column", None)
    end_column = getattr(anchor, "end_column", None)
    if (
        not isinstance(line_number, int)
        or line_number < 1
        or end_line != line_number
        or not isinstance(start_column, int)
        or start_column < 1
        or not isinstance(end_column, int)
        or end_column <= start_column
    ):
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
    source_line = lines[line_number - 1]
    if end_column - 1 > len(source_line):
        return None
    spelling = source_line[start_column - 1:end_column - 1]
    if not spelling.strip():
        return None
    if require_pragma and not spelling.lstrip().startswith("#"):
        return None
    # The raw spelling is intentionally consumed only inside this hash.
    return stable_hash({
        "artifact_sha256": artifact.sha256,
        "anchor": json_ready(anchor),
        "spelling": spelling,
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
    scope_closure = _ownership_closure(graph, scope)
    if scope_closure is None:
        return None
    scope_ownership_hash, scope_owner, scope_owner_relations = scope_closure
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
    spelling_hash = _source_anchor_spelling_hash(
        project_root,
        artifact,
        observation.anchor,
        require_pragma=scope_resolution == "source_ast",
    )
    if spelling_hash is None:
        return None
    operand_id, operand_role = _directive_operand(directive)
    operand = graph.entities.get(operand_id) if operand_id else None
    operand_closure = _ownership_closure(graph, operand) if operand is not None else None
    if kind in {"ARRAY_PARTITION", "STREAM", "INTERFACE"}:
        if operand is None or operand_id != scope_id or operand_closure is None:
            return None
    elif kind == "DEPENDENCE":
        options = directive.attrs.get("options")
        class_selector = isinstance(options, Mapping) and "class" in options
        if class_selector:
            if operand is not None:
                return None
        elif (operand is None or operand_id == scope_id
              or operand_closure is None):
            return None
    if (operand_closure is not None
            and operand_closure[1].id != scope_owner.id):
        return None
    port_owner: Entity | None = None
    port_owner_relation: Any | None = None
    if kind == "INTERFACE":
        if (scope.kind != "hls.port" or scope.stage != "ast"
                or str(scope.authority) != "static_fact"
                or str(scope.completeness) != "complete"):
            return None
        if (scope_owner.kind != "hls.kernel"
                or len(scope_owner_relations) != 1):
            return None
        port_owner_relation = scope_owner_relations[0]
        port_owner = scope_owner
        if (port_owner_relation.snapshot_id != directive.snapshot_id
                or port_owner_relation.stage != "ast"
                or str(port_owner_relation.authority) != "static_fact"
                or str(port_owner_relation.completeness) != "complete"
                or port_owner.snapshot_id != directive.snapshot_id
                or port_owner.stage != "ast"
                or str(port_owner.authority) != "static_fact"
                or str(port_owner.completeness) != "complete"):
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
        "scope_owner_id": scope_owner.id,
        "scope_ownership_hash": scope_ownership_hash,
        "operand_record_hash": operand_hash,
        "operand_owner_id": (
            operand_closure[1].id if operand_closure is not None else None
        ),
        "operand_ownership_hash": (
            operand_closure[0] if operand_closure is not None else None
        ),
        "annotates_relation_hash": relation_hash,
        "requested_observation_hash": observation_hash,
        "port_owner_record_hash": (
            _record_hash(port_owner) if port_owner is not None else None
        ),
        "port_owner_relation_hash": (
            _record_hash(port_owner_relation)
            if port_owner_relation is not None else None
        ),
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
        scope_owner_id=scope_owner.id,
        scope_ownership_hash=scope_ownership_hash,
        operand_record_hash=operand_hash,
        operand_owner_id=(
            operand_closure[1].id if operand_closure is not None else None
        ),
        operand_ownership_hash=(
            operand_closure[0] if operand_closure is not None else None
        ),
        annotates_relation_hash=relation_hash,
        requested_observation_hash=observation_hash,
        port_owner_id=port_owner.id if port_owner is not None else None,
        port_owner_kind=port_owner.kind if port_owner is not None else None,
        port_owner_record_hash=(
            _record_hash(port_owner) if port_owner is not None else None
        ),
        port_owner_relation_hash=(
            _record_hash(port_owner_relation)
            if port_owner_relation is not None else None
        ),
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
        or source.version != "4"
        or type(external) is not ExternalDirectiveExtractor
        or external.name != "directive.external"
        or external.version != "3"
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
    port_owner = (
        graph.entities.get(proof.port_owner_id) if proof.port_owner_id else None
    )
    scope_closure = _ownership_closure(graph, scope) if scope is not None else None
    operand_closure = (
        _ownership_closure(graph, operand) if operand is not None else None
    )
    if (
        directive is None
        or scope is None
        or _record_hash(directive) != proof.directive_record_hash
        or _record_hash(scope) != proof.scope_record_hash
        or scope_closure is None
        or scope_closure[0] != proof.scope_ownership_hash
        or scope_closure[1].id != proof.scope_owner_id
        or (
            proof.operand_record_hash is not None
            and (
                operand is None
                or operand_closure is None
                or _record_hash(operand) != proof.operand_record_hash
                or operand_closure[0] != proof.operand_ownership_hash
                or operand_closure[1].id != proof.operand_owner_id
            )
        )
        or (proof.operand_record_hash is None and operand_closure is not None)
        or (
            proof.port_owner_record_hash is not None
            and (
                port_owner is None
                or port_owner.kind != proof.port_owner_kind
                or _record_hash(port_owner) != proof.port_owner_record_hash
            )
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
    ownership = [
        relation for relation in graph.relations.values()
        if proof.port_owner_id is not None
        and relation.kind == "hls.contains"
        and relation.dst == proof.scope_id
    ]
    if (
        len(annotations) != 1
        or len(requested) != 1
        or _record_hash(annotations[0]) != proof.annotates_relation_hash
        or _record_hash(requested[0]) != proof.requested_observation_hash
        or (
            proof.port_owner_relation_hash is not None
            and (
                len(ownership) != 1
                or ownership[0].src != proof.port_owner_id
                or ownership[0].snapshot_id != graph.snapshot_id
                or ownership[0].stage != "ast"
                or str(ownership[0].authority) != "static_fact"
                or str(ownership[0].completeness) != "complete"
                or _record_hash(ownership[0])
                != proof.port_owner_relation_hash
            )
        )
        or (
            proof.port_owner_relation_hash is None and ownership
        )
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
